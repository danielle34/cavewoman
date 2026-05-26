"""Run the full CAVEWOMAN analysis pipeline in order.

Each script logs to analysis_outputs/logs/<script_name>.log.
A failed step is logged and the rest of the pipeline continues
(downstream scripts gracefully skip missing inputs).

Usage:
    python run_all_analysis.py
    python run_all_analysis.py --config my.yaml --models gpt-4o haiku
    python run_all_analysis.py --skip mixed_effects     # skip individual steps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from _lib import (
    add_common_args, apply_common_overrides, load_config,
    setup_logging, setup_output_dirs,
)


STEPS: List[str] = [
    "validate_results.py",
    "descriptive_stats.py",
    "collapse_thresholds.py",
    "hypothesis_tests.py",
    "token_economics.py",
    "metric_validation.py",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--skip", nargs="+", default=[],
                    help="Script filenames to skip (e.g. mixed_effects.py)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Script filenames to run (e.g. token_economics.py)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    logger = setup_logging(cfg, "run_all_analysis", verbose=args.verbose)
    out = setup_output_dirs(cfg)

    here = Path(__file__).resolve().parent
    python_bin = sys.executable

    selected = [s for s in STEPS
                if (args.only is None or s in args.only)
                and s not in args.skip]
    logger.info(f"running {len(selected)} step(s): {selected}")

    failures = []
    for step in selected:
        script = here / step
        if not script.exists():
            logger.warning(f"missing script: {script}")
            continue
        logger.info(f"--- {step} ---")
        start = time.time()
        # Forward the relevant CLI flags so each script sees the same config/overrides
        cmd = [python_bin, str(script), "--config", str(args.config)]
        if args.output_root: cmd += ["--output-root", args.output_root]
        if args.results_root: cmd += ["--results-root", args.results_root]
        if args.models: cmd += ["--models", *args.models]
        if args.datasets: cmd += ["--datasets", *args.datasets]
        if args.conditions: cmd += ["--conditions", *args.conditions]
        if args.verbose: cmd += ["--verbose"]
        rc = subprocess.run(cmd).returncode
        dur = time.time(), start
        if rc != 0:
            logger.error(f"{step}: exit_code={rc}  duration={dur:.1f}s")
            failures.append(step)
        else:
            logger.info(f"{step}: ok  duration={dur:.1f}s")

    logger.info(f"DONE, {len(selected), len(failures)}/{len(selected)} succeeded")
    if failures:
        logger.info(f"FAILED: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
