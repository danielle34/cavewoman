"""Resolve where the results tree lives.

Standalone analysis scripts read generations from a results directory. That
directory is the repo's own results/ by default, but anyone who unpacked the
data bundle elsewhere, or ran their own experiments to another path, needs to
redirect it. Precedence:

  1. --results-root PATH  on the command line
  2. CAVEWOMAN_RESULTS    in the environment
  3. the default passed by the caller (usually <repo>/results)

Usage:
    from release_paths import results_root
    RESULTS = results_root(_REPO_ROOT / "results")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def results_root(default: Path | str) -> Path:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--results-root", "--results_root") and i + 1 < len(argv):
            return Path(os.path.expanduser(argv[i + 1])).resolve()
        if a.startswith("--results-root=") or a.startswith("--results_root="):
            return Path(os.path.expanduser(a.split("=", 1)[1])).resolve()
    env = os.environ.get("CAVEWOMAN_RESULTS")
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return Path(default).resolve()
