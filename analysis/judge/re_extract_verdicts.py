"""Re-extract verdicts from existing judge JSONLs using the current
`extract_verdict()` parser. Useful when:

, the parser was updated and we want to re-apply it without
    re-running GPU inference,
, we want a quick parse-rate sanity check on a finished cell.

The script reads each input JSONL, applies extract_verdict() to the
`judge_raw_output` field of every record, updates `judge_verdict`, and
writes back to the same file (atomic via tmpfile + rename).

Usage:
    python re_extract_verdicts.py --files ./results/judge_<tag>/*.jsonl
    python re_extract_verdicts.py --files <one_file>.jsonl --no-write   # dry run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Make the judge package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import (  # noqa: E402
    extract_verdict, PAIR_LABELS, RECOVERY_LABELS,
)


def re_extract_file(path: Path, write: bool) -> dict:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        return {"path": str(path), "total": 0, "before_none": 0, "after_none": 0,
                "changed": 0, "delta_verdicts": Counter()}

    # Pick label set from the file's mode (consistent within a file)
    mode = records[0].get("judge_mode", "pair")
    labels = PAIR_LABELS if mode == "pair" else RECOVERY_LABELS

    before_none = sum(1 for r in records if r.get("judge_verdict") is None)
    changed = 0
    after = Counter()
    for r in records:
        old = r.get("judge_verdict")
        new = extract_verdict(r.get("judge_raw_output", ""), labels)
        if new != old:
            r["judge_verdict"] = new
            changed += 1
        after[new] += 1
    after_none = after[None]

    if write:
        # Atomic write: tmp file in same dir, then rename
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(path.parent), delete=False, suffix=".tmp"
        ) as tmp:
            for r in records:
                tmp.write(json.dumps(r) + "\n")
            tmp_name = tmp.name
        os.replace(tmp_name, path)

    return {
        "path": str(path),
        "mode": mode,
        "total": len(records),
        "before_none": before_none,
        "after_none": after_none,
        "changed": changed,
        "delta_verdicts": after,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True,
                    help="One or more JSONL files to re-parse.")
    ap.add_argument("--no-write", action="store_true",
                    help="Dry run: report only, do not overwrite.")
    args = ap.parse_args()

    grand_total = 0
    grand_before_none = 0
    grand_after_none = 0
    grand_changed = 0
    print(f"{'file':<70}{'mode':<10}{'n':>6}{'none→':>8}{'→none':>8}{'changed':>8}")
    print("-" * 110)
    for path_str in args.files:
        path = Path(path_str)
        if not path.exists():
            print(f"  (missing) {path}")
            continue
        s = re_extract_file(path, write=not args.no_write)
        grand_total += s["total"]
        grand_before_none += s["before_none"]
        grand_after_none += s["after_none"]
        grand_changed += s["changed"]
        print(f"{path.name:<70}{s.get('mode',''):<10}{s['total']:>6}"
              f"{s['before_none']:>8}{s['after_none']:>8}{s['changed']:>8}")

    print("-" * 110)
    print(f"{'TOTAL':<80}{grand_total:>6}{grand_before_none:>8}{grand_after_none:>8}{grand_changed:>8}")
    if grand_total > 0:
        old_rate = 100 * (grand_total, grand_before_none) / grand_total
        new_rate = 100 * (grand_total, grand_after_none) / grand_total
        print(f"  extraction rate: {old_rate:.1f}% → {new_rate:.1f}%  "
              f"({grand_changed} records changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
