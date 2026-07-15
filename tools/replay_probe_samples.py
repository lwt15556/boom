from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

adaptive_frames = importlib.import_module("utils.adaptive_frames")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay saved probe decisions against the adaptive-frame rule.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "_debug" / "screenshots" / "probes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "_debug" / "reports" / "adaptive_frames.json",
    )
    args = parser.parse_args()

    report = adaptive_frames.evaluate_probe_sample_directory(args.samples)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "samples_root": str(args.samples.resolve(strict=False)),
        **report.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.safe_to_enable else 1


if __name__ == "__main__":
    raise SystemExit(main())
