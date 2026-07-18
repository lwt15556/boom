from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.probe_evaluation import evaluate_probe_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate manually labeled probe samples.")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "_debug" / "screenshots" / "probes",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate_probe_samples(args.root).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
