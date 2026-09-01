"""Score recorded vision samples without touching runtime state.

Usage:
    .venv\\Scripts\\python.exe tools\\evaluate_vision_samples.py
    .venv\\Scripts\\python.exe tools\\evaluate_vision_samples.py --root _debug --json report.json

Each probe directory may contain ``result.json``.  A human-reviewed sample can
add ``review.json`` with ``{"expected": "hit"}`` (or ``miss``/``unknown``).
The report keeps detector decisions and reviewed accuracy separate so an
unlabelled capture never looks like a correct prediction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def collect_samples(root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for result_path in sorted(root.rglob("result.json")):
        result = _read_json(result_path)
        if result is None:
            continue
        review = _read_json(result_path.with_name("review.json")) or {}
        decision = str(result.get("decision", "unknown")).lower()
        expected = review.get("expected")
        expected = str(expected).lower() if expected is not None else None
        evidence_kind = result.get("evidence_kind")
        if evidence_kind is None:
            frame_kinds = []
            for frame in result.get("frames", []):
                if isinstance(frame, dict):
                    frame_result = frame.get("result", {})
                    if isinstance(frame_result, dict):
                        frame_kinds.append(str(frame_result.get("evidence_kind", "unknown")))
            evidence_kind = Counter(frame_kinds).most_common(1)[0][0] if frame_kinds else "unknown"
        samples.append(
            {
                "path": str(result_path.parent),
                "level": result.get("level"),
                "cell": result.get("cell"),
                "decision": decision,
                "expected": expected,
                "evidence_kind": str(evidence_kind),
                "confidence": result.get("confidence"),
            }
        )
    return samples


def build_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(sample["decision"] for sample in samples)
    evidence = Counter(sample["evidence_kind"] for sample in samples)
    labelled = [sample for sample in samples if sample["expected"] in {"hit", "miss", "unknown"}]
    correct = sum(sample["decision"] == sample["expected"] for sample in labelled)
    by_level: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for sample in labelled:
        level = str(sample["level"])
        by_level[level]["total"] += 1
        by_level[level]["correct"] += int(sample["decision"] == sample["expected"])
    return {
        "sample_count": len(samples),
        "labelled_count": len(labelled),
        "unlabelled_count": len(samples) - len(labelled),
        "decisions": dict(sorted(decisions.items())),
        "evidence_kinds": dict(sorted(evidence.items())),
        "unknown_rate": (decisions.get("unknown", 0) / len(samples)) if samples else 0.0,
        "review_accuracy": (correct / len(labelled)) if labelled else None,
        "by_level": dict(sorted(by_level.items())),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("_debug"))
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()
    report = build_report(collect_samples(args.root))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_path:
        args.json_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
