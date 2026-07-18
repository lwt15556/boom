from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


VALID_DECISIONS = frozenset({"hit", "miss", "unknown"})
VALID_GROUND_TRUTH = frozenset({"hit", "miss"})


@dataclass(frozen=True)
class ProbeEvaluationReport:
    total_samples: int
    labeled_samples: int
    unlabeled_samples: int
    invalid_samples: int
    true_hits: int
    false_hits: int
    missed_hits: int
    true_misses: int
    unknown: int
    hit_precision: float
    hit_recall: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _samples_from_directory(root: str | Path) -> list[Mapping[str, Any]]:
    samples: list[Mapping[str, Any]] = []
    for result_path in sorted(Path(root).rglob("result.json")):
        result = _load_json(result_path)
        if result is None:
            continue
        merged = dict(result)
        status = _load_json(result_path.with_name("status.json"))
        if (
            status is not None
            and status.get("stage") == "complete"
            and status.get("decision") in VALID_DECISIONS
        ):
            merged["decision"] = status["decision"]
        review = _load_json(result_path.with_name("review.json"))
        if review is not None and "ground_truth" in review:
            merged["ground_truth"] = review["ground_truth"]
        samples.append(merged)
    return samples


def evaluate_probe_samples(
    samples_or_root: Iterable[Mapping[str, object]] | str | Path,
) -> ProbeEvaluationReport:
    if isinstance(samples_or_root, (str, Path)):
        samples = _samples_from_directory(samples_or_root)
    else:
        samples = list(samples_or_root)

    labeled = invalid = true_hits = false_hits = 0
    missed_hits = true_misses = unknown = 0
    for sample in samples:
        decision = str(sample.get("decision", "")).strip().lower()
        truth = str(sample.get("ground_truth", "")).strip().lower()
        if decision not in VALID_DECISIONS:
            invalid += 1
            continue
        if not truth:
            continue
        if truth not in VALID_GROUND_TRUTH:
            invalid += 1
            continue

        labeled += 1
        if decision == "unknown":
            unknown += 1
        elif decision == "hit" and truth == "hit":
            true_hits += 1
        elif decision == "hit" and truth == "miss":
            false_hits += 1
        elif decision == "miss" and truth == "hit":
            missed_hits += 1
        else:
            true_misses += 1

    total = len(samples)
    return ProbeEvaluationReport(
        total_samples=total,
        labeled_samples=labeled,
        unlabeled_samples=total - labeled - invalid,
        invalid_samples=invalid,
        true_hits=true_hits,
        false_hits=false_hits,
        missed_hits=missed_hits,
        true_misses=true_misses,
        unknown=unknown,
        hit_precision=_safe_ratio(true_hits, true_hits + false_hits),
        hit_recall=_safe_ratio(true_hits, true_hits + missed_hits),
    )


__all__ = ["ProbeEvaluationReport", "evaluate_probe_samples"]
