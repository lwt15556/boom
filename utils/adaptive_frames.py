from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path


ADAPTIVE_HIT_MIN_FRAMES = 3


@dataclass(frozen=True)
class AdaptiveReplayReport:
    total_samples: int
    eligible_samples: int
    mismatches: int
    safe_to_enable: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def can_stop_after_stable_hit_frames(
    frame_records: Sequence[Mapping[str, object]],
    *,
    min_frames: int = ADAPTIVE_HIT_MIN_FRAMES,
) -> bool:
    min_frames = max(1, int(min_frames))
    if len(frame_records) < min_frames:
        return False

    for record in frame_records[:min_frames]:
        if record.get("dynamic_hit_vetoed") is not False:
            return False
        result = record.get("result")
        if not isinstance(result, Mapping):
            return False
        if result.get("state") != "hit" or result.get("evidence_vetoed") is not False:
            return False
    return True


def evaluate_adaptive_hit_replay(
    samples: Iterable[Mapping[str, object]],
) -> AdaptiveReplayReport:
    total_samples = 0
    eligible_samples = 0
    mismatches = 0
    for sample in samples:
        total_samples += 1
        frames = sample.get("frames")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            continue
        if not can_stop_after_stable_hit_frames(frames):
            continue
        eligible_samples += 1
        if sample.get("decision") != "hit":
            mismatches += 1

    return AdaptiveReplayReport(
        total_samples=total_samples,
        eligible_samples=eligible_samples,
        mismatches=mismatches,
        safe_to_enable=eligible_samples > 0 and mismatches == 0,
    )


def evaluate_probe_sample_directory(root: str | Path) -> AdaptiveReplayReport:
    samples: list[Mapping[str, object]] = []
    for result_path in Path(root).rglob("result.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            samples.append(payload)
    return evaluate_adaptive_hit_replay(samples)
