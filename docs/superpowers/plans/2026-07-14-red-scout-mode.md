# Red Scout Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable red-bomb scouting mode that performs a configurable number of offline multi-cell probes per level, verifies that red ammunition was not consumed, then uses blue bombs to commit the discovered submarine cells.

**Architecture:** Keep red observations inside `SubmarineStrategy` as a separate temporary knowledge layer so they can guide blue targeting without prematurely completing the level. Put pure red-button, ammunition-fingerprint, footprint, analyzer, and planner logic in `utils/red_scout.py`; keep emulator transaction orchestration in `main.py`; extend `AdbController` with an explicit isolation verification API and fail closed after any post-click safety error.

**Tech Stack:** Python 3.11, OpenCV, NumPy, PyQt6, `unittest`, ADB/iptables, existing template matching and probe protocol helpers.

---

## File Map

- Create `utils/red_scout.py`: settings, button location, ammunition fingerprints, multi-cell analyzer, learned footprint, and scout-center planner.
- Create `tests/test_red_scout.py`: pure tests for settings, fingerprints, analyzer, footprint, and planner.
- Create `template/red_bomb_button.png`: repository template copied from the user-provided red-bomb crop.
- Modify `utils/submarine_strategy.py`: temporary scout observations, pending-scout priority, scout-miss pruning, and board states.
- Modify `tests/test_submarine_strategy.py`: tests proving scout data guides blue shots without marking the level complete.
- Create `utils/wreck_detection.py`: move existing positive wreck/red-marker detection out of `main.py` so blue and red analyzers share it.
- Modify `main.py`: import extracted wreck helpers, execute the red scout transaction, integrate per-level scout phase, and latch fail-closed network state.
- Modify `tests/test_main_flow.py`: transaction ordering, safety failure, per-level integration, and blue-only compatibility tests.
- Modify `utils/adb_control.py`: verify IPv4/IPv6 isolation before a red grid click.
- Modify `tests/test_adb_control.py`: isolation verification tests.
- Modify `tools/control_panel.py`: mode selector, count stepper, environment propagation, status, board colors, and locked controls.
- Modify `tests/test_control_panel.py`: mode/count environment and formatter tests.
- Modify `README.md`: document both modes and the red safety stop behavior.

The worktree already contains unrelated and user-authored changes. Before every commit step, inspect `git diff -- <paths>`. If a touched tracked file contains pre-existing changes that cannot be isolated non-interactively, do not commit that task; leave it in the worktree and report it at the final checkpoint. Never stage unrelated files.

### Task 1: Parse Probe Mode And Scout Count

**Files:**
- Create: `utils/red_scout.py`
- Create: `tests/test_red_scout.py`

- [ ] **Step 1: Write failing settings tests**

```python
import unittest

from utils.red_scout import ProbeMode, load_red_scout_settings


class RedScoutSettingsTest(unittest.TestCase):
    def test_defaults_to_blue_only(self):
        settings = load_red_scout_settings({})

        self.assertEqual(settings.mode, ProbeMode.BLUE_ONLY)
        self.assertEqual(settings.count, 2)

    def test_reads_red_mode_and_configurable_count(self):
        settings = load_red_scout_settings(
            {
                "BBMA_PROBE_MODE": "red_scout",
                "BBMA_RED_SCOUT_COUNT": "3",
            }
        )

        self.assertEqual(settings.mode, ProbeMode.RED_SCOUT)
        self.assertEqual(settings.count, 3)

    def test_invalid_values_fall_back_without_enabling_red_mode(self):
        settings = load_red_scout_settings(
            {
                "BBMA_PROBE_MODE": "invalid",
                "BBMA_RED_SCOUT_COUNT": "99",
            }
        )

        self.assertEqual(settings.mode, ProbeMode.BLUE_ONLY)
        self.assertEqual(settings.count, 2)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_red_scout.RedScoutSettingsTest
```

Expected: import failure because `utils.red_scout` does not exist.

- [ ] **Step 3: Implement settings parsing**

Create `utils/red_scout.py` with:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ProbeMode(str, Enum):
    BLUE_ONLY = "blue_only"
    RED_SCOUT = "red_scout"


@dataclass(frozen=True)
class RedScoutSettings:
    mode: ProbeMode = ProbeMode.BLUE_ONLY
    count: int = 2


def load_red_scout_settings(
    environment: Mapping[str, str] | None = None,
) -> RedScoutSettings:
    values = os.environ if environment is None else environment
    raw_mode = str(values.get("BBMA_PROBE_MODE", ProbeMode.BLUE_ONLY.value)).strip()
    try:
        mode = ProbeMode(raw_mode)
    except ValueError:
        return RedScoutSettings()

    raw_count = str(values.get("BBMA_RED_SCOUT_COUNT", "2")).strip()
    try:
        count = int(raw_count)
    except ValueError:
        count = 2
    if not 1 <= count <= 10:
        count = 2
    return RedScoutSettings(mode=mode, count=count)
```

- [ ] **Step 4: Run the settings tests and verify GREEN**

Run the command from Step 2.

Expected: `Ran 3 tests` and `OK`.

- [ ] **Step 5: Commit the isolated new files**

```powershell
git add -- utils/red_scout.py tests/test_red_scout.py
git commit -m "feat: add red scout runtime settings"
```

### Task 2: Add Temporary Scout Knowledge To The Strategy

**Files:**
- Modify: `utils/submarine_strategy.py:45-245`
- Modify: `utils/submarine_strategy.py:272-690`
- Test: `tests/test_submarine_strategy.py`

- [ ] **Step 1: Write failing scout-observation tests**

Add to `SubmarineStrategyTest`:

```python
    def test_scout_hit_is_blue_priority_without_completing_ship(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits={(2, 2), (2, 3)}, misses=set())

        self.assertFalse(strategy.done)
        self.assertEqual(strategy.shots, {})
        self.assertIn(strategy.choose_next_cell(), {(2, 2), (2, 3)})

    def test_blue_result_replaces_scout_state(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits={(2, 2)}, misses={(0, 0)})

        strategy.report_result((2, 2), True)

        self.assertNotIn((2, 2), strategy.scout_observations)
        self.assertTrue(strategy.shots[(2, 2)])
        self.assertEqual(strategy.get_cell_states()[2][2], "hit")
        self.assertEqual(strategy.get_cell_states()[0][0], "scout_miss")

    def test_scout_miss_is_never_selected_or_used_in_placements(self):
        strategy = SubmarineStrategy(4, [2])
        strategy.report_scout_results(hits=set(), misses={(1, 1)})

        for _ in range(8):
            cell = strategy.choose_next_cell()
            self.assertNotEqual(cell, (1, 1))
            if cell is None:
                break
            strategy.report_result(cell, False)
```

- [ ] **Step 2: Run the strategy tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_submarine_strategy.SubmarineStrategyTest.test_scout_hit_is_blue_priority_without_completing_ship tests.test_submarine_strategy.SubmarineStrategyTest.test_blue_result_replaces_scout_state tests.test_submarine_strategy.SubmarineStrategyTest.test_scout_miss_is_never_selected_or_used_in_placements
```

Expected: attribute failure for `report_scout_results`.

- [ ] **Step 3: Implement the temporary scout layer**

In `SubmarineStrategy.__init__` add:

```python
        self.scout_observations: dict[Cell, bool] = {}
```

Add these methods:

```python
    def report_scout_results(
        self,
        *,
        hits: Iterable[Cell],
        misses: Iterable[Cell],
    ) -> None:
        hit_cells = {tuple(cell) for cell in hits}
        miss_cells = {tuple(cell) for cell in misses}
        overlap = hit_cells & miss_cells
        if overlap:
            raise ValueError(f"conflicting scout observations: {sorted(overlap)}")

        for cell in hit_cells | miss_cells:
            self._validate_cell(cell)
        for cell in hit_cells:
            if cell not in self.shots:
                self.scout_observations[cell] = True
        for cell in miss_cells:
            if cell not in self.shots:
                self.scout_observations[cell] = False
        self._hunt_residue_cache.clear()

    def get_scout_hit_cells(self) -> set[Cell]:
        return {cell for cell, hit in self.scout_observations.items() if hit}

    def get_scout_miss_cells(self) -> set[Cell]:
        return {cell for cell, hit in self.scout_observations.items() if not hit}

    def _known_cells(self) -> set[Cell]:
        return set(self.shots) | set(self.scout_observations)

    def _choose_pending_scout_hit(self) -> Optional[Cell]:
        pending = self.get_scout_hit_cells() - set(self.shots) - self.blocked_cells
        if not pending:
            return None

        def score(cell: Cell) -> tuple[int, float]:
            adjacent = sum(1 for neighbor in self._neighbors4(cell) if neighbor in pending)
            return adjacent, self._center_bonus(cell)

        return max(pending, key=score)
```

At the start of `report_result`, remove the temporary observation before writing the real result:

```python
        self.scout_observations.pop(cell, None)
```

At the start of `choose_next_cell`, before the `done` check, add:

```python
        pending_scout_hit = self._choose_pending_scout_hit()
        if pending_scout_hit is not None:
            return pending_scout_hit
```

Change `_miss_cells` to:

```python
    def _miss_cells(self) -> set[Cell]:
        real_misses = {cell for cell, hit in self.shots.items() if not hit}
        return real_misses | self.get_scout_miss_cells()
```

In `_choose_target_cell`, `_choose_oriented_cluster_extension`, `_choose_adjacent_to_recent_hit`, `_choose_hunt_cell`, and `_fallback_unshot_cell`, replace selection checks that only use `self.shots` with `self._known_cells()` so scout misses are never selected. Do not include scout hits in `_unconfirmed_hit_cells` or `_try_confirm_ships`.

Update `get_cell_states` in this order:

```python
        for (row, col), hit in self.scout_observations.items():
            states[row][col] = "scout_hit" if hit else "scout_miss"

        for (row, col), hit in self.shots.items():
            states[row][col] = "hit" if hit else "miss"
```

Keep confirmed ships as the final overlay.

- [ ] **Step 4: Run all strategy tests and verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_submarine_strategy
```

Expected: all strategy tests pass, including the three new tests.

- [ ] **Step 5: Commit only if the strategy diff is isolated**

```powershell
git diff -- utils/submarine_strategy.py tests/test_submarine_strategy.py
git add -- utils/submarine_strategy.py tests/test_submarine_strategy.py
git commit -m "feat: add temporary red scout observations"
```

If the diff includes pre-existing user changes, skip the `git add` and `git commit` commands.

### Task 3: Extract Shared Wreck Detection

**Files:**
- Create: `utils/wreck_detection.py`
- Modify: `main.py:68-120`
- Modify: `main.py:430-640`
- Test: `tests/test_main_flow.py`

- [ ] **Step 1: Add an import-compatibility test before moving code**

Add to `MainFlowTest`:

```python
    def test_main_uses_shared_wreck_detection_helpers(self):
        from utils import wreck_detection

        self.assertIs(self.main.red_hit_marker_visible, wreck_detection.red_hit_marker_visible)
        self.assertIs(
            self.main.visible_wreck_static_detected,
            wreck_detection.visible_wreck_static_detected,
        )
```

- [ ] **Step 2: Run the test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_flow.MainFlowTest.test_main_uses_shared_wreck_detection_helpers
```

Expected: import failure because `utils.wreck_detection` does not exist.

- [ ] **Step 3: Move the existing implementation without behavior changes**

Create `utils/wreck_detection.py` containing the existing constants and functions currently defined in `main.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import TEMPLATE_DIR
from utils.image_match import find_template_multi_scale

SUBMARINE_HIT_WRECK_TEMPLATE = TEMPLATE_DIR / "submarine_hit_wreck.png"
VISIBLE_WRECK_TEMPLATES = tuple(
    TEMPLATE_DIR / f"visible_wreck_{index}.png"
    for index in range(1, 4)
)
WRECK_TEMPLATE_THRESHOLD = 0.70
WRECK_GLOBAL_TEMPLATE_THRESHOLD = 0.90
WRECK_TEMPLATE_SCALES = (0.75, 0.9, 1.0, 1.1, 1.25)
RED_HIT_MARKER_MIN_AREA = 80
RED_HIT_MARKER_MIN_WIDTH = 12
RED_HIT_MARKER_MIN_HEIGHT = 8
FULL_SHIP_MIN_GRAY_RATIO = 0.50
FULL_SHIP_MIN_COMPONENT_AREA = 450
FULL_SHIP_MIN_COMPONENT_WIDTH = 35
FULL_SHIP_MIN_COMPONENT_HEIGHT = 20
FULL_SHIP_MAX_CENTER_OFFSET = 10
```

Move these functions with their current bodies unchanged:

- `wreck_template_visible`
- `red_hit_marker_visible`
- `detect_visible_wreck_cells`
- `visible_wreck_static_detected`

In `main.py`, import the moved names:

```python
from utils.wreck_detection import (
    VISIBLE_WRECK_TEMPLATES,
    detect_visible_wreck_cells,
    red_hit_marker_visible,
    visible_wreck_static_detected,
    wreck_template_visible,
)
```

Remove only the duplicated constants and function bodies from `main.py`. Keep `apply_wreck_template_confirmation` in `main.py` because it mutates the blue probe result.

- [ ] **Step 4: Run focused and full main-flow tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_flow.MainFlowTest.test_main_uses_shared_wreck_detection_helpers tests.test_main_flow.MainFlowTest.test_dynamic_hit_without_positive_visual_evidence_is_vetoed tests.test_main_flow.MainFlowTest.test_new_wreck_evidence_keeps_dynamic_hit
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the new module and isolated import change when safe**

```powershell
git add -- utils/wreck_detection.py
git diff -- main.py tests/test_main_flow.py
```

Commit only if the tracked-file diff can be isolated:

```powershell
git add -- main.py tests/test_main_flow.py
git commit -m "refactor: share wreck detection helpers"
```

### Task 4: Add Red Button Detection And Ammunition Fingerprints

**Files:**
- Create: `template/red_bomb_button.png`
- Modify: `utils/red_scout.py`
- Test: `tests/test_red_scout.py`

- [ ] **Step 1: Build a masked template from the provided binary crop**

```powershell
@'
import cv2
import numpy as np

source = cv2.imread(
    r"C:\Users\lwt\AppData\Local\Temp\codex-clipboard-d0e48224-6dd8-4d56-b475-19e12ba9829f.png",
    cv2.IMREAD_COLOR,
)
if source is None:
    raise SystemExit("red bomb source image was not found")
rgba = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
height, width = rgba.shape[:2]
rgba[:, :, 3] = 255
rgba[int(height * 0.60):height, int(width * 0.62):width, 3] = 0
if not cv2.imwrite(r"template\red_bomb_button.png", rgba):
    raise SystemExit("failed to write red bomb template")
'@ | .\.venv\Scripts\python.exe -
```

The transparent lower-right mask prevents the displayed ammunition number from becoming part of template matching, so the button remains detectable for counts other than `2`.

Verify it loads:

```powershell
.\.venv\Scripts\python.exe -c "import cv2; image=cv2.imread(r'template\red_bomb_button.png'); assert image is not None and image.size > 0; print(image.shape)"
```

Expected: a three-channel image shape is printed.

- [ ] **Step 2: Write failing visual-primitive tests**

Add tests using synthetic 1280 x 720 frames and `MatchResult`:

```python
import cv2
import numpy as np
from pathlib import Path

from utils.image_match import MatchResult
from utils.red_scout import (
    ammo_fingerprint_matches,
    build_ammo_fingerprint,
    red_bomb_selected,
)


class RedBombVisualTest(unittest.TestCase):
    def setUp(self):
        self.match = MatchResult(
            template_path=Path("red_bomb_button.png"),
            top_left=(1174, 620),
            bottom_right=(1261, 707),
            center=(1217, 663),
            score=0.99,
        )

    def test_ammo_fingerprint_ignores_button_highlight(self):
        first = np.zeros((720, 1280, 3), dtype=np.uint8)
        second = first.copy()
        cv2.putText(first, "2", (1247, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(second, "2", (1247, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(second, (1174, 620), (1261, 707), (255, 255, 255), 3)

        before = build_ammo_fingerprint([first, first, first], self.match)
        after = build_ammo_fingerprint([second, second, second], self.match)

        self.assertIsNotNone(before)
        self.assertTrue(ammo_fingerprint_matches(before, after))

    def test_different_number_fails_ammo_verification(self):
        two = np.zeros((720, 1280, 3), dtype=np.uint8)
        one = two.copy()
        cv2.putText(two, "2", (1247, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(one, "1", (1247, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        before = build_ammo_fingerprint([two, two, two], self.match)
        after = build_ammo_fingerprint([one, one, one], self.match)

        self.assertFalse(ammo_fingerprint_matches(before, after))

    def test_selected_button_requires_white_border(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertFalse(red_bomb_selected(image, self.match))
        cv2.rectangle(image, self.match.top_left, self.match.bottom_right, (255, 255, 255), 3)
        self.assertTrue(red_bomb_selected(image, self.match))
```

- [ ] **Step 3: Run the tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_red_scout.RedBombVisualTest
```

Expected: import failures for the fingerprint functions.

- [ ] **Step 4: Implement button and fingerprint helpers**

Add to `utils/red_scout.py`:

```python
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from config import TEMPLATE_DIR
from utils.image_match import MatchResult, find_template_multi_scale

RED_BOMB_TEMPLATE = TEMPLATE_DIR / "red_bomb_button.png"
RED_BOMB_TEMPLATE_SCALES = (0.85, 0.95, 1.0, 1.05, 1.15)
RED_BOMB_TEMPLATE_THRESHOLD = 0.72


@dataclass(frozen=True)
class AmmoFingerprint:
    shape: tuple[int, int]
    packed_mask: bytes
    foreground_pixels: int


def locate_red_bomb_button(image: np.ndarray) -> MatchResult | None:
    height, width = image.shape[:2]
    x1 = int(width * 0.80)
    y1 = int(height * 0.76)
    crop = image[y1:height, x1:width]
    match = find_template_multi_scale(
        crop,
        RED_BOMB_TEMPLATE,
        scales=RED_BOMB_TEMPLATE_SCALES,
        threshold=RED_BOMB_TEMPLATE_THRESHOLD,
    )
    if match is None:
        return None
    top_left = (match.top_left[0] + x1, match.top_left[1] + y1)
    bottom_right = (match.bottom_right[0] + x1, match.bottom_right[1] + y1)
    center = (match.center[0] + x1, match.center[1] + y1)
    return MatchResult(
        template_path=match.template_path,
        top_left=top_left,
        bottom_right=bottom_right,
        center=center,
        score=match.score,
    )


def _ammo_mask(image: np.ndarray, match: MatchResult) -> np.ndarray:
    left, top = match.top_left
    right, bottom = match.bottom_right
    width = right - left
    height = bottom - top
    x1 = left + int(width * 0.78)
    y1 = top + int(height * 0.70)
    crop = image[y1:bottom, x1:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    mask = ((saturation <= 70) & (value >= 175)).astype(np.uint8)
    return cv2.resize(mask, (24, 24), interpolation=cv2.INTER_NEAREST)


def build_ammo_fingerprint(
    frames: Sequence[np.ndarray],
    match: MatchResult,
) -> AmmoFingerprint | None:
    if len(frames) < 3:
        return None
    masks = [_ammo_mask(frame, match) for frame in frames[:3]]
    consensus = (np.sum(masks, axis=0) >= 2).astype(np.uint8)
    foreground = int(np.count_nonzero(consensus))
    if foreground < 3:
        return None
    return AmmoFingerprint(
        shape=consensus.shape,
        packed_mask=np.packbits(consensus).tobytes(),
        foreground_pixels=foreground,
    )


def _unpack_fingerprint(fingerprint: AmmoFingerprint) -> np.ndarray:
    count = fingerprint.shape[0] * fingerprint.shape[1]
    unpacked = np.unpackbits(np.frombuffer(fingerprint.packed_mask, dtype=np.uint8))[:count]
    return unpacked.reshape(fingerprint.shape).astype(np.uint8)


def ammo_fingerprint_matches(
    before: AmmoFingerprint | None,
    after: AmmoFingerprint | None,
    *,
    minimum_iou: float = 0.88,
) -> bool:
    if before is None or after is None or before.shape != after.shape:
        return False
    first = _unpack_fingerprint(before)
    second = _unpack_fingerprint(after)
    union = int(np.count_nonzero(first | second))
    intersection = int(np.count_nonzero(first & second))
    return union > 0 and intersection / union >= minimum_iou


def red_bomb_selected(image: np.ndarray, match: MatchResult) -> bool:
    left, top = match.top_left
    right, bottom = match.bottom_right
    crop = image[max(0, top - 3):bottom + 4, max(0, left - 3):right + 4]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    white = ((saturation <= 45) & (value >= 205)).astype(np.uint8)
    border = np.zeros_like(white)
    border[:5, :] = 1
    border[-5:, :] = 1
    border[:, :5] = 1
    border[:, -5:] = 1
    return int(np.count_nonzero(white & border)) / max(1, int(np.count_nonzero(border))) >= 0.32
```

- [ ] **Step 5: Run visual tests and commit the new asset/helpers**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_red_scout.RedBombVisualTest
git add -- template/red_bomb_button.png utils/red_scout.py tests/test_red_scout.py
git commit -m "feat: detect red bomb and verify ammo badge"
```

### Task 5: Implement Multi-Cell Analysis And Scout Planning

**Files:**
- Modify: `utils/red_scout.py`
- Test: `tests/test_red_scout.py`

- [ ] **Step 1: Write failing analyzer and planner tests**

Use dependency-injected evidence so tests do not require real red ammunition:

```python
from types import SimpleNamespace

from utils.red_scout import RedFootprint, RedScoutAnalyzer, RedScoutPlanner


class RedScoutAnalyzerTest(unittest.TestCase):
    def test_first_analysis_learns_connected_multi_cell_footprint(self):
        changed = {
            (2, 2): (0.96, "miss"),
            (2, 3): (0.94, "hit"),
            (3, 2): (0.93, "miss"),
            (0, 0): (0.89, "hit"),
        }

        def classify(_before, _after, point):
            ratio, state = changed.get(point, (0.10, "miss"))
            return SimpleNamespace(changed_ratio=ratio, state=state, score=ratio)

        analyzer = RedScoutAnalyzer(classifier=classify, hit_detector=lambda _image, point: point == (2, 3))
        points = [(row, col) for row in range(5) for col in range(5)]

        result = analyzer.analyze(
            before_image=np.zeros((20, 20, 3), dtype=np.uint8),
            after_images=[np.zeros((20, 20, 3), dtype=np.uint8)] * 3,
            click_points=points,
            grid_size=5,
            center_cell=(2, 2),
            excluded_cells=set(),
            learned_footprint=None,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.hit_cells, frozenset({(2, 3)}))
        self.assertEqual(result.miss_cells, frozenset({(2, 2), (3, 2)}))
        self.assertNotIn((0, 0), result.affected_cells)
        self.assertEqual(
            result.footprint.offsets,
            frozenset({(0, 0), (0, 1), (1, 0)}),
        )

    def test_planner_avoids_previous_coverage(self):
        planner = RedScoutPlanner(grid_size=5)
        footprint = RedFootprint(offsets=frozenset({(0, 0), (0, 1), (1, 0)}))

        first = planner.choose_center(
            footprint=None,
            known_cells=set(),
            covered_cells=set(),
            cell_scores={},
        )
        second = planner.choose_center(
            footprint=footprint,
            known_cells=set(),
            covered_cells={(2, 2), (2, 3), (3, 2)},
            cell_scores={(0, 0): 10.0},
        )

        self.assertEqual(first, (2, 2))
        self.assertNotEqual(second, first)
```

- [ ] **Step 2: Run analyzer tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_red_scout.RedScoutAnalyzerTest
```

Expected: import failure for analyzer and planner classes.

- [ ] **Step 3: Implement result types, connected filtering, and planner**

Add these public types:

```python
from collections import deque
from collections.abc import Callable
from statistics import median

from utils.diamond_hit import classify_diamond_hit
from utils.wreck_detection import red_hit_marker_visible, visible_wreck_static_detected

Cell = tuple[int, int]


@dataclass(frozen=True)
class RedFootprint:
    offsets: frozenset[Cell]


@dataclass(frozen=True)
class RedScoutResult:
    center_cell: Cell
    affected_cells: frozenset[Cell]
    hit_cells: frozenset[Cell]
    miss_cells: frozenset[Cell]
    unknown_cells: frozenset[Cell]
    footprint: RedFootprint | None
    valid: bool
    confidence_by_cell: Mapping[Cell, float]
```

Implement `RedScoutAnalyzer` with constructor defaults:

```python
class RedScoutAnalyzer:
    def __init__(
        self,
        *,
        classifier: Callable = classify_diamond_hit,
        hit_detector: Callable[[np.ndarray, tuple[int, int]], bool] | None = None,
    ) -> None:
        self.classifier = classifier
        self.hit_detector = hit_detector or (
            lambda image, point: red_hit_marker_visible(image, point)
            or visible_wreck_static_detected(image, point)
        )
```

The `analyze` method must:

- Map `index` to `(row, col)` and the corresponding screen point.
- Collect each frame's `changed_ratio` and `state`.
- Use median change and require at least two frames with the same hit/miss state.
- For the first footprint, keep cells with median change at least `0.72`, then retain only the 8-connected component containing `center_cell`.
- For a learned footprint, evaluate only in-bounds `center + offset` cells and require median change at least `0.45`.
- Mark a cell hit only if the hit detector succeeds in at least two frames.
- Mark it miss only if it is affected, has no positive hit evidence, and at least two classifier frames say `miss`.
- Leave every other affected cell unknown.
- Return `valid=False` if the first footprint has fewer than two affected cells or does not contain the center.

Implement the connected component with:

```python
def _connected_component(cells: set[Cell], start: Cell) -> set[Cell]:
    if start not in cells:
        return set()
    queue = deque([start])
    visited = {start}
    while queue:
        row, col = queue.popleft()
        for row_delta in (-1, 0, 1):
            for col_delta in (-1, 0, 1):
                if row_delta == 0 and col_delta == 0:
                    continue
                neighbor = (row + row_delta, col + col_delta)
                if neighbor in cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return visited
```

Implement `RedScoutPlanner.choose_center` by returning the board center when `footprint is None`; otherwise score every in-bounds center by:

```python
score = new_unknown_cells * 100.0 + placement_score - overlap_cells * 25.0 - clipped_offsets * 40.0
```

Use row-major index as the final deterministic tie-break.

- [ ] **Step 4: Run all red-scout pure tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_red_scout
```

Expected: all settings, visual, analyzer, and planner tests pass.

- [ ] **Step 5: Commit the analyzer and planner**

```powershell
git add -- utils/red_scout.py tests/test_red_scout.py
git commit -m "feat: analyze multi-cell red scout results"
```

### Task 6: Verify Network Isolation Before Red Clicks

**Files:**
- Modify: `utils/adb_control.py:190-230`
- Modify: `utils/adb_control.py:620-750`
- Test: `tests/test_adb_control.py`

- [ ] **Step 1: Write failing isolation tests**

```python
    def test_isolation_is_safe_when_ipv4_drop_exists_and_no_ipv6_route(self):
        controller = AdbController.__new__(AdbController)
        controller._get_package_uid = Mock(return_value=10051)
        controller._is_weak_network_rule_active = Mock(return_value=True)
        controller._run_privileged_script = Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
        )

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertTrue(status.safe)
        self.assertTrue(status.ipv4_blocked)
        self.assertFalse(status.ipv6_route_present)

    def test_isolation_fails_when_ipv6_has_route_without_uid_rule(self):
        controller = AdbController.__new__(AdbController)
        controller._get_package_uid = Mock(return_value=10051)
        controller._is_weak_network_rule_active = Mock(return_value=True)
        controller._run_privileged_script = Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout="default via fe80::1 dev wlan0\n", stderr=""),
                subprocess.CompletedProcess([], 1, stdout="", stderr="owner match unavailable\n"),
            ]
        )

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertFalse(status.safe)
        self.assertTrue(status.ipv6_route_present)
        self.assertFalse(status.ipv6_blocked)

    def test_isolation_fails_when_ipv6_routes_cannot_be_read(self):
        controller = AdbController.__new__(AdbController)
        controller._get_package_uid = Mock(return_value=10051)
        controller._is_weak_network_rule_active = Mock(return_value=True)
        controller._run_privileged_script = Mock(
            return_value=subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr="error: device offline\n",
            )
        )

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertFalse(status.safe)
        self.assertIn("ipv6 route check failed", status.detail)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adb_control.AdbControlTest.test_isolation_is_safe_when_ipv4_drop_exists_and_no_ipv6_route tests.test_adb_control.AdbControlTest.test_isolation_fails_when_ipv6_has_route_without_uid_rule tests.test_adb_control.AdbControlTest.test_isolation_fails_when_ipv6_routes_cannot_be_read
```

Expected: attribute failure for `verify_app_network_isolated`.

- [ ] **Step 3: Implement the isolation status and verifier**

At module level add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkIsolationStatus:
    safe: bool
    ipv4_blocked: bool
    ipv6_route_present: bool
    ipv6_blocked: bool
    detail: str
```

Add to `AdbController`:

```python
    def verify_app_network_isolated(self, package_name: str) -> NetworkIsolationStatus:
        package_name = package_name.strip()
        if not package_name:
            _raise_value_error("包名不能为空")
        uid = self._get_package_uid(package_name)
        ipv4_blocked = self._is_weak_network_rule_active(uid)

        route_result = self._run_privileged_script("ip -6 route show", check=False)
        if route_result.returncode != 0 or route_result.stderr.strip():
            return NetworkIsolationStatus(
                safe=False,
                ipv4_blocked=ipv4_blocked,
                ipv6_route_present=False,
                ipv6_blocked=False,
                detail="ipv6 route check failed",
            )
        ipv6_routes = route_result.stdout.strip()
        ipv6_route_present = any(
            line.startswith("default ") or " via " in line
            for line in ipv6_routes.splitlines()
        )
        ipv6_blocked = False
        if ipv6_route_present:
            rule_result = self._run_privileged_script(
                f"ip6tables -C OUTPUT -m owner --uid-owner {uid} -j BBMA_WEAKNET",
                check=False,
            )
            ipv6_blocked = rule_result.returncode == 0

        safe = ipv4_blocked and (not ipv6_route_present or ipv6_blocked)
        detail = (
            f"ipv4_blocked={int(ipv4_blocked)} "
            f"ipv6_route={int(ipv6_route_present)} "
            f"ipv6_blocked={int(ipv6_blocked)}"
        )
        return NetworkIsolationStatus(
            safe=safe,
            ipv4_blocked=ipv4_blocked,
            ipv6_route_present=ipv6_route_present,
            ipv6_blocked=ipv6_blocked,
            detail=detail,
        )
```

- [ ] **Step 4: Run all ADB-control tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adb_control
```

Expected: all tests pass.

- [ ] **Step 5: Commit only if the tracked-file diff is isolated**

```powershell
git diff -- utils/adb_control.py tests/test_adb_control.py
git add -- utils/adb_control.py tests/test_adb_control.py
git commit -m "feat: verify red scout network isolation"
```

Skip the commit if it would include pre-existing ADB-control changes.

### Task 7: Implement The Fail-Closed Red Scout Transaction

**Files:**
- Modify: `main.py:110-150`
- Modify: `main.py:640-710`
- Modify: `main.py:1516-1990`
- Test: `tests/test_main_flow.py`

- [ ] **Step 1: Write a failing transaction-order test**

Extend `FakeAdb` with `verify_app_network_isolated` and use existing call recording. Add:

```python
    def test_red_scout_discards_request_before_ammo_verification(self):
        settings = self.main.RedScoutSettings(
            mode=self.main.ProbeMode.RED_SCOUT,
            count=1,
        )
        footprint = self.main.RedFootprint(offsets=frozenset({(0, 0), (0, 1)}))
        analysis = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset({(1, 1), (1, 2)}),
            hit_cells=frozenset({(1, 2)}),
            miss_cells=frozenset({(1, 1)}),
            unknown_cells=frozenset(),
            footprint=footprint,
            valid=True,
            confidence_by_cell={(1, 1): 0.95, (1, 2): 0.96},
        )

        def discard_transaction(transaction):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(self.main, "_capture_red_ammo_state", side_effect=[("before", object()), ("after", object())]),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(self.main, "click_template", return_value=True),
            patch.object(self.main, "enter_activity"),
            patch.object(self.main, "_capture_red_result_frames", return_value=[object(), object(), object()]),
            patch.object(self.main, "_analyze_red_result", return_value=analysis),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard_transaction,
            ) as discard,
            patch.object(self.main, "ammo_fingerprint_matches", return_value=True),
        ):
            result = self.main._execute_red_scout_transaction(
                level=1,
                grid_size=3,
                click_points=[(400, 300)] * 9,
                center_cell=(1, 1),
                excluded_cells=set(),
                learned_footprint=None,
                settings=settings,
            )

        self.assertEqual(result, analysis)
        discard.assert_called_once()
```

Add a safety test:

```python
    def test_red_scout_never_clicks_grid_when_isolation_is_unsafe(self):
        self.adb.verify_app_network_isolated = Mock(
            return_value=SimpleNamespace(safe=False, detail="ipv6 unblocked")
        )

        with self.assertRaises(self.main.RedScoutSafetyError):
            self.main._execute_red_scout_transaction(
                level=1,
                grid_size=3,
                click_points=[(400, 300)] * 9,
                center_cell=(1, 1),
                excluded_cells=set(),
                learned_footprint=None,
                settings=self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1),
            )

        self.assertNotIn(("click", 400, 300), self.adb.calls)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_flow.MainFlowTest.test_red_scout_discards_request_before_ammo_verification tests.test_main_flow.MainFlowTest.test_red_scout_never_clicks_grid_when_isolation_is_unsafe
```

Expected: missing red transaction functions and types.

- [ ] **Step 3: Add fail-closed state and red transaction helpers**

Import red-scout types into `main.py`:

```python
from utils.red_scout import (
    AmmoFingerprint,
    ProbeMode,
    RedFootprint,
    RedScoutAnalyzer,
    RedScoutResult,
    RedScoutSettings,
    ammo_fingerprint_matches,
    build_ammo_fingerprint,
    load_red_scout_settings,
    locate_red_bomb_button,
    red_bomb_selected,
)
```

Add:

```python
class RedScoutSafetyError(RuntimeError):
    """Raised when red scouting cannot continue without risking ammunition."""


_network_fail_closed_reason: str | None = None


def latch_network_fail_closed(reason: str) -> None:
    global _network_fail_closed_reason
    _network_fail_closed_reason = reason
    write_runtime_status(network="安全停机：保持断网", last_result=reason)
```

Update `cleanup_weak_network` so it refuses to restore when `_network_fail_closed_reason` is set.

Implement `_capture_red_ammo_state` by taking three frames separated by `0.15` seconds, locating the red button in the first frame, and building one fingerprint from all three frames. Raise `RedScoutSafetyError` if either operation fails.

Implement `_select_red_bomb` by clicking the matched button center, waiting `0.25` seconds, reading a screenshot, and returning `red_bomb_selected(selected_frame, match)`.

Implement `_execute_red_scout_transaction` with this exact safety order:

```python
    enable_weak_network(PROBE_DROP_SETTLE_SECONDS)
    isolation = adb.verify_app_network_isolated(GAME_PACKAGE_NAME)
    if not isolation.safe:
        latch_network_fail_closed(f"红色侦察断网验证失败: {isolation.detail}")
        raise RedScoutSafetyError(_network_fail_closed_reason)

    _button_match, before_ammo = _capture_red_ammo_state()
    before_image = adb.read_screenshot()
    if not _select_red_bomb(_button_match):
        raise RedScoutSafetyError("无法确认红色炮弹已选中")

    transaction = ProbeTransaction(level=level, cell=center_cell, index=index)
    _active_probe = transaction
    transaction.advance(ProbePhase.REQUEST_PENDING)
    adb.click(*click_points[index])
```

After result capture, advance through `RESULT_VISIBLE` and `RESULT_RECORDED`, call the existing `_discard_pending_request_and_prepare_next_probe(transaction)`, then capture the new ammunition fingerprint. If the fingerprints differ:

```python
    _stop_and_latch_red_safety_failure("红色炮弹数量无法确认")
```

Implement `_stop_and_latch_red_safety_failure(reason)` to install both network rules, close the game, verify process exit, latch the reason, and raise `RedScoutSafetyError`. Wrap both the post-restart fingerprint capture and fingerprint comparison with this helper so a missing button, unreadable badge, or changed badge all re-isolate before returning control to `main()`.

Clear `_active_probe` only after a complete safe transaction. If an exception occurs after the grid click, preserve the pending transaction so exit cleanup remains fail closed.

- [ ] **Step 4: Run focused transaction and existing blue transaction tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_flow.MainFlowTest.test_red_scout_discards_request_before_ammo_verification tests.test_main_flow.MainFlowTest.test_red_scout_never_clicks_grid_when_isolation_is_unsafe tests.test_main_flow.MainFlowTest.test_miss_discard_force_stops_before_network_restore tests.test_main_flow.MainFlowTest.test_hit_transaction_restores_network_without_reject
```

Expected: all four tests pass.

- [ ] **Step 5: Commit only if the main-flow diff is isolated**

```powershell
git diff -- main.py tests/test_main_flow.py
git add -- main.py tests/test_main_flow.py
git commit -m "feat: add fail-closed red scout transaction"
```

Skip the commit if pre-existing main-flow changes would be included.

### Task 8: Run Configured Scout Attempts Before Blue Strategy

**Files:**
- Modify: `main.py:850-1050`
- Modify: `main.py:1170-1515`
- Modify: `main.py:2300-2395`
- Test: `tests/test_main_flow.py`

- [ ] **Step 1: Write failing per-level integration tests**

```python
    def test_red_mode_runs_configured_attempts_then_seeds_strategy(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        results = [
            self.main.RedScoutResult(
                center_cell=(1, 1),
                affected_cells=frozenset({(1, 1), (1, 2)}),
                hit_cells=frozenset({(1, 2)}),
                miss_cells=frozenset({(1, 1)}),
                unknown_cells=frozenset(),
                footprint=self.main.RedFootprint(frozenset({(0, 0), (0, 1)})),
                valid=True,
                confidence_by_cell={(1, 1): 0.95, (1, 2): 0.96},
            )
        ] * 3

        with (
            patch.object(self.main, "_execute_red_scout_transaction", side_effect=results) as execute,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                level=1,
                hit_map=[[0] * 3 for _ in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_hits=set(),
                settings=settings,
            )

        self.assertTrue(completed)
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], {(1, 2)})
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {(1, 1)})

    def test_blue_only_mode_never_enters_red_transaction(self):
        with (
            patch.object(self.main, "_execute_red_scout_transaction") as red,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            self.main._run_red_scout_and_blue_strategy(
                level=1,
                hit_map=[[0] * 3 for _ in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_hits=set(),
                settings=self.main.RedScoutSettings(),
            )

        red.assert_not_called()
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_flow.MainFlowTest.test_red_mode_runs_configured_attempts_then_seeds_strategy tests.test_main_flow.MainFlowTest.test_blue_only_mode_never_enters_red_transaction
```

Expected: missing orchestration helper and scan arguments.

- [ ] **Step 3: Integrate scout observations into the strategy scan**

Extend `_scan_level_by_strategy` with:

```python
    initial_scout_hits: set[Cell] | None = None,
    initial_scout_misses: set[Cell] | None = None,
```

After applying visual hits:

```python
    strategy.report_scout_results(
        hits=initial_scout_hits or set(),
        misses=initial_scout_misses or set(),
    )
```

Record `initial_real_observation_count = len(strategy.shots)` after saved and visual state is loaded. Use `len(strategy.shots) - initial_real_observation_count` plus later fallback probes for the current run's real blue progress; never count `scout_observations` and never add them to `save_level_shots`. Add scout misses to fallback skip cells:

```python
    fallback_skip_cells = (
        set(strategy.shots)
        | set(strategy.blocked_cells)
        | strategy.get_scout_miss_cells()
    )
```

Implement `_run_red_scout_and_blue_strategy`:

- Return directly to `_scan_level_by_strategy` in blue-only mode.
- Create `RedScoutPlanner(grid_size)` and `learned_footprint=None`.
- Loop at most `settings.count` times.
- Publish `red_scout_capture` status with `red_scout_current` and `red_scout_total`.
- Ask the planner for a center; stop the red phase if no center exists.
- Merge only `result.valid` hit and miss sets.
- Carry the first valid learned footprint into later attempts.
- Track covered cells so subsequent centers avoid duplicate coverage.
- Enter `_scan_level_by_strategy` with the merged temporary observations.

Parse settings once in `main()`:

```python
    settings = load_red_scout_settings()
```

Publish `probe_mode=settings.mode.value` and `red_scout_total=settings.count` in runtime status. Pass settings into `handle_game_level`, and reset red progress in `reset_runtime_level_status` so every new level starts clean.

- [ ] **Step 4: Run main-flow tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_main_flow
```

Expected: all main-flow tests pass, including unchanged blue-only tests.

- [ ] **Step 5: Commit only if the integration diff is isolated**

```powershell
git diff -- main.py tests/test_main_flow.py
git add -- main.py tests/test_main_flow.py
git commit -m "feat: run red scouting before blue attacks"
```

Skip the commit if it would include unrelated worktree changes.

### Task 9: Add Control Panel Mode And Count Controls

**Files:**
- Modify: `tools/control_panel.py:20-45`
- Modify: `tools/control_panel.py:575-760`
- Modify: `tools/control_panel.py:1062-1110`
- Modify: `tools/control_panel.py:1180-1280`
- Test: `tests/test_control_panel.py`

- [ ] **Step 1: Write failing pure helper tests**

Add a helper contract that avoids requiring a visible Qt window:

```python
from tools.control_panel import build_main_environment, format_probe_mode


class ControlPanelProbeModeTest(unittest.TestCase):
    def test_build_main_environment_includes_red_settings(self):
        values = build_main_environment("red_scout", 3)

        self.assertEqual(values["BBMA_PROBE_MODE"], "red_scout")
        self.assertEqual(values["BBMA_RED_SCOUT_COUNT"], "3")
        self.assertEqual(values["PYTHONUTF8"], "1")

    def test_probe_mode_formatter_is_user_facing(self):
        self.assertEqual(format_probe_mode("blue_only"), "仅蓝色炮弹")
        self.assertEqual(format_probe_mode("red_scout"), "红色侦察 + 蓝色攻击")
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_control_panel.ControlPanelProbeModeTest
```

Expected: import failure for the two helpers.

- [ ] **Step 3: Implement environment and UI controls**

Import `QSpinBox`. Add:

```python
PROBE_MODE_NAMES = {
    "blue_only": "仅蓝色炮弹",
    "red_scout": "红色侦察 + 蓝色攻击",
}


def format_probe_mode(value: object) -> str:
    return PROBE_MODE_NAMES.get(str(value), "仅蓝色炮弹")


def build_main_environment(mode: str, red_count: int) -> dict[str, str]:
    normalized_mode = mode if mode in PROBE_MODE_NAMES else "blue_only"
    normalized_count = min(10, max(1, int(red_count)))
    return {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "BBMA_PROBE_MODE": normalized_mode,
        "BBMA_RED_SCOUT_COUNT": str(normalized_count),
    }
```

In `_create_widgets` create:

```python
        self.probe_mode_combo = QComboBox()
        self.probe_mode_combo.addItem("仅蓝色炮弹", "blue_only")
        self.probe_mode_combo.addItem("红色侦察 + 蓝色攻击", "red_scout")
        self.red_scout_count = QSpinBox()
        self.red_scout_count.setRange(1, 10)
        self.red_scout_count.setValue(2)
        self.red_scout_count.setSuffix(" 次/关")
        self.red_scout_count.setEnabled(False)
        self.probe_mode_value = QLabel("仅蓝色炮弹")
        self.red_scout_progress_value = QLabel("--")
```

Place mode and count above the start/stop buttons in the existing `运行控制` group. Connect mode changes so the stepper is enabled only when item data is `red_scout` and the program is stopped.

In `start_program`, replace direct environment inserts with:

```python
        values = build_main_environment(
            str(self.probe_mode_combo.currentData()),
            self.red_scout_count.value(),
        )
        for name, value in values.items():
            environment.insert(name, value)
```

Read `probe_mode`, `red_scout_current`, and `red_scout_total` from runtime status. Add current mode and red progress to the status group. Disable both controls while running.

Extend constants:

```python
PHASE_NAMES.update(
    {
        "red_scout_preflight": "红色侦察准备",
        "red_scout_capture": "红色侦察",
        "red_scout_discard": "丢弃红色请求",
        "red_scout_verify_ammo": "验证红色炮弹",
        "blue_attack": "蓝色攻击",
    }
)
BOARD_STATE_NAMES.update(
    {
        "scout_miss": "侦察未命中",
        "scout_hit": "侦察命中",
    }
)
BOARD_STATE_COLORS.update(
    {
        "scout_miss": QColor("#aab7be"),
        "scout_hit": QColor("#d9822b"),
    }
)
```

Draw a small hollow circle for `scout_miss` and a white diamond for `scout_hit`. Add both states to the board summary and legend without nesting a new card.

- [ ] **Step 4: Run control-panel tests and compile the module**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_control_panel
.\.venv\Scripts\python.exe -m py_compile tools\control_panel.py
```

Expected: tests and compilation pass.

- [ ] **Step 5: Commit only if control-panel changes are isolated**

```powershell
git diff -- tools/control_panel.py tests/test_control_panel.py
git add -- tools/control_panel.py tests/test_control_panel.py
git commit -m "feat: configure red scouting in control panel"
```

Because both files may already be untracked or user-modified, skip this commit if it would capture unrelated work.

### Task 10: Documentation, Full Verification, And Safe Handoff

**Files:**
- Modify: `README.md`
- Verify: all files above

- [ ] **Step 1: Document user-visible behavior**

Add a UTF-8 section to `README.md` containing these exact operational points:

```markdown
## 探测模式

控制台支持两种模式：

- `仅蓝色炮弹`：保持原有单格探测流程。
- `红色侦察 + 蓝色攻击`：每关先执行设定次数的离线红色侦察，再使用蓝色炮弹提交可靠命中。

红色侦察次数可在控制台设置为 1 至 10。红色请求不会主动提交；如果断网、游戏进程退出或红色炮弹数量无法确认，程序会停止并保持游戏断网。此时先查看日志，不要直接恢复网络。
```

- [ ] **Step 2: Run the complete automated test suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 3: Run syntax and whitespace verification**

```powershell
.\.venv\Scripts\python.exe -m py_compile main.py tools\control_panel.py utils\adb_control.py utils\red_scout.py utils\wreck_detection.py utils\submarine_strategy.py
git diff --check
```

Expected: Python compilation exits `0`; `git diff --check` reports no whitespace errors. Existing LF/CRLF warnings are acceptable.

- [ ] **Step 4: Verify no live process or network rule was changed by tests**

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'main\.py' -and $_.CommandLine -match 'BoomBeachSonarAuto-main' } | Select-Object ProcessId,CommandLine
.\tools\platform-tools\adb.exe -s 127.0.0.1:5555 shell iptables -S OUTPUT
```

Expected: automated tests did not start `main.py`; no unexpected `BBMA_WEAKNET` or `BBMA_REJECTNET` jump remains unless the user intentionally has the program running.

- [ ] **Step 5: Review the final diff against the specification**

Check every acceptance item in `docs/superpowers/specs/2026-07-14-red-scout-mode-design.md`. Confirm that:

- Blue-only is the default.
- Red count is selectable from 1 to 10.
- Red observations do not make the strategy complete before blue verification.
- Unsafe isolation prevents a grid click.
- Post-click exceptions preserve network isolation.
- Ammunition mismatch re-isolates and stops.
- Scout data resets for each level.

- [ ] **Step 6: Do not run a real red scout without explicit user authorization**

The first live validation consumes a real interaction even though the protocol is designed to roll it back. Leave the program stopped and report that automated verification is complete. When the user explicitly asks for live validation, use `红色侦察 + 蓝色攻击`, set the count to `1`, record the red count before starting, and stop immediately after the first completed scout transaction for visual comparison.

- [ ] **Step 7: Commit documentation or report deferred commits**

```powershell
git diff -- README.md
git add -- README.md
git commit -m "docs: explain red scout mode"
```

Skip the commit if the README contains unrelated pre-existing changes. In the final handoff, list any tasks whose commits were deferred because the worktree was dirty.
