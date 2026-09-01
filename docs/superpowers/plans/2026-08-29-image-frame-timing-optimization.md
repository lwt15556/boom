# Image Recognition Frame Timing Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the red-scout `RESULT_VISIBLE -> RESULT_RECORDED` time by tuning only the result screenshot frame schedule, while keeping the existing image classifier, evidence rules, safety checks, and transaction order unchanged.

**Architecture:** Keep `_capture_red_result_frames()` as the only production timing boundary. Add measurement around each scheduled delay and ADB screenshot, then compare a small set of fixed three-frame schedules on the same device. Select the fastest schedule that preserves the existing three-frame analyzer output and never reduce the frame count below three; retain the current `(0.55, 0.15, 0.20)` schedule as the immediate rollback profile.

**Tech Stack:** Python 3.11, ADB screenshot capture, OpenCV image arrays, `unittest`, existing `main.py` red-scout orchestration.

---

## Scope Map

**Files to modify when this plan is executed:**

- `main.py:118-135`: name the current and candidate result-frame schedules and add a profile selector with a safe default.
- `main.py:1180-1210`: instrument and apply the selected delays in `_capture_red_result_frames()` without changing screenshot paths or frame ordering.
- `tests/test_main_flow.py:2560-2630`: verify schedule selection, delay order, exactly three captures, and rollback behavior.
- `tests/test_red_scout.py`: replay the same captured frames through the unchanged analyzer to prove that timing changes do not alter the result tuple.

**Files explicitly out of scope:**

- `utils/diamond_hit.py` (no classifier, mask, center-search, or feature-cache changes).
- `utils/red_scout.py` (no candidate filtering, voting, evidence, or state-resolution changes).
- `utils/sidebar_progress.py`, `utils/wreck_detection.py`, and `utils/image_match.py` (no recognition changes).
- Network isolation, pending-request discard, red-ammunition fingerprint verification, and transaction transitions in `main.py`.

## Timing Model

The delay tuple is applied immediately before each screenshot. For a tuple `(d0, d1, d2)`, the elapsed capture stage is approximately:

```text
d0 + screenshot_0 + d1 + screenshot_1 + d2 + screenshot_2
```

The current schedule is `(0.55, 0.15, 0.20)`. Because a real ADB screenshot currently costs about 0.54–0.88 seconds, the first delay must remain long enough for the result screen to render, while later delays only need to separate frames enough to observe the animation settling.

Candidate schedules for the controlled comparison are:

```python
RED_SCOUT_RESULT_FRAME_DELAYS_CURRENT = (0.55, 0.15, 0.20)
RED_SCOUT_RESULT_FRAME_DELAYS_BALANCED = (0.35, 0.10, 0.15)
RED_SCOUT_RESULT_FRAME_DELAYS_FAST = (0.25, 0.08, 0.12)
```

The candidate values are test profiles, not an unconditional production change. A profile is accepted only after the device replay and safety checks in Task 4 pass.

## Task 1: Add Per-Frame Timing Evidence

**Files:**
- Modify: `main.py:1180-1210` in `_capture_red_result_frames()`.
- Test: `tests/test_main_flow.py`.

- [ ] **Step 1: Log the scheduled delay and actual screenshot duration.**

Wrap only the delay and `adb.read_screenshot()` call with `monotonic()` and emit informational records in this shape:

```python
capture_started = monotonic()
adb.delay(frame_delay)
delay_finished = monotonic()
screenshot = adb.read_screenshot(output_path)
capture_finished = monotonic()
logger.info(
    "red scout frame_timing index=%d scheduled_delay=%.3f "
    "delay_elapsed=%.3f screenshot_elapsed=%.3f",
    frame_index,
    frame_delay,
    delay_finished - capture_started,
    capture_finished - delay_finished,
)
```

Do not change the returned list, output paths, exception behavior, or number of captures.

- [ ] **Step 2: Add a unit test for timing log inputs without sleeping.**

Patch `main.monotonic` with a deterministic sequence and replace `adb.delay`/`adb.read_screenshot` with mocks. Assert that three timing records receive the three configured delays in order and that the returned frame list is unchanged.

- [ ] **Step 3: Run the focused test.**

Run:

```powershell
py -3.11 -m unittest tests.test_main_flow -q
```

Expected: all existing tests and the new timing-evidence test pass.

## Task 2: Make Frame Schedules Explicit and Reversible

**Files:**
- Modify: `main.py:118-135` and `_capture_red_result_frames()`.
- Test: `tests/test_main_flow.py`.

- [ ] **Step 1: Add named schedules with the current profile as default.**

Keep the existing public constant for compatibility and introduce a private profile map:

```python
RED_SCOUT_RESULT_FRAME_DELAYS = (0.55, 0.15, 0.20)
_RED_SCOUT_FRAME_DELAY_PROFILES = {
    "current": RED_SCOUT_RESULT_FRAME_DELAYS,
    "balanced": (0.35, 0.10, 0.15),
    "fast": (0.25, 0.08, 0.12),
}
```

The selector must reject unknown names, non-finite values, negative values, and tuples with fewer than three frames, returning the current profile in every invalid case.

- [ ] **Step 2: Add an internal schedule argument without changing callers.**

Use this signature so tests and a later controlled run can choose a profile while all existing callers retain the current behavior:

```python
def _capture_red_result_frames(
    sample_dir: Path | None = None,
    *,
    frame_delays: Sequence[float] = RED_SCOUT_RESULT_FRAME_DELAYS,
) -> list[np.ndarray]:
```

Validate the sequence before the first screenshot. A validation failure must raise the same safety exception used for capture failures and must not partially capture a result.

- [ ] **Step 3: Test profile selection and rollback.**

Cover these exact cases:

```text
current -> (0.55, 0.15, 0.20)
balanced -> (0.35, 0.10, 0.15)
fast -> (0.25, 0.08, 0.12)
unknown/invalid -> current
explicit frame_delays -> exact caller-provided order
```

- [ ] **Step 4: Run the focused tests and compile check.**

Run:

```powershell
py -3.11 -m unittest tests.test_main_flow -q
py -3.11 -m py_compile main.py
```

Expected: tests pass and `py_compile` exits successfully.

## Task 3: Verify That Faster Timing Does Not Change Recognition

**Files:**
- Test: `tests/test_red_scout.py` and `tests/test_main_flow.py`.
- Inspect: `_debug/red_scout_samples/` and any existing `after_*.png` evidence directories.

- [ ] **Step 1: Build a deterministic schedule replay fixture.**

For each sample with at least three recorded result frames, load the existing `before_*.png` and `after_*.png` files and call the unchanged analyzer with the metadata already stored in that sample. Do not regenerate frames or alter analyzer thresholds.

- [ ] **Step 2: Compare the exact result tuple for each sample.**

For every sample, compare the current baseline output with the output using the same first three frames:

```python
def result_tuple(result):
    return (
        result.valid,
        result.affected_cells,
        result.hit_cells,
        result.miss_cells,
        result.unknown_cells,
    )
```

The test fails on any new invalid result, new `unknown_cells`, or changed hit/miss/affected set. This isolates timing selection from classifier changes.

- [ ] **Step 3: Record the replay gate.**

Print or assert a fixed count of samples where the first three frames match the baseline. The fast profile cannot be enabled if the available evidence does not prove three-frame equivalence.

- [ ] **Step 4: Run the replay tests.**

Run:

```powershell
py -3.11 -m unittest tests.test_red_scout tests.test_main_flow -q
```

Expected: all samples produce the same result tuple before and after schedule selection.

## Task 4: Controlled Real-Device A/B Timing Run

**Files:**
- No new recognition code; inspect `run_stderr.log` and generated red-scout sample metadata.

- [ ] **Step 1: Run the current profile as the baseline.**

Use the same emulator, 1280x720 resolution, server, level, and network conditions. Run three red scouts on level 5 and three on level 6 with `current` and record:

```text
REQUEST_PENDING -> RESULT_VISIBLE
RESULT_VISIBLE -> RESULT_RECORDED
per-frame scheduled delay and screenshot duration
frame count
valid / invalid / unknown counts
affected / hit / miss sets
```

- [ ] **Step 2: Run the balanced profile.**

Repeat the exact six-run matrix with `(0.35, 0.10, 0.15)`. Stop the comparison if any red-ammunition mismatch, pending-request violation, fail-closed latch, or changed recognition tuple appears.

- [ ] **Step 3: Run the fast profile only if balanced is equivalent.**

Repeat with `(0.25, 0.08, 0.12)` only when the balanced profile has identical recognition output and no safety event. This prevents an aggressive schedule from becoming the first production candidate.

- [ ] **Step 4: Select by p50/p90, not one lucky run.**

Accept the fastest profile that satisfies all of these initial gates:

```text
RESULT_VISIBLE -> RESULT_RECORDED p50 <= 3.2 s
p90 is at least 0.5 s below the current-profile p90
valid results do not decrease
unknown or invalid results do not increase
affected/hit/miss sets match the baseline exactly
all red-ammunition and fail-closed checks remain unchanged
```

If no candidate meets every gate, keep `current` and report the measured bottleneck instead of lowering delays further.

## Task 5: Enable the Chosen Schedule With a One-Line Rollback

**Files:**
- Modify: `main.py:118-135` and the call site around `main.py:1684`.
- Test: `tests/test_main_flow.py`.

- [ ] **Step 1: Set only the accepted profile as the runtime default.**

Change the default tuple or profile selection to the profile accepted in Task 4. Do not change `HIT_RESULT_FRAME_DELAYS`, `ONLINE_SCOUT_HIT_FRAME_DELAYS`, analyzer thresholds, or any transaction timing.

- [ ] **Step 2: Preserve an immediate rollback.**

Keep the current tuple as a named constant and make rollback a single configuration change:

```python
RED_SCOUT_RESULT_FRAME_DELAYS = RED_SCOUT_RESULT_FRAME_DELAYS_CURRENT
```

No rollback may require reverting classifier or state-machine code.

- [ ] **Step 3: Add a regression test for the selected default.**

Assert that the default schedule has exactly three non-negative finite delays, that the first delay is not shorter than the selected later-frame delays, and that `_capture_red_result_frames()` performs exactly three `delay -> read_screenshot` pairs.

- [ ] **Step 4: Run the final focused checks.**

Run:

```powershell
py -3.11 -m unittest tests.test_main_flow tests.test_red_scout -q
py -3.11 -m py_compile main.py
```

Expected: all tests pass and the selected schedule remains isolated to red result screenshot timing.

## Self-Review Checklist

- [ ] The plan changes only result screenshot timing and does not alter image-recognition algorithms.
- [ ] The first-frame render wait is measured before shortening.
- [ ] Every faster profile has a current-profile fallback.
- [ ] Exactly three frames remain mandatory; no timing profile silently reduces evidence count.
- [ ] Recognition output and fail-closed behavior are compared against a baseline.
- [ ] Acceptance uses repeated real-device p50/p90 measurements, not a single run.

