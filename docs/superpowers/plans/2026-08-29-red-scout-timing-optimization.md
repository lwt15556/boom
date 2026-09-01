# Red Scout Timing Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the two measured red-scout stages, `REQUEST_PENDING -> RESULT_VISIBLE` and `RESULT_VISIBLE -> RESULT_RECORDED`, without weakening the pending-request, network-isolation, red-ammunition, or fail-closed contracts.

**Architecture:** Keep the existing red-scout transaction and state machine. Make activity re-entry event-driven with a fast path and an explicit fallback, then make result-frame capture data-driven and conditionally adaptive only after replay proves that three frames preserve the same red-scout result as four. Keep network recovery and red-ammunition verification serial with the same safety order.

**Tech Stack:** Python 3.11, ADB, OpenCV, `unittest`, existing `main.py` orchestration and `utils/red_scout.py` analyzer.

---

## Scope Map

**Files to modify:**

- `main.py`: timing boundaries, activity re-entry fast path, shared connection-prompt frame, and guarded red-result capture.
- `tests/test_main_flow.py`: unit tests for fast-path/fallback behavior, prompt-frame reuse, capture schedules, and fail-closed behavior.
- `tests/test_red_scout.py`: replay-level checks that compare three-frame and four-frame analyzer output where pure analyzer behavior is involved.

**Files to inspect but not change initially:**

- `utils/adb_control.py`: confirm screenshot and input command costs; do not change ADB semantics in this optimization.
- `docs/superpowers/specs/2026-07-14-red-scout-mode-design.md`: preserve the transaction order in steps 8-16.
- `_debug/logs/bbma.log` and `_debug/red_scout_samples/`: real timing and replay evidence.

## Task 1: Establish a Reproducible Baseline

**Files:**
- Modify: `main.py:1599-1709` only for stage timing logs.
- Test: `tests/test_main_flow.py`.

- [ ] **Step 1: Add monotonic stage markers without changing control flow.**

Record elapsed time from the existing `REQUEST_PENDING` transition at these boundaries:

```python
red_started_at = monotonic()
logger.info("red scout timing stage=request_pending")

# immediately after _reenter_activity_for_probe_result() returns
logger.info(
    "red scout timing stage=result_visible elapsed=%.3f",
    monotonic() - red_started_at,
)

# immediately after _capture_red_result_frames() returns
logger.info(
    "red scout timing stage=result_recorded elapsed=%.3f",
    monotonic() - red_started_at,
)
```

The markers must be informational only and must not alter timeout values, network calls, or transaction transitions.

- [ ] **Step 2: Add a unit test that verifies marker calls do not change the transaction order.**

Patch `monotonic()` with a deterministic sequence and assert the existing order remains:

```python
self.assertEqual(
    phases,
    [
        "red_scout_preflight",
        "red_scout_capture",
        "red_scout_discard",
        "red_scout_verify_ammo",
    ],
)
```

- [ ] **Step 3: Run the focused baseline test.**

Run:

```powershell
py -3.11 -m unittest tests.test_main_flow -q
```

Expected: all existing tests pass.

- [ ] **Step 4: Run one real-device baseline with red count 1 on level 5 or 6.**

Record these log intervals separately:

```text
REQUEST_PENDING -> RESULT_VISIBLE
RESULT_VISIBLE -> RESULT_RECORDED
RESULT_RECORDED -> LOGIN_RECOVERING
LOGIN_RECOVERING -> COMPLETE
```

Do not proceed to a timing change if the run has a pending probe, red-ammunition mismatch, or fail-closed latch.

## Task 2: Add a Fast Activity Re-entry Path

**Files:**
- Modify: `main.py:1819-1927` and `main.py:4564-4613`.
- Test: `tests/test_main_flow.py`.

- [ ] **Step 1: Define the fast-path contract.**

Add an internal option to `enter_activity()` with this exact behavior:

```python
def enter_activity(
    re_enter: bool = False,
    max_retries: int = 5,
    *,
    activity_button_timeout: float | None = None,
    prepare_activity_list: bool | None = None,
    fast_prepare_activity_list: bool = False,
) -> bool:
```

When `fast_prepare_activity_list=True` and the call is a post-retry `re_enter=True` call:

1. Keep the existing activity-button detection and click.
2. Click `ACTIVITY_DETAIL_POINT` immediately without the two list swipes.
3. Wait up to 3 seconds for `QUIT_ACTIVITY_TEMPLATE`.
4. Return immediately when detail is confirmed.
5. If detail is not confirmed, execute the current two-swipe path and existing detail wait as the fallback.

The fast path must never disable `DROP` or `REJECT`, and it must not be used while a request is still pending.

- [ ] **Step 2: Use the fast path only after the red request is discarded and network is restored.**

Change the `enter_activity()` call in `_discard_pending_request_and_prepare_next_probe()` to pass:

```python
fast_prepare_activity_list=True
```

Keep `prepare_activity_list=True` so the existing swipe path remains available as the fallback. Do not change the order of `disable_weak_network()`, `disable_reject_network()`, retry click, and activity entry.

- [ ] **Step 3: Test fast success without swipes.**

Mock the activity button and detail template as immediately visible. Assert:

```python
self.assertEqual(self.adb.swipe.call_count, 0)
self.assertTrue(result)
```

Also assert that `disable_weak_network()` and `disable_reject_network()` were called before the retry click, preserving the current safety order.

- [ ] **Step 4: Test fallback after fast-path timeout.**

Make the first 3-second detail wait fail, then make the existing post-swipe wait succeed. Assert exactly two swipes and one successful detail confirmation. The test must fail if the function raises before the fallback is attempted.

- [ ] **Step 5: Test pending-request protection.**

Use an active `ProbeTransaction` in `REQUEST_DISCARDED` and force the fast path to fail. Assert that `DROP/REJECT` remain enabled and the function raises `DiscardRecoveryError` or `ProbeProtocolError` according to the existing branch; it must not call a network-restoring helper before the request is safely discarded.

- [ ] **Step 6: Run focused tests.**

Run:

```powershell
py -3.11 -m unittest tests.test_main_flow -q
```

Expected: existing tests plus the fast-path and fallback tests pass.

## Task 3: Reuse the First Connection-Dialog Screenshot

**Files:**
- Modify: `main.py:4842-4870` and `main.py:4564-4592`.
- Test: `tests/test_main_flow.py`.

- [ ] **Step 1: Add a private wait helper that returns both match and frame.**

Add:

```python
def _wait_for_connection_interrupted_dialog_frame(
    timeout: float,
) -> tuple[MatchResult, np.ndarray] | None:
```

It must use the same ROI, threshold, polling interval, and timeout as `wait_until_connection_interrupted_dialog()`. On success it returns the already-captured screenshot together with the dialog match. On timeout it returns `None` without restoring network.

- [ ] **Step 2: Let retry-button detection inspect the first frame before taking another screenshot.**

Add an optional first-frame argument to the private path used by `_discard_pending_request_and_prepare_next_probe()`:

```python
def _wait_for_retry_button(
    timeout: float,
    *,
    first_screenshot: np.ndarray | None = None,
) -> MatchResult | None:
```

The first operation must call `find_connection_retry_button(first_screenshot, require_dialog=True)`. Only if it is absent may the loop capture another screenshot. Keep the current 4-second retry-button budget.

- [ ] **Step 3: Wire the shared frame into the red discard flow.**

Replace the two independent waits with the private helpers while preserving these outcomes:

```text
dialog timeout -> latch fail-closed, leave DROP/REJECT enabled
retry timeout -> latch fail-closed, leave DROP/REJECT enabled
retry found -> restore network, click retry, enter activity
```

Do not increase the polling rate or lower the timeout as a substitute for frame reuse.

- [ ] **Step 4: Test same-frame retry detection.**

Provide one screenshot containing both the full connection panel and retry button. Assert that the retry helper returns without calling `adb.read_screenshot()` a second time.

- [ ] **Step 5: Test delayed retry detection and fail-closed timeout.**

Provide a first frame containing only the dialog, then a later frame containing the button. Assert one later capture occurs and the retry match is returned. Provide only dialog frames until timeout and assert no network-disabling call is made.

- [ ] **Step 6: Run focused tests.**

Run:

```powershell
py -3.11 -m unittest tests.test_main_flow -q
```

Expected: all prompt and fail-closed tests pass.

## Task 4: Validate Three-Frame Red Result Analysis Offline

**Files:**
- Modify: `tests/test_red_scout.py` and `tests/test_main_flow.py`.
- Inspect: `_debug/red_scout_samples/` real sample directories.

- [ ] **Step 1: Build a replay fixture from existing red-scout samples.**

For each sample with `before_*.png`, at least three `after_*.png`, and `analysis.json`, load the images and call the existing analyzer with the same click points, grid size, excluded cells, learned footprint, and submarine lengths recorded by the sample.

- [ ] **Step 2: Compare three-frame and four-frame outputs.**

For every sample, compare this exact tuple:

```python
(
    result.valid,
    result.affected_cells,
    result.hit_cells,
    result.miss_cells,
    result.unknown_cells,
)
```

The replay test passes only when the three-frame tuple equals the four-frame tuple for every candidate sample. A mismatch, new `unknown`, or new invalid result blocks runtime frame reduction.

- [ ] **Step 3: Record the evidence distribution.**

Count how many samples are safe for three frames and how many require four. Keep the result in the test output or a short comment in the test so future timing changes have a fixed evidence baseline.

- [ ] **Step 4: Run the replay tests.**

Run:

```powershell
py -3.11 -m unittest tests.test_red_scout tests.test_main_flow -q
```

Expected: current samples either prove three-frame equivalence or explicitly block the adaptive runtime step.

## Task 5: Add Guarded Red Result Capture

**Files:**
- Modify: `main.py:1185-1195` and the red transaction around `main.py:1677-1682`.
- Test: `tests/test_main_flow.py`.

- [ ] **Step 1: Keep the existing four-frame schedule as the fallback.**

Refactor `_capture_red_result_frames()` to accept an explicit schedule while preserving the current default:

```python
def _capture_red_result_frames(
    sample_dir: Path | None = None,
    *,
    frame_delays: Sequence[float] = HIT_RESULT_FRAME_DELAYS,
) -> list[np.ndarray]:
```

No caller may receive fewer than three valid frames. Any capture or analysis error must use the existing interrupted-transaction path.

- [ ] **Step 2: Add a conservative early-stop decision.**

Only enable the three-frame schedule when Task 4 proves equivalence and the first three frames produce a complete, valid six-cell result with no unknown cells. Otherwise capture the fourth frame and analyze all four. The decision must be fail-closed: uncertainty selects four frames, not three.

- [ ] **Step 3: Preserve result-file evidence.**

When three frames are accepted, write `frame_count=3` and `adaptive_frames_stopped=True` to the red-scout sample metadata. When the gate is not satisfied, write `frame_count=4` and leave the existing evidence intact.

- [ ] **Step 4: Test schedule selection.**

Add tests for all three branches:

```text
equivalence proven + valid complete six-cell result -> three frames
unknown/invalid/incomplete result -> four frames
capture exception -> interrupted transaction and fail-closed cleanup
```

- [ ] **Step 5: Run focused and full tests.**

Run:

```powershell
py -3.11 -m unittest tests.test_main_flow tests.test_red_scout -q
py -3.11 -m py_compile main.py utils/red_scout.py utils/adb_control.py
```

Expected: all tests pass and compilation succeeds.

## Task 6: Real-Device Acceptance and Rollback Criteria

**Files:**
- No production file changes in this task; inspect `_debug/logs/bbma.log` and runtime status.

- [ ] **Step 1: Run three red-scout attempts on level 5 and three on level 6.**

Use the same emulator, resolution, server, and network conditions as the baseline. Do not mix unit-test log output with the runtime log.

- [ ] **Step 2: Compare p50 and p90 stage timings.**

Use the stage markers from Task 1. The initial targets are:

```text
REQUEST_PENDING -> RESULT_VISIBLE: reduce from about 8s to 6.5-7s
RESULT_VISIBLE -> RESULT_RECORDED: reduce from about 5s to 4-4.5s only when the adaptive gate is proven
```

These are goals, not permission to lower safety timeouts.

- [ ] **Step 3: Verify safety invariants after every run.**

Confirm all of the following:

```text
no red ammunition fingerprint mismatch
no pending probe marker remains after a successful transaction
no unexpected REQUEST_COMMITTED transition in a red transaction
no fail-closed latch on a normal successful run
no increase in unknown or invalid red results
```

- [ ] **Step 4: Roll back a stage independently when its acceptance criteria fail.**

Disable only the fast activity path if detail re-entry failures increase. Disable only the three-frame gate if any replay or real run changes affected/hit/miss/unknown output. Keep prompt-frame reuse if it changes no state or classification behavior.

- [ ] **Step 5: Update the timing report.**

Record baseline versus optimized p50/p90 values, frame counts, fallback counts, and any fail-closed events. Do not claim an average improvement from a single successful run.

## Self-Review Checklist

- [ ] Both requested stages have an independent optimization task.
- [ ] Every speedup has an explicit fallback path.
- [ ] No plan step removes network isolation, request discard, or ammunition verification.
- [ ] Three-frame capture is blocked until real sample replay proves equivalent output.
- [ ] Tests cover fast success, fallback, timeout, exceptions, and fail-closed behavior.
- [ ] Acceptance uses p50/p90 real-device timings rather than fixed timeout values.
