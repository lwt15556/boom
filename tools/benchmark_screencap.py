from __future__ import annotations

"""Benchmark the PNG vs raw-framebuffer screencap paths on the target device.

Usage (run on the machine with the emulator connected, from the project root)::

    .venv\\Scripts\\python.exe tools\\benchmark_screencap.py [runs_per_path]

It measures wall-clock time for a fixed number of screenshots through:
  * capture_screenshot_png()  -- ``exec-out screencap -p`` (PNG decode)
  * _try_capture_raw_screencap() -- ``exec-out screencap`` (raw fb decode)

It also saves one example image from each path to ``_debug/`` so you can eyeball
that the raw decode produced the right colors and orientation, and reports the
dimensions and mean absolute pixel difference between the two paths.
"""

import os
import sys
from pathlib import Path
from time import perf_counter
from statistics import median

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.adb_control import AdbController, RAW_SCREENCAP_ENV, raw_screencap_enabled

DEBUG_DIR = PROJECT_ROOT / "_debug"
OUT_PNG = DEBUG_DIR / "bench_screencap_png.png"
OUT_RAW = DEBUG_DIR / "bench_screencap_raw.png"


def stats(values: list[float]) -> str:
    if not values:
        return "--"
    mean = sum(values) / len(values)
    return (
        f"n={len(values)} mean={mean*1000:.0f}ms "
        f"med={median(values)*1000:.0f}ms min={min(values)*1000:.0f}ms max={max(values)*1000:.0f}ms"
    )


def mean_abs_diff(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(diff.mean())


def main() -> int:
    try:
        runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    except ValueError:
        runs = 20
    runs = max(3, min(200, runs))
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    adb = AdbController()
    width, height = adb.get_screen_size()
    print(f"device screen size: {width}x{height}")
    print(f"raw_screencap_env default: raw_screencap_enabled()={raw_screencap_enabled()} ({RAW_SCREENCAP_ENV})")
    print(f"runs per path: {runs}\n")

    # Warm-up so the first (cold) adb/screencap round-trip is not counted.
    try:
        adb.read_screenshot()
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"warm-up screenshot failed: {exc}")

    # ---- PNG path ----
    png_times: list[float] = []
    png_cap = None
    for _ in range(runs):
        t0 = perf_counter()
        png_cap = adb.capture_screenshot_png()
        png_times.append(perf_counter() - t0)
    png_image = png_cap.image if png_cap is not None else None
    if png_image is not None:
        try:
            png_cap.save(OUT_PNG)
            print(f"png path example saved: {OUT_PNG}")
        except OSError as exc:
            print(f"png save failed: {exc}")

    # ---- raw path ----
    raw_times: list[float] = []
    raw_cap = None
    for _ in range(runs):
        t0 = perf_counter()
        raw_cap = adb._try_capture_raw_screencap()
        if raw_cap is None:
            print("raw capture returned None (falling back); aborting raw timing")
            break
        raw_times.append(perf_counter() - t0)
    raw_image = raw_cap.image if raw_cap is not None else None
    if raw_image is not None:
        try:
            raw_cap.save(OUT_RAW)
            print(f"raw path example saved: {OUT_RAW}")
        except OSError as exc:
            print(f"raw save failed: {exc}")

    print("\n--- timing ---")
    print(f"png : {stats(png_times)}")
    print(f"raw : {stats(raw_times)}")

    if png_image is not None and raw_image is not None:
        print("\n--- image comparison ---")
        print(f"png shape: {png_image.shape}")
        print(f"raw shape: {raw_image.shape}")
        diff = mean_abs_diff(png_image, raw_image)
        print(f"mean abs diff (png vs raw): {diff if diff is None else round(diff, 3)}")
        print("NOTE: a non-zero diff usually means the screen changed between the two")
        print("captures, OR the raw channel order/stride is wrong. Inspect the two saved")
        print("PNGs to confirm both look identical in colour and orientation.")
    elif raw_image is None:
        print("\nraw path did not produce an image; keep using the PNG path (BBMA_RAW_SCREENCAP not set).")

    # ---- end-to-end env toggle check ----
    print("\n--- env toggle check ---")
    for value in ("1", "0"):
        os.environ[RAW_SCREENCAP_ENV] = value
        cap = adb.capture_screenshot()
        print(f"BBMA_RAW_SCREENCAP={value!r}: enabled={raw_screencap_enabled()} "
              f"got_image={cap is not None and cap.image is not None} shape={None if cap is None or cap.image is None else cap.image.shape}")
    os.environ.pop(RAW_SCREENCAP_ENV, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
