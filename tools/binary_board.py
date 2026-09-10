from __future__ import annotations

"""Binarize game board screenshots with a fixed crop region (for training).

Usage::

    # single image
    .venv\\Scripts\\python.exe tools\\binary_board.py <image.png>
    # a whole directory of images
    .venv\\Scripts\\python.exe tools\\binary_board.py <input_dir>
    # specify output dir
    .venv\\Scripts\\python.exe tools\\binary_board.py <image.png> --out <out_dir>

For every input image it applies Otsu binarization (cells black on white) and
crops the fixed board region ``FIXED_REGION`` (a normalized 1280x720 frame), then
writes ``<stem>_binary.png`` into the output directory.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fixed board crop region for a 1280x720 screenshot: (x0, y0, x1, y1).
# Chosen so all board cells stay in frame while removing title/left/right/bottom UI.
FIXED_REGION = (230, 40, 1100, 650)

# Default output directory for processed binary boards.
DEFAULT_OUT = PROJECT_ROOT / "识图" / "二值法识图"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_DIMENSION = 1920

# Black components smaller than this (in a ~1280x720 frame) are sea-foam speckle
# noise and are dropped; board cells / submarines are several hundred px+.
MIN_COMPONENT_AREA = 300


def normalize_hue_gray(gray: np.ndarray) -> np.ndarray:
    """Otsu binarize then invert -> white cells on a black background."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


def clean_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Drop small white regions (sea-foam speckle) keeping the real board cells.

    With white-cells-on-black, the sea highlight becomes small white speckles on
    the black background; keep only components with area >= ``min_area`` so the
    background stays uniform black.
    """
    if min_area <= 0:
        return binary
    white = (binary == 255).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    out = np.zeros_like(binary)
    for index in range(1, num):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            out[labels == index] = 255
    return out


def process_image(
    path: Path,
    out_dir: Path,
    region: tuple[int, int, int, int],
    min_area: int = MIN_COMPONENT_AREA,
) -> Path | None:
    try:
        image = cv2.imread(str(path))
    except Exception:
        return None
    if image is None:
        return None

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = normalize_hue_gray(gray)

    # The fixed region is specified for a 1280x720 frame; scale it proportionally
    # if the input has a different resolution (keeps the same relative position).
    sx = width / 1280.0 if width else 1.0
    sy = height / 720.0 if height else 1.0
    x0, y0, x1, y1 = region
    x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
    y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = binary[y0:y1, x0:x1]
    crop = clean_small_components(crop, min_area)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_binary.png"
    cv2.imwrite(str(out_path), crop)
    return out_path


def iter_inputs(path: Path) -> Path:
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                yield child
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input image file or directory")
    parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--region",
        default=", ".join(str(v) for v in FIXED_REGION),
        help=f"fixed crop (x0,y0,x1,y1) in a 1280x720 frame (default: {FIXED_REGION})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=MIN_COMPONENT_AREA,
        help=f"drop black components smaller than this (sea-foam); 0 disables (default: {MIN_COMPONENT_AREA})",
    )
    args = parser.parse_args(argv)

    try:
        region = tuple(int(v) for v in str(args.region).replace("，", ",").split(","))
        if len(region) != 4:
            raise ValueError("region must be x0,y0,x1,y1")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"input does not exist: {input_path}")

    processed = 0
    for image_path in iter_inputs(input_path):
        result = process_image(image_path, out_dir, region, min_area=args.min_area)
        if result is not None:
            print(f"  {image_path.name} -> {result}")
            processed += 1
    print(f"done: {processed} image(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
