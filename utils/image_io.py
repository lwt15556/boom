"""OpenCV image I/O helpers that also work with non-ASCII Windows paths."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image_compat(
    path: str | Path,
    flags: int = cv2.IMREAD_COLOR,
    *,
    cv2_module: Any | None = None,
) -> np.ndarray | None:
    """Read an image, falling back to byte decoding for Unicode paths."""
    module = cv2_module or cv2
    path_text = str(path)
    image = module.imread(path_text, flags)
    if image is not None:
        return image

    try:
        data = np.fromfile(path_text, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    try:
        return module.imdecode(data, flags)
    except cv2.error:
        return None


def write_image_compat(
    path: str | Path,
    image: np.ndarray,
    *,
    cv2_module: Any | None = None,
) -> bool:
    """Write an image, falling back to encoded bytes for Unicode paths."""
    module = cv2_module or cv2
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_text = str(path_obj)
    try:
        if module.imwrite(path_text, image):
            return True
    except cv2.error:
        pass

    suffix = path_obj.suffix.lower() or ".png"
    try:
        ok, buffer = module.imencode(suffix, image)
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buffer.tofile(path_text)
    except OSError:
        return False
    return True
