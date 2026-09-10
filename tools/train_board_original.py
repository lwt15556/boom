from __future__ import annotations

"""Train the board-cell recognizer on ORIGINAL (non-binarized) screenshots.

与 ``train_board_real.py`` 的区别：特征来自**原始整帧截图**（灰度归一化），
而不是二值化裁剪。原图保留更多信息（明暗/纹理），上限更高。

样本配对方式（同名，靠时间戳）：
    识图/原图训练样本/<stem>.png   <- 原始整帧截图（颜色）
    识图/训练样本/<stem>.json      <- 同一时刻的标签 + 棋盘 quad

训练出的模型 json 会带 ``"feature_mode": "gray"``，推理端
(``utils/board_recognizer.classify_board_cells``) 会自动按灰度特征提取，
保证训练/推理一致。

用法：
    .venv\\Scripts\\python.exe tools\\train_board_original.py [--epochs 200] [--sub-weight 6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from train_board_recognition import FEATURE_SIZE, HIDDEN, OUTPUTS, MLP, WEIGHTS_PATH  # noqa: E402

from utils.board_recognizer import (  # noqa: E402
    FEATURE_MODE_GRAY,
    PAD,
    cell_centers_perspective,
    cell_feature,
    prepare_frame,
    quad_crop_box,
)
from utils.image_io import read_image_compat  # noqa: E402

ORIGINAL_DIR = PROJECT_ROOT / "识图" / "原图训练样本"
LABEL_DIR = PROJECT_ROOT / "识图" / "训练样本"


def load_samples(original_dir: Path, label_dir: Path):
    """读原图 + 配对 json，返回 (feats, labels, 用到的样本数)。"""
    feats: list[np.ndarray] = []
    labels: list[int] = []
    used = 0
    missing_original = missing_json = 0
    for json_path in sorted(label_dir.glob("*.json")):
        original_path = original_dir / f"{json_path.stem}.png"
        if not original_path.exists():
            missing_original += 1
            continue
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
            labels_grid = record.get("labels")
            quad = record.get("quad")
            grid_size = record.get("grid_size")
            if not labels_grid or not quad or not grid_size:
                missing_json += 1
                continue
            n = int(grid_size)
            image = read_image_compat(original_path)
            if image is None:
                continue
            gray = prepare_frame(image, FEATURE_MODE_GRAY)
            x0, y0, x1, y1 = quad_crop_box(quad, gray.shape)
            crop = gray[y0:y1, x0:x1]
            crop_h, crop_w = crop.shape[:2]
            quad_crop = np.asarray(quad, dtype=np.float32) - np.array([x0, y0], dtype=np.float32)
            centers, _ = cell_centers_perspective(quad_crop, n)
            qx0 = min(p[0] for p in quad_crop); qy0 = min(p[1] for p in quad_crop)
            w = max(p[0] for p in quad_crop) - qx0
            h = max(p[1] for p in quad_crop) - qy0
            if w <= 0 or h <= 0:
                continue
            a = w / n
            b = h / n
            for index, (cx, cy) in enumerate(centers):
                row, col = index // n, index % n
                left = int(max(0, cx - a / 2 - PAD)); right = int(min(crop_w, cx + a / 2 + PAD))
                top = int(max(0, cy - b / 2 - PAD)); bottom = int(min(crop_h, cy + b / 2 + PAD))
                if right <= left or bottom <= top:
                    continue
                box = crop[top:bottom, left:right]
                feats.append(cell_feature(box, FEATURE_MODE_GRAY))
                lbl = int(labels_grid[row][col]) if row < len(labels_grid) and col < len(labels_grid[row]) else 0
                labels.append(lbl)
            used += 1
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
            print(f"  skip {json_path.name}: {exc}")
    if missing_original:
        print(f"  [提示] {missing_original} 个 json 没有配对的原图（{original_dir}），已跳过")
    if not feats:
        raise RuntimeError(
            f"no paired original samples found; 把原图按同名放进 {original_dir} 后再训练"
        )
    return np.array(feats, np.float32), np.array(labels, np.int64), used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original-dir", default=str(ORIGINAL_DIR))
    parser.add_argument("--label-dir", default=str(LABEL_DIR))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sub-weight", type=float, default=6.0)
    args = parser.parse_args(argv)

    feats, labels, used = load_samples(Path(args.original_dir), Path(args.label_dir))
    n_sub = int(labels.sum())
    print(f"loaded {used} original sample(s), {len(feats)} cells ({n_sub} submarine / {len(feats) - n_sub} water)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(feats))
    split = int(len(feats) * 0.8)
    tr, te = perm[:split], perm[split:]
    Xtr, ytr = feats[tr], labels[tr]
    Xte, yte = feats[te], labels[te]

    model = MLP(FEATURE_SIZE * FEATURE_SIZE, HIDDEN, OUTPUTS)
    print(f"training on original samples (feature=gray, sub_weight={args.sub_weight}) ...")
    model.train(Xtr, ytr, epochs=args.epochs, sub_weight=args.sub_weight)
    pred = model.predict(Xte)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum()); fn = int(((pred == 0) & (yte == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"test acc: {(pred == yte).mean():.3f}  precision={prec:.3f}  recall={rec:.3f} (tp={tp} fp={fp} fn={fn})")

    payload = model.to_dict()
    payload["feature_mode"] = FEATURE_MODE_GRAY
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(payload))
    print(f"saved weights (feature_mode=gray) -> {WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
