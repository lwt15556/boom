from __future__ import annotations

"""Train the board-cell recognizer on REAL captured samples.

Reads samples written by ``_capture_training_sample`` (in ``识图\\训练样本``):
each sample is a fixed-region binarized board crop (.png) + a .json with the
per-cell label grid and the start-of-level board quad.  It aligns each cell via
the saved quad (perspective transform), extracts a resized cell crop, and trains
the same numpy MLP used by the synthetic trainer.

Usage:
    .venv\\Scripts\\python.exe tools\\train_board_real.py [--dir 识图/训练样本] [--epochs 150]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from train_board_recognition import FEATURE_SIZE, HIDDEN, OUTPUTS, MLP, WEIGHTS_PATH  # noqa: E402

DEFAULT_DIR = PROJECT_ROOT / "识图" / "训练样本"
ORIGINAL_DIR = PROJECT_ROOT / "识图" / "原图训练样本"
CROP_ORIGIN = (230, 40)  # the fixed crop origin (x, y) used by the collector
CROP_SIZE = (1100 - 230, 650 - 40)  # (w, h) of the fixed crop


def load_wave_samples(original_dir: Path, label_dir: Path, n_per_frame: int = 15, patch: int = 52):
    """从原图**右上角（棋盘菱形之外）**采"海浪"样本，返回特征列表（标签为 3）。

    这一片是浅水/白泡沫，是确定的海浪，用来教模型"海浪长什么样"，
    这样棋盘内的格子若也是海浪外观，就会被判成海浪类而不是残骸。
    """
    from utils.board_recognizer import FEATURE_MODE_BINARY, cell_feature, prepare_frame
    from utils.image_io import read_image_compat

    feats: list[np.ndarray] = []
    for op in sorted(original_dir.glob("*.png")):
        jp = label_dir / f"{op.stem}.json"
        if not jp.exists():
            continue
        try:
            record = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        quad = record.get("quad")
        if not quad:
            continue
        img = read_image_compat(op)
        if img is None:
            continue
        binary = prepare_frame(img, FEATURE_MODE_BINARY)
        h, w = binary.shape[:2]
        poly = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
        rng = np.random.default_rng(abs(hash(op.stem)) % (2**32))
        got = 0
        for _ in range(3000):
            x = int(rng.integers(int(0.60 * w), max(int(0.60 * w) + 1, w - patch)))
            y = int(rng.integers(0, max(1, int(0.40 * h) - patch)))
            if x + patch > w or y + patch > h:
                continue
            # 只取棋盘菱形之外（那片是海浪/浅水）
            if cv2.pointPolygonTest(poly, (float(x + patch / 2), float(y + patch / 2)), False) >= 0:
                continue
            box = binary[y : y + patch, x : x + patch]
            if box.shape[0] < 20 or box.shape[1] < 20:
                continue
            feats.append(cell_feature(box, FEATURE_MODE_BINARY))
            got += 1
            if got >= n_per_frame:
                break
    return feats


def _quad_to_crop(quad, width: int, height: int) -> np.ndarray:
    """Map the full-frame quad (1280x720) into the fixed-crop coordinate system."""
    sx = width / CROP_SIZE[0]
    sy = height / CROP_SIZE[1]
    pts = []
    for point in quad:
        x = float(point[0]) - CROP_ORIGIN[0]
        y = float(point[1]) - CROP_ORIGIN[1]
        pts.append((x * sx, y * sy))
    return np.array(pts, np.float32)


def load_samples(sample_dir: Path, only_v2: bool = False, labels_key: str | None = None):
    feats, labels = [], []
    count = 0
    for json_path in sorted(sample_dir.glob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
            png_path = json_path.with_suffix(".png")
            if not png_path.exists():
                continue
            # 优先用"红旗+模板"重打的标签（labels_v2），没有才退回旧标签。
            v2 = record.get("labels_v2")
            if only_v2 and not v2:
                continue
            img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # 跳过近全白/全黑的空样本（棋盘不在画面/过渡帧），避免污染模型。
            black_frac = float((img < 128).mean())
            if black_frac < 0.02 or black_frac > 0.98:
                print(f"  skip blank sample: {json_path.name} (black_frac={black_frac:.3f})")
                continue
            labels_grid = record.get(labels_key) if labels_key else (v2 or record.get("labels"))
            quad = record.get("quad")
            grid_size = record.get("grid_size")
            if not labels_grid or not quad or not grid_size:
                continue
            n = int(grid_size)
            crop_origin = record.get("crop_origin")
            if crop_origin:
                ox, oy = float(crop_origin[0]), float(crop_origin[1])
                quad_crop = np.asarray(quad, np.float32) - np.array([ox, oy], np.float32)
            else:
                quad_crop = _quad_to_crop(quad, img.shape[1], img.shape[0])
            from utils.board_recognizer import cell_centers_perspective

            centers, _matrix = cell_centers_perspective(quad_crop, n)
            x0 = min(p[0] for p in quad_crop); y0 = min(p[1] for p in quad_crop)
            w = max(p[0] for p in quad_crop) - x0
            h = max(p[1] for p in quad_crop) - y0
            a = w / n; b = h / n
            pad = 6
            for idx, (cx, cy) in enumerate(centers):
                row, col = idx // n, idx % n
                left = int(max(0, cx - a / 2 - pad)); right = int(min(img.shape[1], cx + a / 2 + pad))
                top = int(max(0, cy - b / 2 - pad)); bottom = int(min(img.shape[0], cy + b / 2 + pad))
                if right <= left or bottom <= top:
                    continue
                box = img[top:bottom, left:right]
                box = cv2.resize(box, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
                feats.append((box < 128).astype(np.float32).reshape(-1))
                lbl = int(labels_grid[row][col]) if row < len(labels_grid) and col < len(labels_grid[row]) else 0
                labels.append(lbl)
            count += 1
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"  skip {json_path.name}: {exc}")
    if not feats:
        raise RuntimeError(f"no samples found in {sample_dir}; run the game to collect them")
    return np.array(feats, np.float32), np.array(labels, np.int64), count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help=f"sample dir (default {DEFAULT_DIR})")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--sub-weight", type=float, default=6.0,
                        help="放大潜艇类的梯度，抵消类别不平衡（真实样本潜艇占~3.5%）；1=不放大")
    parser.add_argument("--only-v2", action="store_true",
                        help="只用「红旗+模板」重打的标签（labels_v2）的样本训练")
    parser.add_argument("--classes", type=int, default=2, choices=(2, 3),
                        help="2=水/内容；3=水/潜艇/残骸（用 labels_v2_3class）")
    parser.add_argument("--wave", action="store_true",
                        help="额外从原图右上角（棋盘外）采海浪样本，训练 4 类：水/潜艇/残骸/海浪")
    parser.add_argument("--original-dir", default=str(ORIGINAL_DIR), help="原图目录（--wave 用）")
    args = parser.parse_args(argv)

    n_classes = int(args.classes)
    if args.wave:
        n_classes = max(n_classes, 4)
    labels_key = "labels_v2_3class" if n_classes >= 3 else None
    sample_dir = Path(args.dir)
    feats, labels, count = load_samples(
        sample_dir, only_v2=bool(args.only_v2), labels_key=labels_key
    )
    if args.wave:
        wave_feats = load_wave_samples(Path(args.original_dir), sample_dir)
        if wave_feats:
            feats = np.concatenate([feats, np.stack(wave_feats).astype(np.float32)], axis=0)
            labels = np.concatenate([labels, np.full(len(wave_feats), 3, np.int64)], axis=0)
        print(f"海浪样本（右上角，棋盘外）: {len(wave_feats)} 个 -> class 3")
    dist = [int((labels == k).sum()) for k in range(n_classes)]
    print(f"loaded {count} sample(s), {len(feats)} cells; 类别分布 {dist}")

    # 随机打乱切分，避免按关卡排序导致训练/测试分布不同（顺序切分会让测试集全是高关卡）。
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(feats))
    split = int(len(feats) * 0.8)
    tr, te = perm[:split], perm[split:]
    Xtr, ytr = feats[tr], labels[tr]
    Xte, yte = feats[te], labels[te]
    model = MLP(FEATURE_SIZE * FEATURE_SIZE, HIDDEN, n_classes)
    if n_classes >= 3:
        total = sum(dist)
        cw = np.array([total / (n_classes * max(1, d)) for d in dist], np.float32)
        print(f"training ({n_classes} 类, class_weight={[round(float(v), 2) for v in cw]}) ...")
        model.train(Xtr, ytr, epochs=args.epochs, class_weight=cw)
    else:
        print(f"training (2 类, sub_weight={args.sub_weight}) ...")
        model.train(Xtr, ytr, epochs=args.epochs, sub_weight=args.sub_weight)
    pred = model.predict(Xte)
    print(f"test acc: {(pred == yte).mean():.3f}")
    for k in range(n_classes):
        tp = int(((pred == k) & (yte == k)).sum())
        fp = int(((pred == k) & (yte != k)).sum())
        fn = int(((pred != k) & (yte == k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  class {k}: precision={prec:.3f} recall={rec:.3f} (tp={tp} fp={fp} fn={fn})")
    from utils.board_recognizer import FEATURE_MODE_BINARY

    payload = model.to_dict()
    payload["feature_mode"] = FEATURE_MODE_BINARY
    payload["n_classes"] = n_classes
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(payload))
    print(f"saved weights (feature_mode=binary, n_classes={n_classes}) -> {WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
