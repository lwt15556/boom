"""用规则标签训练开局识图模型（3 类：水/海浪、完整潜艇、残骸）。

数据来源：``识图/原图训练样本`` 里的**全部原图**，标签取自
``tools/label_board_rules.py`` 写入的 ``labels_rules_3class``。

特征与线上推理完全一致：``utils.board_recognizer.extract_cell_features``
（整帧预处理 → quad 透视定位 → 逐格固定框 → 32x32 特征）。

用法::

    .venv\\Scripts\\python.exe tools\\train_board_rules.py --feature-mode gray
    .venv\\Scripts\\python.exe tools\\train_board_rules.py --feature-mode binary --epochs 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from train_board_recognition import FEATURE_SIZE, HIDDEN, MLP, WEIGHTS_PATH  # noqa: E402

from utils.board_recognizer import (  # noqa: E402
    FEATURE_MODE_BINARY,
    FEATURE_MODE_GRAY,
    extract_cell_features,
)
from utils.image_io import read_image_compat  # noqa: E402

IMAGE_DIR = PROJECT_ROOT / "识图" / "原图训练样本"
LABEL_DIR = PROJECT_ROOT / "识图" / "训练样本"
LABEL_KEY = "labels_rules_3class"
CLASS_NAMES = ("水/海浪", "潜艇", "残骸")


def load_dataset(mode: str, image_dir: Path, label_dir: Path):
    """返回 ``(frames, labels_by_frame, stats)``；frames 是每帧的特征数组。"""
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    stats: Counter = Counter()
    skipped = 0
    for image_path in sorted(image_dir.glob("*.png")):
        json_path = label_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            skipped += 1
            continue
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped += 1
            continue
        grid = record.get(LABEL_KEY)
        quad = record.get("quad")
        grid_size = record.get("grid_size")
        if not grid or not quad or not grid_size:
            skipped += 1
            continue
        image = read_image_compat(image_path)
        if image is None:
            skipped += 1
            continue
        n = int(grid_size)
        features = extract_cell_features(image, quad, n, mode)
        if len(features) != n * n:
            skipped += 1
            continue
        feats = np.stack([feat for _, feat in features]).astype(np.float32)
        cell_labels = np.array(
            [int(grid[row][col]) for (row, col), _ in features], dtype=np.int64
        )
        frames.append(feats)
        labels.append(cell_labels)
        stats.update(cell_labels.tolist())
    print(f"载入 {len(frames)} 帧（跳过 {skipped}）；类别分布 {dict(stats)}")
    return frames, labels, stats


def _metrics(pred: np.ndarray, truth: np.ndarray, n_classes: int) -> dict:
    out = {}
    for k in range(n_classes):
        tp = int(((pred == k) & (truth == k)).sum())
        fp = int(((pred == k) & (truth != k)).sum())
        fn = int(((pred != k) & (truth == k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[k] = (prec, rec, f1, tp, fp, fn)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-mode", choices=(FEATURE_MODE_BINARY, FEATURE_MODE_GRAY),
                        default=FEATURE_MODE_GRAY, help="特征模式（写入模型，推理自动跟随）")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--save", action="store_true", help="写入 识图/board_cell_model.json（会先备份）")
    parser.add_argument("--image-dir", default=str(IMAGE_DIR))
    parser.add_argument("--label-dir", default=str(LABEL_DIR))
    args = parser.parse_args(argv)

    frames, labels, stats = load_dataset(
        args.feature_mode, Path(args.image_dir), Path(args.label_dir)
    )
    if len(frames) < 10:
        print("样本太少", file=sys.stderr)
        return 2

    # 按帧切分：同一帧的格子高度相关，按格子随机切会让测试集泄漏。
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(frames))
    split = max(1, int(len(frames) * (1.0 - args.test_frac)))
    train_idx, test_idx = order[:split], order[split:]
    Xtr = np.concatenate([frames[i] for i in train_idx])
    ytr = np.concatenate([labels[i] for i in train_idx])
    Xte = np.concatenate([frames[i] for i in test_idx])
    yte = np.concatenate([labels[i] for i in test_idx])
    print(f"训练帧 {len(train_idx)}（{len(Xtr)} 格） / 测试帧 {len(test_idx)}（{len(Xte)} 格）")

    n_classes = 3
    total = sum(stats.values())
    cw = np.array([total / (n_classes * max(1, stats.get(k, 0))) for k in range(n_classes)], np.float32)
    print(f"class_weight = {[round(float(v), 2) for v in cw]}")
    model = MLP(FEATURE_SIZE * FEATURE_SIZE, int(args.hidden), n_classes, seed=args.seed)
    model.train(Xtr, ytr, epochs=args.epochs, lr=args.lr, class_weight=cw)

    pred = model.predict(Xte)
    print(f"\n=== 格子级（{args.feature_mode}）===")
    print(f"整体准确率 {float((pred == yte).mean()):.4f}")
    for k, (prec, rec, f1, tp, fp, fn) in _metrics(pred, yte, n_classes).items():
        print(f"  {CLASS_NAMES[k]:>6}: precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} "
              f"(tp={tp} fp={fp} fn={fn})")
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(yte, pred):
        conf[t, p] += 1
    print("  混淆矩阵（行=真值，列=预测）")
    print("        " + "".join(f"{name:>9}" for name in CLASS_NAMES))
    for k in range(n_classes):
        print(f"  {CLASS_NAMES[k]:>6}" + "".join(f"{conf[k, j]:9d}" for j in range(n_classes)))

    # 帧级：按帧统计"预测格集合 vs 真值格集合"的精确率/召回率。
    print(f"\n=== 帧级（{len(test_idx)} 帧）===")
    agg = {1: [0, 0, 0], 2: [0, 0, 0]}   # class -> [tp, fp, fn]
    for i in test_idx:
        p = model.predict(frames[i])
        t = labels[i]
        for k in (1, 2):
            agg[k][0] += int(((p == k) & (t == k)).sum())
            agg[k][1] += int(((p == k) & (t != k)).sum())
            agg[k][2] += int(((p != k) & (t == k)).sum())
    for k in (1, 2):
        tp, fp, fn = agg[k]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {CLASS_NAMES[k]:>6}: precision={prec:.3f} recall={rec:.3f} (tp={tp} fp={fp} fn={fn})")

    if args.save:
        if WEIGHTS_PATH.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = WEIGHTS_PATH.with_name(f"{WEIGHTS_PATH.stem}.bak_{stamp}{WEIGHTS_PATH.suffix}")
            shutil.copy2(WEIGHTS_PATH, backup)
            print(f"旧模型已备份 -> {backup}")
        payload = model.to_dict()
        payload["feature_mode"] = args.feature_mode
        payload["n_classes"] = n_classes
        payload["classes"] = list(CLASS_NAMES)
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_PATH.write_text(json.dumps(payload), encoding="utf-8")
        print(f"模型已保存 -> {WEIGHTS_PATH} (feature_mode={args.feature_mode}, n_classes={n_classes})")
    else:
        print("\n（未加 --save，仅评估）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
