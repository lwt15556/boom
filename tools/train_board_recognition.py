from __future__ import annotations

"""
Train a per-cell board recognizer on synthetically generated binarized boards.

Pipeline
--------
1. Generate synthetic boards: place submarines (from a level config) on an n x n
   isometric grid, render a binarized look (black diamond cells + black submarine
   content blobs), and label every cell as water / submarine.
2. Extract per-cell crops (resized binarized crop) + labels.
3. Train a small 2-layer MLP in pure numpy (no external ML deps).
4. Save normalized weights to JSON for later inference.

Usage:
    .venv\\Scripts\\python.exe tools\\train_board_recognition.py
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

CELL_W = 66.0
CELL_H = 44.0
FEATURE_SIZE = 32
HIDDEN = 64
OUTPUTS = 2  # indices: 0=water, 1=submarine
WEIGHTS_PATH = PROJECT_ROOT / "识图" / "board_cell_model.json"


def place_submarines(n: int, lengths: list[int], rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Place straight submarines (length L, horizontal or vertical), no touching (incl diagonal)."""
    occupied: set[tuple[int, int]] = set()
    placements: list[list[tuple[int, int]]] = []

    def touches(seg: list[tuple[int, int]]) -> bool:
        for r, c in seg:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (r + dr, c + dc) in occupied:
                        return True
        return False

    for length in lengths:
        placed = False
        for _ in range(8000):
            horizontal = bool(rng.integers(0, 2))
            if horizontal:
                r = int(rng.integers(0, n))
                c = int(rng.integers(0, n - length + 1))
                seg = [(r, c + i) for i in range(length)]
            else:
                r = int(rng.integers(0, n - length + 1))
                c = int(rng.integers(0, n))
                seg = [(r + i, c) for i in range(length)]
            if touches(seg):
                continue
            placements.append(seg)
            occupied.update(seg)
            placed = True
            break
        if not placed:
            raise RuntimeError(f"could not place length {length} on {n}x{n}")
    return placements


def grid_geometry(n: int, canvas_w: int, canvas_h: int) -> tuple[float, float, float, float]:
    a = CELL_W / 2.0
    b = CELL_H / 2.0
    ox = canvas_w / 2.0
    oy = canvas_h / 2.0 - b * (n - 1)
    return ox, oy, a, b


def isometric_pos(ox: float, oy: float, a: float, b: float, row: int, col: int) -> tuple[float, float]:
    return ox + a * (col - row), oy + b * (col + row)


def render_board(n: int, placements: list[list[tuple[int, int]]]) -> np.ndarray:
    """Render a binarized board: black diamond cells + black submarine blobs on white."""
    canvas_w, canvas_h = 1280, 720
    img = np.full((canvas_h, canvas_w), 255, np.uint8)
    ox, oy, a, b = grid_geometry(n, canvas_w, canvas_h)

    def diamond(cx: float, cy: float, scale: float):
        ah, bh = a * scale, b * scale
        return np.array(
            [[cx, cy - bh], [cx + ah, cy], [cx, cy + bh], [cx - ah, cy]], dtype=np.int32
        )

    for row in range(n):
        for col in range(n):
            cx, cy = isometric_pos(ox, oy, a, b, row, col)
            cv2.fillPoly(img, [diamond(cx, cy, 0.9).reshape(-1, 1, 2)], 0)

    for seg in placements:
        row_set = {r for r, _ in seg}
        if len(row_set) == 1:  # horizontal
            row = seg[0][0]
            cols = [c for _, c in seg]
            x0, y0 = isometric_pos(ox, oy, a, b, row, cols[0])
            x1, y1 = isometric_pos(ox, oy, a, b, row, cols[-1])
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cv2.ellipse(img, (int(cx), int(cy)), (int(abs(x1 - x0) / 2 + a * 0.8), int(b * 1.5)), 0,
                        0, 360, 0, -1)
        else:  # vertical
            col = seg[0][1]
            rows = [r for r, _ in seg]
            x0, y0 = isometric_pos(ox, oy, a, b, rows[0], col)
            x1, y1 = isometric_pos(ox, oy, a, b, rows[-1], col)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cv2.ellipse(img, (int(cx), int(cy)), (int(a * 1.5), int(abs(y1 - y0) / 2 + b * 0.8)), 90,
                        0, 360, 0, -1)
    return img


def cell_crops(n: int, placements: list[list[tuple[int, int]]], img: np.ndarray):
    """Yield (feature_vector, label) for every cell; label 1 == submarine."""
    ox, oy, a, b = grid_geometry(n, img.shape[1], img.shape[0])
    sub_cells = {(r, c) for seg in placements for (r, c) in seg}
    pad = 6
    for row in range(n):
        for col in range(n):
            cx, cy = isometric_pos(ox, oy, a, b, row, col)
            x0 = int(max(0, cx - a - pad)); x1 = int(min(img.shape[1], cx + a + pad))
            y0 = int(max(0, cy - b - pad)); y1 = int(min(img.shape[0], cy + b + pad))
            crop = img[y0:y1, x0:x1]
            crop = cv2.resize(crop, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
            feat = (crop < 128).astype(np.float32)  # 1 where black
            label = 1 if (row, col) in sub_cells else 0
            yield feat.reshape(-1), label


class MLP:
    """2-layer network (input -> hidden relu -> softmax) trained with SGD + momentum."""

    def __init__(self, input_dim: int, hidden: int, output: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (input_dim + hidden))
        self.W1 = rng.uniform(-limit, limit, (input_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        limit2 = np.sqrt(6.0 / (hidden + output))
        self.W2 = rng.uniform(-limit2, limit2, (hidden, output)).astype(np.float32)
        self.b2 = np.zeros(output, np.float32)

    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2 + self.b2
        z2 = z2 - z2.max(axis=1, keepdims=True)
        e = np.exp(z2)
        p = e / e.sum(axis=1, keepdims=True)
        return a1, p

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 150, lr: float = 0.05, batch: int = 64,
              sub_weight: float = 1.0, class_weight: np.ndarray | None = None):
        """SGD + momentum.

        ``sub_weight`` 放大"潜艇"类（label==1）的梯度，用于抵消类别不平衡。
        ``class_weight`` 给定时按类别逐类加权（用于 3 类：水/潜艇/残骸）。
        """
        n_out = int(self.W2.shape[1])
        y1h = np.zeros((len(y), n_out), np.float32)
        y1h[np.arange(len(y)), y] = 1
        vW1 = np.zeros_like(self.W1); vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2); vb2 = np.zeros_like(self.b2)
        mom = 0.9
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s : s + batch]
                Xb, yb = X[bi], y1h[bi]
                a1, p = self._forward(Xb)
                # 按真值类别加权（抵消类别不平衡）
                if class_weight is not None:
                    weight = np.asarray(class_weight, np.float32)[y[bi]][:, None]
                else:
                    weight = np.where(y[bi] == 1, sub_weight, 1.0).astype(np.float32)[:, None]
                dz2 = (p - yb) * weight
                dW2 = a1.T @ dz2 / len(Xb) + 1e-4 * self.W2
                db2 = dz2.mean(axis=0) + 1e-4 * self.b2
                dz1 = (dz2 @ self.W2.T) * (a1 > 0)
                dW1 = Xb.T @ dz1 / len(Xb) + 1e-4 * self.W1
                db1 = dz1.mean(axis=0) + 1e-4 * self.b1
                vW1 = mom * vW1 + (1 - mom) * dW1; vb1 = mom * vb1 + (1 - mom) * db1
                vW2 = mom * vW2 + (1 - mom) * dW2; vb2 = mom * vb2 + (1 - mom) * db2
                self.W1 -= lr * vW1; self.b1 -= lr * vb1
                self.W2 -= lr * vW2; self.b2 -= lr * vb2
            if epoch % 25 == 0 or epoch == epochs - 1:
                _, p = self._forward(X)
                pr = p.argmax(1)
                tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                print(f"  epoch {epoch}: acc={float((pr == y).mean()):.3f} prec={prec:.3f} recall={rec:.3f} (tp={tp} fp={fp} fn={fn})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, p = self._forward(X)
        return p.argmax(1)

    def to_dict(self) -> dict:
        return {"W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist()}


def generate_dataset(n: int, lengths: list[int], samples: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for _ in range(samples):
        placements = place_submarines(n, lengths, rng)
        img = render_board(n, placements)
        for feat, lab in cell_crops(n, placements, img):
            feats.append(feat)
            labels.append(lab)
    return np.array(feats, np.float32), np.array(labels, np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="grid size (default 10)")
    parser.add_argument("--submarines", type=str, default="5,4,3,3", help="comma list of submarine lengths")
    parser.add_argument("--samples", type=int, default=300, help="number of synthetic boards")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    lengths = [int(v) for v in args.submarines.replace("，", ",").split(",")]
    print(f"generating {args.samples} synthetic {args.n}x{args.n} boards ...")
    feats, labels = generate_dataset(args.n, lengths, args.samples, args.seed)
    n_sub = int(labels.sum())
    print(f"dataset: {len(feats)} cells  ({n_sub} submarine / {len(feats) - n_sub} water)")

    split = int(len(feats) * 0.8)
    Xtr, ytr = feats[:split], labels[:split]
    Xte, yte = feats[split:], labels[split:]

    model = MLP(FEATURE_SIZE * FEATURE_SIZE, HIDDEN, OUTPUTS)
    print("training ...")
    model.train(Xtr, ytr, epochs=args.epochs)
    pred = model.predict(Xte)
    print(f"test acc: {(pred == yte).mean():.3f}")
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(model.to_dict()))
    print(f"saved weights -> {WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
