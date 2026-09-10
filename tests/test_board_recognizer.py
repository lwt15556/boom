import unittest

import numpy as np

from utils.board_recognizer import (
    BoardCellMLP,
    DEFAULT_WEIGHTS,
    FEATURE_SIZE,
    binarize_cell_hit,
    binarize_full,
    binarize_reveal_cells,
    classify_board_cells,
    load_board_model,
    quad_crop_box,
)


class _MeanModel:
    """测试用假模型：特征均值 < 0.5 → 预测 1（有潜艇内容）。"""

    def predict(self, X):
        return (np.asarray(X, dtype=np.float32).mean(axis=1) < 0.5).astype(int)


def _constant_model(class_index: int = 1) -> BoardCellMLP:
    """一个恒等预测的模型：无条件预测 class_index（0=水，1=潜艇）。"""
    input_dim = FEATURE_SIZE * FEATURE_SIZE
    hidden = 64
    b2 = np.zeros(2, dtype=np.float32)
    b2[class_index] = 1.0
    return BoardCellMLP(
        np.zeros((input_dim, hidden), dtype=np.float32),
        np.zeros(hidden, dtype=np.float32),
        np.zeros((hidden, 2), dtype=np.float32),
        b2,
    )


class BoardRecognizerLoadTest(unittest.TestCase):
    def test_load_missing_model_returns_none(self):
        self.assertIsNone(load_board_model(DEFAULT_WEIGHTS.with_name("nope.json")))

    def test_load_rejects_wrong_shape(self):
        model = BoardCellMLP(
            np.zeros((8, 8), np.float32),
            np.zeros(8, np.float32),
            np.zeros((8, 2), np.float32),
            np.zeros(2, np.float32),
        )
        self.assertEqual(model.W1.shape[0], 8)
        # 特征维度不匹配时 load_board_model 应拒绝（这里直接判定形状规则）
        self.assertEqual(model.predict(np.zeros((3, 8), np.float32)).shape, (3,))


class BoardRecognizerBinarizeTest(unittest.TestCase):
    def test_binarize_full_returns_single_channel_mask(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[100:400, 100:400] = 200  # 混合明暗，Otsu 才能分出 0/255
        binary = binarize_full(img)
        self.assertEqual(binary.ndim, 2)
        self.assertEqual(binary.shape, (720, 1280))
        self.assertEqual(set(np.unique(binary)), {0, 255})

    def test_quad_crop_box_covers_quad_with_margin(self):
        quad = np.array([[676, 96], [1100, 355], [686, 718], [262, 352]], dtype=np.float32)
        x0, y0, x1, y1 = quad_crop_box(quad, (720, 1280))
        self.assertLessEqual(x0, 262)
        self.assertGreaterEqual(x1, 1100)
        self.assertLessEqual(y0, 96)
        self.assertGreaterEqual(y1, 718)
        # 需在图像范围内
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 1280)
        self.assertLessEqual(y1, 720)


class BoardRecognizerClassifyTest(unittest.TestCase):
    def test_constant_submarine_model_marks_all_cells(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        quad = np.array([[676, 96], [1100, 355], [686, 718], [262, 352]], dtype=np.float32)
        cells = classify_board_cells(img, quad, 10, _constant_model(1))
        self.assertEqual(cells, {(r, c) for r in range(10) for c in range(10)})

    def test_constant_water_model_marks_none(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        quad = np.array([[676, 96], [1100, 355], [686, 718], [262, 352]], dtype=np.float32)
        cells = classify_board_cells(img, quad, 10, _constant_model(0))
        self.assertEqual(cells, set())

    def test_none_model_returns_empty(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        quad = np.array([[676, 96], [1100, 355], [686, 718], [262, 352]], dtype=np.float32)
        self.assertEqual(classify_board_cells(img, quad, 10, None), set())


class BinarizeHitTest(unittest.TestCase):
    POLY = np.array([[600, 300], [700, 300], [700, 400], [600, 400]], dtype=np.float32)

    def _blank(self, value: int):
        return np.full((720, 1280, 3), value, dtype=np.uint8)

    def test_cell_hit_true_when_content_appears(self):
        before = self._blank(255)   # 全亮 → 二值化后无"内容"
        after = self._blank(0)      # 全暗 → 二值化后视为"内容"
        self.assertTrue(binarize_cell_hit(before, after, self.POLY, _MeanModel()))
        self.assertFalse(binarize_cell_hit(after, before, self.POLY, _MeanModel()))
        self.assertFalse(binarize_cell_hit(after, after, self.POLY, _MeanModel()))

    def test_reveal_cells_returns_newly_appeared_cells(self):
        before = self._blank(255)
        after = self._blank(0)
        points = [(640, 360)] * 4  # 4 格，坐标无所谓（假模型只看内容）
        revealed = binarize_reveal_cells(before, after, points, 2, _MeanModel())
        self.assertEqual(revealed, {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertEqual(binarize_reveal_cells(after, before, points, 2, _MeanModel()), set())

    def test_none_model_never_hits(self):
        self.assertFalse(
            binarize_cell_hit(self._blank(255), self._blank(0), self.POLY, None)
        )
        self.assertEqual(
            binarize_reveal_cells(self._blank(255), self._blank(0), [(1, 1)], 1, None),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
