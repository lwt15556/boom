"""规则识别（残骸 / 完整潜艇 / 高亮海浪）的合成帧测试。

测试用一张 1280x720 的合成棋盘，几何与生产帧一致（10x10，格心间距约 44px）：

* 水面：中等饱和度的蓝色（不在灰白掩码里）。
* 网格线：细白线（低饱和高亮，但太细，被开运算滤掉）。
* 残骸：格内一块带暗部碎片的灰白实体（面积占比大、填充率高、纹理高）。
* 完整潜艇：跨两格的灰白艇身 + 红色指挥塔。
* 高亮海浪：细长的白色泡沫条纹（开运算后剩不下东西）。
"""

import unittest

import cv2
import numpy as np

from save_points.points import points_from_quad
from utils.submarine_detector import (
    classify_cells_by_rules,
    detect_board_sprites,
    detect_ship_cells_by_red_tower,
    detect_wreck_cells_by_features,
)

FRAME_SHAPE = (720, 1280, 3)
QUAD = ((676, 96), (1100, 355), (686, 718), (262, 352))
GRID = 10
WATER_BGR = (170, 125, 85)          # 蓝色水面：HSV 饱和度约 127，不属于灰白实体
RUBBLE_CELL = (3, 3)
SHIP_CELLS = ((7, 4), (7, 5))
WAVE_CELL = (1, 8)


def _frame() -> np.ndarray:
    frame = np.full(FRAME_SHAPE, WATER_BGR, dtype=np.uint8)
    rng = np.random.default_rng(20260909)
    # 轻微水面纹理，避免整幅图完全均匀。
    noise = rng.integers(-12, 12, size=(FRAME_SHAPE[0], FRAME_SHAPE[1], 1), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def _draw_grid_lines(frame: np.ndarray, points) -> None:
    for index in range(len(points)):
        row, col = divmod(index, GRID)
        if col + 1 < GRID:
            cv2.line(frame, points[index], points[index + 1], (255, 255, 255), 3)
        if row + 1 < GRID:
            cv2.line(frame, points[index], points[index + GRID], (255, 255, 255), 3)


def _draw_rubble(frame: np.ndarray, centre: tuple[int, int]) -> None:
    rng = np.random.default_rng(7)
    cv2.circle(frame, centre, 22, (205, 205, 205), -1)
    for _ in range(14):
        dx = int(rng.integers(-18, 18))
        dy = int(rng.integers(-14, 14))
        w = int(rng.integers(5, 11))
        h = int(rng.integers(4, 9))
        shade = int(rng.integers(60, 115))
        cv2.rectangle(
            frame,
            (centre[0] + dx, centre[1] + dy),
            (centre[0] + dx + w, centre[1] + dy + h),
            (shade, shade, shade),
            -1,
        )
    for _ in range(8):
        dx = int(rng.integers(-20, 20))
        dy = int(rng.integers(-16, 16))
        cv2.circle(frame, (centre[0] + dx, centre[1] + dy), 3, (250, 250, 250), -1)


def _draw_submarine(frame: np.ndarray, centres: tuple[tuple[int, int], tuple[int, int]]) -> None:
    start = np.asarray(centres[0], dtype=float)
    end = np.asarray(centres[1], dtype=float)
    middle = ((start + end) / 2.0).astype(int)
    angle = float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])))
    cv2.ellipse(frame, tuple(middle), (78, 30), angle, 0, 360, (198, 198, 198), -1)
    # 弹痕暗斑：艇身被切成几块，验证"按距离归并到潜艇"的逻辑。
    cv2.ellipse(frame, tuple(middle), (10, 24), angle, 0, 360, (70, 70, 70), -1)
    # 红色指挥塔：位于两格交界处。
    cv2.rectangle(
        frame,
        (middle[0] - 10, middle[1] - 16),
        (middle[0] + 10, middle[1] + 16),
        (32, 32, 214),
        -1,
    )


def _draw_wave(frame: np.ndarray, centre: tuple[int, int]) -> None:
    for offset in (-30, -10, 12, 34):
        cv2.line(
            frame,
            (centre[0] - 46, centre[1] + offset),
            (centre[0] + 46, centre[1] + offset - 18),
            (238, 246, 250),
            3,
        )


def _scene() -> tuple[np.ndarray, list[tuple[int, int]]]:
    frame = _frame()
    points = points_from_quad([tuple(map(float, corner)) for corner in QUAD], GRID)
    _draw_grid_lines(frame, points)
    _draw_rubble(frame, points[RUBBLE_CELL[0] * GRID + RUBBLE_CELL[1]])
    _draw_submarine(
        frame,
        tuple(points[cell[0] * GRID + cell[1]] for cell in SHIP_CELLS),
    )
    _draw_wave(frame, points[WAVE_CELL[0] * GRID + WAVE_CELL[1]])
    return frame, points


class SpriteRuleDetectionTest(unittest.TestCase):
    def setUp(self):
        self.frame, self.points = _scene()

    def test_wreck_cell_detected(self):
        wrecks = detect_wreck_cells_by_features(self.frame, self.points, GRID)
        self.assertIn(RUBBLE_CELL, wrecks)

    def test_wave_cell_is_not_a_wreck(self):
        wrecks = detect_wreck_cells_by_features(self.frame, self.points, GRID)
        self.assertNotIn(WAVE_CELL, wrecks)

    def test_submarine_cells_detected_and_not_wrecks(self):
        ships, wrecks = classify_cells_by_rules(self.frame, self.points, GRID)
        for cell in SHIP_CELLS:
            self.assertIn(cell, ships)
            self.assertNotIn(cell, wrecks)

    def test_wave_cell_is_not_a_ship(self):
        ships = detect_ship_cells_by_red_tower(self.frame, self.points, GRID)
        self.assertNotIn(WAVE_CELL, ships)

    def test_plain_water_cells_stay_empty(self):
        ships, wrecks = classify_cells_by_rules(self.frame, self.points, GRID)
        self.assertNotIn((0, 0), wrecks)
        self.assertNotIn((9, 9), wrecks)
        self.assertNotIn((0, 0), ships)
        self.assertNotIn((9, 9), ships)

    def test_ship_sprite_is_one_component_with_red_tower(self):
        sprites = detect_board_sprites(self.frame, self.points, GRID)
        ship_sprites = [sprite for sprite in sprites if sprite.has_red_tower]
        self.assertTrue(ship_sprites)
        covered = {
            cell
            for sprite in ship_sprites
            for cell, coverage in sprite.cells
            if coverage >= 0.10
        }
        for cell in SHIP_CELLS:
            self.assertIn(cell, covered)

    def test_detached_hull_fragment_merges_into_the_ship(self):
        # 弹痕暗斑会把艇身切成两块，但两块都算潜艇，不会退化成残骸。
        ships, wrecks = classify_cells_by_rules(self.frame, self.points, GRID)
        for cell in SHIP_CELLS:
            self.assertIn(cell, ships)
        for cell in ships:
            self.assertLessEqual(abs(cell[0] - 7) + abs(cell[1] - 4), 3)

    def test_textured_rubble_near_ship_is_not_absorbed_as_ship(self):
        """高纹理的独立残骸堆即使落在红塔附近，也不能被并入潜艇。

        Guards the level-20 (8,5) regression: a standalone textured wreck inside
        ``hull_reach`` of a supplied submarine was pulled into the ship by the
        hull-fragment merge, so the wreck was demoted from the confirmed-hit set
        to a provisional candidate.  Only smooth hull fragments belong to the
        submarine.
        """
        frame = _frame()
        points = points_from_quad([tuple(map(float, corner)) for corner in QUAD], GRID)
        _draw_grid_lines(frame, points)
        _draw_submarine(
            frame,
            tuple(points[cell[0] * GRID + cell[1]] for cell in SHIP_CELLS),
        )
        near_rubble = (6, 5)
        _draw_rubble(frame, points[near_rubble[0] * GRID + near_rubble[1]])

        ships = detect_ship_cells_by_red_tower(frame, points, GRID)
        for cell in SHIP_CELLS:
            self.assertIn(cell, ships)
        self.assertNotIn(near_rubble, ships)

    def test_surface_classifier_separates_wreck_wave_water_and_hull(self):
        """四类地表判定：残骸/艇身/海浪/深水，用中心 patch 口径。"""
        from utils.submarine_detector import (
            SURFACE_HULL,
            SURFACE_WATER,
            SURFACE_WAVE,
            SURFACE_WRECK,
            classify_surface_cell,
        )

        water = np.full((60, 60, 3), (170, 125, 85), dtype=np.uint8)
        self.assertEqual(classify_surface_cell(water, (30, 30)), SURFACE_WATER)

        wave = water.copy()
        cv2.rectangle(wave, (24, 24), (35, 35), (250, 250, 250), -1)  # 白沫
        self.assertEqual(classify_surface_cell(wave, (30, 30)), SURFACE_WAVE)

        wreck = np.full((60, 60, 3), (205, 205, 205), dtype=np.uint8)
        self.assertEqual(classify_surface_cell(wreck, (30, 30)), SURFACE_WRECK)

        hull = wreck.copy()
        cv2.rectangle(hull, (24, 24), (35, 35), (60, 60, 60), -1)  # 弹痕暗斑
        self.assertEqual(classify_surface_cell(hull, (30, 30)), SURFACE_HULL)

    def test_wreck_sprite_has_no_red_tower_and_is_textured(self):
        sprites = detect_board_sprites(self.frame, self.points, GRID)
        rubble = [
            sprite
            for sprite in sprites
            if not sprite.has_red_tower
            and RUBBLE_CELL in {cell for cell, _ in sprite.cells}
        ]
        self.assertTrue(rubble)
        self.assertGreaterEqual(rubble[0].texture, 30.0)

    def test_empty_frame_is_rejected_without_error(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(detect_wreck_cells_by_features(empty, self.points, GRID), set())
        self.assertEqual(classify_cells_by_rules(empty, self.points, GRID), (set(), set()))

    def test_short_click_point_list_is_rejected(self):
        self.assertEqual(detect_wreck_cells_by_features(self.frame, [], GRID), set())
        self.assertEqual(detect_board_sprites(self.frame, [], GRID), [])


if __name__ == "__main__":
    unittest.main()
