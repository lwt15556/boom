import unittest

from utils.submarine_strategy import (
    SubmarineStrategy,
    get_configured_submarines,
    play_with_strategy,
)


def play_board(n, submarines, ships):
    ship_cells = {cell for ship in ships for cell in ship}
    strategy = SubmarineStrategy(n, submarines)
    steps = 0

    while not strategy.done and steps < n * n:
        cell = strategy.choose_next_cell()
        if cell is None:
            break

        strategy.report_result(cell, cell in ship_cells)
        steps += 1

    return strategy, steps


class SubmarineStrategyTest(unittest.TestCase):
    def test_scout_hit_is_blue_priority_without_completing_ship(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits={(2, 2), (2, 3)}, misses=set())

        self.assertFalse(strategy.done)
        self.assertEqual(strategy.shots, {})
        self.assertIn(strategy.choose_next_cell(), {(2, 2), (2, 3)})

    def test_blue_result_replaces_scout_state(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits={(2, 2)}, misses={(0, 0)})

        strategy.report_result((2, 2), True)

        self.assertNotIn((2, 2), strategy.scout_observations)
        self.assertTrue(strategy.shots[(2, 2)])
        self.assertEqual(strategy.get_cell_states()[2][2], "hit")
        self.assertEqual(strategy.get_cell_states()[0][0], "scout_miss")

    def test_scout_miss_is_never_selected_or_used_in_placements(self):
        strategy = SubmarineStrategy(4, [2])
        strategy.report_scout_results(hits=set(), misses={(1, 1)})

        placements = strategy._all_placements(2, include_scout_misses=True)
        self.assertFalse(
            any((1, 1) in placement.cells for placement in placements)
        )

        selected = set()
        for _ in range(strategy.n * strategy.n):
            cell = strategy.choose_next_cell()
            self.assertNotEqual(cell, (1, 1))
            if cell is None:
                break
            selected.add(cell)
            strategy.report_result(cell, False)

        self.assertEqual(len(selected), strategy.n * strategy.n - 1)
        self.assertIsNone(strategy.choose_next_cell())

    def test_malformed_scout_cells_leave_observations_unchanged(self):
        malformed_batches = (
            ([(2, 2), (1.0, 2)], []),
            ([(2, 2), (1.5, 2)], []),
            ([(2, 2), (True, 2)], []),
            ([(2, 2), ("1", 2)], []),
            ([(2, 2), (1,)], []),
            ([(2, 2), (1, 2, 3)], []),
            ([(2, 2), 5], []),
            ([(2, 2), "12"], []),
            ([(2, 2)], [(3, 3), (2.0, 1)]),
            ([(2, 2), (5, 0)], []),
            (5, []),
        )

        for hits, misses in malformed_batches:
            with self.subTest(hits=hits, misses=misses):
                strategy = SubmarineStrategy(5, [2])
                strategy.report_scout_results(hits={(0, 0)}, misses={(0, 1)})
                original = dict(strategy.scout_observations)
                with self.assertRaises((TypeError, ValueError)):
                    strategy.report_scout_results(hits=hits, misses=misses)
                self.assertEqual(strategy.scout_observations, original)

    def test_numpy_integral_scout_cells_normalize_to_builtin_ints(self):
        import numpy as np

        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(
            hits=[(np.int64(1), np.int32(2))],
            misses=[(np.int32(3), np.int64(4))],
        )

        self.assertEqual(strategy.get_scout_hit_cells(), {(1, 2)})
        self.assertEqual(strategy.get_scout_miss_cells(), {(3, 4)})
        for cell in strategy.scout_observations:
            self.assertTrue(all(type(coordinate) is int for coordinate in cell))

    def test_scout_hit_then_miss_is_rejected(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits={(1, 1)}, misses=set())

        with self.assertRaises(ValueError):
            strategy.report_scout_results(hits=set(), misses={(1, 1)})

        self.assertEqual(strategy.scout_observations, {(1, 1): True})

    def test_scout_miss_then_hit_is_rejected(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits=set(), misses={(1, 1)})

        with self.assertRaises(ValueError):
            strategy.report_scout_results(hits={(1, 1)}, misses=set())

        self.assertEqual(strategy.scout_observations, {(1, 1): False})

    def test_repeated_identical_scout_observations_are_idempotent(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits={(1, 1)}, misses={(3, 3)})

        strategy.report_scout_results(hits={(1, 1)}, misses={(3, 3)})

        self.assertEqual(
            strategy.scout_observations,
            {(1, 1): True, (3, 3): False},
        )

    def test_contradictory_mixed_scout_batch_is_atomic(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_scout_results(hits={(1, 1)}, misses={(2, 2)})
        original = dict(strategy.scout_observations)

        with self.assertRaises(ValueError):
            strategy.report_scout_results(
                hits={(2, 2), (3, 3)},
                misses={(4, 4)},
            )

        self.assertEqual(strategy.scout_observations, original)

    def test_scout_results_do_not_override_authoritative_shots(self):
        strategy = SubmarineStrategy(5, [2])
        strategy.report_result((1, 1), False)

        strategy.report_scout_results(hits={(1, 1)}, misses=set())

        self.assertFalse(strategy.shots[(1, 1)])
        self.assertNotIn((1, 1), strategy.scout_observations)

    def test_scout_misses_do_not_authoritatively_confirm_short_ship(self):
        strategy = SubmarineStrategy(6, [2, 4])
        real_hits = {(2, 2), (2, 3)}
        for cell in real_hits:
            strategy.report_result(cell, True)
        strategy.report_scout_results(hits=set(), misses={(2, 1), (2, 4)})

        authoritative = strategy._all_placements(4)
        selectable = strategy._all_placements(4, include_scout_misses=True)
        self.assertTrue(
            any(real_hits.issubset(placement.cells) for placement in authoritative)
        )
        self.assertFalse(
            any(real_hits.issubset(placement.cells) for placement in selectable)
        )

        strategy.choose_next_cell()

        self.assertFalse(strategy.done)
        self.assertEqual(strategy.get_confirmed_ships(), [])
        self.assertEqual(sorted(strategy.remaining.elements()), [2, 4])

    def test_pending_scout_hit_ties_use_row_major_order(self):
        observation_orders = (
            [(0, 0), (0, 6)],
            [(0, 6), (0, 0)],
        )

        for observations in observation_orders:
            with self.subTest(observations=observations):
                strategy = SubmarineStrategy(7, [2])
                strategy.report_scout_results(hits=observations, misses=set())
                self.assertEqual(strategy.choose_next_cell(), (0, 0))

    def test_finds_all_ships_before_full_scan(self):
        ships = [
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(1, 7), (2, 7), (3, 7)],
            [(7, 3), (7, 4)],
        ]

        strategy, steps = play_board(8, [2, 3, 4], ships)

        self.assertTrue(strategy.done)
        self.assertLess(steps, 64)
        self.assertEqual(
            sorted(ship.length for ship in strategy.get_confirmed_ships()),
            [2, 3, 4],
        )

    def test_repeated_lengths_are_counted_independently(self):
        ships = [
            [(0, 0), (0, 1), (0, 2)],
            [(3, 6), (4, 6)],
            [(6, 0), (6, 1)],
        ]

        strategy, _ = play_board(7, [2, 2, 3], ships)

        self.assertTrue(strategy.done)
        self.assertEqual(
            sorted(ship.length for ship in strategy.get_confirmed_ships()),
            [2, 2, 3],
        )

    def test_confirmed_safety_area_is_not_selected_again(self):
        strategy = SubmarineStrategy(5, [2, 2])
        strategy.report_result((0, 0), True)
        strategy.report_result((0, 1), True)

        confirmed = strategy.get_confirmed_ships()
        self.assertEqual(len(confirmed), 1)

        safety = confirmed[0].safety_area
        for _ in range(10):
            cell = strategy.choose_next_cell()
            self.assertIsNotNone(cell)
            self.assertNotIn(cell, safety)
            strategy.report_result(cell, False)

    def test_repeated_result_is_idempotent_and_conflict_errors(self):
        strategy = SubmarineStrategy(4, [2])
        strategy.report_result((1, 1), True)
        strategy.report_result((1, 1), True)

        with self.assertRaises(ValueError):
            strategy.report_result((1, 1), False)

    def test_after_hit_next_probe_prefers_adjacent_cell(self):
        strategy = SubmarineStrategy(6, [3])
        strategy.report_result((3, 3), True)

        next_cell = strategy.choose_next_cell()

        self.assertIn(next_cell, {(2, 3), (4, 3), (3, 2), (3, 4)})

    def test_two_hits_lock_direction_to_line_extension(self):
        strategy = SubmarineStrategy(6, [4])
        strategy.report_result((3, 2), True)
        strategy.report_result((3, 3), True)

        next_cell = strategy.choose_next_cell()

        self.assertIn(next_cell, {(3, 1), (3, 4)})

    def test_complete_straight_ship_blocks_surrounding_ring(self):
        strategy = SubmarineStrategy(6, [3])
        for cell in [(2, 2), (2, 3), (2, 4)]:
            strategy.report_result(cell, True)

        confirmed = strategy.get_confirmed_ships()
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].cells, ((2, 2), (2, 3), (2, 4)))

        blocked_ring = {
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 1), (2, 5),
            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        }
        self.assertTrue(blocked_ring.issubset(strategy.blocked_cells))
        for _ in range(10):
            cell = strategy.choose_next_cell()
            if cell is None:
                break
            self.assertNotIn(cell, blocked_ring)
            strategy.report_result(cell, False)

    def test_structured_board_states_distinguish_probe_results(self):
        strategy = SubmarineStrategy(5, [2, 3])
        strategy.report_result((2, 1), True)
        strategy.report_result((2, 2), True)
        strategy.confirm_completed_lengths((2,), anchor=(2, 2))
        strategy.report_result((0, 4), False)

        states = strategy.get_cell_states()

        self.assertEqual(states[2][1], "ship")
        self.assertEqual(states[2][2], "ship")
        self.assertEqual(states[1][1], "blocked")
        self.assertEqual(states[0][4], "miss")
        self.assertEqual(states[4][4], "unknown")

    def test_partial_long_ship_is_not_confirmed_as_short_ship(self):
        strategy = SubmarineStrategy(6, [2, 4])
        strategy.report_result((2, 2), True)
        strategy.report_result((2, 3), True)

        self.assertEqual(strategy.get_confirmed_ships(), [])

    def test_sidebar_completed_length_confirms_ambiguous_short_ship(self):
        strategy = SubmarineStrategy(6, [2, 4])
        strategy.report_result((2, 2), True)
        strategy.report_result((2, 3), True)

        confirmed = strategy.confirm_completed_lengths((2,), anchor=(2, 3))

        self.assertEqual(confirmed, (2,))
        self.assertEqual(
            [ship.length for ship in strategy.get_confirmed_ships()],
            [2],
        )
        self.assertIn((1, 1), strategy.blocked_cells)
        self.assertIn((3, 4), strategy.blocked_cells)

    def test_unlocated_sidebar_completion_is_removed_from_hunt_fleet(self):
        strategy = SubmarineStrategy(6, [2, 4])
        strategy.report_result((1, 1), True)

        located, unlocated = strategy.reconcile_completed_lengths(
            (4,),
            observed_completed_cells={(1, 1)},
        )

        self.assertEqual(located, ())
        self.assertEqual(unlocated, (4,))
        self.assertEqual(list(strategy.remaining.elements()), [2])
        self.assertEqual(strategy.get_accounted_completed_lengths(), [4])
        self.assertIn((1, 1), strategy.blocked_cells)
        self.assertNotIn((1, 1), strategy._unconfirmed_hit_cells())

    def test_midgame_visual_state_reconciles_located_and_unlocated_ships(self):
        strategy = SubmarineStrategy(9, [2, 2, 3, 3, 4, 5])
        visual_hits = {
            (1, 3),
            (3, 3),
            (3, 4),
            (5, 4),
            (7, 3),
            (8, 4),
        }
        partial_wrecks = {(1, 3)}
        for cell in visual_hits:
            strategy.report_result(cell, True)

        located, unlocated = strategy.reconcile_completed_lengths(
            (4, 2),
            observed_completed_cells=visual_hits - partial_wrecks,
        )

        self.assertEqual(located, (2,))
        self.assertEqual(unlocated, (4,))
        self.assertEqual(
            sorted(strategy.get_accounted_completed_lengths()),
            [2, 4],
        )
        self.assertEqual(
            sorted(strategy.remaining.elements()),
            [2, 3, 3, 5],
        )
        self.assertIn(
            strategy.choose_next_cell(),
            {(0, 3), (2, 3), (1, 2), (1, 4)},
        )

    def test_missing_level_config_returns_none_for_fallback(self):
        self.assertEqual(get_configured_submarines(1, {1: [2, 3]}), [2, 3])
        self.assertIsNone(get_configured_submarines(9, {1: [2, 3]}))

    def test_play_with_strategy_uses_callback(self):
        ships = [[(0, 0), (0, 1)]]
        ship_cells = {cell for ship in ships for cell in ship}

        confirmed = play_with_strategy(3, [2], lambda cell: cell in ship_cells)

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].cells, ((0, 0), (0, 1)))


if __name__ == "__main__":
    unittest.main()
