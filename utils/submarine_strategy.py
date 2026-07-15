from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, FrozenSet, Iterable, Mapping, Optional, Sequence


Cell = tuple[int, int]


@dataclass(frozen=True)
class Placement:
    """一艘潜艇在当前信息下的一个可能摆放方案。"""

    length: int
    direction: str
    cells: tuple[Cell, ...]


@dataclass(frozen=True)
class ConfirmedShip:
    """已经根据命中反馈唯一确认的潜艇及其安全区。"""

    length: int
    direction: str
    cells: tuple[Cell, ...]
    safety_area: FrozenSet[Cell]


def get_configured_submarines(
    level: int,
    configs: Mapping[int, Sequence[int]],
) -> list[int] | None:
    """读取关卡潜艇长度配置；缺少配置时返回 None 供主流程回退逐格扫描。"""
    submarines = configs.get(int(level))
    if submarines is None:
        return None
    return [int(length) for length in submarines]


class SubmarineStrategy:
    """根据命中/未命中反馈选择下一格，尽量减少实际探测次数。"""

    def __init__(
        self,
        n: int,
        submarines: Sequence[int],
        use_safety_rule: bool = True,
    ) -> None:
        """初始化 N x N 棋盘策略，并记录待确认的潜艇长度计数。"""
        if n <= 0:
            raise ValueError("n must be positive")

        lengths = [int(length) for length in submarines]
        if not lengths:
            raise ValueError("submarines cannot be empty")
        if any(length <= 0 for length in lengths):
            raise ValueError("submarine lengths must be positive")

        self.n = int(n)
        self.remaining = Counter(lengths)
        self.use_safety_rule = use_safety_rule
        self.shots: dict[Cell, bool] = {}
        self.scout_observations: dict[Cell, bool] = {}
        self.confirmed_ships: list[ConfirmedShip] = []
        self.unlocated_completed: Counter[int] = Counter()
        self.accounted_hit_cells: set[Cell] = set()
        self.blocked_cells: set[Cell] = set()
        self._hunt_residue_cache: dict[int, int] = {}

    @property
    def done(self) -> bool:
        """是否已经确认配置中的全部潜艇。"""
        return sum(self.remaining.values()) == 0

    def report_scout_results(
        self,
        *,
        hits: Iterable[Cell],
        misses: Iterable[Cell],
    ) -> None:
        """Record temporary scout observations without counting them as shots."""
        hit_cells = self._normalize_scout_cells(hits, label="hits")
        miss_cells = self._normalize_scout_cells(misses, label="misses")
        overlap = hit_cells & miss_cells
        if overlap:
            raise ValueError(f"scout hits and misses overlap: {sorted(overlap)}")

        incoming = {cell: True for cell in hit_cells}
        incoming.update({cell: False for cell in miss_cells})
        contradictions = {
            cell
            for cell, hit in incoming.items()
            if cell not in self.shots
            and cell in self.scout_observations
            and self.scout_observations[cell] != hit
        }
        if contradictions:
            raise ValueError(
                f"scout observations contradict existing state: {sorted(contradictions)}"
            )

        for cell, hit in incoming.items():
            if cell not in self.shots:
                self.scout_observations[cell] = hit

        self._hunt_residue_cache.clear()

    def _normalize_scout_cells(
        self,
        cells: Iterable[Cell],
        *,
        label: str,
    ) -> set[Cell]:
        try:
            iterator = iter(cells)
        except TypeError as exc:
            raise TypeError(f"scout {label} must be an iterable of cells") from exc

        return {
            self._normalize_scout_cell(cell, label=label)
            for cell in iterator
        }

    def _normalize_scout_cell(self, cell: object, *, label: str) -> Cell:
        try:
            coordinates = tuple(cell)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"scout {label} cell must be an iterable") from exc

        if len(coordinates) != 2:
            raise ValueError(
                f"scout {label} cell must contain exactly two coordinates: {cell!r}"
            )

        normalized: list[int] = []
        for coordinate in coordinates:
            if isinstance(coordinate, bool) or not isinstance(coordinate, Integral):
                raise TypeError(
                    f"scout {label} coordinates must be integers: {cell!r}"
                )
            normalized.append(int(coordinate))

        normalized_cell = (normalized[0], normalized[1])
        self._validate_cell(normalized_cell)
        return normalized_cell

    def get_scout_hit_cells(self) -> set[Cell]:
        return {
            cell
            for cell, hit in self.scout_observations.items()
            if hit
        }

    def get_scout_miss_cells(self) -> set[Cell]:
        return {
            cell
            for cell, hit in self.scout_observations.items()
            if not hit
        }

    def _known_cells(self) -> set[Cell]:
        return set(self.shots) | set(self.scout_observations)

    def report_result(self, cell: Cell, hit: bool) -> None:
        """记录一次真实探测反馈，并尝试根据新信息确认潜艇。"""
        self._validate_cell(cell)
        self.scout_observations.pop(cell, None)

        if cell in self.shots:
            old = self.shots[cell]
            if old != hit:
                raise ValueError(f"conflicting result for cell {cell}: old={old}, new={hit}")
            return

        self.shots[cell] = bool(hit)
        self._hunt_residue_cache.clear()
        self._try_confirm_ships()

    def confirm_completed_lengths(
        self,
        completed_lengths: Sequence[int],
        *,
        anchor: Cell | None = None,
    ) -> tuple[int, ...]:
        """Confirm placements backed by an external completed-length signal."""
        confirmed_lengths: list[int] = []
        self._try_confirm_ships()

        for raw_length in completed_lengths:
            length = int(raw_length)
            if length <= 0 or self.remaining.get(length, 0) <= 0:
                continue

            placement = self._find_explicit_completion_placement(
                length,
                anchor=anchor if not confirmed_lengths else None,
            )
            if placement is None:
                continue

            self._confirm_placement(placement)
            confirmed_lengths.append(length)

        if confirmed_lengths:
            self._try_confirm_ships()
        return tuple(confirmed_lengths)

    def reconcile_completed_lengths(
        self,
        completed_lengths: Sequence[int],
        *,
        anchor: Cell | None = None,
        observed_completed_cells: Iterable[Cell] = (),
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Reconcile the full sidebar completion state with strategy state."""
        observed = Counter(
            int(length)
            for length in completed_lengths
            if int(length) > 0
        )
        accounted = Counter(ship.length for ship in self.confirmed_ships)
        accounted.update(self.unlocated_completed)
        missing = observed - accounted

        located: list[int] = []
        unlocated: list[int] = []
        anchor_available = anchor
        for length in missing.elements():
            if self.remaining.get(length, 0) <= 0:
                continue

            placement = self._find_explicit_completion_placement(
                length,
                anchor=anchor_available,
            )
            if placement is not None:
                self._confirm_placement(placement)
                located.append(length)
                anchor_available = None
                continue

            self._consume_remaining_length(length)
            self.unlocated_completed[length] += 1
            unlocated.append(length)

        if unlocated:
            confirmed_cells = {
                cell
                for ship in self.confirmed_ships
                for cell in ship.cells
            }
            recovered_cells = {
                cell
                for cell in observed_completed_cells
                if self._inside(cell) and cell not in confirmed_cells
            }
            self.accounted_hit_cells.update(recovered_cells)
            self.blocked_cells.update(recovered_cells)

        if located or unlocated:
            self._hunt_residue_cache.clear()
            self._try_confirm_ships()
        return tuple(located), tuple(unlocated)

    def choose_next_cell(self) -> Optional[Cell]:
        """返回下一次建议探测的格子；全部潜艇确认后返回 None。"""
        pending_scout_hit = self._choose_pending_scout_hit()
        if pending_scout_hit is not None:
            return pending_scout_hit

        self._try_confirm_ships()

        if self.done:
            return None

        extension = self._choose_oriented_cluster_extension()
        if extension is not None:
            return extension

        adjacent = self._choose_adjacent_to_recent_hit()
        if adjacent is not None:
            return adjacent

        target = self._choose_target_cell()
        if target is not None:
            return target

        return self._choose_hunt_cell()

    def get_confirmed_ships(self) -> list[ConfirmedShip]:
        """返回已经确认完整位置的潜艇列表副本。"""
        return list(self.confirmed_ships)

    def get_accounted_completed_lengths(self) -> list[int]:
        lengths = [ship.length for ship in self.confirmed_ships]
        lengths.extend(self.unlocated_completed.elements())
        return lengths

    def get_debug_board(self) -> list[str]:
        """生成文本调试棋盘，用于观察命中、未命中、已确认和安全区状态。"""
        confirmed_cells = set()
        for ship in self.confirmed_ships:
            confirmed_cells.update(ship.cells)

        board = []
        for row in range(self.n):
            chars = []
            for col in range(self.n):
                cell = (row, col)
                if cell in confirmed_cells or cell in self.accounted_hit_cells:
                    chars.append("S")
                elif self.shots.get(cell) is True:
                    chars.append("X")
                elif self.shots.get(cell) is False:
                    chars.append(".")
                elif cell in self.blocked_cells:
                    chars.append("-")
                else:
                    chars.append("?")
            board.append("".join(chars))
        return board

    def get_cell_states(self) -> list[list[str]]:
        """Return structured cell states for runtime monitoring."""
        states = [["unknown" for _col in range(self.n)] for _row in range(self.n)]

        for row, col in self.blocked_cells:
            if self._inside((row, col)):
                states[row][col] = "blocked"

        for (row, col), hit in self.scout_observations.items():
            states[row][col] = "scout_hit" if hit else "scout_miss"

        for (row, col), hit in self.shots.items():
            states[row][col] = "hit" if hit else "miss"

        confirmed_cells = set(self.accounted_hit_cells)
        for ship in self.confirmed_ships:
            confirmed_cells.update(ship.cells)
        for row, col in confirmed_cells:
            if self._inside((row, col)):
                states[row][col] = "ship"

        return states

    def _validate_cell(self, cell: Cell) -> None:
        """校验格子坐标是否位于当前棋盘内。"""
        row, col = cell
        if not (0 <= row < self.n and 0 <= col < self.n):
            raise ValueError(f"cell out of bounds: {cell}")

    def _inside(self, cell: Cell) -> bool:
        """判断格子坐标是否在当前棋盘范围内。"""
        row, col = cell
        return 0 <= row < self.n and 0 <= col < self.n

    def _neighbors4(self, cell: Cell) -> Iterable[Cell]:
        """枚举一个格子的上下左右四连通邻居。"""
        row, col = cell
        for next_cell in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if self._inside(next_cell):
                yield next_cell

    def _unconfirmed_hit_cells(self) -> set[Cell]:
        """返回已命中但尚未归属到确认潜艇的格子集合。"""
        confirmed_cells = set()
        for ship in self.confirmed_ships:
            confirmed_cells.update(ship.cells)

        return {
            cell
            for cell, hit in self.shots.items()
            if hit
            and cell not in confirmed_cells
            and cell not in self.accounted_hit_cells
        }

    def _miss_cells(self) -> set[Cell]:
        """返回所有已探测且判定为未命中的格子。"""
        return self._real_miss_cells() | self.get_scout_miss_cells()

    def get_priority_scout_miss_recheck_targets(
        self,
        already_rechecked: Iterable[Cell] = (),
    ) -> list[Cell]:
        """Return line ends first, then neighbors around isolated real hits."""
        rechecked = {
            self._normalize_scout_cell(cell, label="rechecked")
            for cell in already_rechecked
        }
        scout_miss_cells = self.get_scout_miss_cells()
        scheduled = set(rechecked)
        targets: list[Cell] = []

        clusters = sorted(
            self._get_hit_clusters(),
            key=lambda cluster: (-len(cluster), tuple(sorted(cluster))),
        )
        for cluster in clusters:
            cells = self._straight_contiguous_cells(cluster)
            if cells is None or len(cells) < 2:
                continue

            if len({row for row, _ in cells}) == 1:
                candidates = (
                    (cells[0][0], cells[0][1] - 1),
                    (cells[-1][0], cells[-1][1] + 1),
                )
            else:
                candidates = (
                    (cells[0][0] - 1, cells[0][1]),
                    (cells[-1][0] + 1, cells[-1][1]),
                )

            for cell in candidates:
                if (
                    not self._inside(cell)
                    or cell not in scout_miss_cells
                    or cell in self.shots
                    or cell in scheduled
                ):
                    continue
                scheduled.add(cell)
                targets.append(cell)

        if targets:
            return targets
        return self.get_isolated_hit_scout_miss_neighbors_for_recheck(scheduled)

    def get_isolated_hit_scout_miss_neighbors_for_recheck(
        self,
        already_rechecked: Iterable[Cell] = (),
    ) -> list[Cell]:
        """Return scout-miss neighbors that fully surround an unresolved real hit."""
        rechecked = {
            self._normalize_scout_cell(cell, label="rechecked")
            for cell in already_rechecked
        }
        scout_miss_cells = self.get_scout_miss_cells()
        confirmed_hit_cells = set(self.accounted_hit_cells)
        for ship in self.confirmed_ships:
            confirmed_hit_cells.update(ship.cells)
        known_hit_cells = (
            {cell for cell, hit in self.shots.items() if hit}
            | self.get_scout_hit_cells()
            | confirmed_hit_cells
        )

        targets: list[Cell] = []
        scheduled = set(rechecked)
        for hit_cell in sorted(self._unconfirmed_hit_cells()):
            neighbors = tuple(self._neighbors4(hit_cell))
            if not neighbors or not all(
                neighbor in scout_miss_cells for neighbor in neighbors
            ):
                continue
            if any(neighbor in known_hit_cells for neighbor in neighbors):
                continue

            for neighbor in neighbors:
                if (
                    neighbor in self.shots
                    or neighbor in scheduled
                ):
                    continue
                scheduled.add(neighbor)
                targets.append(neighbor)
        return targets

    def _real_miss_cells(self) -> set[Cell]:
        return {cell for cell, hit in self.shots.items() if not hit}

    def _get_hit_clusters(self) -> list[set[Cell]]:
        """按四连通关系聚合未确认命中格，较大的命中簇优先处理。"""
        hits = self._unconfirmed_hit_cells()
        visited: set[Cell] = set()
        clusters: list[set[Cell]] = []

        for start in hits:
            if start in visited:
                continue

            queue = deque([start])
            visited.add(start)
            cluster = {start}

            while queue:
                current = queue.popleft()
                for next_cell in self._neighbors4(current):
                    if next_cell in hits and next_cell not in visited:
                        visited.add(next_cell)
                        cluster.add(next_cell)
                        queue.append(next_cell)

            clusters.append(cluster)

        clusters.sort(key=lambda item: -len(item))
        return clusters

    def _all_placements(
        self,
        length: int,
        *,
        include_scout_misses: bool = False,
    ) -> list[Placement]:
        """生成指定长度潜艇在当前已知信息下仍可能存在的全部位置。"""
        misses = (
            self._miss_cells()
            if include_scout_misses
            else self._real_miss_cells()
        )
        invalid = misses | self.blocked_cells
        result: list[Placement] = []
        seen: set[tuple[Cell, ...]] = set()

        for row in range(self.n):
            for col_start in range(self.n - length + 1):
                cells = tuple((row, col) for col in range(col_start, col_start + length))
                if any(cell in invalid for cell in cells):
                    continue
                if cells not in seen:
                    seen.add(cells)
                    result.append(Placement(length=length, direction="H", cells=cells))

        for col in range(self.n):
            for row_start in range(self.n - length + 1):
                cells = tuple((row, col) for row in range(row_start, row_start + length))
                if any(cell in invalid for cell in cells):
                    continue
                if cells not in seen:
                    seen.add(cells)
                    result.append(Placement(length=length, direction="V", cells=cells))

        return result

    def _candidate_placements_for_cluster(
        self,
        cluster: set[Cell],
        *,
        include_scout_misses: bool = False,
    ) -> list[Placement]:
        """找出所有能覆盖指定命中簇的剩余潜艇摆放方案。"""
        candidates: list[Placement] = []
        cluster_cells = frozenset(cluster)

        for length, count in self.remaining.items():
            if count <= 0:
                continue

            for placement in self._all_placements(
                length,
                include_scout_misses=include_scout_misses,
            ):
                if cluster_cells.issubset(frozenset(placement.cells)):
                    candidates.append(placement)

        return candidates

    def _try_confirm_ships(self) -> None:
        """当某个命中簇只剩唯一完整解释时，确认潜艇并屏蔽安全区。"""
        changed = True

        while changed:
            changed = False
            for cluster in self._get_hit_clusters():
                exact_placement = self._exact_complete_cluster_placement(cluster)
                if exact_placement is not None:
                    self._confirm_placement(exact_placement)
                    changed = True
                    break

                candidates = self._candidate_placements_for_cluster(cluster)
                if len(candidates) != 1:
                    continue

                placement = candidates[0]
                if not all(self.shots.get(cell) is True for cell in placement.cells):
                    continue
                if self.remaining[placement.length] <= 0:
                    continue

                self._confirm_placement(placement)
                changed = True
                break

    def _confirm_placement(self, placement: Placement) -> None:
        safety = self._calc_safety_area(placement)
        self.confirmed_ships.append(
            ConfirmedShip(
                length=placement.length,
                direction=placement.direction,
                cells=placement.cells,
                safety_area=frozenset(safety),
            )
        )

        self._consume_remaining_length(placement.length)

        if self.use_safety_rule:
            self.blocked_cells.update(safety)
        else:
            self.blocked_cells.update(placement.cells)
        self._hunt_residue_cache.clear()

    def _consume_remaining_length(self, length: int) -> None:
        if self.remaining.get(length, 0) <= 0:
            raise ValueError(f"no remaining submarine of length {length}")
        self.remaining[length] -= 1
        if self.remaining[length] == 0:
            del self.remaining[length]

    def _find_explicit_completion_placement(
        self,
        length: int,
        *,
        anchor: Cell | None = None,
    ) -> Optional[Placement]:
        """Find one fully observed placement for a sidebar-completed ship."""
        clusters = self._get_hit_clusters()
        if anchor is not None:
            clusters = [cluster for cluster in clusters if anchor in cluster]
            if not clusters:
                return None

        for cluster in clusters:
            if len(cluster) > length:
                continue

            candidates: list[Placement] = []
            for placement in self._all_placements(length):
                placement_cells = frozenset(placement.cells)
                if not frozenset(cluster).issubset(placement_cells):
                    continue
                if not all(self.shots.get(cell) is True for cell in placement.cells):
                    continue
                candidates.append(placement)

            if len(candidates) == 1:
                return candidates[0]

        return None

    def _exact_complete_cluster_placement(self, cluster: set[Cell]) -> Optional[Placement]:
        if not cluster:
            return None

        cells = self._straight_contiguous_cells(cluster)
        if cells is None:
            return None

        length = len(cells)
        if self.remaining.get(length, 0) <= 0:
            return None

        cluster_cells = frozenset(cells)
        for longer_length, count in self.remaining.items():
            if count <= 0 or longer_length <= length:
                continue
            for placement in self._all_placements(longer_length):
                if cluster_cells.issubset(frozenset(placement.cells)):
                    return None

        direction = "H" if len({row for row, _ in cells}) == 1 else "V"
        return Placement(length=length, direction=direction, cells=cells)

    def _straight_contiguous_cells(self, cluster: set[Cell]) -> tuple[Cell, ...] | None:
        rows = {row for row, _ in cluster}
        cols = {col for _, col in cluster}
        if len(rows) == 1:
            row = next(iter(rows))
            ordered = tuple(sorted(cluster, key=lambda cell: cell[1]))
            expected = tuple((row, col) for col in range(ordered[0][1], ordered[-1][1] + 1))
            return ordered if ordered == expected else None
        if len(cols) == 1:
            col = next(iter(cols))
            ordered = tuple(sorted(cluster, key=lambda cell: cell[0]))
            expected = tuple((row, col) for row in range(ordered[0][0], ordered[-1][0] + 1))
            return ordered if ordered == expected else None
        return None

    def _calc_safety_area(self, placement: Placement) -> set[Cell]:
        """按上浮规则计算潜艇周围一圈安全区，并裁剪到棋盘范围内。"""
        rows = [row for row, _ in placement.cells]
        cols = [col for _, col in placement.cells]
        area: set[Cell] = set()

        for row in range(min(rows) - 1, max(rows) + 2):
            for col in range(min(cols) - 1, max(cols) + 2):
                cell = (row, col)
                if self._inside(cell):
                    area.add(cell)

        return area

    def _choose_target_cell(self) -> Optional[Cell]:
        """命中后进入追击模式，优先选择能最快确认方向和长度的邻近格。"""
        clusters = self._get_hit_clusters()
        if not clusters:
            return None

        known_cells = self._known_cells()
        best_cell: Optional[Cell] = None
        best_score = -1.0

        for cluster in clusters:
            candidates = self._candidate_placements_for_cluster(
                cluster,
                include_scout_misses=True,
            )
            if not candidates:
                continue

            freq: Counter[Cell] = Counter()
            for placement in candidates:
                for cell in placement.cells:
                    if cell in known_cells or cell in self.blocked_cells:
                        continue
                    freq[cell] += 1

            if not freq:
                continue

            frontier = {
                next_cell
                for hit_cell in cluster
                for next_cell in self._neighbors4(hit_cell)
                if next_cell in freq
            }
            selectable = frontier if frontier else set(freq.keys())

            for cell in selectable:
                score = float(freq[cell])
                if cell in frontier:
                    score += 100.0
                score += self._center_bonus(cell)

                if score > best_score:
                    best_score = score
                    best_cell = cell

        return best_cell

    def _choose_oriented_cluster_extension(self) -> Optional[Cell]:
        """If two or more hits reveal direction, probe only the two line ends first."""
        known_cells = self._known_cells()
        for cluster in self._get_hit_clusters():
            cells = self._straight_contiguous_cells(cluster)
            if cells is None or len(cells) < 2:
                continue

            direction = "H" if len({row for row, _ in cells}) == 1 else "V"
            first = cells[0]
            last = cells[-1]
            if direction == "H":
                candidates = [
                    (first[0], first[1] - 1),
                    (last[0], last[1] + 1),
                ]
            else:
                candidates = [
                    (first[0] - 1, first[1]),
                    (last[0] + 1, last[1]),
                ]

            valid = [
                cell
                for cell in candidates
                if self._inside(cell)
                and cell not in known_cells
                and cell not in self.blocked_cells
                and self._cell_can_extend_cluster(cell, cluster)
            ]
            if valid:
                return max(valid, key=self._center_bonus)

        return None

    def _cell_can_extend_cluster(self, cell: Cell, cluster: set[Cell]) -> bool:
        expanded = frozenset(cluster | {cell})
        for length, count in self.remaining.items():
            if count <= 0:
                continue
            for placement in self._all_placements(
                length,
                include_scout_misses=True,
            ):
                placement_cells = frozenset(placement.cells)
                if expanded.issubset(placement_cells):
                    return True
        return False

    def _choose_adjacent_to_recent_hit(self) -> Optional[Cell]:
        """After a hit, immediately probe an untested 4-neighbor to find ship direction."""
        unconfirmed_hits = self._unconfirmed_hit_cells()
        if not unconfirmed_hits:
            return None

        known_cells = self._known_cells()
        placement_heat: Counter[Cell] = Counter()
        for length, count in self.remaining.items():
            if count <= 0:
                continue
            for placement in self._all_placements(
                length,
                include_scout_misses=True,
            ):
                if not any(cell in unconfirmed_hits for cell in placement.cells):
                    continue
                for cell in placement.cells:
                    if cell not in known_cells and cell not in self.blocked_cells:
                        placement_heat[cell] += count

        best_cell: Optional[Cell] = None
        best_score = -1.0
        recent_hits = [
            cell
            for cell, hit in reversed(self.shots.items())
            if hit and cell in unconfirmed_hits
        ]
        for hit_cell in recent_hits:
            for cell in self._neighbors4(hit_cell):
                if cell in known_cells or cell in self.blocked_cells:
                    continue
                score = float(placement_heat.get(cell, 0)) + self._center_bonus(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell

            if best_cell is not None:
                return best_cell

        return None

    def _choose_hunt_cell(self) -> Optional[Cell]:
        """未处于追击模式时，用候选段热力图和最短潜艇跳格规则巡航。"""
        if not self.remaining:
            return None

        known_cells = self._known_cells()
        heat: Counter[Cell] = Counter()
        for length, count in self.remaining.items():
            for placement in self._all_placements(
                length,
                include_scout_misses=True,
            ):
                for cell in placement.cells:
                    if cell not in known_cells and cell not in self.blocked_cells:
                        heat[cell] += count * length

        if not heat:
            return self._fallback_unshot_cell()

        min_len = min(self.remaining.keys())
        if min_len not in self._hunt_residue_cache:
            residue_scores = Counter()
            for cell, value in heat.items():
                row, col = cell
                residue_scores[(row + col) % min_len] += value
            self._hunt_residue_cache[min_len] = residue_scores.most_common(1)[0][0]

        residue = self._hunt_residue_cache[min_len]
        best_cell: Optional[Cell] = None
        best_score = -1.0

        for cell, value in heat.items():
            row, col = cell
            if (row + col) % min_len != residue:
                continue

            score = float(value) + self._center_bonus(cell)
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is not None:
            return best_cell

        for cell, value in heat.items():
            score = float(value) + self._center_bonus(cell)
            if score > best_score:
                best_score = score
                best_cell = cell

        return best_cell

    def _center_bonus(self, cell: Cell) -> float:
        """给靠近中心的格子极小加权，用于同分时减少边界优先级。"""
        row, col = cell
        center = (self.n - 1) / 2
        distance = abs(row - center) + abs(col - center)
        return -distance * 0.001

    def _choose_pending_scout_hit(self) -> Optional[Cell]:
        pending_hits = {
            cell
            for cell in self.get_scout_hit_cells()
            if cell not in self.shots and cell not in self.blocked_cells
        }
        if not pending_hits:
            return None

        return max(
            pending_hits,
            key=lambda cell: (
                sum(neighbor in pending_hits for neighbor in self._neighbors4(cell)),
                self._center_bonus(cell),
                -cell[0],
                -cell[1],
            ),
        )

    def _fallback_unshot_cell(self) -> Optional[Cell]:
        """热力图无解时，按行优先返回第一个未探测且未屏蔽格子。"""
        known_cells = self._known_cells()
        for row in range(self.n):
            for col in range(self.n):
                cell = (row, col)
                if cell not in known_cells and cell not in self.blocked_cells:
                    return cell
        return None


def play_with_strategy(
    n: int,
    submarines: Sequence[int],
    fire_once: Callable[[Cell], bool],
    max_steps: int | None = None,
) -> list[ConfirmedShip]:
    """用回调执行完整策略循环，主要供纯逻辑测试或外部接入复用。"""
    strategy = SubmarineStrategy(n=n, submarines=submarines)
    limit = max_steps if max_steps is not None else n * n

    for _ in range(limit):
        if strategy.done:
            break

        cell = strategy.choose_next_cell()
        if cell is None:
            break

        strategy.report_result(cell, fire_once(cell))

    return strategy.get_confirmed_ships()
