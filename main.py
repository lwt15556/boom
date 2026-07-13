import atexit
import json
import signal
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from config import (
    AUTO_DETECT_LEVEL,
    DEFAULT_LEVEL,
    GAME_PACKAGE_NAME,
    LEVEL_GRID_SIZES,
    LEVEL_REFERENCE_DIR,
    MAX_LEVEL,
    OUTPUT_DIR,
    REQUIRE_CONFIDENT_LEVEL_DETECTION,
    SCREENSHOT_DIR,
    SUBMARINES,
    TEMPLATE_DIR,
    USE_SAVED_POINTS,
)
from save_points.points import read_saved_points, read_saved_quad
from utils import AdbController, MatchResult, find_template, get_logger
from utils.diamond_centers import detect_diamond_centers
from utils.diamond_hit import classify_diamond_hit
from utils.hit_map import save_hit_map_image
from utils.image_match import find_template_multi_scale
from utils.level_recognition import recognize_level_from_screenshot
from utils.level_title_recognition import recognize_level_title
from utils.progress import (
    SearchProgress,
    fixed_progress_bar,
    format_elapsed,
    update_fixed_progress,
)
from utils.probe_protocol import (
    ProbeNotReadyError,
    ProbePhase,
    ProbeProtocolError,
    ProbeTransaction,
)
from utils.submarine_strategy import Cell, SubmarineStrategy, get_configured_submarines

logger = get_logger(__name__)
adb = AdbController()

ACTIVITY_BUTTON_TEMPLATE = TEMPLATE_DIR / "activity_button.png"
LOGIN_TEMPLATE = TEMPLATE_DIR / "login.png"
QUIT_ACTIVITY_TEMPLATE = TEMPLATE_DIR / "quit_activity.png"
RETRY_TEMPLATE = TEMPLATE_DIR / "retry.png"
CONNECTION_INTERRUPTED_TEMPLATE = TEMPLATE_DIR / "connection_interrupted.png"
CONNECTION_RETRY_TEMPLATE = TEMPLATE_DIR / "connection_retry.png"
VICTORY_BANNER_TEMPLATE = TEMPLATE_DIR / "victory_banner.png"
RETRY_TEMPLATE_SCALES = (0.85, 0.95, 1.0, 1.05, 1.15)
RETRY_TEMPLATE_LOOSE_THRESHOLD = 0.72
CONNECTION_TEMPLATE_SCALES = (0.9, 1.0, 1.1)
CONNECTION_DIALOG_THRESHOLD = 0.78
CONNECTION_RETRY_THRESHOLD = 0.74
VICTORY_TEMPLATE_SCALES = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.65, 1.8)
VICTORY_BANNER_THRESHOLD = 0.80
VICTORY_WAIT_AFTER_HIT_SECONDS = 10.0
VICTORY_WAIT_BEFORE_LEVEL_SECONDS = 3.0
VICTORY_SKIP_SETTLE_SECONDS = 2.0
HIT_RESULT_FRAME_DELAYS = (1.0, 0.35, 0.45, 0.55)
SUSPECT_HIT_EXTRA_FRAME_DELAYS = (0.45, 0.55, 0.65)
MIN_HIT_RESULT_VOTES = 2
SUSPECT_HIT_SCORE_THRESHOLD = 0.78

ACTIVITY_DETAIL_POINT = (1205, 644)
ACTIVITY_LIST_SWIPE = (1000, 660, 1000, 180)
SCREEN_CONTINUE_POINT = (640, 360)
RUN_DEBUG_DIR = SCREENSHOT_DIR / "run_debug"
PROBE_SAMPLE_DIR = SCREENSHOT_DIR / "probes"

_weak_network_cleanup_done = False
_active_probe: "ProbeTransaction | None" = None


def _has_pending_probe_request() -> bool:
    return _active_probe is not None and _active_probe.request_may_be_pending


def _create_probe_sample_dir(level: int, cell: Cell, index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    row, col = cell
    sample_dir = PROBE_SAMPLE_DIR / f"level_{level}_cell_{index}_r{row}_c{col}_{timestamp}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def _write_probe_status(sample_dir: Path, stage: str, **extra) -> None:
    payload = {
        "stage": stage,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    (sample_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _hit_result_to_dict(result) -> dict:
    return {
        "state": result.state,
        "confidence": float(getattr(result, "confidence", 0.0)),
        "score": float(result.score),
        "rough_center": list(result.rough_center),
        "refined_center": list(result.refined_center),
        "changed_ratio": float(result.changed_ratio),
        "center_gray_ratio": float(result.center_gray_ratio),
        "ring_gray_ratio": float(getattr(result, "ring_gray_ratio", 0.0)),
        "gray_excess": float(result.gray_excess),
        "component_ratio": float(result.component_ratio),
        "s_center": float(getattr(result, "s_center", 0.0)),
        "s_ring": float(getattr(result, "s_ring", 0.0)),
        "s_drop": float(result.s_drop),
        "edge_density": float(result.edge_density),
    }


def _save_probe_result_json(
    sample_dir: Path,
    *,
    level: int,
    cell: Cell,
    index: int,
    point: tuple[int, int],
    hit: bool,
    hit_votes: int,
    frames: list[dict],
    suspect_extra_checked: bool,
) -> None:
    payload = {
        "level": level,
        "cell": list(cell),
        "index": index,
        "point": list(point),
        "decision": "hit" if hit else "miss",
        "hit_votes": hit_votes,
        "frame_count": len(frames),
        "min_hit_votes": MIN_HIT_RESULT_VOTES,
        "suspect_extra_checked": suspect_extra_checked,
        "frames": frames,
    }
    (sample_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def enable_weak_network(second: float = 0) -> None:
    """弢启游戏弱网，并按霢等待网络状生效"""
    adb.enable_weak_network(GAME_PACKAGE_NAME)
    if second > 0:
        sleep(second)


def disable_weak_network(second: float = 0) -> None:
    """安全关闭游戏弱网；存在待丢弃请求时拒绝恢复网络"""
    if _has_pending_probe_request():
        transaction = _active_probe
        raise ProbeProtocolError(
            "pending probe request may still exist; refuse to disable DROP weak network "
            f"cell={transaction.cell if transaction else None} "
            f"phase={transaction.phase.name if transaction else None}"
        )
    adb.disable_weak_network(GAME_PACKAGE_NAME)
    if second > 0:
        sleep(second)


def cleanup_weak_network(reason: str = "脚本逢") -> None:
    """仅在不存在待发探测请求时关闭 DROP 弱网"""
    global _weak_network_cleanup_done
    if _weak_network_cleanup_done:
        return

    if _has_pending_probe_request():
        transaction = _active_probe
        logger.critical(
            "%s，但格子 %s 的探测处?%s；为避免暂存请求补发，保?DROP 弱网",
            reason,
            transaction.cell if transaction else None,
            transaction.phase.name if transaction else None,
        )
        return

    try:
        logger.info("%s, disabling weak network", reason)
        disable_weak_network()
    except Exception as exc:
        logger.error("关闭弱网失败: %s", exc)
    else:
        _weak_network_cleanup_done = True


def cleanup_reject_network(reason: str = "脚本逢") -> None:
    """关闭游戏 REJECT 断网残留，避免影响本次或下次运行"""
    try:
        logger.info("%s，正在清?REJECT 断网", reason)
        adb.disable_reject_network(GAME_PACKAGE_NAME)
    except Exception as exc:
        logger.error("清理 REJECT 断网失败: %s", exc)


def handle_exit_signal(signum: int, _frame) -> None:
    """收到逢出信号时先关闭弱网再逢出"""
    cleanup_weak_network(f"收到逢出信?{signum}")
    raise SystemExit(128 + signum)


def register_exit_cleanup() -> None:
    """注册脚本逢出清理，尽量避免弱网规则残留"""
    atexit.register(cleanup_weak_network)
    for signame in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, signame, None)
        if signum is not None:
            signal.signal(signum, handle_exit_signal)


def enter_activity(re_enter: bool = False, max_retries: int = 5) -> None:
    """进入活动详情页?

    ``re_enter=False`` 用于没有待验证请求的普进入，允许重启恢复?
    ``re_enter=True`` 用于点击后的第二次进入，此时 DROP 下可能仍有暂存请求，
    任何失败都必须立即中止，不能复用会关闭弱网的普恢复流程?
    """
    if max_retries <= 0:
        raise ValueError(f"max_retries 必须大于 0: {max_retries}")

    last_failure = "进入活动失败"
    for attempt in range(1, max_retries + 1):
        adb.delay(0.5)
        if not re_enter:
            screenshot = adb.read_screenshot()
            if handle_victory_prompt(timeout=0.0, screenshot=screenshot):
                logger.info("victory banner skipped before entering activity")
                continue

        res = wait_until_occur(ACTIVITY_BUTTON_TEMPLATE, timeout=20)
        if res is None:
            last_failure = "activity button not found"
            if re_enter:
                raise ProbeProtocolError(
                    f"第二次进入活动失败: {last_failure}; keep DROP weak network and stop probing"
                )
            logger.warning(
                "%s，无法进入活动界面，正在重试 (%s/%s)",
                last_failure,
                attempt,
                max_retries,
            )
            _restart_game_for_activity_retry()
            continue

        adb.click(*res.center)  # 点击活动按钮进入活动界面
        if not re_enter:
            enable_weak_network(0.2)
            adb.delay(0.4).swipe(*ACTIVITY_LIST_SWIPE)  # 首次进入霢要展示全部项
            adb.delay(0.2).swipe(*ACTIVITY_LIST_SWIPE)

        adb.delay(0.7).click(*ACTIVITY_DETAIL_POINT)
        if wait_until_occur(QUIT_ACTIVITY_TEMPLATE, timeout=15) is not None:
            return

        recovery = recover_activity_detail_timeout(re_enter=re_enter)
        if recovery == "ready":
            return
        if recovery == "retry":
            continue

        last_failure = "进入活动详情界面失败"
        if re_enter:
            raise ProbeProtocolError(
                f"第二次进入活动失败: {last_failure}; keep DROP weak network and stop probing"
            )
        logger.warning(
            "%s，正在重试进入活?(%s/%s)",
            last_failure,
            attempt,
            max_retries,
        )
        _restart_game_for_activity_retry()

    message = f"{last_failure}，已达到最大重试次数 {max_retries}"
    logger.error(message)
    raise RuntimeError(message)


def recover_activity_detail_timeout(re_enter: bool) -> str:
    """Return 'ready', 'retry', or 'unhandled' after an activity-detail timeout."""
    if re_enter:
        return "unhandled"

    screenshot = adb.read_screenshot()
    if find_template(screenshot, QUIT_ACTIVITY_TEMPLATE) is not None:
        logger.info("activity detail was detected after timeout; continuing")
        return "ready"

    if handle_victory_prompt(timeout=0.0, screenshot=screenshot):
        logger.info("victory banner handled after activity-detail timeout; retrying entry")
        return "retry"

    try:
        if handle_connection_interrupted_prompt(timeout=6.0):
            logger.info("connection dialog handled after activity-detail timeout; retrying entry")
            return "retry"
    except ProbeProtocolError as exc:
        logger.warning("connection dialog recovery after activity-detail timeout failed: %s", exc)

    return "unhandled"


def _restart_game_for_activity_retry() -> None:
    """在没有待验证请求的普通进入阶段重启游戏"""
    if _has_pending_probe_request():
        raise ProbeProtocolError("存在待发送探测请求，禁止通过重启游戏恢复活动入口")

    adb.close_app(GAME_PACKAGE_NAME)
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    disable_weak_network()
    adb.delay(1.5).open_app(GAME_PACKAGE_NAME)
    login_img = wait_until_occur(LOGIN_TEMPLATE, timeout=30)
    if login_img is None:
        logger.warning("restarted game but login button was not found; continuing")
        return
    adb.click(*login_img.center)  # 点击登录按钮


def get_level_grid_size(level: int) -> int:
    """读取指定关卡的菱形网格边长"""
    if level not in LEVEL_GRID_SIZES:
        raise ValueError(f"未配置第 {level} 关的网格边长")
    return LEVEL_GRID_SIZES[level]


def get_click_points(
    level: int, grid_img: np.ndarray
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """按配置读取人工点位，失败时回逢到自动识别"""
    grid_size = get_level_grid_size(level)

    if USE_SAVED_POINTS:
        try:
            saved_points = read_saved_points(level, expected_n=grid_size)
            saved_quad = read_saved_quad(level)
        except Exception as exc:
            logger.warning("failed to read saved points for level %s; falling back to auto detection: %s", level, exc)
        else:
            if saved_points is not None and saved_quad is not None:
                logger.info("level %s uses saved calibration points: %s", level, len(saved_points))
                return saved_points, saved_quad
            logger.warning("?%s 关人工点位不存在或数量不正确，回逢自动识别", level)

    grid_result = detect_diamond_centers(grid_img, grid_size)
    logger.info("level %s uses auto-detected points: %s", level, len(grid_result.points))
    return grid_result.points, grid_result.global_quad


def handle_game_level(
    level: int,
    hit_map: list[list[int]],
    run_started_at: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """处理单个关卡：有潜艇配置时策略点，缺少配置时回逐格扫描"""
    adb.delay(1.5)
    grid_img = adb.read_screenshot()
    click_points, grid_quad = get_click_points(level, grid_img)

    submarines = get_configured_submarines(level, SUBMARINES)
    if submarines is None:
        message = f"?{level} 关缺少潜艇长度配置，回逐格扫描"
        logger.warning(message)
        _scan_level_by_grid_order(
            level,
            hit_map,
            click_points,
            run_started_at=run_started_at,
        )
    else:
        _scan_level_by_strategy(
            level,
            hit_map,
            click_points,
            submarines,
            run_started_at=run_started_at,
        )

    return grid_img, grid_quad


def _scan_level_by_grid_order(
    level: int,
    hit_map: list[list[int]],
    click_points: list[tuple[int, int]],
    skip_cells: set[Cell] | None = None,
    run_started_at: float | None = None,
) -> None:
    """按行优先顺序逐格探测，可跳过策略阶段已获得真实反馈的格子"""
    grid_size = get_level_grid_size(level)
    skip_cells = skip_cells or set()
    targets = [
        (index, point, (index // grid_size, index % grid_size))
        for index, point in enumerate(click_points)
        if (index // grid_size, index % grid_size) not in skip_cells
    ]
    if not targets:
        logger.info("level %s grid scan has no remaining targets", level)
        return

    progress = SearchProgress(
        level=level,
        max_probes=len(targets),
        started_at=run_started_at if run_started_at is not None else monotonic(),
    )
    with fixed_progress_bar(
        total=len(targets),
        description=f"Level {level} grid scan",
        unit="cell",
    ) as bar:
        update_fixed_progress(
            bar,
            0,
            progress.grid_postfix(
                completed=0,
                total=len(targets),
                now=monotonic(),
            ),
        )
        for completed, (index, point, cell) in enumerate(targets, start=1):
            _probe_cell(level, hit_map, cell, point, index)
            update_fixed_progress(
                bar,
                current=completed,
                postfix=progress.grid_postfix(
                    completed=completed,
                    total=len(targets),
                    now=monotonic(),
                ),
            )


def _scan_level_by_strategy(
    level: int,
    hit_map: list[list[int]],
    click_points: list[tuple[int, int]],
    submarines: list[int],
    run_started_at: float | None = None,
) -> None:
    """使用潜艇策略选择探测格；策略无法完成时回逢扫描剩余未探测格"""
    grid_size = get_level_grid_size(level)
    strategy = SubmarineStrategy(grid_size, submarines)
    max_attempts = grid_size * grid_size
    attempts = 0
    progress = SearchProgress(
        level=level,
        max_probes=max_attempts,
        total_ship_cells=sum(submarines),
        total_ships=len(submarines),
        started_at=run_started_at if run_started_at is not None else monotonic(),
    )

    with fixed_progress_bar(
        total=sum(submarines),
        description=f"Level {level} strategy scan",
        unit="cell",
    ) as bar:
        logger.info(
            "level %s strategy enabled: grid=%s submarines=%s",
            level,
            grid_size,
            submarines,
        )
        update_fixed_progress(
            bar,
            0,
            progress.strategy_postfix(
                attempts=0,
                confirmed_lengths=[],
                remaining_lengths=list(submarines),
                now=monotonic(),
            ),
        )

        while not strategy.done and attempts < max_attempts:
            cell = strategy.choose_next_cell()
            if cell is None:
                logger.warning("?%s 关策略已无可选方格，提前结束", level)
                break

            row, col = cell
            index = row * grid_size + col
            hit = _probe_cell(level, hit_map, cell, click_points[index], index)
            attempts += 1
            strategy.report_result(cell, hit)
            confirmed_lengths = [
                ship.length for ship in strategy.get_confirmed_ships()
            ]
            hit_cells = sum(1 for shot_hit in strategy.shots.values() if shot_hit)
            update_fixed_progress(
                bar,
                hit_cells,
                progress.strategy_postfix(
                    attempts=attempts,
                    confirmed_lengths=confirmed_lengths,
                    remaining_lengths=list(strategy.remaining.elements()),
                    now=monotonic(),
                ),
            )

        if strategy.done:
            logger.info("level %s strategy confirmed all submarines, attempts=%s", level, attempts)
        else:
            logger.warning(
                "level %s strategy did not confirm all submarines; falling back to grid scan",
                level,
            )

    if not strategy.done:
        known_cells = set(strategy.shots) | strategy.blocked_cells
        _scan_level_by_grid_order(
            level,
            hit_map,
            click_points,
            skip_cells=known_cells,
            run_started_at=run_started_at,
        )


def _probe_cell(
    level: int,
    hit_map: list[list[int]],
    cell: Cell,
    point: tuple[int, int],
    index: int,
) -> bool:
    """准备页面并执行一次完整探测；点击前异常只重试当前格"""
    max_preflight_retries = 3
    for attempt in range(1, max_preflight_retries + 1):
        try:
            return _execute_probe_transaction(level, hit_map, cell, point, index)
        except ProbeNotReadyError as exc:
            if attempt >= max_preflight_retries:
                raise ProbeProtocolError(
                    f"cell {cell} was not ready before click after {max_preflight_retries} retries"
                ) from exc
            logger.warning(
                "cell %s was not ready before click; retrying same cell (%s/%s): %s",
                cell,
                attempt,
                max_preflight_retries,
                exc,
            )
            enter_activity()

    raise AssertionError("探测重试循环意外结束")


def _execute_probe_transaction(
    level: int,
    hit_map: list[list[int]],
    cell: Cell,
    point: tuple[int, int],
    index: int,
) -> bool:
    """按固?DROP/二次进入/REJECT/登录顺序执行单格探测事务"""
    global _active_probe

    if _active_probe is not None:
        raise ProbeProtocolError(
            f"上一轮探测尚未结束，禁止弢始格?{cell}: "
            f"cell={_active_probe.cell} phase={_active_probe.phase.name}"
        )

    if wait_until_occur(QUIT_ACTIVITY_TEMPLATE, timeout=6) is None:
        raise ProbeNotReadyError("当前不在活动详情界面")

    transaction = ProbeTransaction(level=level, cell=cell, index=index)
    _active_probe = transaction
    x, y = point
    sample_dir: Path | None = None

    try:
        sample_dir = _create_probe_sample_dir(level, cell, index)
        _write_probe_status(
            sample_dir,
            "started",
            level=level,
            cell=list(cell),
            index=index,
            point=list(point),
            phase=transaction.phase.name,
        )
        before_img = adb.read_screenshot(sample_dir / "before.png")
        _write_probe_status(sample_dir, "before_captured", phase=transaction.phase.name)

        # 点击命令丢旦发出，就保守地认为客户端可能已经暂存验证请求?
        transaction.advance(ProbePhase.REQUEST_PENDING)
        _write_probe_status(sample_dir, "request_pending", phase=transaction.phase.name)
        adb.click(x, y)
        adb.delay(0.3)

        if not click_template(
            QUIT_ACTIVITY_TEMPLATE,
            RUN_DEBUG_DIR / "debug_quit1.png",
        ):
            raise ProbeProtocolError(
                "点击格子后未找到逢出按钮；待发送请求状态未知，保留 DROP 弱网"
            )

        _write_probe_status(sample_dir, "activity_exited", phase=transaction.phase.name)
        enter_activity(re_enter=True, max_retries=1)
        _write_probe_status(sample_dir, "activity_reentered", phase=transaction.phase.name)
        hit_results = []
        frame_records = []
        for frame_index, frame_delay in enumerate(HIT_RESULT_FRAME_DELAYS, start=1):
            screenshot_path = sample_dir / f"after_{frame_index}.png"
            after_img = adb.delay(frame_delay).read_screenshot(screenshot_path)
            result = classify_diamond_hit(before_img, after_img, (x, y))
            hit_results.append(result)
            frame_records.append(
                {
                    "frame": frame_index,
                    "delay": frame_delay,
                    "path": str(screenshot_path),
                    "result": _hit_result_to_dict(result),
                }
            )
            _write_probe_status(
                sample_dir,
                "frame_captured",
                phase=transaction.phase.name,
                frame=frame_index,
                state=result.state,
                score=float(result.score),
            )
        transaction.advance(ProbePhase.RESULT_VISIBLE)
        _write_probe_status(sample_dir, "result_visible", phase=transaction.phase.name)

        hit_votes = sum(1 for result in hit_results if result.state == "hit")
        best_result = max(hit_results, key=lambda result: result.score)
        suspect_extra_checked = False
        if hit_votes < MIN_HIT_RESULT_VOTES and (
            hit_votes == 1 or best_result.score >= SUSPECT_HIT_SCORE_THRESHOLD
        ):
            suspect_extra_checked = True
            logger.info(
                "suspect hit cell=%s index=%s votes=%s/%s best_score=%.3f; collecting extra frames",
                cell,
                index,
                hit_votes,
                len(hit_results),
                best_result.score,
            )
            for extra_index, frame_delay in enumerate(
                SUSPECT_HIT_EXTRA_FRAME_DELAYS,
                start=len(hit_results) + 1,
            ):
                screenshot_path = sample_dir / f"after_{extra_index}.png"
                after_img = adb.delay(frame_delay).read_screenshot(screenshot_path)
                result = classify_diamond_hit(before_img, after_img, (x, y))
                hit_results.append(result)
                frame_records.append(
                    {
                        "frame": extra_index,
                        "delay": frame_delay,
                        "path": str(screenshot_path),
                        "result": _hit_result_to_dict(result),
                    }
                )
                _write_probe_status(
                    sample_dir,
                    "extra_frame_captured",
                    phase=transaction.phase.name,
                    frame=extra_index,
                    state=result.state,
                    score=float(result.score),
                )
            hit_votes = sum(1 for result in hit_results if result.state == "hit")
            best_result = max(hit_results, key=lambda result: result.score)
        first_result = hit_results[0]
        logger.info(
            "hit check cell=%s index=%s votes=%s/%s states=%s scores=%s changed=%s "
            "best_gray=%.3f best_excess=%.3f best_component=%.3f best_s_drop=%.1f best_edge=%.3f "
            "center=%s refined=%s",
            cell,
            index,
            hit_votes,
            len(hit_results),
            "/".join(result.state for result in hit_results),
            "/".join(f"{result.score:.3f}" for result in hit_results),
            "/".join(f"{result.changed_ratio:.3f}" for result in hit_results),
            best_result.center_gray_ratio,
            best_result.gray_excess,
            best_result.component_ratio,
            best_result.s_drop,
            best_result.edge_density,
            first_result.rough_center,
            best_result.refined_center,
        )
        hit = hit_votes >= MIN_HIT_RESULT_VOTES
        _save_probe_result_json(
            sample_dir,
            level=level,
            cell=cell,
            index=index,
            point=point,
            hit=hit,
            hit_votes=hit_votes,
            frames=frame_records,
            suspect_extra_checked=suspect_extra_checked,
        )
        transaction.hit = hit
        transaction.advance(ProbePhase.RESULT_RECORDED)
        _write_probe_status(
            sample_dir,
            "result_recorded",
            phase=transaction.phase.name,
            decision="hit" if hit else "miss",
            hit_votes=hit_votes,
        )

        if hit:
            row, col = cell
            hit_map[row][col] = 1
            logger.info("level %s cell %s result: hit", level, index)
            _commit_hit_request_and_prepare_next_probe(transaction)
        else:
            logger.info("level %s cell %s result: miss", level, index)
            _discard_pending_request_and_prepare_next_probe(transaction)

        _write_probe_status(
            sample_dir,
            "complete",
            phase=transaction.phase.name,
            decision="hit" if hit else "miss",
        )
        return hit
    except Exception as exc:
        if sample_dir is not None:
            _write_probe_status(
                sample_dir,
                "interrupted",
                phase=transaction.phase.name,
                error=repr(exc),
            )
        raise
    finally:
        if transaction.phase in {ProbePhase.PREPARING, ProbePhase.COMPLETE}:
            _active_probe = None
        elif transaction.request_may_be_pending:
            logger.critical(
                "cell %s probe interrupted at %s; pending request may remain; keep DROP weak network",

                transaction.cell,
                transaction.phase.name,
            )


def _commit_hit_request_and_prepare_next_probe(
    transaction: ProbeTransaction,
) -> None:
    """Restore network immediately on hit so the pending request is submitted."""
    transaction.advance(ProbePhase.REQUEST_COMMITTED)
    logger.info("hit detected; restoring network immediately to submit the pending request")
    transaction.advance(ProbePhase.LOGIN_RECOVERING)
    restart_process()
    transaction.advance(ProbePhase.COMPLETE)


def _discard_pending_request_and_prepare_next_probe(
    transaction: ProbeTransaction,
) -> None:
    """Force-stop the game while offline so a miss request cannot be retried."""
    adb.enable_reject_network(GAME_PACKAGE_NAME)
    adb.delay(0.5)
    logger.info("miss detected; force-stopping game before restoring network")
    adb.close_app(GAME_PACKAGE_NAME)

    transaction.advance(ProbePhase.REQUEST_DISCARDED)
    transaction.advance(ProbePhase.LOGIN_RECOVERING)

    restart_process(reopen_game=True)
    transaction.advance(ProbePhase.COMPLETE)


def restart_process(reopen_game: bool = False) -> None:
    """在请求确认丢弃后恢复网络登录，并进入下一轮探测页靃69"""
    if reopen_game:
        logger.info("pending miss request discarded; reopening game before next probe")
        adb.close_app(GAME_PACKAGE_NAME)
        adb.disable_reject_network(GAME_PACKAGE_NAME)
        disable_weak_network()
        adb.delay(1.5).open_app(GAME_PACKAGE_NAME)
        login_img = wait_until_occur(LOGIN_TEMPLATE, timeout=30)
        if login_img is not None:
            adb.click(*login_img.center)
        else:
            logger.warning("reopened game but login button was not found; continuing to activity entry")
        enter_activity()
        return

    disable_weak_network()
    handle_victory_prompt(timeout=VICTORY_WAIT_AFTER_HIT_SECONDS)
    enter_activity()


def find_victory_banner(screenshot: np.ndarray) -> MatchResult | None:
    """Detect the victory banner in a screenshot."""
    if not isinstance(screenshot, np.ndarray):
        return None

    victory = find_template(
        screenshot,
        VICTORY_BANNER_TEMPLATE,
        threshold=VICTORY_BANNER_THRESHOLD,
    )
    if victory is not None:
        return victory

    return find_template_multi_scale(
        screenshot,
        VICTORY_BANNER_TEMPLATE,
        scales=VICTORY_TEMPLATE_SCALES,
        threshold=VICTORY_BANNER_THRESHOLD,
    )


def handle_victory_prompt(
    timeout: float = 4.0,
    screenshot: np.ndarray | None = None,
) -> bool:
    """Skip the victory banner after a committed hit, if it appears."""
    victory = find_victory_banner(screenshot) if screenshot is not None else None
    if victory is None:
        if timeout > 0:
            logger.info("waiting up to %.1f seconds for victory banner", timeout)
        victory = wait_until_victory_banner(timeout=timeout)
    if victory is None:
        return False

    logger.info("victory banner detected; restoring network and tapping screen to continue")
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    disable_weak_network()
    adb.click(*SCREEN_CONTINUE_POINT)
    adb.delay(VICTORY_SKIP_SETTLE_SECONDS)
    return True


def handle_connection_interrupted_prompt(timeout: float = 20.0) -> bool:
    """Detect the connection-interrupted dialog, reconnect, and click retry."""
    dialog = wait_until_connection_interrupted_dialog(timeout=min(4.0, float(timeout)))
    if dialog is None:
        return False

    logger.info("connection-interrupted dialog detected; reconnecting and clicking retry")
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    retry = wait_until_retry_button(timeout=max(0.0, float(timeout) - 4.0))
    if retry is None:
        raise ProbeProtocolError("connection-interrupted dialog found, but retry button was not found")

    adb.delay(0.8).click(*retry.center)
    return True


def wait_until_victory_banner(timeout: float = 4.0) -> MatchResult | None:
    """Wait briefly for the victory banner shown after the final submarine is hit."""
    deadline = monotonic() + max(0.0, float(timeout))
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        victory = find_victory_banner(screenshot)
        if victory is not None:
            return victory
        sleep(0.3)
    return None


def wait_until_connection_interrupted_dialog(timeout: float = 20.0) -> MatchResult | None:
    """Wait for the larger connection-interrupted dialog."""
    exact_wait = min(3.0, max(0.0, float(timeout)))
    dialog = wait_until_occur(CONNECTION_INTERRUPTED_TEMPLATE, timeout=exact_wait)
    if dialog is not None:
        return dialog

    deadline = monotonic() + max(0.0, float(timeout) - exact_wait)
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        dialog = find_template_multi_scale(
            screenshot,
            CONNECTION_INTERRUPTED_TEMPLATE,
            scales=CONNECTION_TEMPLATE_SCALES,
            threshold=CONNECTION_DIALOG_THRESHOLD,
        )
        if dialog is not None:
            return dialog
        sleep(0.5)
    return None


def wait_until_retry_button(timeout: float = 20.0) -> MatchResult | None:
    """Wait for the current connection dialog retry button or the legacy retry button."""
    exact_wait = min(3.0, max(0.0, float(timeout)))
    retry = wait_until_occur(CONNECTION_RETRY_TEMPLATE, timeout=exact_wait)
    if retry is not None:
        return retry

    legacy_wait = min(5.0, max(0.0, float(timeout) - exact_wait))
    retry = wait_until_occur(RETRY_TEMPLATE, timeout=legacy_wait)
    if retry is not None:
        return retry

    deadline = monotonic() + max(0.0, float(timeout) - exact_wait - legacy_wait)
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        retry = find_template_multi_scale(
            screenshot,
            CONNECTION_RETRY_TEMPLATE,
            scales=RETRY_TEMPLATE_SCALES,
            threshold=CONNECTION_RETRY_THRESHOLD,
        )
        if retry is None:
            retry = find_template_multi_scale(
                screenshot,
                RETRY_TEMPLATE,
                scales=RETRY_TEMPLATE_SCALES,
                threshold=RETRY_TEMPLATE_LOOSE_THRESHOLD,
            )
        if retry is not None:
            return retry
        sleep(0.5)
    return None


def wait_until_retry_prompt(timeout: float = 20.0) -> MatchResult | None:
    """Wait for the retry prompt using the consolidated retry-button helper."""
    retry = wait_until_retry_button(timeout=timeout)
    if retry is None:
        logger.warning("retry button wait timed out (%s seconds)", timeout)
    return retry


def wait_until_occur(
    template_path: str | Path,
    timeout: float = 30.0,
) -> MatchResult | None:
    """等待直到指定模板出现，返回匹配结果或 None（超时）"""
    logger.info("正在等待模板 '%s' 出现，超时时?%s ?..", template_path, timeout)
    start_time = monotonic()
    while monotonic() - start_time < timeout:
        screenshot = adb.read_screenshot()
        match_result = find_template(screenshot, template_path)
        if match_result is not None:
            return match_result
        sleep(0.5)  # 每隔 0.5 秒检查一?
    logger.warning("等待模板 '%s' 超时 (%s ?", template_path, timeout)
    return None


def click_template(
    template_path: str | Path,
    screenshot_path: str | Path | None = None,
    threshold: float = 0.85,
) -> bool:
    """查找模板并点击中心点，找不到时返?False"""
    img = adb.read_screenshot(screenshot_path)
    match_result = find_template(img, template_path, threshold=threshold)
    if match_result is None:
        return False

    adb.delay(0.5).click(*match_result.center)
    return True


def resolve_current_level(
    screenshot: np.ndarray,
    fallback_level: int = DEFAULT_LEVEL,
    fallback_is_manual: bool = False,
) -> int:
    """Detect the current level from the activity page, or use the fallback."""
    if not AUTO_DETECT_LEVEL:
        logger.info("level auto detection disabled; using fallback level %s", fallback_level)
        return fallback_level

    title_result = recognize_level_title(
        screenshot,
        reference_dir=LEVEL_REFERENCE_DIR,
    )
    if title_result is not None:
        logger.info(
            "level title detection: best=%s score=%.3f second=%s score=%.3f confident=%s",
            title_result.level,
            title_result.score,
            title_result.second_level,
            title_result.second_score,
            title_result.confident,
        )
        if title_result.confident and title_result.level in LEVEL_GRID_SIZES:
            return title_result.level
        if title_result.confident:
            logger.warning(
                "level title detection returned unsupported level %s; falling back to image detection",
                title_result.level,
            )
    else:
        logger.info("level title detection: title number not readable in current screenshot")

    result = recognize_level_from_screenshot(
        screenshot,
        reference_dir=LEVEL_REFERENCE_DIR,
        candidate_levels=LEVEL_GRID_SIZES.keys(),
    )
    if result is None:
        logger.warning("level auto detection found no reference images; using fallback level %s", fallback_level)
        return fallback_level

    logger.info(
        "level auto detection: best=%s score=%.3f second=%s score=%.3f confident=%s",
        result.level,
        result.score,
        result.second_level,
        result.second_score,
        result.confident,
    )
    if result.confident:
        return result.level

    if REQUIRE_CONFIDENT_LEVEL_DETECTION and not fallback_is_manual:
        raise RuntimeError(
            "level auto detection is uncertain; stop before probing to avoid wasting bombs "
            f"(detected={result.level} score={result.score:.3f}, "
            f"second={result.second_level} score={result.second_score:.3f})"
        )

    logger.warning(
        "level auto detection is uncertain; using fallback level %s instead of detected level %s",
        fallback_level,
        result.level,
    )
    return fallback_level


def resolve_current_level_from_device(
    fallback_level: int = DEFAULT_LEVEL,
    fallback_is_manual: bool = False,
    attempts: int = 8,
) -> int:
    """Take several screenshots until the level title is stable enough to read."""
    if attempts <= 0:
        raise ValueError(f"attempts must be positive: {attempts}")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        adb.delay(1.0)
        screenshot = adb.read_screenshot()
        if handle_victory_prompt(
            timeout=VICTORY_WAIT_BEFORE_LEVEL_SECONDS,
            screenshot=screenshot,
        ):
            logger.info(
                "level detection attempt %s/%s skipped a victory banner",
                attempt,
                attempts,
            )
            continue
        try:
            return resolve_current_level(
                screenshot,
                fallback_level=fallback_level,
                fallback_is_manual=fallback_is_manual,
            )
        except RuntimeError as exc:
            last_error = exc
            logger.warning(
                "level detection attempt %s/%s was uncertain: %s",
                attempt,
                attempts,
                exc,
            )

    if last_error is not None:
        raise last_error
    return fallback_level


def main(level: int | None = None) -> Path | None:
    """执行指定关卡的辑探测并输出命中图"""
    run_started_at = monotonic()
    fallback_is_manual = level is not None
    fallback_level = DEFAULT_LEVEL if level is None else int(level)
    last_out_path: Path | None = None
    try:
        disable_weak_network()

        screenshot = adb.read_screenshot()
        if handle_victory_prompt(timeout=0.0, screenshot=screenshot):
            screenshot = adb.delay(1.0).read_screenshot()

        already_in_activity_detail = find_template(screenshot, QUIT_ACTIVITY_TEMPLATE) is not None
        if already_in_activity_detail:
            logger.info("current screen is already the activity detail; skipping activity entry")
        elif find_template(screenshot, ACTIVITY_BUTTON_TEMPLATE) is None:
            logger.error("当前不在海岛主界面，无法启动脚本")
            return None

        if not already_in_activity_detail:
            enter_activity()
        current_level = resolve_current_level_from_device(
            fallback_level=fallback_level,
            fallback_is_manual=fallback_is_manual,
        )
        while current_level <= MAX_LEVEL:
            grid_size = get_level_grid_size(current_level)
            hit_map = [[0] * grid_size for _ in range(grid_size)]
            base_img, quad = handle_game_level(
                current_level,
                hit_map,
                run_started_at=run_started_at,
            )
            out_path = OUTPUT_DIR / f"hit_map_level_{current_level}.png"
            save_hit_map_image(base_img, quad, hit_map, out_path)
            logger.info("hit map: %s", hit_map)
            logger.info("hit map image saved: %s", out_path)
            last_out_path = out_path

            if current_level >= MAX_LEVEL:
                logger.info("reached max level %s; stopping", MAX_LEVEL)
                break

            next_fallback_level = min(current_level + 1, MAX_LEVEL)
            logger.info(
                "level %s finished; trying to continue to next level (fallback=%s)",
                current_level,
                next_fallback_level,
            )
            try:
                next_level = resolve_current_level_from_device(
                    fallback_level=next_fallback_level,
                    fallback_is_manual=False,
                )
            except Exception as exc:
                logger.warning(
                    "failed to resolve next level after level %s; stopping progression: %s",
                    current_level,
                    exc,
                )
                break

            if next_level <= current_level:
                logger.warning(
                    "next level detection did not advance beyond %s (detected=%s); stopping progression",
                    current_level,
                    next_level,
                )
                break

            current_level = next_level

        return last_out_path
    finally:
        logger.info("脚本总运行时间：%s", format_elapsed(monotonic() - run_started_at))


if __name__ == "__main__":
    register_exit_cleanup()
    try:
        adb.ensure_root_shell()
        cleanup_reject_network("main startup")
        main()
    finally:
        cleanup_weak_network("main finished")
        cleanup_reject_network("main finished")

