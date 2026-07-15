import unittest
from unittest.mock import patch

import tools.control_panel as control_panel

from tools.control_panel import (
    build_main_environment,
    decode_log_bytes,
    format_cell,
    format_phase,
    format_probe_mode,
    format_reason,
    format_result,
    should_show_log_line,
)


class ControlPanelHelperTest(unittest.TestCase):
    def test_build_main_environment_configures_red_scout(self):
        environment = build_main_environment("red_scout", 3)
        self.assertEqual(environment["BBMA_PROBE_MODE"], "red_scout")
        self.assertEqual(environment["BBMA_RED_SCOUT_COUNT"], "3")

    def test_probe_mode_formatter_has_fallback(self):
        self.assertEqual(format_probe_mode("blue_only"), "仅蓝色炮弹")
        self.assertEqual(format_probe_mode("red_scout"), "红色侦察 + 蓝色攻击")
        self.assertEqual(format_probe_mode("bad"), "仅蓝色炮弹")

    def test_decode_log_bytes_handles_utf8_and_removes_ansi(self):
        data = "\x1b[32mINFO\x1b[0m 网络已恢复".encode("utf-8")

        self.assertEqual(decode_log_bytes(data), "INFO 网络已恢复")

    def test_decode_log_bytes_handles_gbk(self):
        data = "模拟器已连接".encode("gbk")

        self.assertEqual(decode_log_bytes(data), "模拟器已连接")

    def test_key_log_filter_hides_noisy_lines(self):
        self.assertFalse(should_show_log_line("INFO 截图已保存", show_detail=False))
        self.assertFalse(should_show_log_line("DEBUG frame score=0.5", show_detail=False))
        self.assertTrue(should_show_log_line("INFO level 7 cell 76 result: hit", show_detail=False))
        self.assertTrue(should_show_log_line("INFO 截图已保存", show_detail=True))

    def test_status_formatters_are_user_facing(self):
        self.assertEqual(format_cell(76, 81), "#76（第 9 行，第 5 列）")
        self.assertEqual(format_phase("strategy_scan"), "智能寻路")
        self.assertEqual(format_result("hit"), "命中")
        self.assertEqual(format_result("level_complete"), "关卡已完成")
        self.assertEqual(format_result("scout_valid"), "侦察结果已累计")
        self.assertEqual(format_reason("hit_votes_4"), "4 帧确认")
        self.assertEqual(
            format_reason("victory_banner_during_reentry"),
            "重新进入时检测到胜利",
        )

    def test_red_scout_and_board_states_have_localized_names(self):
        self.assertEqual(control_panel.PHASE_NAMES["red_scout_preflight"], "红色侦察准备")
        self.assertEqual(control_panel.PHASE_NAMES["blue_attack"], "蓝色攻击")
        self.assertEqual(control_panel.BOARD_STATE_NAMES["scout_miss"], "侦察未命中")
        self.assertEqual(control_panel.BOARD_STATE_NAMES["scout_hit"], "侦察命中")

    def test_restore_network_refuses_while_main_is_running(self):
        with (
            patch.object(control_panel, "get_main_process", return_value=(1234, True)),
            patch.object(control_panel, "AdbController") as adb_controller,
        ):
            with self.assertRaisesRegex(RuntimeError, "主程序运行中"):
                control_panel.restore_network()

        adb_controller.assert_not_called()

    def test_fail_closed_runtime_status_requires_safe_offline_recovery(self):
        with (
            patch.object(control_panel, "has_pending_probe", return_value=False),
            patch.object(
                control_panel,
                "read_runtime_status",
                return_value={"network": "fail_closed"},
            ),
        ):
            self.assertTrue(control_panel._runtime_status_is_offline())

    def test_stop_program_discards_pending_request_before_restoring_network(self):
        events = []

        class FakeAdb:
            def __init__(self, serial):
                events.append(("adb_init", serial))

            def ensure_root_shell(self):
                events.append(("ensure_root",))

            def enable_reject_network(self, package_name):
                events.append(("enable_reject", package_name))

            def enable_weak_network(self, package_name):
                events.append(("enable_drop", package_name))

            def delay(self, seconds):
                events.append(("delay", seconds))
                return self

            def close_app(self, package_name):
                events.append(("close_app", package_name))

            def wait_until_app_stopped(self, package_name, timeout, poll_interval):
                events.append(("wait_stopped", package_name, timeout, poll_interval))
                return True

            def disable_weak_network(self, package_name):
                events.append(("disable_drop", package_name))

            def disable_reject_network(self, package_name):
                events.append(("disable_reject", package_name))

        with (
            patch.object(control_panel, "get_main_process", return_value=(1234, True)),
            patch.object(control_panel, "AdbController", FakeAdb),
            patch.object(
                control_panel,
                "stop_pid",
                side_effect=lambda pid: events.append(("stop_pid", pid)),
            ),
            patch.object(
                control_panel,
                "remove_pid",
                side_effect=lambda **kwargs: events.append(("remove_pid", kwargs["pid"])),
            ),
            patch.object(
                control_panel,
                "clear_pending_probe",
                side_effect=lambda: events.append(("clear_pending",)),
            ),
        ):
            message = control_panel.stop_program()

        package_name = control_panel.GAME_PACKAGE_NAME
        self.assertIn("PID=1234", message)
        self.assertEqual(
            events,
            [
                ("adb_init", control_panel.ADB_SERIAL),
                ("ensure_root",),
                ("enable_drop", package_name),
                ("enable_reject", package_name),
                ("delay", control_panel.NETWORK_BLOCK_SETTLE_SECONDS),
                ("stop_pid", 1234),
                ("close_app", package_name),
                (
                    "wait_stopped",
                    package_name,
                    control_panel.APP_STOP_TIMEOUT_SECONDS,
                    control_panel.APP_STOP_POLL_SECONDS,
                ),
                ("delay", control_panel.POST_FORCE_STOP_GUARD_SECONDS),
                ("remove_pid", 1234),
                ("clear_pending",),
                ("disable_drop", package_name),
                ("disable_reject", package_name),
            ],
        )

    def test_stop_program_keeps_network_blocked_when_game_does_not_stop(self):
        events = []

        class FakeAdb:
            def __init__(self, _serial):
                pass

            def ensure_root_shell(self):
                pass

            def enable_weak_network(self, package_name):
                events.append(("enable_drop", package_name))

            def enable_reject_network(self, package_name):
                events.append(("enable_reject", package_name))

            def delay(self, seconds):
                events.append(("delay", seconds))
                return self

            def close_app(self, package_name):
                events.append(("close_app", package_name))

            def wait_until_app_stopped(self, package_name, timeout, poll_interval):
                events.append(("wait_stopped", package_name, timeout, poll_interval))
                return False

            def disable_weak_network(self, package_name):
                events.append(("disable_drop", package_name))

            def disable_reject_network(self, package_name):
                events.append(("disable_reject", package_name))

        with (
            patch.object(control_panel, "get_main_process", return_value=(1234, True)),
            patch.object(control_panel, "AdbController", FakeAdb),
            patch.object(control_panel, "stop_pid"),
        ):
            with self.assertRaisesRegex(RuntimeError, "游戏进程未完全退出"):
                control_panel.stop_program()

        self.assertFalse(any(event[0].startswith("disable_") for event in events))

    def test_stop_program_never_kills_stale_unowned_pid(self):
        with (
            patch.object(control_panel, "get_main_process", return_value=(1234, False)),
            patch.object(control_panel, "remove_pid") as remove_pid,
            patch.object(control_panel, "restore_network", return_value="网络已恢复"),
            patch.object(control_panel, "stop_pid") as stop_pid,
            patch.object(control_panel, "AdbController") as adb_controller,
        ):
            message = control_panel.stop_program()

        self.assertIn("过期 PID", message)
        remove_pid.assert_called_once_with(pid=1234)
        stop_pid.assert_not_called()
        adb_controller.assert_not_called()

    def test_restore_network_discards_stale_offline_request_after_main_dies(self):
        events = []

        class FakeAdb:
            def __init__(self, serial):
                events.append(("adb_init", serial))

            def ensure_root_shell(self):
                events.append(("ensure_root",))

            def enable_reject_network(self, package_name):
                events.append(("enable_reject", package_name))

            def enable_weak_network(self, package_name):
                events.append(("enable_drop", package_name))

            def delay(self, seconds):
                events.append(("delay", seconds))
                return self

            def close_app(self, package_name):
                events.append(("close_app", package_name))

            def wait_until_app_stopped(self, package_name, timeout, poll_interval):
                events.append(("wait_stopped", package_name, timeout, poll_interval))
                return True

            def disable_weak_network(self, package_name):
                events.append(("disable_drop", package_name))

            def disable_reject_network(self, package_name):
                events.append(("disable_reject", package_name))

        with (
            patch.object(control_panel, "get_main_process", return_value=None),
            patch.object(
                control_panel,
                "read_runtime_status",
                return_value={"running": True, "network": "断网中"},
            ),
            patch.object(control_panel, "AdbController", FakeAdb),
            patch.object(control_panel, "has_pending_probe", return_value=False),
            patch.object(
                control_panel,
                "clear_pending_probe",
                side_effect=lambda: events.append(("clear_pending",)),
            ),
        ):
            message = control_panel.restore_network()

        package_name = control_panel.GAME_PACKAGE_NAME
        self.assertIn("已丢弃中断请求", message)
        self.assertEqual(
            events,
            [
                ("adb_init", control_panel.ADB_SERIAL),
                ("ensure_root",),
                ("enable_drop", package_name),
                ("enable_reject", package_name),
                ("delay", control_panel.NETWORK_BLOCK_SETTLE_SECONDS),
                ("close_app", package_name),
                (
                    "wait_stopped",
                    package_name,
                    control_panel.APP_STOP_TIMEOUT_SECONDS,
                    control_panel.APP_STOP_POLL_SECONDS,
                ),
                ("delay", control_panel.POST_FORCE_STOP_GUARD_SECONDS),
                ("clear_pending",),
                ("disable_drop", package_name),
                ("disable_reject", package_name),
            ],
        )


if __name__ == "__main__":
    unittest.main()
