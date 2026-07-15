import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from utils.adb_control import AdbController, NetworkIsolationStatus


class AdbControlTest(unittest.TestCase):
    @staticmethod
    def _png_bytes(image: np.ndarray) -> bytes:
        encoded, buffer = cv2.imencode(".png", image)
        if not encoded:
            raise AssertionError("test image could not be encoded")
        return buffer.tobytes()

    def test_read_screenshot_uses_exec_out_without_remote_file_or_pull(self):
        controller = AdbController.__new__(AdbController)
        expected = np.zeros((3, 4, 3), dtype=np.uint8)
        expected[1, 2] = (10, 80, 220)
        controller._run_binary = Mock(return_value=self._png_bytes(expected))
        controller._run = Mock()

        actual = controller.read_screenshot()

        np.testing.assert_array_equal(actual, expected)
        controller._run_binary.assert_called_once_with(["exec-out", "screencap", "-p"])
        controller._run.assert_not_called()

    def test_read_screenshot_writes_requested_debug_png_from_memory(self):
        controller = AdbController.__new__(AdbController)
        expected = np.full((2, 5, 3), 123, dtype=np.uint8)
        payload = self._png_bytes(expected)
        controller._run_binary = Mock(return_value=payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "frame.png"
            actual = controller.read_screenshot(output_path)

            self.assertEqual(output_path.read_bytes(), payload)
            np.testing.assert_array_equal(actual, expected)

    def test_capture_screenshot_defers_writing_and_preserves_original_png(self):
        controller = AdbController.__new__(AdbController)
        expected = np.full((3, 5, 3), 91, dtype=np.uint8)
        payload = self._png_bytes(expected)
        controller._run_binary = Mock(return_value=payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "deferred" / "frame.png"
            capture = controller.capture_screenshot()

            self.assertFalse(output_path.exists())
            np.testing.assert_array_equal(capture.image, expected)
            self.assertEqual(capture.png_bytes, payload)

            capture.save(output_path)

            self.assertEqual(output_path.read_bytes(), payload)

    def test_read_screenshot_falls_back_to_pull_when_exec_out_is_not_decodable(self):
        controller = AdbController.__new__(AdbController)
        expected = np.full((2, 2, 3), 77, dtype=np.uint8)
        controller._run_binary = Mock(return_value=b"not-a-png")
        controller._read_screenshot_via_pull = Mock(return_value=expected)

        actual = controller.read_screenshot()

        np.testing.assert_array_equal(actual, expected)
        controller._read_screenshot_via_pull.assert_called_once()
        self.assertEqual(
            controller._read_screenshot_via_pull.call_args.args[0].name,
            "screen.png",
        )

    def _controller_for_isolation(self, *, ipv4_blocked, ipv6_result):
        controller = AdbController.__new__(AdbController)
        controller._get_package_uid = Mock(return_value=10042)
        controller._is_weak_network_rule_active = Mock(return_value=ipv4_blocked)
        controller._run_privileged_script = Mock(return_value=ipv6_result)
        return controller

    def test_verify_network_isolated_for_safe_ipv4_only_app(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=True,
            ipv6_result=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        )

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertEqual(status, NetworkIsolationStatus(True, True, False, False, status.detail))
        controller._run_privileged_script.assert_called_once_with("ip -6 route show", check=False)

    def test_verify_network_isolated_rejects_ipv6_route_without_rule(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=True,
            ipv6_result=subprocess.CompletedProcess([], 0, stdout="default via fe80::1 dev wlan0\n", stderr=""),
        )
        controller._run_privileged_script.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="default via fe80::1 dev wlan0\n", stderr=""),
            subprocess.CompletedProcess([], 1, stdout="", stderr=""),
        ]

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertFalse(status.safe)
        self.assertTrue(status.ipv6_route_present)
        self.assertFalse(status.ipv6_blocked)
        self.assertEqual(controller._run_privileged_script.call_args_list[1].args[0],
                         "ip6tables -C OUTPUT -m owner --uid-owner 10042 -j BBMA_WEAKNET "
                         "&& ip6tables -C BBMA_WEAKNET -j DROP")

    def test_verify_network_isolated_marks_route_read_failure_unsafe(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=True,
            ipv6_result=subprocess.CompletedProcess([], 1, stdout="", stderr="permission denied"),
        )

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertFalse(status.safe)
        self.assertIn("ipv6 route check failed", status.detail)

    def test_verify_network_isolated_marks_unblocked_ipv4_unsafe(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=False,
            ipv6_result=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        )

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertFalse(status.safe)
        self.assertFalse(status.ipv4_blocked)

    def test_verify_network_isolated_accepts_ipv6_route_with_successful_rule(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=True,
            ipv6_result=subprocess.CompletedProcess([], 0, stdout="2001:db8::/64 via fe80::1 dev wlan0\n", stderr=""),
        )
        controller._run_privileged_script.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="2001:db8::/64 via fe80::1 dev wlan0\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertEqual(status.safe, True)
        self.assertTrue(status.ipv6_route_present)
        self.assertTrue(status.ipv6_blocked)

    def test_verify_network_isolated_detects_default_ipv6_route_without_via(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=True,
            ipv6_result=subprocess.CompletedProcess([], 0, stdout="default dev wlan0\n", stderr=""),
        )
        controller._run_privileged_script.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="default dev wlan0\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertTrue(status.ipv6_route_present)
        self.assertTrue(status.ipv6_blocked)
        self.assertTrue(status.safe)

    def test_verify_network_isolated_detects_direct_ipv6_route(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=True,
            ipv6_result=subprocess.CompletedProcess([], 0, stdout="2001:db8::/64 dev wlan0\n", stderr=""),
        )
        controller._run_privileged_script.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="2001:db8::/64 dev wlan0\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertTrue(status.ipv6_route_present)
        self.assertTrue(status.ipv6_blocked)
        self.assertTrue(status.safe)

    def test_verify_network_isolated_rejects_ipv6_rule_success_with_stderr(self):
        controller = self._controller_for_isolation(
            ipv4_blocked=True,
            ipv6_result=subprocess.CompletedProcess([], 0, stdout="default dev wlan0\n", stderr=""),
        )
        controller._run_privileged_script.side_effect = [
            subprocess.CompletedProcess([], 0, stdout="default dev wlan0\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr="warning\n"),
        ]

        status = controller.verify_app_network_isolated("com.example.game")

        self.assertTrue(status.ipv6_route_present)
        self.assertFalse(status.ipv6_blocked)
        self.assertFalse(status.safe)

    def test_verify_network_isolated_rejects_empty_package(self):
        controller = AdbController.__new__(AdbController)

        with self.assertRaises(ValueError):
            controller.verify_app_network_isolated("  ")

    def test_weak_network_rule_check_requires_jump_and_drop_rule(self):
        controller = AdbController.__new__(AdbController)
        controller._run_privileged_script = Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
        )

        active = controller._is_weak_network_rule_active(10042)

        self.assertTrue(active)
        controller._run_privileged_script.assert_called_once_with(
            "iptables -C OUTPUT -m owner --uid-owner 10042 -j BBMA_WEAKNET "
            "&& iptables -C BBMA_WEAKNET -j DROP",
            check=False,
        )

    def test_reject_network_rule_check_requires_jump_and_reject_rules(self):
        controller = AdbController.__new__(AdbController)
        controller._run_privileged_script = Mock(
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
        )

        active = controller._is_reject_network_rule_active(10042)

        self.assertTrue(active)
        controller._run_privileged_script.assert_called_once_with(
            "iptables -C OUTPUT -m owner --uid-owner 10042 -j BBMA_REJECTNET "
            "&& iptables -C BBMA_REJECTNET -p tcp -j REJECT --reject-with tcp-reset "
            "&& iptables -C BBMA_REJECTNET -j REJECT --reject-with icmp-port-unreachable",
            check=False,
        )

    def test_network_rule_checks_treat_stderr_as_unsafe(self):
        controller = AdbController.__new__(AdbController)
        controller._run_privileged_script = Mock(
            return_value=subprocess.CompletedProcess(
                [],
                0,
                stdout="",
                stderr="iptables warning",
            )
        )

        self.assertFalse(controller._is_weak_network_rule_active(10042))
        self.assertFalse(controller._is_reject_network_rule_active(10042))

    def test_wait_until_app_stopped_polls_until_pid_disappears(self):
        controller = AdbController.__new__(AdbController)
        controller._run = Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout="1234\n", stderr=""),
                subprocess.CompletedProcess([], 1, stdout="", stderr=""),
            ]
        )

        with patch("utils.adb_control.sleep") as sleep:
            stopped = controller.wait_until_app_stopped(
                "com.example.game",
                timeout=1.0,
                poll_interval=0.05,
            )

        self.assertTrue(stopped)
        self.assertEqual(controller._run.call_count, 2)
        sleep.assert_called_once_with(0.05)

    def test_wait_until_app_stopped_does_not_accept_adb_error(self):
        controller = AdbController.__new__(AdbController)
        controller._run = Mock(
            return_value=subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr="error: device offline\n",
            )
        )

        stopped = controller.wait_until_app_stopped(
            "com.example.game",
            timeout=0.0,
            poll_interval=0.05,
        )

        self.assertFalse(stopped)

    def test_wait_until_app_stopped_limits_each_query_to_remaining_deadline(self):
        controller = AdbController.__new__(AdbController)
        controller._run = Mock(
            return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="")
        )

        with patch("utils.adb_control.monotonic", side_effect=[100.0, 100.25]):
            stopped = controller.wait_until_app_stopped(
                "com.example.game",
                timeout=0.5,
                poll_interval=0.05,
            )

        self.assertTrue(stopped)
        self.assertAlmostEqual(controller._run.call_args.kwargs["timeout"], 0.25)

    def test_run_uses_finite_timeout_and_reports_timeout(self):
        controller = AdbController.__new__(AdbController)
        controller.serial = "127.0.0.1:5555"

        with patch(
            "utils.adb_control.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["adb", "devices"], 20.0),
        ) as run:
            with self.assertRaises(TimeoutError):
                controller._run(["devices"], device=False)

        self.assertEqual(run.call_args.kwargs["timeout"], 20.0)

    def test_recover_connection_uses_configured_adb_runner_for_disconnect(self):
        controller = AdbController.__new__(AdbController)
        controller.serial = "127.0.0.1:5555"
        controller._touch_device_info = ("old", 0, 1, 0, 1)
        controller._root_shell_ready = True
        controller._package_uid_cache = {"com.example.game": 10042}
        controller._ip6tables_available = True
        controller._weak_network_enabled_uids = {10042}
        controller._reject_network_enabled_uids = {10042}
        controller._run = Mock()

        with patch("utils.adb_control.sleep"):
            controller._recover_connection()

        self.assertEqual(
            controller._run.call_args_list,
            [
                unittest.mock.call(
                    ["disconnect", "127.0.0.1:5555"],
                    device=False,
                    check=False,
                ),
                unittest.mock.call(["connect", "127.0.0.1:5555"], device=False),
            ],
        )
        self.assertIsNone(controller._touch_device_info)
        self.assertFalse(controller._root_shell_ready)
        self.assertEqual(controller._package_uid_cache, {})
        self.assertIsNone(controller._ip6tables_available)
        self.assertEqual(controller._weak_network_enabled_uids, set())
        self.assertEqual(controller._reject_network_enabled_uids, set())


if __name__ == "__main__":
    unittest.main()
