import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.setup_preflight import prepare_adb


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def __call__(self, command, *, timeout):
        self.commands.append((command, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected command: {command}")
        return self.responses.pop(0)


def result(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class SetupAdbPreflightTest(unittest.TestCase):
    def test_script_can_load_project_config_from_any_working_directory(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "setup_preflight.py"
        with tempfile.TemporaryDirectory() as working_directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import runpy; runpy.run_path({str(script)!r})",
                ],
                cwd=working_directory,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_connects_requests_root_waits_for_reconnect_and_verifies_uid_zero(self):
        runner = FakeRunner(
            [
                result("connected to 127.0.0.1:5555\n"),
                result("List of devices attached\n127.0.0.1:5555\tdevice\n"),
                result("restarting adbd as root\n"),
                result("connected to 127.0.0.1:5555\n"),
                result("offline\n", returncode=1),
                result("already connected to 127.0.0.1:5555\n"),
                result("device\n"),
                result("0\n"),
            ]
        )

        preflight = prepare_adb(
            "adb.exe",
            "127.0.0.1:5555",
            runner=runner,
            sleep=lambda _seconds: None,
            reconnect_attempts=2,
        )

        self.assertTrue(preflight.ready)
        self.assertIn("Root 验证通过", preflight.message)
        commands = [command for command, _timeout in runner.commands]
        self.assertIn(
            ["adb.exe", "-s", "127.0.0.1:5555", "root"],
            commands,
        )
        self.assertEqual(
            commands[-1],
            ["adb.exe", "-s", "127.0.0.1:5555", "shell", "id", "-u"],
        )

    def test_reports_missing_target_and_lists_other_detected_devices(self):
        runner = FakeRunner(
            [
                result("cannot connect to 127.0.0.1:5555\n", returncode=1),
                result("List of devices attached\n127.0.0.1:7555\tdevice\n"),
            ]
        )

        preflight = prepare_adb(
            "adb.exe",
            "127.0.0.1:5555",
            runner=runner,
            sleep=lambda _seconds: None,
        )

        self.assertFalse(preflight.ready)
        self.assertIn("127.0.0.1:5555", preflight.message)
        self.assertIn("127.0.0.1:7555", preflight.message)
        self.assertIn("config.py", preflight.message)

    def test_reports_unauthorized_target_without_requesting_root(self):
        runner = FakeRunner(
            [
                result("already connected to 127.0.0.1:5555\n"),
                result("List of devices attached\n127.0.0.1:5555\tunauthorized\n"),
            ]
        )

        preflight = prepare_adb(
            "adb.exe",
            "127.0.0.1:5555",
            runner=runner,
            sleep=lambda _seconds: None,
        )

        self.assertFalse(preflight.ready)
        self.assertIn("unauthorized", preflight.message)
        self.assertEqual(len(runner.commands), 2)

    def test_rejects_shell_uid_other_than_zero(self):
        runner = FakeRunner(
            [
                result("already connected to 127.0.0.1:5555\n"),
                result("List of devices attached\n127.0.0.1:5555\tdevice\n"),
                result("adbd is already running as root\n"),
                result("already connected to 127.0.0.1:5555\n"),
                result("device\n"),
                result("2000\n"),
            ]
        )

        preflight = prepare_adb(
            "adb.exe",
            "127.0.0.1:5555",
            runner=runner,
            sleep=lambda _seconds: None,
            reconnect_attempts=1,
        )

        self.assertFalse(preflight.ready)
        self.assertIn("2000", preflight.message)
        self.assertIn("Root", preflight.message)


if __name__ == "__main__":
    unittest.main()
