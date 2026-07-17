import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptSafetyTest(unittest.TestCase):
    def test_one_click_launcher_runs_setup_with_execution_policy_bypass(self):
        launcher = PROJECT_ROOT / "一键安装并启动.bat"
        script = launcher.read_text(encoding="ascii")

        self.assertIn("-ExecutionPolicy Bypass", script)
        self.assertIn('setup.ps1', script)

    def test_setup_creates_venv_installs_requirements_and_validates_adb(self):
        script = (PROJECT_ROOT / "setup.ps1").read_text(encoding="utf-8")

        self.assertIn('Python.Python.3.11', script)
        self.assertIn('PrefixArguments @("-3")', script)
        self.assertIn('"-m", "venv"', script)
        self.assertIn('"-m", "pip", "install", "-r"', script)
        self.assertIn('"import cv2, numpy, PyQt6;', script)
        self.assertIn('Arguments @(\"version\")', script)
        self.assertIn('.venv.invalid.', script)
        self.assertIn('[switch]$SkipLaunch', script)

    def test_control_panel_launcher_bootstraps_missing_environment(self):
        script = (PROJECT_ROOT / "run_control_panel.ps1").read_text(encoding="utf-8")

        self.assertIn('$setup = Join-Path $root "setup.ps1"', script)
        self.assertIn('& $setup -SkipLaunch', script)

    def test_stop_script_delegates_main_shutdown_to_locked_process_control(self):
        script = (PROJECT_ROOT / "stop_all.ps1").read_text(encoding="utf-8")

        self.assertIn("from tools.control_panel import stop_program", script)
        self.assertIn("stop_program()", script)
        self.assertNotIn("_debug\\runtime\\main.pid", script)
        self.assertNotIn('CommandLine = "main.py pid file"', script)
        self.assertNotIn('*main.py*', script)

    def test_overlay_script_uses_utf8_and_does_not_leave_overlay_after_main_rejection(self):
        script = (PROJECT_ROOT / "run_with_overlay.ps1").read_text(encoding="utf-8")

        self.assertIn('$env:PYTHONUTF8 = "1"', script)
        self.assertIn('$env:PYTHONIOENCODING = "utf-8"', script)
        self.assertIn("$mainProcess = Start-Process", script)
        self.assertIn("if ($mainProcess.HasExited)", script)
        self.assertLess(
            script.index("$mainProcess = Start-Process"),
            script.index("-ArgumentList @($overlay)"),
        )

    def test_control_panel_launcher_uses_utf8_and_hides_console_window(self):
        script = (PROJECT_ROOT / "run_control_panel.ps1").read_text(encoding="utf-8")

        self.assertIn('$env:PYTHONUTF8 = "1"', script)
        self.assertIn('$env:PYTHONIOENCODING = "utf-8"', script)
        self.assertIn("-WindowStyle Hidden", script)


if __name__ == "__main__":
    unittest.main()
