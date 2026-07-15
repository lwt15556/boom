import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptSafetyTest(unittest.TestCase):
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
