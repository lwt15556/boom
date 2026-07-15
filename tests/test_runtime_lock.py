import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import utils.runtime_lock as runtime_lock


LOCK_HOLDER_SCRIPT = r"""
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
handle = path.open("a+b")
handle.seek(0)
handle.write(b"not-a-pid")
handle.flush()
handle.seek(4096)

if os.name == "nt":
    import msvcrt

    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
else:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

print("READY", flush=True)
sys.stdin.readline()
"""


class RuntimeLockTest(unittest.TestCase):
    def test_os_lock_blocks_acquisition_when_pid_content_is_unreadable(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "main.pid"
            holder = subprocess.Popen(
                [sys.executable, "-c", LOCK_HOLDER_SCRIPT, str(pid_file)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(holder.stdout.readline().strip(), "READY")

                with self.assertRaises(runtime_lock.AlreadyRunningError):
                    runtime_lock.acquire_main_lock(pid_file)
            finally:
                holder.communicate("\n", timeout=5)

    def test_live_stale_pid_without_os_lock_does_not_block_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "main.pid"
            pid_file.write_text(str(os.getpid()), encoding="utf-8")

            acquired_pid = runtime_lock.acquire_main_lock(pid_file)
            try:
                self.assertEqual(acquired_pid, os.getpid())
            finally:
                release = getattr(runtime_lock, "release_main_lock", None)
                if release is not None:
                    release(pid_file=pid_file, pid=acquired_pid)
                else:
                    runtime_lock.remove_pid(pid_file=pid_file, pid=acquired_pid)

    def test_locked_file_with_invalid_pid_is_reported_as_active(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "main.pid"
            holder = subprocess.Popen(
                [sys.executable, "-c", LOCK_HOLDER_SCRIPT, str(pid_file)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(holder.stdout.readline().strip(), "READY")

                self.assertEqual(
                    runtime_lock.get_main_process(pid_file=pid_file),
                    (None, True),
                )
            finally:
                holder.communicate("\n", timeout=5)


if __name__ == "__main__":
    unittest.main()
