from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import os
import re
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter.scrolledtext import ScrolledText


DEFAULT_TITLE_KEYWORDS = (
    "模拟器",
    "MuMu",
    "雷电",
    "夜神",
    "BlueStacks",
    "逍遥",
    "LDPlayer",
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _find_window_rect(title_keywords: tuple[str, ...]) -> tuple[int, int, int, int] | None:
    user32 = ctypes.windll.user32
    hwnds: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_proc(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if any(keyword.lower() in title.lower() for keyword in title_keywords):
            hwnds.append(hwnd)
            return False
        return True

    user32.EnumWindows(enum_proc, None)
    if not hwnds:
        return None

    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnds[0], ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def _read_existing_tail(log_file: Path, max_lines: int) -> list[str]:
    if not log_file.exists():
        return []
    try:
        return log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return []


def _read_log_chunk(log_file: Path, position: int) -> tuple[list[str], int]:
    try:
        size = log_file.stat().st_size
        start = int(position)
        if start < 0 or start > size:
            start = 0
        with log_file.open("rb") as file:
            file.seek(start)
            data = file.read()
    except OSError:
        return [], position

    if not data:
        return [], start
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        return [], start

    consumed = data[: last_newline + 1]
    next_position = start + last_newline + 1
    text = consumed.decode("utf-8", errors="replace")
    return [_strip_ansi(line) for line in text.splitlines()], next_position


class LogOverlay:
    def __init__(
        self,
        log_file: Path,
        *,
        width: int,
        height: int,
        opacity: float,
        max_lines: int,
        title_keywords: tuple[str, ...],
    ) -> None:
        self.log_file = log_file
        self.width = width
        self.height = height
        self.opacity = opacity
        self.max_lines = max_lines
        self.title_keywords = title_keywords
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.pending: deque[str] = deque()
        self.stop_event = threading.Event()

        self.root = tk.Tk()
        self.root.title("BBMA Logs")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", opacity)
        self.root.configure(bg="#101418")

        self.text = ScrolledText(
            self.root,
            bg="#101418",
            fg="#d7f9d8",
            insertbackground="#d7f9d8",
            font=("Consolas", 10),
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        self._place_window()
        self._load_initial_lines()
        self.root.after(200, self._flush_pending)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _place_window(self) -> None:
        rect = _find_window_rect(self.title_keywords)
        if rect:
            left, top, _, bottom = rect
            x = max(0, left - self.width - 8)
            y = max(0, top)
            height = min(self.height, max(240, bottom - top))
        else:
            x = 20
            y = 80
            height = self.height
        self.root.geometry(f"{self.width}x{height}+{x}+{y}")

    def _load_initial_lines(self) -> None:
        for line in _read_existing_tail(self.log_file, self.max_lines):
            self.lines.append(_strip_ansi(line))
        self._render()

    def start_tail_thread(self) -> None:
        thread = threading.Thread(target=self._tail_log_file, daemon=True)
        thread.start()

    def _tail_log_file(self) -> None:
        position = self.log_file.stat().st_size if self.log_file.exists() else 0
        while not self.stop_event.is_set():
            lines, position = _read_log_chunk(self.log_file, position)
            self.pending.extend(lines)
            time.sleep(0.25)

    def _flush_pending(self) -> None:
        changed = False
        while self.pending:
            self.lines.append(self.pending.popleft())
            changed = True
        if changed:
            self._render()
        if not self.stop_event.is_set():
            self.root.after(200, self._flush_pending)

    def _render(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", "\n".join(self.lines))
        self.text.see("end")
        self.text.configure(state="disabled")

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.start_tail_thread()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    default_log = _project_root() / "_debug" / "logs" / "bbma.log"
    parser = argparse.ArgumentParser(description="Show BBMA log output in a topmost overlay window.")
    parser.add_argument("--log-file", type=Path, default=default_log)
    parser.add_argument("--width", type=int, default=430)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--opacity", type=float, default=0.86)
    parser.add_argument("--max-lines", type=int, default=120)
    parser.add_argument(
        "--title-keywords",
        default=",".join(DEFAULT_TITLE_KEYWORDS),
        help="Comma-separated emulator window title keywords.",
    )
    return parser.parse_args()


def main() -> None:
    if os.name != "nt":
        raise SystemExit("log_overlay.py only supports Windows desktop overlay mode.")
    args = parse_args()
    keywords = tuple(keyword.strip() for keyword in args.title_keywords.split(",") if keyword.strip())
    overlay = LogOverlay(
        args.log_file,
        width=args.width,
        height=args.height,
        opacity=max(0.2, min(1.0, args.opacity)),
        max_lines=args.max_lines,
        title_keywords=keywords or DEFAULT_TITLE_KEYWORDS,
    )
    overlay.run()


if __name__ == "__main__":
    main()
