"""后台播放 WAV 提示，紧急提示可抢占当前播放。"""

import logging
import shutil
import subprocess
import threading
import time
from collections import deque

from dexmani_real import ASSET_DIR

logger = logging.getLogger(__name__)

EVENT_FILES = {
    "begin": "遥操作启动.wav",
    "pause": "操作暂停.wav",
    "resume": "工作继续.wav",
    "quit": "准备退出遥操作.wav",
    "save": "成功保存轨迹.wav",
    "discard": "放弃保存轨迹.wav",
    "home": "即将回到初始姿态.wav",
    "home_done": "已经回到初始姿态.wav",
    "end": "操作结束.wav",
    "emergency": "意外的事情出现了.wav",
    "calibrated": "轴向已标定.wav",
}


class AudioFeedback:
    def __init__(self):
        self.condition = threading.Condition()
        self.current_proc = None
        self.active_request = None
        self.pending = deque()
        self.generation = 0
        self.closed = False
        self.paths = {event: ASSET_DIR / "audio" / name for event, name in EVENT_FILES.items()}
        self.player = next((name for name in ("aplay", "paplay") if shutil.which(name)), None)
        self.worker = None
        if self.player is None:
            logger.warning("No audio player; audio disabled")
            return
        for event, path in self.paths.items():
            if not path.is_file():
                logger.warning("Audio file not found: %s", path)
                self.paths[event] = None
        self.worker = threading.Thread(target=self.run, daemon=True, name="audio-feedback")
        self.worker.start()

    def play(self, event, *, interrupt=True):
        path = self.paths[event]
        with self.condition:
            if self.closed or self.player is None or path is None:
                return
            if interrupt:
                self.generation += 1
                self.pending.clear()
                self.kill_current()
            self.pending.append((path, self.generation))
            self.condition.notify_all()

    def wait_until_idle(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.active_request is not None or self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(timeout=remaining)
        return True

    def close(self):
        with self.condition:
            self.closed = True
            self.generation += 1
            self.pending.clear()
            self.kill_current()
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
            if self.worker.is_alive():
                raise RuntimeError("audio worker did not stop within 1s")

    def kill_current(self):
        """持有 condition 时只发终止信号，不等待子进程。"""
        if self.current_proc is not None and self.current_proc.poll() is None:
            try:
                self.current_proc.kill()
            except OSError:
                logger.warning("Audio process termination failed", exc_info=True)

    def run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed and self.player is not None:
                    self.condition.wait()
                if self.closed or self.player is None:
                    return
                self.active_request = self.pending.popleft()
            try:
                self.play_one(*self.active_request)
            except Exception:
                logger.warning("Audio failed; disabling playback", exc_info=True)
                with self.condition:
                    self.player = None
                    self.pending.clear()
            finally:
                with self.condition:
                    self.active_request = None
                    self.condition.notify_all()

    def play_one(self, path, generation):
        with self.condition:
            if generation != self.generation or self.closed:
                return
        command = [self.player, "-q", str(path)] if self.player == "aplay" else [self.player, str(path)]
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            with self.condition:
                self.current_proc = proc
                # Popen 在锁外；创建期间到达的抢占也必须终止旧请求。
                if generation != self.generation or self.closed:
                    self.kill_current()
            _, stderr = proc.communicate()
            with self.condition:
                cancelled = generation != self.generation or self.closed
            if proc.returncode and not cancelled:
                detail = " ".join(stderr.decode("utf-8", errors="replace").split())[:512]
                raise RuntimeError(f"{self.player} exited {proc.returncode}: {detail}")
        finally:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
            finally:
                with self.condition:
                    self.current_proc = None
