"""后台尽力播放 WAV；待播提示只保留最新一条。"""

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import cast

from dexmani_real import ASSET_DIR

logger = logging.getLogger(__name__)

EVENT_FILES = {
    "begin": "遥操作启动.wav",
    "quit": "准备退出遥操作.wav",
    "save": "成功保存轨迹.wav",
    "discard": "放弃保存轨迹.wav",
    "home": "即将回到初始姿态.wav",
    "home_done": "已经回到初始姿态.wav",
    "emergency": "意外的事情出现了.wav",
}


class AudioFeedback:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.paths: dict[str, Path | None] = {}
        self.player = next((name for name in ("aplay", "paplay") if shutil.which(name)), None)
        self.worker = None
        if self.player is None:
            logger.warning("No audio player; audio disabled")
            return
        for event, name in EVENT_FILES.items():
            path = ASSET_DIR / "audio" / name
            if path.is_file():
                self.paths[event] = path
            else:
                logger.warning("Audio file not found: %s", path)
                self.paths[event] = None
        self.worker = threading.Thread(target=self.run, daemon=True, name="audio-feedback")
        self.worker.start()

    def play(self, event):
        with self.condition:
            if self.closed or self.player is None or self.paths[event] is None:
                return
            self.pending = self.paths[event]
            self.condition.notify()

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
            if self.worker.is_alive():
                logger.warning("Audio worker still exiting after 1s")

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    return
                path, self.pending = self.pending, None
            try:
                self.play_one(path)
            except Exception:
                logger.warning("Audio failed; disabling playback", exc_info=True)
                with self.condition:
                    self.player = None
                    self.pending = None
                return

    def play_one(self, path):
        player = cast(str, self.player)  # 仅在播放器已发现的工作线程调用。
        command = [player, "-q", str(path)] if player == "aplay" else [player, str(path)]
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            while True:
                with self.condition:
                    if self.closed:
                        return
                try:
                    _, stderr = proc.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if proc.returncode:
                detail = " ".join(stderr.decode("utf-8", errors="replace").split())[:512]
                raise RuntimeError(f"{self.player} exited {proc.returncode}: {detail}")
        finally:
            if proc.poll() is None:
                try:
                    proc.kill()
                    proc.communicate(timeout=0.5)
                except Exception:
                    logger.warning("Audio process cleanup failed", exc_info=True)
