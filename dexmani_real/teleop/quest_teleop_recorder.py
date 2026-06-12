"""Quest VR teleoperation recorder with keyboard-controlled recording.

States:
  IDLE      — Quest tracking active, visualization only, no recording
  RECORDING — Recording VR frames to buffer, visualization active

Keyboard:
  R — Start recording (from IDLE)
  Q — Stop recording
  Esc — Exit
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from dexmani_real.teleop.arm_wrist_mapper import ArmWristMapper
from dexmani_real.teleop.quest_hand_tracker import QuestHandTracker
from dexmani_real.teleop.quest_hand_visualizer import QuestHandVisualizer
from dexmani_real.teleop.trajectory_buffer import (
    TrajectoryBuffer,
    get_next_episode_path,
)

try:
    from pynput import keyboard as pynput_keyboard

    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

DEFAULT_EEF_POS = np.array([0.4, 0.0, 0.3])
DEFAULT_EEF_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


class QuestTeleopRecorder:
    def __init__(
        self,
        tracker: QuestHandTracker | None = None,
        visualizer: QuestHandVisualizer | None = None,
        mapper: ArmWristMapper | None = None,
        eef_pos0: np.ndarray | None = None,
        eef_quat_wxyz0: np.ndarray | None = None,
    ) -> None:
        self.tracker = tracker or QuestHandTracker(
            transport="tcp_server",
            host="0.0.0.0",
            port=8000,
            hand_side="right",
            output_frame="flu",
            verbose=True,
        )
        self.visualizer = visualizer or QuestHandVisualizer(show_axes=True)
        self.mapper = mapper or ArmWristMapper(
            pos_scale=1.0,
            rot_scale=1.0,
            eef_delta_bounds=np.array([
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-0.2, 0.2],
            ]),
        )
        self.eef_pos0 = eef_pos0 or DEFAULT_EEF_POS.copy()
        self.eef_quat_wxyz0 = eef_quat_wxyz0 or DEFAULT_EEF_QUAT_WXYZ.copy()

        self.buffer = TrajectoryBuffer()
        self.key_queue: queue.Queue[str] = queue.Queue()
        self.listener: pynput_keyboard.Listener | None = None

        self.state = "IDLE"
        self.last_save_path: str | None = None
        self.recording_start_time = 0.0
        self.frame_count = 0
        self._mapper_ready = False

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        self.buffer.clear()
        self._reset_mapper()
        self.state = "RECORDING"
        self.recording_start_time = time.time()
        self.frame_count = 0
        print("[RECORDING] Started. Press Q to stop.")

    def stop_recording(self) -> None:
        path = get_next_episode_path()
        try:
            self.buffer.save(path)
            self.last_save_path = str(path)
            duration = time.time() - self.recording_start_time
            print(
                f"[RECORDING] Saved {len(self.buffer)} frames "
                f"({duration:.1f}s) to {path}"
            )
        except ValueError:
            print("[RECORDING] No frames recorded, nothing saved.")
        self.buffer.clear()
        self.state = "IDLE"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, start_mode: str = "idle") -> None:
        if not HAS_PYNPUT:
            print("pynput is required. Install with: pip install pynput")
            sys.exit(1)

        self._start_keyboard_listener()
        self._print_banner()

        try:
            with self.tracker:
                self._wait_for_first_frame()
                if start_mode == "record":
                    print("[AUTO] Starting recording...")
                    self.start_recording()
                self._main_loop()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            if self.listener is not None:
                self.listener.stop()
            print("Done.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_keyboard_listener(self) -> None:
        def on_press(key):
            try:
                if hasattr(key, "char") and key.char is not None:
                    ch = key.char.lower()
                    if ch in ("r", "q"):
                        self.key_queue.put(ch)
                elif key == pynput_keyboard.Key.esc:
                    self.key_queue.put("esc")
            except Exception:
                pass

        self.listener = pynput_keyboard.Listener(on_press=on_press)
        self.listener.start()

    def _print_banner(self) -> None:
        print("=" * 50)
        print("Quest Teleop Recorder")
        print("  R   — Start recording")
        print("  Q   — Stop recording")
        print("  Esc — Exit")
        print("=" * 50)

    def _wait_for_first_frame(self) -> None:
        print("Waiting for Quest VR frames...")
        while True:
            if self.tracker.get_latest() is not None:
                print("Quest hand tracking active.")
                return
            self._drain_keys()
            time.sleep(0.05)

    def _reset_mapper(self) -> None:
        vr_frame = self.tracker.get_latest()
        if vr_frame is None:
            return
        self.mapper.reset(
            wrist_pos=vr_frame["wrist_pos"],
            wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
            eef_pos=self.eef_pos0,
            eef_quat_wxyz=self.eef_quat_wxyz0,
        )
        self._mapper_ready = True

    def _drain_keys(self) -> None:
        try:
            while True:
                ch = self.key_queue.get_nowait()
                if ch == "esc":
                    raise KeyboardInterrupt
        except queue.Empty:
            pass

    def _process_keys(self) -> None:
        try:
            while True:
                ch = self.key_queue.get_nowait()

                if ch == "esc":
                    raise KeyboardInterrupt

                if ch == "r" and self.state == "IDLE":
                    self.start_recording()

                elif ch == "q" and self.state == "RECORDING":
                    self.stop_recording()

        except queue.Empty:
            pass

    def _main_loop(self) -> None:
        while True:
            self._process_keys()
            self._tick()
            time.sleep(0.003)

    def _tick(self) -> None:
        vr_frame = self.tracker.get_latest()
        if vr_frame is None:
            return

        self.visualizer.log_frame(vr_frame, path="vr/right_hand")

        if not self._mapper_ready:
            self._reset_mapper()

        target = self.mapper.map(vr_frame["wrist_pos"], vr_frame["wrist_quat_wxyz"])
        if target is not None:
            self.visualizer.log_axes(
                "ee_target/right_hand",
                target["pos"],
                target["quat_wxyz"],
            )

        if self.state == "RECORDING":
            eef_pos = target["pos"] if target is not None else np.zeros(3)
            eef_quat_wxyz = target["quat_wxyz"] if target is not None else np.zeros(4)
            self.buffer.add_frame(
                timestamp=time.time(),
                wrist_pos=vr_frame["wrist_pos"],
                wrist_quat_wxyz=vr_frame["wrist_quat_wxyz"],
                landmarks=vr_frame["landmarks"],
                sequence_id=vr_frame["sequence_id"],
                eef_pos=eef_pos,
                eef_quat_wxyz=eef_quat_wxyz,
            )
            self.frame_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Quest VR Teleop Recorder")
    parser.add_argument(
        "--record", "-r",
        action="store_true",
        help="Auto-start recording on first frame",
    )
    args = parser.parse_args()

    recorder = QuestTeleopRecorder()
    recorder.run(start_mode="record" if args.record else "idle")


if __name__ == "__main__":
    main()
