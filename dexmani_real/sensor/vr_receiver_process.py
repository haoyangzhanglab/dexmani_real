"""VR receiver background process — crash-isolated HTS SDK wrapper.

Runs QuestHandTracker's HTS receive loop in a dedicated multiprocessing.Process
to eliminate GIL contention with the control loop and isolate HTS SDK crashes.

Outputs VR frames to SharedMemoryRingBuffer (zero-copy) instead of the
thread-local dict accessed through threading.Lock.

Ref: ManiUniCon process isolation pattern.
     BunnyVisionPro separate VR process design.

Architecture:
    ┌───────────────────────────┐       SharedMemory        ┌──────────────────┐
    │ VRReceiverProcess        │ ──── RingBuffer(FILO) ───► │ TeleopController │
    │ (独立 mp.Process)        │                             │ (主进程, 50Hz)   │
    │                           │                             │                  │
    │ HTS SDK iter_events()    │                             │ read_latest_vr() │
    │ → shm.write()            │                             │                  │
    └───────────────────────────┘                             └──────────────────┘

Usage:
    proc = VRReceiverProcess(transport="tcp_server", port=8000)
    proc.start()
    # In controller loop:
    frame = proc.shm.read_latest_vr()
    proc.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class VRReceiverConfig:
    """Configuration for VRReceiverProcess."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000
    hand_side: str = "both"  # "both" needed for HeadFrame (heading calibration)
    strict: bool = False


class VRReceiverProcess:
    """Crash-isolated VR frame receiver running in a separate process.

    Connects to the Meta Quest HTS hand-tracking app and writes every
    received frame to a SharedMemoryRingBuffer for the main control
    process to consume.

    Attributes:
        shm: SharedMemoryFrameManager (accessible from both processes).
    """

    def __init__(self, config: VRReceiverConfig | None = None) -> None:
        self.config = config or VRReceiverConfig()
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()
        self._crashed = mp.Event()

        # Shared memory frame manager (created in current process,
        # attached in child process via shared name).
        from dexmani_real.shm.frame_manager import SharedMemoryFrameManager

        self.shm = SharedMemoryFrameManager(
            camera_hw=(480, 640),
            n_cameras=0,  # VR-only; no cameras
            vr_maxlen=3,
            create=True,
        )

        # Stats (accessible from main process)
        self._received_count = mp.Value("Q", 0, lock=False)
        self._error_count = mp.Value("Q", 0, lock=False)
        self._ignored_count = mp.Value("Q", 0, lock=False)
        self._sdk_lines_received = mp.Value("Q", 0, lock=False)
        self._sdk_parse_errors = mp.Value("Q", 0, lock=False)
        self._sdk_dropped_lines = mp.Value("Q", 0, lock=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the VR receiver process. Returns True on success."""
        if self._process is not None and self._process.is_alive():
            logger.warning("VRReceiverProcess already running.")
            return False

        self._stop_event.clear()
        self._crashed.clear()
        self._received_count.value = 0
        self._error_count.value = 0

        self._process = mp.Process(
            target=self._run,
            name="vr-receiver",
            daemon=True,
        )
        self._process.start()
        logger.info(
            "VRReceiverProcess started (transport=%s, port=%d)",
            self.config.transport, self.config.port,
        )
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop and wait for process exit."""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning(
                    "VRReceiverProcess did not exit within %.1fs, terminating.", timeout
                )
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = None
        self.shm.close()
        self.shm.unlink()
        logger.info("VRReceiverProcess stopped.")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def crashed(self) -> bool:
        if self._process is not None and not self._process.is_alive():
            self._crashed.set()
        return self._crashed.is_set()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def received_frames(self) -> int:
        return int(self._received_count.value)

    @property
    def error_frames(self) -> int:
        return int(self._error_count.value)

    @property
    def ignored_events(self) -> int:
        return int(self._ignored_count.value)

    @property
    def sdk_lines_received(self) -> int:
        return int(self._sdk_lines_received.value)

    @property
    def sdk_parse_errors(self) -> int:
        return int(self._sdk_parse_errors.value)

    @property
    def sdk_dropped_lines(self) -> int:
        return int(self._sdk_dropped_lines.value)

    def get_stats(self) -> dict:
        """Diagnostic stats (mirrors QuestHandTracker.get_status)."""
        return {
            "running": self.running,
            "crashed": self.crashed,
            "received_frames": self.received_frames,
            "error_frames": self.error_frames,
            "ignored_events": self.ignored_events,
            "sdk_lines_received": self.sdk_lines_received,
            "sdk_parse_errors": self.sdk_parse_errors,
            "sdk_dropped_lines": self.sdk_dropped_lines,
        }

    # ------------------------------------------------------------------
    # Frame access (from main process)
    # ------------------------------------------------------------------

    def read_latest(self) -> dict | None:
        """Non-blocking read of the latest VR frame from shared memory."""
        return self.shm.read_latest_vr()

    def read_latest_with_age(self) -> tuple[dict | None, float]:
        """Read latest VR frame with age in seconds."""
        return self.shm.read_latest_vr_with_age()

    def frame_age_s(self) -> float:
        return self.shm.vr_age_s()

    # ------------------------------------------------------------------
    # Internal (runs in child process)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main VR receive loop (runs in child process)."""
        try:
            from hand_tracking_sdk import (
                HandFilter,
                HandFrame,
                HeadFrame,
                HTSClient,
                HTSClientConfig,
                StreamOutput,
                TransportMode,
                unity_left_to_flu_position,
                unity_left_to_flu_rotation,
            )

            # Convert (qx,qy,qz,qw) → (qw,qx,qy,qz)
            def xyzw_to_wxyz(qx, qy, qz, qw):
                return (qw, qx, qy, qz)

            client = HTSClient(
                HTSClientConfig(
                    transport_mode=TransportMode(self.config.transport),
                    host=self.config.host,
                    port=self.config.port,
                    timeout_s=1.0,
                    output=StreamOutput.FRAMES,
                    hand_filter=HandFilter(self.config.hand_side),
                    error_policy=0,  # TOLERANT
                    include_wall_time=True,
                )
            )

            logger.info(
                "VRReceiverProcess: connected to HTS @ %s://%s:%d",
                self.config.transport, self.config.host, self.config.port,
            )

            # ── Monitor thread: periodically log HTS SDK stats ──
            _monitor_stop = threading.Event()

            def _monitor_loop():
                """Log HTS client stats every 10s for diagnosing no-frame issues."""
                while not _monitor_stop.wait(10.0):
                    try:
                        stats = client.get_stats()
                        self._sdk_lines_received.value = getattr(stats, "lines_received", -1)
                        self._sdk_parse_errors.value = getattr(stats, "parse_errors", -1)
                        self._sdk_dropped_lines.value = getattr(stats, "dropped_lines", -1)
                        logger.info(
                            "VRReceiverProcess diag: sdk_lines=%d parse_err=%d dropped=%d "
                            "frames=%d ignored=%d errors=%d",
                            self._sdk_lines_received.value,
                            self._sdk_parse_errors.value,
                            self._sdk_dropped_lines.value,
                            self._received_count.value,
                            self._ignored_count.value,
                            self._error_count.value,
                        )
                    except Exception:
                        logger.debug("VRReceiverProcess: stat polling failed (client may not support get_stats)")

            monitor = threading.Thread(target=_monitor_loop, daemon=True, name="vr-monitor")
            monitor.start()

            # ── Event loop ──
            _first_event = True
            _logged_event_types: set[str] = set()

            # Head pose cache — updated from HeadFrame events, bundled into each VR frame
            _latest_head_pos = np.zeros(3, dtype=np.float64)
            _latest_head_quat_wxyz = np.zeros(4, dtype=np.float64)
            _latest_head_quat_wxyz[0] = 1.0  # identity quaternion

            for event in client.iter_events():
                if self._stop_event.is_set():
                    break

                if _first_event:
                    _first_event = False
                    logger.info(
                        "VRReceiverProcess: first event received (type=%s), "
                        "iter_events is yielding data",
                        type(event).__name__,
                    )

                # ── HeadFrame: cache latest head pose for heading calibration ──
                if isinstance(event, HeadFrame):
                    head_flu_pos = unity_left_to_flu_position(
                        event.head.x, event.head.y, event.head.z,
                    )
                    head_flu_quat = unity_left_to_flu_rotation(
                        event.head.qx, event.head.qy, event.head.qz, event.head.qw,
                    )
                    _latest_head_pos = np.asarray(head_flu_pos, dtype=np.float64)
                    _latest_head_quat_wxyz = np.asarray(
                        xyzw_to_wxyz(*head_flu_quat), dtype=np.float64,
                    )
                    continue

                if not isinstance(event, HandFrame):
                    self._ignored_count.value += 1
                    # Log the first occurrence of each non-HandFrame event type
                    etype = type(event).__name__
                    if etype not in _logged_event_types:
                        _logged_event_types.add(etype)
                        logger.info(
                            "VRReceiverProcess: non-HandFrame event type=%s (x%d total)",
                            etype, self._ignored_count.value,
                        )
                    continue

                try:
                    # Extract geometry
                    wrist = event.wrist

                    # Skip LEFT hand frames (we only want RIGHT + HEAD)
                    _side_str = str(event.side.value).lower()
                    if "left" in _side_str:
                        continue

                    pos = (wrist.x, wrist.y, wrist.z)
                    quat_xyzw = (wrist.qx, wrist.qy, wrist.qz, wrist.qw)
                    landmarks = event.landmarks.points

                    # Convert to FLU
                    flu_pos = unity_left_to_flu_position(*pos)
                    flu_quat = unity_left_to_flu_rotation(*quat_xyzw)

                    # Map side string→int for SHM dtype (int32)
                    _side_str = str(event.side.value).lower()
                    _side_int = 0 if "right" in _side_str else (1 if "left" in _side_str else -1)

                    frame_dict = {
                        "side": _side_int,
                        "wrist_pos": np.asarray(flu_pos, dtype=np.float64),
                        "wrist_quat_wxyz": np.asarray(
                            xyzw_to_wxyz(*flu_quat), dtype=np.float64
                        ),
                        "landmarks": np.asarray(
                            [unity_left_to_flu_position(*p) for p in landmarks],
                            dtype=np.float64,
                        ).reshape(21, 3),
                        "head_pos": _latest_head_pos.copy(),
                        "head_quat_wxyz": _latest_head_quat_wxyz.copy(),
                        "recv_ts_ns": event.recv_ts_ns,
                        "source_ts_ns": event.source_ts_ns,
                        "sequence_id": event.sequence_id,
                        "source_frame_seq": event.source_frame_seq,
                        "coordinate_frame": "flu",
                        "local_recv_ns": time.monotonic_ns(),
                    }

                    # Write to shared memory
                    self.shm.write_vr_frame(frame_dict)
                    self._received_count.value += 1

                except (ValueError, TypeError, AttributeError) as exc:
                    self._error_count.value += 1
                    # Log the first 3 errors with full traceback for diagnosis
                    if self._error_count.value <= 3:
                        logger.warning(
                            "VRReceiverProcess: frame conversion error #%d: %s: %s\n%s",
                            self._error_count.value, type(exc).__name__, exc,
                            traceback.format_exc(),
                        )
                    if self.config.strict:
                        raise
                    continue

            _monitor_stop.set()
            logger.info("VRReceiverProcess: receive loop exited cleanly.")

        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            logger.exception("VRReceiverProcess crashed: %s", exc)
            self._crashed.set()
        except ImportError as exc:
            logger.error("VRReceiverProcess: hand_tracking_sdk not available: %s", exc)
            self._crashed.set()
