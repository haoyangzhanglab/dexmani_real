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
import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.log import get_logger

logger = get_logger(__name__)


@dataclass
class VRReceiverConfig:
    """Configuration for VRReceiverProcess."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000
    hand_side: str = "right"
    output_frame: str = "flu"
    max_frame_age_s: float = 0.20
    strict: bool = False
    verbose: bool = False


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

            for event in client.iter_events():
                if self._stop_event.is_set():
                    break

                if not isinstance(event, HandFrame):
                    continue

                try:
                    # Extract geometry
                    wrist = event.wrist
                    pos = (wrist.x, wrist.y, wrist.z)
                    quat_xyzw = (wrist.qx, wrist.qy, wrist.qz, wrist.qw)
                    landmarks = event.landmarks.points

                    # Convert to FLU
                    flu_pos = unity_left_to_flu_position(*pos)
                    flu_quat = unity_left_to_flu_rotation(*quat_xyzw)

                    frame_dict = {
                        "side": event.side.value,
                        "wrist_pos": np.asarray(flu_pos, dtype=np.float64),
                        "wrist_quat_wxyz": np.asarray(
                            xyzw_to_wxyz(*flu_quat), dtype=np.float64
                        ),
                        "landmarks": np.asarray(
                            [unity_left_to_flu_position(*p) for p in landmarks],
                            dtype=np.float64,
                        ).reshape(21, 3),
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
                    if self.config.strict:
                        raise
                    continue

            logger.info("VRReceiverProcess: receive loop exited cleanly.")

        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            logger.exception("VRReceiverProcess crashed: %s", exc)
            self._crashed.set()
        except ImportError as exc:
            logger.error("VRReceiverProcess: hand_tracking_sdk not available: %s", exc)
            self._crashed.set()
