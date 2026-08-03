"""VR receiver process — crash-isolated HTS SDK wrapper.

Primary entry point: ``vr_loop(shared)`` — mp.Process target, writes directly
to SharedStorage.vr_ring.

``VRReceiverProcess`` (class) is a legacy implementation retained only for
deprecated entry points (vr_teleop_arm_only*.py).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    """Convert xyzw quaternion to wxyz."""
    return (qw, qx, qy, qz)


@dataclass
class VRReceiverConfig:
    """Configuration for VR receiver."""

    transport: str = "tcp_server"
    host: str = "0.0.0.0"
    port: int = 8000
    hand_side: str = "both"  # "both" needed for HeadFrame (heading calibration)


# ═══════════════════════════════════════════════════════════════════
# Legacy: VRReceiverProcess (old entry point compatibility)
# ═══════════════════════════════════════════════════════════════════


class VRReceiverProcess:
    """Crash-isolated VR frame receiver.

    Connects to Meta Quest HTS hand-tracking app, writes frames to
    SharedMemoryFrameManager for consumption by the main process.
    """

    def __init__(self, config: VRReceiverConfig | None = None) -> None:
        self.config = config or VRReceiverConfig()
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()
        self._crashed = mp.Event()

        # Cross-process stats counters (written by child, read by parent)
        self._received_frames = mp.Value("i", 0)
        self._ignored_events = mp.Value("i", 0)
        self._error_frames = mp.Value("i", 0)

        from dexmani_real.shm.frame_manager import SharedMemoryFrameManager
        self.shm = SharedMemoryFrameManager(vr_maxlen=3, create=True)

    # ── Lifecycle ──

    def start(self) -> bool:
        if self._process is not None and self._process.is_alive():
            logger.warning("VRReceiverProcess already running.")
            return False

        self._stop_event.clear()
        self._crashed.clear()

        self._process = mp.Process(target=self._run, name="vr-receiver", daemon=True)
        self._process.start()
        logger.info("VRReceiverProcess started (port=%d)", self.config.port)
        return True

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("VRReceiverProcess did not exit — terminating.")
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = None
        self.shm.close()
        self.shm.unlink()
        logger.info("VRReceiverProcess stopped.")

    # ── Health ──

    @property
    def crashed(self) -> bool:
        if self._process is not None and not self._process.is_alive():
            self._crashed.set()
        return self._crashed.is_set()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ── Frame access ──

    def read_latest(self) -> dict | None:
        return self.shm.read_latest_vr()

    def read_latest_with_age(self) -> tuple[dict | None, float]:
        return self.shm.read_latest_vr_with_age()

    def frame_age_s(self) -> float:
        return self.shm.vr_age_s()

    def get_stats(self) -> dict:
        """Return diagnostic stats dict (cross-process safe)."""
        return {
            "received_frames": self._received_frames.value,
            "ignored_events": self._ignored_events.value,
            "error_frames": self._error_frames.value,
            "sdk_lines_received": 0,  # not tracked in current impl
            "sdk_parse_errors": 0,
            "sdk_dropped_lines": 0,
            "running": self.running,
            "crashed": self.crashed,
        }

    # ── Internal ──

    def _run(self) -> None:
        """Main VR receive loop (runs in child process)."""
        try:
            from hand_tracking_sdk import (
                HandFilter, HandFrame, HeadFrame, HTSClient, HTSClientConfig,
                StreamOutput, TransportMode,
                unity_left_to_flu_position, unity_left_to_flu_rotation,
            )

            client = HTSClient(HTSClientConfig(
                transport_mode=TransportMode(self.config.transport),
                host=self.config.host, port=self.config.port,
                timeout_s=1.0, output=StreamOutput.FRAMES,
                hand_filter=HandFilter(self.config.hand_side),
                error_policy=0, include_wall_time=True,
            ))

            logger.info("VRReceiverProcess: connected to HTS port=%d", self.config.port)

            _latest_head_pos = np.zeros(3, dtype=np.float64)
            _latest_head_quat_wxyz = np.zeros(4, dtype=np.float64)
            _latest_head_quat_wxyz[0] = 1.0

            for event in client.iter_events():
                if self._stop_event.is_set():
                    break

                # Update parent-visible stats (mp.Value, cross-process safe)
                self._ignored_events.value += 1

                if isinstance(event, HeadFrame):
                    head_flu_pos = unity_left_to_flu_position(event.head.x, event.head.y, event.head.z)
                    head_flu_quat = unity_left_to_flu_rotation(event.head.qx, event.head.qy, event.head.qz, event.head.qw)
                    _latest_head_pos = np.asarray(head_flu_pos, dtype=np.float64)
                    _latest_head_quat_wxyz = np.asarray(xyzw_to_wxyz(*head_flu_quat), dtype=np.float64)
                    continue

                if not isinstance(event, HandFrame):
                    continue

                try:
                    wrist = event.wrist
                    _side_str = str(event.side.value).lower()
                    if "left" in _side_str:
                        continue

                    flu_pos = unity_left_to_flu_position(wrist.x, wrist.y, wrist.z)
                    flu_quat = unity_left_to_flu_rotation(wrist.qx, wrist.qy, wrist.qz, wrist.qw)

                    frame_dict = {
                        "side": 0 if "right" in _side_str else -1,
                        "wrist_pos": np.asarray(flu_pos, dtype=np.float64),
                        "wrist_quat_wxyz": np.asarray(xyzw_to_wxyz(*flu_quat), dtype=np.float64),
                        "landmarks": np.asarray(
                            [unity_left_to_flu_position(*p) for p in event.landmarks.points],
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
                    self.shm.write_vr_frame(frame_dict)
                    # Ignored count was already incremented at event start —
                    # this is a valid frame, so transfer the count to received.
                    self._ignored_events.value -= 1
                    self._received_frames.value += 1

                except (ValueError, TypeError, AttributeError):
                    logger.warning("VR: frame conversion error", exc_info=True)

            logger.info("VRReceiverProcess: receive loop exited.")

        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            logger.exception("VRReceiverProcess crashed: %s", exc)
            self._crashed.set()
        except ImportError as exc:
            logger.error("VRReceiverProcess: hand_tracking_sdk not available: %s", exc)
            self._crashed.set()


# ═══════════════════════════════════════════════════════════════════
# New architecture: vr_loop (mp.Process target)
# ═══════════════════════════════════════════════════════════════════


def vr_loop(shared, config: VRReceiverConfig | None = None) -> None:
    """VR process entry point — writes directly to SharedStorage.vr_ring.

    No SharedMemoryFrameManager, no stats counters, no monitor thread.
    """
    cfg = config or VRReceiverConfig()

    from dexmani_real.shm.shared_storage import new_frame, vr_frame_dtype

    try:
        from hand_tracking_sdk import (
            HandFilter, HandFrame, HeadFrame, HTSClient, HTSClientConfig,
            StreamOutput, TransportMode,
            unity_left_to_flu_position, unity_left_to_flu_rotation,
        )

        client = HTSClient(HTSClientConfig(
            transport_mode=TransportMode(cfg.transport),
            host=cfg.host, port=cfg.port,
            timeout_s=1.0, output=StreamOutput.FRAMES,
            hand_filter=HandFilter(cfg.hand_side),
            error_policy=0, include_wall_time=True,
        ))
    except ImportError as e:
        logger.error("vr_loop: SDK import failed: %s", e)
        return
    except Exception as e:
        logger.error("vr_loop: connect failed: %s", e)
        return

    logger.info("vr_loop: connected to HTS port=%d", cfg.port)

    _latest_head_pos = np.zeros(3, dtype=np.float64)
    _latest_head_quat_wxyz = np.zeros(4, dtype=np.float64)
    _latest_head_quat_wxyz[0] = 1.0

    # Write heartbeat BEFORE ready signal — prevents false FAULT on startup
    # (Main's supervisor checks heartbeats immediately after ready events).
    shared.vr_heartbeat_s.value = time.monotonic()
    shared.vr_ready.set()
    logger.info("vr_loop: ready")

    dtype = vr_frame_dtype()

    for event in client.iter_events():
        if not shared.is_running.value:
            break

        # Heartbeat — written on every event to prove VR process is alive + receiving data
        shared.vr_heartbeat_s.value = time.monotonic()

        if isinstance(event, HeadFrame):
            head_flu_pos = unity_left_to_flu_position(event.head.x, event.head.y, event.head.z)
            head_flu_quat = unity_left_to_flu_rotation(event.head.qx, event.head.qy, event.head.qz, event.head.qw)
            _latest_head_pos = np.asarray(head_flu_pos, dtype=np.float64)
            _latest_head_quat_wxyz = np.asarray(xyzw_to_wxyz(*head_flu_quat), dtype=np.float64)
            continue

        if not isinstance(event, HandFrame):
            continue

        try:
            wrist = event.wrist
            _side_str = str(event.side.value).lower()
            if "left" in _side_str:
                continue

            flu_pos = unity_left_to_flu_position(wrist.x, wrist.y, wrist.z)
            flu_quat = unity_left_to_flu_rotation(wrist.qx, wrist.qy, wrist.qz, wrist.qw)

            frame = new_frame(dtype)
            frame["wrist_pos"][0] = np.asarray(flu_pos, dtype=np.float64)
            frame["wrist_quat_wxyz"][0] = np.asarray(xyzw_to_wxyz(*flu_quat), dtype=np.float64)
            frame["landmarks"][0] = np.asarray(
                [unity_left_to_flu_position(*p) for p in event.landmarks.points],
                dtype=np.float64,
            ).reshape(21, 3)
            frame["head_pos"][0] = _latest_head_pos.copy()
            frame["head_quat_wxyz"][0] = _latest_head_quat_wxyz.copy()
            frame["recv_ts_ns"][0] = np.uint64(event.recv_ts_ns)
            frame["source_ts_ns"][0] = np.uint64(event.source_ts_ns)
            frame["sequence_id"][0] = np.uint64(event.sequence_id)
            frame["source_frame_seq"][0] = np.uint64(event.source_frame_seq)
            frame["local_recv_ns"][0] = np.uint64(time.monotonic_ns())
            frame["side"][0] = np.int32(0 if "right" in _side_str else -1)

            shared.vr_ring.write(frame)

        except (ValueError, TypeError, AttributeError):
            logger.warning("vr_loop: frame conversion error", exc_info=True)
            continue

    logger.info("vr_loop: exited")
