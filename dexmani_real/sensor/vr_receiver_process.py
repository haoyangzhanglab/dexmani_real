"""VR receiver process — crash-isolated HTS SDK wrapper.

Primary entry point: ``vr_loop(shared)`` — mp.Process target, writes directly
to SharedStorage.vr_ring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.config.defaults import vr
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    """Convert xyzw quaternion to wxyz."""
    return (qw, qx, qy, qz)


@dataclass
class VRReceiverConfig:
    """Configuration for VR receiver — defaults from vr singleton."""

    transport: str = field(default_factory=lambda: vr.transport)
    host: str = field(default_factory=lambda: vr.host)
    port: int = field(default_factory=lambda: vr.port)
    hand_side: str = field(default_factory=lambda: vr.hand_side)  # "both" needed for HeadFrame



def vr_loop(shared, config: VRReceiverConfig | None = None) -> None:
    """VR process entry point — writes directly to SharedStorage.vr_ring.

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

    # vr_ready is deferred to the first event (HeadFrame or HandFrame).
    # TCP connect alone is not enough — data must actually be flowing before
    # Main considers VR "ready", otherwise the 5 s heartbeat timeout fires
    # before the operator has time to put on the headset.

    dtype = vr_frame_dtype()

    for event in client.iter_events():
        if not shared.is_running.value:
            break

        # Heartbeat on every event — proves VR process is alive + receiving data.
        # Written *before* vr_ready so the supervisor sees a fresh heartbeat
        # immediately (avoids false FAULT on the first check).
        shared.vr_heartbeat_s.value = time.monotonic()

        if not shared.vr_ready.is_set():
            shared.vr_ready.set()
            logger.info("vr_loop: ready (first event received)")

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
            frame["source_ts_ns"][0] = np.uint64(event.source_ts_ns or 0)
            frame["sequence_id"][0] = np.uint64(event.sequence_id)
            frame["source_frame_seq"][0] = np.uint64(event.source_frame_seq or 0)
            frame["local_recv_ns"][0] = np.uint64(time.monotonic_ns())
            frame["side"][0] = np.int32(0 if "right" in _side_str else -1)

            shared.vr_ring.write(frame)

        except (ValueError, TypeError, AttributeError):
            logger.warning("vr_loop: frame conversion error", exc_info=True)
            continue

    logger.info("vr_loop: exited")
