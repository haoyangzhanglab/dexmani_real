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

_QUAT_NORM_EPS = 1e-12


def xyzw_to_wxyz(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float, float]:
    """Convert xyzw quaternion to wxyz."""
    return (qw, qx, qy, qz)


def _finite_vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Validate one converted SDK payload before cross-process publication."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


def _normalized_wxyz(value: object, name: str) -> np.ndarray:
    quat = _finite_vector(value, (4,), name)
    norm = float(np.linalg.norm(quat))
    if norm < _QUAT_NORM_EPS:
        raise ValueError(f"{name} norm is too small")
    return quat / norm


@dataclass
class VRReceiverConfig:
    """Configuration for VR receiver — defaults from vr singleton."""

    transport: str = field(default_factory=lambda: vr.transport)
    host: str = field(default_factory=lambda: vr.host)
    port: int = field(default_factory=lambda: vr.port)
    hand_side: str = field(default_factory=lambda: vr.hand_side)  # "both" needed for HeadFrame

    @classmethod
    def from_runtime(cls, runtime: object) -> "VRReceiverConfig":
        cfg = getattr(runtime, "vr")
        return cls(transport=str(cfg.transport), host=str(cfg.host), port=int(cfg.port), hand_side=str(cfg.hand_side))


def vr_loop(shared, config: VRReceiverConfig | None = None) -> None:
    """VR process entry point — writes directly to SharedStorage.vr_ring."""

    cfg = config or VRReceiverConfig()

    from dexmani_real.runtime.status import ComponentPhase, FaultCode
    from dexmani_real.shm.shared_storage import new_frame, publish_component_status, vr_frame_dtype

    publish_component_status(shared, "vr", ComponentPhase.LOADING)

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

        client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode(cfg.transport),
                host=cfg.host,
                port=cfg.port,
                timeout_s=1.0,
                output=StreamOutput.FRAMES,
                hand_filter=HandFilter(cfg.hand_side),
                error_policy=0,
                include_wall_time=True,
            )
        )
    except ImportError as e:
        logger.error("vr_loop: SDK import failed: %s", e)
        publish_component_status(
            shared, "vr", ComponentPhase.FAULT, fault_code=FaultCode.STARTUP_FAILED, detail="VR SDK import failed"
        )
        return
    except Exception as e:
        logger.error("vr_loop: connect failed: %s", e)
        publish_component_status(
            shared, "vr", ComponentPhase.FAULT, fault_code=FaultCode.STARTUP_FAILED, detail="VR connect failed"
        )
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
            publish_component_status(shared, "vr", ComponentPhase.READY)
            logger.info("vr_loop: ready (first event received)")

        if isinstance(event, HeadFrame):
            try:
                head_flu_pos = unity_left_to_flu_position(event.head.x, event.head.y, event.head.z)
                head_flu_quat = unity_left_to_flu_rotation(event.head.qx, event.head.qy, event.head.qz, event.head.qw)
                _latest_head_pos = _finite_vector(head_flu_pos, (3,), "head_pos")
                _latest_head_quat_wxyz = _normalized_wxyz(xyzw_to_wxyz(*head_flu_quat), "head_quat_wxyz")
            except (ValueError, TypeError, AttributeError):
                logger.warning("vr_loop: invalid head pose rejected", exc_info=True)
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

            wrist_pos = _finite_vector(flu_pos, (3,), "wrist_pos")
            wrist_quat_wxyz = _normalized_wxyz(xyzw_to_wxyz(*flu_quat), "wrist_quat_wxyz")
            landmarks = _finite_vector(
                [unity_left_to_flu_position(*p) for p in event.landmarks.points],
                (21, 3),
                "landmarks",
            )

            frame = new_frame(dtype)
            frame["wrist_pos"][0] = wrist_pos
            frame["wrist_quat_wxyz"][0] = wrist_quat_wxyz
            frame["landmarks"][0] = landmarks
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

    publish_component_status(shared, "vr", ComponentPhase.STOPPED)
    logger.info("vr_loop: exited")
