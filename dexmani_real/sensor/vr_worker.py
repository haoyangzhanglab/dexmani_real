"""VR receiver worker — crash-isolated HTS SDK wrapper.

Primary entry point: ``run_vr_worker(shared, config)`` — mp.Process target, writes directly
to RuntimeChannels.vr_ring.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.config.hardware import VRParams
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


def run_vr_worker(shared, config: VRParams) -> None:
    """VR process entry point — writes directly to RuntimeChannels.vr_ring."""

    cfg = config

    from dexmani_real.ipc.schema import VR_FRAME_DTYPE

    logger.debug("run_vr_worker: LOADING")

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
        logger.error("run_vr_worker: SDK import failed: %s", e)
        raise
    except Exception as e:
        logger.error("run_vr_worker: connect failed: %s", e)
        raise

    logger.info("run_vr_worker: connected to HTS port=%d", cfg.port)

    _latest_head_pos = np.full(3, np.nan)
    _latest_head_quat_wxyz = np.full(4, np.nan)
    _latest_head_sequence_id = 0
    _latest_head_recv_ts_ns = 0

    # vr_ready guarantees that the ring already contains one validated
    # right-hand frame; consumers may read it immediately after readiness.

    for event in client.iter_events():
        if not shared.is_running.value:
            break

        if isinstance(event, HeadFrame):
            try:
                head_flu_pos = unity_left_to_flu_position(event.head.x, event.head.y, event.head.z)
                head_flu_quat = unity_left_to_flu_rotation(
                    event.head.qx, event.head.qy, event.head.qz, event.head.qw
                )
                head_pos = _finite_vector(head_flu_pos, (3,), "head_pos")
                head_quat_wxyz = _normalized_wxyz(xyzw_to_wxyz(*head_flu_quat), "head_quat_wxyz")
                head_sequence_id = int(event.sequence_id)
                head_recv_ts_ns = int(event.recv_ts_ns)
                if head_sequence_id < 0 or head_recv_ts_ns <= 0:
                    raise ValueError("head sequence/timestamp must be non-negative and nonzero")
                _latest_head_pos = head_pos
                _latest_head_quat_wxyz = head_quat_wxyz
                _latest_head_sequence_id = head_sequence_id
                _latest_head_recv_ts_ns = head_recv_ts_ns
            except (ValueError, TypeError, AttributeError):
                logger.warning("run_vr_worker: invalid head pose rejected", exc_info=True)
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

            frame = np.zeros(1, dtype=VR_FRAME_DTYPE)
            frame["wrist_pos"][0] = wrist_pos
            frame["wrist_quat_wxyz"][0] = wrist_quat_wxyz
            frame["landmarks"][0] = landmarks
            frame["head_pos"][0] = _latest_head_pos.copy()
            frame["head_quat_wxyz"][0] = _latest_head_quat_wxyz.copy()
            frame["head_sequence_id"][0] = np.uint64(_latest_head_sequence_id)
            frame["head_recv_ts_ns"][0] = np.uint64(_latest_head_recv_ts_ns)
            frame["recv_ts_ns"][0] = np.uint64(event.recv_ts_ns)
            frame["source_ts_ns"][0] = np.uint64(event.source_ts_ns or 0)
            frame["sequence_id"][0] = np.uint64(event.sequence_id)
            frame["source_frame_seq"][0] = np.uint64(event.source_frame_seq or 0)
            frame["side"][0] = np.int32(0 if "right" in _side_str else -1)

            shared.vr_ring.write(frame)
            if not shared.vr_ready.is_set():
                shared.vr_ready.set()
                logger.debug("run_vr_worker: READY")
                logger.info("run_vr_worker: ready (first valid right-hand frame received)")

        except (ValueError, TypeError, AttributeError):
            logger.warning("run_vr_worker: frame conversion error", exc_info=True)
            continue

    logger.debug("run_vr_worker: STOPPED")
    logger.info("run_vr_worker: exited")
