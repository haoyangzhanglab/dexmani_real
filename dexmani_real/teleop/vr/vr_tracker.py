"""Meta Quest hand-tracking receiver based on hand-tracking-sdk."""

from __future__ import annotations

__all__ = ["QuestHandTracker"]

import threading
import time
from typing import Any

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.planning.pose_utils import xyzw_to_wxyz
from hand_tracking_sdk import (
    ErrorPolicy,
    HandFilter,
    HandFrame,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
    unity_left_to_flu_position,
    unity_left_to_flu_rotation,
    unity_left_to_rfu_position,
    unity_left_to_rfu_rotation,
)

logger = get_logger(__name__)

# ── Magic numbers ──
_CONNECT_DEADLINE_S = 3.0
_THREAD_JOIN_CLIENT_TIMEOUT_S = 2.0
_THREAD_JOIN_SERVER_TIMEOUT_S = 6.0


class QuestHandTracker:
    """Receive wrist pose and 21 hand landmarks from HTS."""

    def __init__(
        self,
        transport: str = "tcp_server",
        host: str = "0.0.0.0",
        port: int = 8000,
        hand_side: str = "right",
        output_frame: str = "flu",
        max_frame_age_s: float = 0.20,
        strict: bool = False,
        verbose: bool = False,
    ) -> None:
        self.transport = transport
        self.host = host
        self.port = port
        self.hand_side = hand_side
        self.output_frame = output_frame
        self.max_frame_age_s = max_frame_age_s
        self.strict = strict
        self.verbose = verbose

        self.client: HTSClient | None = None
        self.thread: threading.Thread | None = None
        self.started = False
        self.running = False

        self.latest_frame: dict[str, Any] | None = None
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.last_read_key = None

        self.received_frames = 0
        self.ignored_events = 0
        self.malformed_frames = 0
        self.last_error: str | None = None

    def connect(self) -> bool:
        if self.running:
            return True

        self.client = self.create_client()
        self.running = True
        self.last_error = None

        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

        if self.verbose:
            logger.info("[QuestHandTracker] %s://%s:%s", self.transport, self.host, self.port)

        # Verify the receive thread is actually alive.
        # In tcp_client mode we also wait for the first event to confirm the
        # server is reachable; in tcp_server/udp mode the client may connect
        # later, so we only verify the thread didn't crash immediately.
        deadline = time.monotonic() + _CONNECT_DEADLINE_S
        while time.monotonic() < deadline:
            if not self.running:
                self.started = False
                return False
            if self.received_frames > 0 or self.ignored_events > 0:
                self.started = True
                return True
            # In server modes the first event may take arbitrarily long
            # (waiting for Quest to connect).  Only require it for client mode.
            if self._is_server_mode():
                if time.monotonic() > deadline:
                    break
            time.sleep(0.05)

        if self._is_server_mode():
            # Server mode: thread is alive and listening.  Client connects later.
            self.started = True
            return True

        # Client mode timed out — server unreachable.
        self.running = False
        self.event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=_THREAD_JOIN_CLIENT_TIMEOUT_S)
        self.started = False
        self.last_error = (
            f"No events received within {_CONNECT_DEADLINE_S}s ({self.transport}://{self.host}:{self.port}). "
            "Check: adb reverse (USB) or HTS app mode (TCP Server vs Client)."
        )
        return False

    def _is_server_mode(self) -> bool:
        return self.transport in ("tcp_server", "udp")

    def disconnect(self) -> None:
        self.running = False
        self.event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=_THREAD_JOIN_SERVER_TIMEOUT_S)

        self.started = False
        self.thread = None
        self.client = None

    def is_connected(self) -> bool:
        return self.started and self.running and self.received_frames > 0

    def get_latest(self, max_age_s: float | None = None) -> dict[str, Any] | None:
        max_age_s = self.max_frame_age_s if max_age_s is None else max_age_s

        with self.lock:
            if self.latest_frame is None:
                return None
            if self.frame_age(self.latest_frame) > max_age_s:
                return None
            return self.copy_frame(self.latest_frame)

    def read(self, timeout_s: float = 1.0) -> dict[str, Any]:
        start = time.monotonic()

        while True:
            frame = self.get_latest()
            if frame is not None:
                key = (frame["sequence_id"], frame["recv_ts_ns"])
                if key != self.last_read_key:
                    self.last_read_key = key
                    return frame

            remain = timeout_s - (time.monotonic() - start)
            if remain <= 0:
                raise TimeoutError("No new VR frame received.")

            self.event.wait(timeout=remain)
            self.event.clear()

    def clear(self) -> None:
        with self.lock:
            self.latest_frame = None
        self.last_read_key = None
        self.event.set()

    def get_status(self) -> dict[str, Any]:
        age = None
        sequence_id = None

        with self.lock:
            if self.latest_frame is not None:
                age = self.frame_age(self.latest_frame)
                sequence_id = self.latest_frame["sequence_id"]

        stats = self.client.get_stats() if self.client is not None else None

        return {
            "started": self.started,
            "running": self.running,
            "transport": self.transport,
            "host": self.host,
            "port": self.port,
            "hand_side": self.hand_side,
            "output_frame": self.output_frame,
            "received_frames": self.received_frames,
            "ignored_events": self.ignored_events,
            "malformed_frames": self.malformed_frames,
            "sdk_lines_received": getattr(stats, "lines_received", None),
            "sdk_parse_errors": getattr(stats, "parse_errors", None),
            "sdk_dropped_lines": getattr(stats, "dropped_lines", None),
            "sequence_id": sequence_id,
            "frame_age_s": age,
            "last_error": self.last_error,
        }

    def create_client(self) -> HTSClient:
        return HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode(self.transport),
                host=self.host,
                port=self.port,
                timeout_s=1.0,
                output=StreamOutput.FRAMES,
                hand_filter=HandFilter(self.hand_side),
                error_policy=ErrorPolicy.STRICT if self.strict else ErrorPolicy.TOLERANT,
                include_wall_time=True,
            )
        )

    def _receive_loop(self) -> None:
        assert self.client is not None

        try:
            for event in self.client.iter_events():
                if not self.running:
                    break

                if not isinstance(event, HandFrame):
                    self.ignored_events += 1
                    continue

                try:
                    frame = self.convert_frame(event)
                except (ValueError, TypeError) as exc:
                    self.malformed_frames += 1
                    self.last_error = str(exc)
                    if self.strict:
                        break
                    continue

                with self.lock:
                    self.latest_frame = frame
                    self.received_frames += 1
                self.event.set()

        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            self.last_error = str(exc)
        finally:
            self.running = False
            self.started = False

    def convert_frame(self, frame: HandFrame) -> dict[str, Any]:
        pos, quat_wxyz, landmarks = self.extract_geometry(frame)

        return {
            "side": frame.side.value,
            "wrist_pos": np.asarray(pos, dtype=np.float64),
            "wrist_quat_wxyz": np.asarray(quat_wxyz, dtype=np.float64),
            "landmarks": np.asarray(landmarks, dtype=np.float64).reshape(21, 3),
            "recv_ts_ns": frame.recv_ts_ns,
            "source_ts_ns": frame.source_ts_ns,
            "sequence_id": frame.sequence_id,
            "source_frame_seq": frame.source_frame_seq,
            "coordinate_frame": self.output_frame,
            "local_recv_ns": time.monotonic_ns(),
        }

    def extract_geometry(self, frame: HandFrame):
        wrist = frame.wrist
        pos = (wrist.x, wrist.y, wrist.z)
        quat_xyzw = (wrist.qx, wrist.qy, wrist.qz, wrist.qw)
        landmarks = frame.landmarks.points

        if self.output_frame == "unity":
            return pos, xyzw_to_wxyz(quat_xyzw), landmarks

        if self.output_frame == "rfu":
            return (
                unity_left_to_rfu_position(*pos),
                xyzw_to_wxyz(unity_left_to_rfu_rotation(*quat_xyzw)),
                [unity_left_to_rfu_position(*point) for point in landmarks],
            )

        if self.output_frame == "flu":
            return (
                unity_left_to_flu_position(*pos),
                xyzw_to_wxyz(unity_left_to_flu_rotation(*quat_xyzw)),
                [unity_left_to_flu_position(*point) for point in landmarks],
            )

        raise ValueError(f"Unsupported output_frame: {self.output_frame}")

    def frame_age(self, frame: dict[str, Any]) -> float:
        return (time.monotonic_ns() - frame["local_recv_ns"]) * 1e-9

    def copy_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Deep-copy a VR frame dict, copying numpy arrays by value.

        Uses a generic pattern instead of listing every key manually (P2.2).
        Automatically handles new keys added to the frame dict without
        requiring code changes here.
        """
        return {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in frame.items()
        }

    def __enter__(self) -> "QuestHandTracker":
        if not self.connect():
            raise RuntimeError(f"QuestHandTracker connect failed: {self.last_error}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

