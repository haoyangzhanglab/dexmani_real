"""VR frame publisher process — decoupled from control loop via ZMQ PUB/SUB.

Runs QuestHandTracker in a separate multiprocessing.Process, publishing VR
frames over ZMQ PUB socket so the control loop can subscribe without GIL
contention from HTS SDK I/O.

Ref: Open-Teach OculusVRHandDetector (openteach/components/detector/oculus.py:74-104)
     ManiUniCon SharedStorage write_action/read_all_action pattern.

Architecture:
    ┌──────────────────────┐   ZMQ PUB (5555)   ┌──────────────────────┐
    │ VR Publisher Process │───────────────────►│ TeleopController     │
    │ (独立 mp.Process)    │  topic="vr_frame"  │ (主进程, 50Hz)       │
    │                      │  JSON metadata +   │                      │
    │ QuestHandTracker     │  base64 numpy      │ ZMQ SUB → get_latest │
    │ _receive_loop()      │                    │ → Arm IK → Hand     │
    └──────────────────────┘                    └──────────────────────┘
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.teleop.vr.vr_tracker import QuestHandTracker

logger = get_logger(__name__)


class VRFramePublisher:
    """Publishes VR frames over ZMQ PUB socket from a separate process.

    Usage:
        tracker = QuestHandTracker(...)
        publisher = VRFramePublisher(tracker, pub_port=5555, hz=60.0)
        publisher.start()
        # ... run controller with ZMQ subscription ...
        publisher.stop()
    """

    def __init__(
        self,
        tracker: QuestHandTracker,
        pub_port: int = 5555,
        hz: float = 60.0,
    ) -> None:
        self.tracker = tracker
        self.pub_port = pub_port
        self.hz = float(hz)
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the publisher process (non-blocking)."""
        if self._process is not None:
            logger.warning("VRFramePublisher already running.")
            return
        self._stop_event.clear()
        self._process = mp.Process(target=self._run, name="vr-publisher", daemon=True)
        self._process.start()
        logger.info("VRFramePublisher started on port %d @ %.0f Hz", self.pub_port, self.hz)

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the publisher process to stop and wait for exit."""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.warning("VRFramePublisher did not exit within %.1fs, terminating.", timeout)
                self._process.terminate()
        self._process = None
        logger.info("VRFramePublisher stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop: read VR frames → serialize → publish over ZMQ."""
        try:
            import zmq
        except ImportError:
            logger.error("pyzmq not installed — VR publisher cannot run.")
            return

        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(f"tcp://127.0.0.1:{self.pub_port}")
        except zmq.ZMQError as e:
            logger.error("Failed to bind ZMQ PUB on port %d: %s", self.pub_port, e)
            socket.close()
            ctx.term()
            return

        interval = 1.0 / self.hz
        frame_count = 0
        last_ts = time.monotonic()

        try:
            while not self._stop_event.is_set():
                frame = self.tracker.get_latest()
                if frame is not None:
                    msg = self._serialize_frame(frame, frame_count)
                    socket.send_string("vr_frame", zmq.SNDMORE)
                    socket.send_json(msg)
                    frame_count += 1

                elapsed = time.monotonic() - last_ts
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_ts = time.monotonic()
        except (RuntimeError, ConnectionError, TimeoutError, ValueError, KeyError, TypeError):
            logger.exception("VRFramePublisher run loop crashed.")
        finally:
            socket.close()
            ctx.term()

    @staticmethod
    def _serialize_frame(frame: dict, seq: int) -> dict:
        """Serialize VR frame dict to JSON-safe format (numpy → base64)."""
        msg: dict = {"meta": {}, "arrays": {}, "seq": seq}

        for key, val in frame.items():
            if isinstance(val, np.ndarray):
                # Encode numpy arrays as base64 strings
                msg["arrays"][key] = {
                    "dtype": str(val.dtype),
                    "shape": list(val.shape),
                    "data": base64.b64encode(val.tobytes()).decode("ascii"),
                }
            elif isinstance(val, (int, float, str, bool, type(None))):
                msg["meta"][key] = val
            else:
                msg["meta"][key] = str(val)

        return msg


class VRFrameSubscriber:
    """Receives VR frames from a ZMQ PUB socket.

    Provides a get_latest()-like interface compatible with TeleopController.

    Usage in controller._read_vr_frame():
        if self._vr_subscriber is not None:
            return self._vr_subscriber.recv_latest()
        else:
            return self.tracker.get_latest()
    """

    def __init__(self, sub_port: int = 5555, topic: str = "vr_frame") -> None:
        self.sub_port = sub_port
        self.topic = topic
        self._ctx: object | None = None
        self._socket: object | None = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to VR publisher. Returns True on success."""
        try:
            import zmq
        except ImportError:
            logger.error("pyzmq not installed — ZMQ subscription unavailable.")
            return False

        try:
            self._ctx = zmq.Context()
            self._socket = self._ctx.socket(zmq.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
            self._socket.setsockopt(zmq.RCVTIMEO, 5)  # 5ms receive timeout
            self._socket.connect(f"tcp://127.0.0.1:{self.sub_port}")
            self._connected = True
            logger.info("VRFrameSubscriber connected to tcp://127.0.0.1:%d", self.sub_port)
            return True
        except zmq.ZMQError as e:
            logger.error("VRFrameSubscriber connect failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect and release ZMQ resources."""
        if self._socket is not None:
            try:
                self._socket.close()
            except (RuntimeError, ConnectionError, TimeoutError):
                pass
        if self._ctx is not None:
            try:
                self._ctx.term()
            except (RuntimeError, ConnectionError, TimeoutError):
                pass
        self._socket = None
        self._ctx = None
        self._connected = False

    def recv_latest(self) -> dict | None:
        """Non-blocking receive of latest VR frame.

        Returns the deserialized frame dict, or None if no frame available.
        """
        if not self._connected or self._socket is None:
            return None

        try:
            import zmq
        except ImportError:
            return None

        try:
            # Drain all available frames, keep only the latest
            latest: dict | None = None
            while True:
                try:
                    topic = self._socket.recv_string(flags=zmq.NOBLOCK)
                    msg = self._socket.recv_json(flags=zmq.NOBLOCK)
                    if topic == self.topic:
                        latest = msg
                except zmq.Again:
                    break

            if latest is not None:
                return self._deserialize_frame(latest)
            return None
        except zmq.ZMQError:
            return None

    @staticmethod
    def _deserialize_frame(msg: dict) -> dict:
        """Reconstruct VR frame dict from serialized message (base64 → numpy)."""
        frame: dict = dict(msg.get("meta", {}))
        arrays = msg.get("arrays", {})
        for key, info in arrays.items():
            frame[key] = np.frombuffer(
                base64.b64decode(info["data"]),
                dtype=np.dtype(info["dtype"]),
            ).reshape(info["shape"])
        return frame
