"""ZMQ REP server for dex-retargeting inference.

Listens on port 5556, receives hand-landmark JSON requests,
runs XHandRetargeter.retarget(), and sends back joint positions.

Usage:
    python -m dexmani_real.services.retarget_server [--port PORT] [--hand-type right|left] [--retargeting-type dexpilot|position]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from typing import Any

import numpy as np
import zmq

from dexmani_real.utils.log import get_logger
from dexmani_real.teleop.vr.hand_retarget import XHandRetargeter
from dexmani_real.utils.hand_utils import OPERATOR2MANO_RIGHT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("retarget_server")

# ---------------------------------------------------------------------------
# Globals for graceful shutdown
# ---------------------------------------------------------------------------

_retargeter: XHandRetargeter | None = None
_context: zmq.Context | None = None
_socket: zmq.Socket | None = None


def _shutdown() -> None:
    global _socket, _context
    logger.info("Shutting down retarget server ...")
    try:
        if _socket is not None:
            _socket.close(linger=500)
            _socket = None
    except Exception:
        pass
    try:
        if _context is not None:
            _context.term()
            _context = None
    except Exception:
        pass
    logger.info("Retarget server stopped.")


_shutdown_requested = False


def _signal_handler(signum: int, frame: Any) -> None:
    global _shutdown_requested
    logger.info("Received signal %s, shutting down ...", signal.Signals(signum).name)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Request processing
# ---------------------------------------------------------------------------


def _validate_landmarks(data: dict) -> np.ndarray | None:
    """Validate and extract 21x3 hand landmarks from request data.

    Accepts ``landmarks`` (list-of-lists) or ``keypoints`` (flat list, reshaped).
    """
    if "landmarks" in data:
        arr = np.asarray(data["landmarks"], dtype=np.float64)
    elif "keypoints" in data:
        arr = np.asarray(data["keypoints"], dtype=np.float64).reshape(21, 3)
    else:
        logger.warning("Request missing 'landmarks' or 'keypoints' key.")
        return None

    if arr.shape != (21, 3):
        logger.warning("Expected shape (21, 3), got %s.", arr.shape)
        return None

    return arr


def _process_request(request: dict) -> dict:
    """Process a single retargeting request and return a JSON-serializable response."""
    landmarks = _validate_landmarks(request)
    if landmarks is None:
        return {"status": "error", "message": "invalid landmarks"}

    # Apply operator-to-mano coordinate transform
    landmarks_mano = landmarks @ OPERATOR2MANO_RIGHT.T

    try:
        start = time.perf_counter()
        joints = _retargeter.retarget(landmarks_mano)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
    except Exception:
        logger.exception("Retargeting failed.")
        return {"status": "error", "message": "retargeting exception", "traceback": traceback.format_exc()}

    if joints is None:
        return {"status": "error", "message": "retargeting returned None"}

    return {
        "status": "ok",
        "joints": joints.tolist(),
        "elapsed_ms": round(elapsed_ms, 3),
    }


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


def run_server(port: int = 5556, hand_type: str = "right", retargeting_type: str = "dexpilot") -> None:
    global _retargeter, _context, _socket

    # Install signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "Initializing XHandRetargeter (hand_type=%s, retargeting_type=%s) ...",
        hand_type,
        retargeting_type,
    )
    _retargeter = XHandRetargeter(hand_type=hand_type, retargeting_type=retargeting_type)
    logger.info("Retargeter ready.")

    _context = zmq.Context()
    _socket = _context.socket(zmq.REP)
    _socket.bind(f"tcp://*:{port}")
    logger.info("Retarget server listening on port %d ...", port)

    poller = zmq.Poller()
    poller.register(_socket, zmq.POLLIN)
    while not _shutdown_requested:
        socks = dict(poller.poll(timeout=100))  # 100ms poll interval
        if _socket not in socks:
            continue
        try:
            raw = _socket.recv_json()
        except zmq.ZMQError as e:
            logger.error("ZMQ recv error: %s", e)
            break

        request = json.loads(raw) if isinstance(raw, str) else raw
        logger.debug("Received request: %s", {k: v for k, v in request.items() if k != "landmarks"})

        response = _process_request(request)
        _socket.send_json(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Dex-Retargeting ZMQ REP Server")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ bind port (default: 5556)")
    parser.add_argument("--hand-type", type=str, default="right", choices=["right", "left"], help="Hand side")
    parser.add_argument(
        "--retargeting-type",
        type=str,
        default="dexpilot",
        choices=["dexpilot", "position"],
        help="Retargeting algorithm",
    )
    args = parser.parse_args()

    try:
        run_server(port=args.port, hand_type=args.hand_type, retargeting_type=args.retargeting_type)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception:
        logger.exception("Fatal error in retarget server.")
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
