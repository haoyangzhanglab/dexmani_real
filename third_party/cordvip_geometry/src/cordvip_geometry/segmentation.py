"""One persistent Grounded-SAM2 process with cancellable, frame-matched replies."""

import os
from concurrent.futures import CancelledError
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

import numpy as np

from .segmentation_protocol import receive, send


class GroundedSAM2Segmenter:
    """One owner-thread request; start loads the child model and cancel revokes its process group.
    The owner calls close after either successful or failed start. Parent FoundationPose
    CUDA work must finish on its own thread before cleanup."""

    def __init__(self, sam_python, sam_root, checkpoint, *, sam_config, dino_weights, log_path):
        self.python = Path(sam_python).expanduser().absolute()
        self.root = Path(sam_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.log_path = Path(log_path)
        self.config = sam_config
        self.dino_weights = Path(dino_weights).expanduser().resolve()
        self.provenance = None
        self.last_metrics = {}
        self._process = self._log = None
        self._session = uuid.uuid4().hex
        self._counter = 0
        self._ids = None
        self._shape = None
        self._cancelled = threading.Event()
        self._lifecycle = threading.Lock()
        self._request = threading.Lock()

    def start(self):
        with self._lifecycle:
            if self._cancelled.is_set():
                raise CancelledError("SAM2 session was cancelled")
            if self._process is not None:
                return self
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("xb")
            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).parents[1]), str(self.root)))
            env["PYTHONNOUSERSITE"] = "1"
            command = [
                str(self.python),
                "-u",
                "-m",
                "cordvip_geometry.segmentation_worker",
                "--sam-root",
                str(self.root),
                "--checkpoint",
                str(self.checkpoint),
                "--sam-config",
                self.config,
                "--dino-weights",
                str(self.dino_weights),
            ]
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self._log,
                    env=env,
                    start_new_session=True,
                )
            except BaseException:
                self._log.close()
                self._log = None
                raise
        reply, _ = self.call("load")
        self.provenance = reply["provenance"]
        return self

    def call(self, operation, arrays=None, *, frame_key=None, **parameters):
        if not self._request.acquire(blocking=False):
            raise RuntimeError("SAM2 allows exactly one in-flight request")
        try:
            if self._cancelled.is_set():
                raise CancelledError("SAM2 session was cancelled")
            if self._process is None:
                raise RuntimeError("SAM2 session is not running")
            self._counter += 1
            identity = dict(session=self._session, request=self._counter, frame_key=frame_key)
            started = time.monotonic_ns()
            send(self._process.stdin, dict(**identity, operation=operation, **parameters), arrays)
            reply, result = receive(self._process.stdout)
            if any(reply.get(k) != v for k, v in identity.items()):
                raise RuntimeError("Rejected mismatched SAM2 reply")
            if reply.get("error"):
                raise RuntimeError(f"SAM2 {operation}: {reply['error']}; log: {self.log_path}")
            if self._cancelled.is_set():
                raise CancelledError("Rejected cancelled SAM2 reply")
            self.last_metrics = reply["metrics"]
            self.last_metrics["round_trip_ms"] = (time.monotonic_ns() - started) / 1e6
            self.last_metrics["ipc_and_dispatch_ms"] = max(
                0.0, self.last_metrics["round_trip_ms"] - self.last_metrics["compute_ms"]
            )
            return reply, result
        except (EOFError, BrokenPipeError) as error:
            if self._cancelled.is_set():
                raise CancelledError("SAM2 request was cancelled") from error
            self.cancel()
            raise RuntimeError(f"SAM2 worker stopped; log: {self.log_path}") from error
        except CancelledError:
            raise
        except Exception:
            self.cancel()
            raise
        finally:
            self._request.release()

    @staticmethod
    def rgb(rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("SAM2 expects RGB uint8 [H,W,3]")
        return rgb

    def predict(self, rgb, text, *, box_threshold=0.3, text_threshold=0.25, frame_key=None):
        if self._ids is not None:
            raise RuntimeError("Text detection is only allowed before fixed-ID tracking starts")
        rgb = self.rgb(rgb)
        if (
            not isinstance(text, str)
            or not text.strip()
            or any(not 0 < x < 1 for x in (box_threshold, text_threshold))
        ):
            raise ValueError("Provide a nonempty prompt and detection thresholds in (0,1)")
        reply, arrays = self.call(
            "predict",
            {"rgb": rgb},
            text=text.strip(),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            frame_key=frame_key,
        )
        try:
            masks, labels = arrays["masks"], reply["labels"]
            count = len(masks)
            if (
                masks.dtype != np.bool_
                or masks.shape != (count, *rgb.shape[:2])
                or arrays["boxes"].shape != (count, 4)
                or arrays["scores"].shape != (count,)
                or arrays["mask_scores"].shape != (count,)
                or len(labels) != count
                or any(not isinstance(label, str) for label in labels)
                or any(not np.isfinite(arrays[k]).all() for k in ("boxes", "scores", "mask_scores"))
            ):
                raise RuntimeError("Grounded-SAM2 returned invalid candidates")
            self.provenance.update(reply["provenance"])
            return [
                dict(
                    mask=mask,
                    box_xyxy=arrays["boxes"][i],
                    score=float(arrays["scores"][i]),
                    mask_score=float(arrays["mask_scores"][i]),
                    label=labels[i],
                )
                for i, mask in enumerate(masks)
            ]
        except Exception:
            self.cancel()
            raise

    def initialize(self, rgb, masks, *, frame_key=None):
        rgb = self.rgb(rgb)
        if self._ids is not None:
            raise RuntimeError("Create a new SAM2 session for explicit reinitialization")
        if not isinstance(masks, dict) or not masks:
            raise ValueError("Provide nonempty fixed named masks for the scene")
        for name, mask in masks.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Object IDs must be nonempty strings")
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2] or not mask.any():
                raise ValueError(f"Initial mask {name} must be nonempty bool [H,W]")
        ids = list(masks)
        reply, arrays = self.call(
            "initialize",
            {"rgb": rgb, "masks": np.stack(list(masks.values()))},
            ids=ids,
            frame_key=frame_key,
        )
        self._ids, self._shape = ids, rgb.shape[:2]
        return self.masks(reply, arrays)

    def masks(self, reply, arrays):
        try:
            masks = arrays["masks"]
            if (
                reply["ids"] != self._ids
                or masks.dtype != np.bool_
                or masks.shape != (len(self._ids), *self._shape)
                or np.any(masks.sum(axis=0) > 1)
            ):
                raise RuntimeError("SAM2 returned invalid masks or changed fixed identities")
            return dict(zip(self._ids, masks))
        except Exception:
            self.cancel()
            raise

    def track(self, rgb, *, frame_key=None):
        rgb = self.rgb(rgb)
        if self._ids is None or rgb.shape[:2] != self._shape:
            raise ValueError("Initialize first; frame shape must remain fixed")
        reply, arrays = self.call("track", {"rgb": rgb}, frame_key=frame_key)
        return self.masks(reply, arrays)

    def cancel(self):
        self._cancelled.set()
        with self._lifecycle:
            if self._process is not None:
                self.signal_group(signal.SIGTERM)

    def signal_group(self, sig):
        # The worker owns a separate session. Never signal the caller or unrelated processes.
        try:
            os.killpg(self._process.pid, sig)
        except ProcessLookupError:
            pass

    def close(self):
        self.cancel()
        with self._lifecycle:
            if self._process is not None:
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.signal_group(signal.SIGKILL)
                    self._process.wait(timeout=2)
                self.signal_group(signal.SIGKILL)
                self._process.stdin.close()
                self._process.stdout.close()
                self._process = None
            if self._log is not None:
                self._log.close()
                self._log = None
