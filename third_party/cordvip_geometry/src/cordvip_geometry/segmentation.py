"""一个私有 Grounded-SAM2 子进程；单线程顺序请求，失败后由调用方关闭。"""

import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .segmentation_protocol import receive, send

logger = logging.getLogger(__name__)


class GroundedSAM2Segmenter:
    """每个实例拥有一个模型进程和管道；调用方必须在 finally 中 close。"""

    def __init__(self, sam_python, sam_root, checkpoint, *, sam_config, dino_weights, log_path):
        self.python = Path(sam_python).expanduser().absolute()
        self.root = Path(sam_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.log_path = Path(log_path)
        self.config = sam_config
        self.dino_weights = Path(dino_weights).expanduser().resolve()
        self.provenance = None
        self._process = self._log = None
        self._ids = None
        self._shape = None

    def start(self):
        if self._process is not None:
            raise RuntimeError("SAM2 process is already started")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("xb")
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).parents[1]), str(self.root)))
        env["PYTHONNOUSERSITE"] = "1"
        command = [
            str(self.python), "-u", "-m", "cordvip_geometry.segmentation_worker",
            "--sam-root", str(self.root), "--checkpoint", str(self.checkpoint),
            "--sam-config", self.config, "--dino-weights", str(self.dino_weights),
        ]
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            env=env, start_new_session=True,
        )
        reply, _ = self.call("load")
        self.provenance = reply["provenance"]
        return self

    def call(self, operation, arrays=None, *, frame_key=None, **parameters):
        if self._process is None:
            raise RuntimeError("SAM2 process is not running")
        try:
            send(self._process.stdin, dict(frame_key=frame_key, operation=operation, **parameters), arrays)
            reply, result = receive(self._process.stdout)
        except (EOFError, BrokenPipeError) as error:
            raise RuntimeError(f"SAM2 worker stopped; log: {self.log_path}") from error
        if reply["frame_key"] != frame_key:
            raise RuntimeError("Rejected mismatched SAM2 frame reply")
        if reply.get("error"):
            raise RuntimeError(f"SAM2 {operation}: {reply['error']}; log: {self.log_path}")
        return reply, result

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
        masks = arrays["masks"]
        if (
            reply["ids"] != self._ids
            or masks.dtype != np.bool_
            or masks.shape != (len(self._ids), *self._shape)
            or np.any(masks.sum(axis=0) > 1)
        ):
            raise RuntimeError("SAM2 returned invalid masks or changed fixed identities")
        return dict(zip(self._ids, masks))

    def track(self, rgb, *, frame_key=None):
        rgb = self.rgb(rgb)
        if self._ids is None or rgb.shape[:2] != self._shape:
            raise ValueError("Initialize first; frame shape must remain fixed")
        reply, arrays = self.call("track", {"rgb": rgb}, frame_key=frame_key)
        return self.masks(reply, arrays)

    def signal_group(self, sig):
        # The worker owns a separate session. Never signal the caller or unrelated processes.
        try:
            os.killpg(self._process.pid, sig)
        except ProcessLookupError:
            pass

    def close(self):
        failure = None
        process = self._process
        try:
            if process is not None:
                self.signal_group(signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.signal_group(signal.SIGKILL)
                    process.wait(timeout=2)
                # Terminate any descendants left in the private process group.
                self.signal_group(signal.SIGKILL)
        except BaseException as error:
            failure = error
        streams: list[BinaryIO] = []
        if process is not None and process.returncode is not None:
            streams.extend((process.stdin, process.stdout))
            self._process = None
        if self._log is not None:
            streams.append(self._log)
            self._log = None
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    logger.exception("SAM2 pipe/log close also failed")
        if failure is not None:
            raise failure
