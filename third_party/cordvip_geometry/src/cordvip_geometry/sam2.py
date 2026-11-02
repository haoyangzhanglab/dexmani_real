"""Grounded-SAM2 的独立解释器与顺序请求。"""

import logging
import os
import pickle
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from .models import GROUNDING_MODEL

logger = logging.getLogger(__name__)


class SAM2Client:
    def __init__(
        self, sam_python, workspace, models_dir, *, checkpoint, config, log_path=None
    ):
        self.python = Path(sam_python).expanduser().absolute()
        self.root = (
            Path(workspace).expanduser().resolve() / "third_party/Grounded-SAM-2"
        )
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.config = config
        self.dino_weights = (
            Path(models_dir).expanduser().resolve() / GROUNDING_MODEL["directory"]
        )
        self.log_path = Path(log_path) if log_path is not None else None
        self.provenance: dict[str, Any] = {}
        self._process = self._log = None
        self._ids = self._shape = None

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            logger.exception("SAM2 cleanup also failed")

    def start(self):
        if self._process is not None:
            raise RuntimeError("SAM2 process is already started")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("xb")
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(Path(__file__).parents[1]), str(self.root))
        )
        env["PYTHONNOUSERSITE"] = "1"
        command = [
            str(self.python),
            "-u",
            "-m",
            "cordvip_geometry.sam2_worker",
            "--sam-root",
            str(self.root),
            "--checkpoint",
            str(self.checkpoint),
            "--sam-config",
            self.config,
            "--dino-weights",
            str(self.dino_weights),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            env=env,
            start_new_session=True,
        )
        diagnostics = f"log: {self.log_path}" if self.log_path else "stderr"
        try:
            reply, _ = pickle.load(cast(BinaryIO, self._process.stdout))
        except (EOFError, pickle.UnpicklingError) as error:
            raise RuntimeError(
                f"SAM2 worker stopped during startup; {diagnostics}"
            ) from error
        if reply.get("error"):
            raise RuntimeError(f"SAM2 startup: {reply['error']}; {diagnostics}")
        self.provenance = reply["provenance"]
        return self

    def _call(self, operation, arrays=None, **parameters):
        if self._process is None:
            raise RuntimeError("SAM2 process is not running")
        diagnostics = f"log: {self.log_path}" if self.log_path else "stderr"
        try:
            pickle.dump(
                (dict(operation=operation, **parameters), arrays or {}),
                self._process.stdin,
                protocol=5,
            )
            self._process.stdin.flush()
            reply, result = pickle.load(self._process.stdout)
        except (EOFError, BrokenPipeError, pickle.UnpicklingError) as error:
            raise RuntimeError(f"SAM2 worker stopped; {diagnostics}") from error
        if reply.get("error"):
            raise RuntimeError(f"SAM2 {operation}: {reply['error']}; {diagnostics}")
        return reply, result

    @staticmethod
    def _rgb(rgb):
        rgb = np.asarray(rgb)
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 2
        ):
            raise ValueError("SAM2 expects RGB uint8 [H,W,3]")
        return rgb

    def predict(self, rgb, text, *, box_threshold=0.3, text_threshold=0.25):
        if self._ids is not None:
            raise RuntimeError("Text detection must precede fixed-ID tracking")
        rgb = self._rgb(rgb)
        if any(not 0 < value < 1 for value in (box_threshold, text_threshold)):
            raise ValueError("Detection thresholds must be in (0,1)")
        reply, arrays = self._call(
            "predict",
            {"rgb": rgb},
            text=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        self.provenance.update(reply["provenance"])
        return [
            {
                "mask": mask,
                "box_xyxy": arrays["boxes"][i],
                "score": float(arrays["scores"][i]),
                "mask_score": float(arrays["mask_scores"][i]),
                "label": reply["labels"][i],
            }
            for i, mask in enumerate(arrays["masks"])
        ]

    def initialize(self, rgb, masks):
        rgb = self._rgb(rgb)
        if self._ids is not None:
            raise RuntimeError("Create a new SAM2 session to reinitialize")
        if not masks:
            raise ValueError("Provide fixed named masks for the scene")
        for name, mask in masks.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Object IDs must be nonempty strings")
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2] or not mask.any():
                raise ValueError(f"Initial mask {name} must be nonempty bool [H,W]")
        ids = list(masks)
        _, arrays = self._call(
            "initialize",
            {"rgb": rgb, "masks": np.stack(list(masks.values()))},
            ids=ids,
        )
        self._ids, self._shape = ids, rgb.shape[:2]
        return dict(zip(ids, arrays["masks"]))

    def track(self, rgb):
        rgb = self._rgb(rgb)
        if self._ids is None or rgb.shape[:2] != self._shape:
            raise ValueError("Initialize first; frame shape must remain fixed")
        _, arrays = self._call("track", {"rgb": rgb})
        return dict(zip(self._ids, arrays["masks"]))

    @staticmethod
    def _signal(pid, sig):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass

    def close(self):
        process, failure = self._process, None
        try:
            if process is not None:
                self._signal(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._signal(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                self._signal(process.pid, signal.SIGKILL)
        except BaseException as error:  # noqa: BLE001 - 关闭管道后重抛首错
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
