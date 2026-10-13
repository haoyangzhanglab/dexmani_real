"""Local Grounding DINO + SAM 2.1 image segmentation for an example subprocess.

Importing this module loads neither model nor device. Model files must already
exist locally; downloading and choosing among detections belong to the caller.
"""

from contextlib import nullcontext
import gc
import hashlib
from pathlib import Path
import sys
import zipfile

import numpy as np


def _file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty local model file: {path}")
    pending = [path.with_name(path.name + suffix) for suffix in (".aria2", ".part", ".incomplete")]
    if any(marker.exists() for marker in pending):
        raise ValueError(f"Local model download is unfinished: {path}")
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(prompt):
    """Grounding DINO expects lowercase, period-separated object descriptions."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Object prompt must be nonempty text")
    prompt = prompt.strip().lower()
    if not prompt.rstrip(". "):
        raise ValueError("Object prompt must contain an object description")
    return prompt if prompt.endswith(".") else prompt + "."


class GroundedSegmenter:
    """Detect all text-matching candidates and return their full-resolution masks.

    SAM and Grounding DINO scores are model outputs, not tracking confidence or
    proof of mask correctness. The caller must explicitly select a candidate.
    Use in a fresh example process because SAM initializes Hydra globally.
    """

    def __init__(self, dino_weights, sam_root, sam_checkpoint,
                 sam_config="configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda"):
        self._processor = self._detector = self._predictor = None
        self._closed = False
        self.dino_weights = Path(dino_weights).expanduser().resolve()
        self.sam_root = Path(sam_root).expanduser().resolve()
        self.sam_checkpoint = _file(sam_checkpoint)
        if not zipfile.is_zipfile(self.sam_checkpoint):
            raise ValueError(f"SAM 2.1 checkpoint is incomplete or not a PyTorch ZIP archive: {self.sam_checkpoint}")
        self.sam_config = str(sam_config)
        config_path = (self.sam_root / "sam2" / self.sam_config).resolve()
        if Path(self.sam_config).is_absolute() or not config_path.is_relative_to(self.sam_root / "sam2"):
            raise ValueError("SAM config must be relative to the checkout's sam2 package")
        _file(config_path)
        _file(self.sam_root / "sam2" / "build_sam.py")
        _file(self.dino_weights / "config.json")
        dino_checkpoints = sorted(self.dino_weights.glob("*.safetensors"))
        if not dino_checkpoints:
            raise FileNotFoundError(f"Local safetensors Grounding DINO weights missing: {self.dino_weights}")
        for checkpoint in dino_checkpoints:
            _file(checkpoint)

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only explicit cpu or cuda devices are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; CPU mode must be requested explicitly")
            if self.device.index is not None and self.device.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device does not exist: {self.device}")

        loaded = sys.modules.get("sam2")
        if loaded is not None:
            source = getattr(loaded, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(self.sam_root):
                raise RuntimeError("A different sam2 package is loaded; use a fresh subprocess")
        previous_path = sys.path.copy()
        try:
            sys.path.insert(0, str(self.sam_root))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self._processor = AutoProcessor.from_pretrained(
                str(self.dino_weights), local_files_only=True, trust_remote_code=False,
            )
            self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(self.dino_weights), local_files_only=True,
                trust_remote_code=False, use_safetensors=True, disable_custom_kernels=True,
            ).to(self.device).eval()
            self._predictor = SAM2ImagePredictor(build_sam2(
                self.sam_config, str(self.sam_checkpoint), device=str(self.device),
            ))
        except BaseException:
            self.close()
            raise
        finally:
            sys.path[:] = previous_path

        self.provenance = {
            "method": "Grounding DINO + SAM 2.1",
            "dino_weights": str(self.dino_weights),
            "dino_weights_sha256": {path.name: _sha256(path) for path in dino_checkpoints},
            "dino_configuration_sha256": {
                path.name: _sha256(path) for path in sorted(self.dino_weights.glob("*.json"))
            },
            "sam_root": str(self.sam_root),
            "sam_config": self.sam_config,
            "sam_config_sha256": _sha256(config_path),
            "sam_checkpoint": str(self.sam_checkpoint),
            "sam_checkpoint_sha256": _sha256(self.sam_checkpoint),
            "device": str(self.device),
            "dino_disable_custom_kernels": True,
        }

    def predict(self, rgb, prompt, box_threshold=0.3, text_threshold=0.25):
        """Return score-sorted dicts: box_xyxy, mask, score, mask_score, label.

        Boxes are pixel coordinates in the original RGB image. Masks are bool
        arrays [H,W]; score/mask_score/label are ordinary Python values. No
        detection returns [] rather than a made-up whole-image foreground.
        """
        if self._closed:
            raise RuntimeError("Segmenter has been closed")
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("RGB must be a uint8 [H,W,3] array with nonempty dimensions")
        prompt = normalize_prompt(prompt)
        for value in (box_threshold, text_threshold):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Detection thresholds must lie in [0,1]")

        import torch
        from PIL import Image

        inputs = self._processor(
            images=Image.fromarray(rgb), text=prompt, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self._detector(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=[rgb.shape[:2]],
            )[0]
        boxes = result["boxes"].detach().float().cpu().numpy()
        detection_scores = result["scores"].detach().float().cpu().numpy()
        labels = result["labels"]
        if len(boxes) == 0:
            return []
        if boxes.shape != (len(detection_scores), 4) or len(labels) != len(boxes):
            raise RuntimeError("Grounding DINO returned inconsistent detections")
        if not np.isfinite(boxes).all() or not np.isfinite(detection_scores).all():
            raise RuntimeError("Grounding DINO returned nonfinite boxes/scores")
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, rgb.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, rgb.shape[0])
        valid = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        if len(valid) == 0:
            return []
        boxes = boxes[valid]
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda" else nullcontext())
        with torch.inference_mode(), autocast:
            self._predictor.set_image(np.ascontiguousarray(rgb))
            masks, mask_scores, _ = self._predictor.predict(
                point_coords=None, point_labels=None, box=boxes, multimask_output=False,
            )
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        mask_scores = np.asarray(mask_scores).reshape(-1)
        if masks.shape != (len(boxes), *rgb.shape[:2]) or mask_scores.shape != (len(boxes),):
            raise RuntimeError("SAM returned an unexpected mask shape")
        if not np.isfinite(mask_scores).all():
            raise RuntimeError("SAM returned nonfinite mask scores")
        candidates = []
        for index, source in enumerate(valid):
            mask = masks[index].astype(bool)
            if not mask.any():
                continue
            candidates.append({
                "box_xyxy": boxes[index].astype(np.float32, copy=True),
                "mask": mask,
                "score": float(detection_scores[source]),
                "mask_score": float(mask_scores[index]),
                "label": str(labels[source]),
            })
        return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)

    def close(self):
        """Release model references. A subprocess exit also releases CUDA state."""
        self._predictor = self._detector = self._processor = None
        self._closed = True
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and getattr(self, "device", None) is not None and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
