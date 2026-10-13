"""Causal, bounded multi-object masks using the local SAM 2 video predictor.

Construction loads a model; importing this module does not. Callers supply RGB
frames in acquisition order, with every returned mask belonging to that call's
frame. No old mask is substituted when a model predicts an empty mask.
"""

from contextlib import nullcontext
import gc
from pathlib import Path
import sys

import numpy as np

from .segmentation import _file


class StreamSegmenter:
    """Track a fixed set of visible instances until explicit reinitialization.

    IDs are nonempty strings. Initialization takes full-resolution bool masks;
    all subsequent RGB uint8 frames must have the same size. Output masks are
    mutually exclusive according to SAM's logits, not geometric confidence.
    This wrapper supports the pinned checkout's ``infer_single_frame`` API and
    deliberately does not use its unbounded ``add_new_frame`` implementation.
    """

    def __init__(self, sam_root, checkpoint,
                 config="configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda"):
        self._predictor = self._state = None
        self._closed = False
        self._ids = ()
        self._shape = None
        self.frame_index = -1
        root = Path(sam_root).expanduser().resolve()
        checkpoint = _file(checkpoint)
        config = str(config)
        config_path = (root / "sam2" / config).resolve()
        if Path(config).is_absolute() or not config_path.is_relative_to(root / "sam2"):
            raise ValueError("SAM config must be relative to the checkout's sam2 package")
        _file(config_path)
        _file(root / "sam2" / "build_sam.py")

        import torch

        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only explicit cpu or cuda devices are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU mode must be requested explicitly")
        loaded = sys.modules.get("sam2")
        if loaded is not None:
            source = getattr(loaded, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(root):
                raise RuntimeError("A different sam2 package is loaded; use a fresh subprocess")
        previous_path = sys.path.copy()
        try:
            sys.path.insert(0, str(root))
            from sam2.build_sam import build_sam2_video_predictor
            from sam2.utils.misc import process_stream_frame

            self._preprocess = process_stream_frame
            self._predictor = build_sam2_video_predictor(
                config, str(checkpoint), device=str(self.device),
                # Basket lattice openings are observations, not holes to fill.
                apply_postprocessing=False,
                hydra_overrides_extra=[
                    "++model.non_overlap_masks=true",
                    "++model.non_overlap_masks_for_mem_enc=true",
                    "++model.fill_hole_area=0",
                ],
            )
            if not callable(getattr(self._predictor, "infer_single_frame", None)):
                raise RuntimeError("SAM checkout lacks the pinned streaming API")
            stride = int(self._predictor.memory_temporal_stride_for_eval)
            if stride < 1:
                raise ValueError("SAM memory stride must be positive")
            # Include every mask memory and object pointer requested by future
            # forward frames, including stride-aligned frames not used just now.
            self.history_frames = max(
                1, (int(self._predictor.num_maskmem) - 1) * stride,
                int(self._predictor.max_obj_ptrs_in_encoder) - 1,
            )
        except BaseException:
            self.close()
            raise
        finally:
            sys.path[:] = previous_path
        self.provenance = {
            "method": "SAM 2 causal multi-object propagation",
            "sam_root": str(root), "sam_checkpoint": str(checkpoint),
            "sam_config": config, "device": str(self.device),
            "history_frames": self.history_frames,
            "non_overlap_masks": True, "fill_hole_area": 0,
        }

    def _validate_rgb(self, rgb):
        if self._closed:
            raise RuntimeError("Stream segmenter has been closed")
        rgb = np.asarray(rgb)
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or min(rgb.shape[:2]) < 2):
            raise ValueError("RGB must be a uint8 [H,W,3] array")
        return np.ascontiguousarray(rgb)

    def _autocast(self):
        import torch

        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda" else nullcontext())

    def _append(self, rgb):
        tensor, height, width = self._preprocess(
            img_array=rgb, image_size=self._predictor.image_size,
            offload_to_cpu=False, compute_device=self.device,
        )
        if (height, width) != self._shape:
            raise RuntimeError("SAM preprocessing changed the source image dimensions")
        frame_index = self.frame_index + 1
        # _get_image_feature only requires absolute-index lookup. Keeping a
        # dict avoids a growing image tensor without renumbering temporal memory.
        self._state["images"] = {frame_index: tensor}
        self._state["cached_features"] = {}
        self._state["num_frames"] = frame_index + 1
        self.frame_index = frame_index

    def _output(self, result):
        import torch

        frame_index, ids, logits = result
        if frame_index != self.frame_index or tuple(ids) != self._ids:
            raise RuntimeError("SAM returned mismatched frame or instance IDs")
        if tuple(logits.shape) != (len(self._ids), 1, *self._shape):
            raise RuntimeError("SAM returned an unexpected mask shape")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("SAM returned nonfinite mask logits")
        masks = (logits[:, 0] > 0).detach().cpu().numpy()
        return {obj_id: masks[index].copy() for index, obj_id in enumerate(self._ids)}

    def _prune(self):
        cutoff = self.frame_index - self.history_frames + 1
        output_dicts = [self._state["output_dict"]]
        output_dicts.extend(self._state["output_dict_per_obj"].values())
        output_dicts.extend(self._state["temp_output_dict_per_obj"].values())
        for output in output_dicts:
            non_cond = output["non_cond_frame_outputs"]
            for index in list(non_cond):
                if index < cutoff:
                    del non_cond[index]
        # Only frame 0 has user input. Retaining it is also required by SAM's
        # preflight consistency check between conditioning frames and inputs.
        tracked = self._state["frames_already_tracked"]
        for index in list(tracked):
            if index != 0 and index < cutoff:
                del tracked[index]

    def initialize(self, rgb, masks):
        """Start a new sequence; discard all previous identity and image history."""
        rgb = self._validate_rgb(rgb)
        if not isinstance(masks, dict) or not masks:
            raise ValueError("Provide at least one named initialization mask")
        for obj_id, mask in masks.items():
            if not isinstance(obj_id, str) or not obj_id.strip():
                raise ValueError("Instance IDs must be nonempty strings")
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2] or not mask.any():
                raise ValueError(f"Initialization mask {obj_id!r} must be nonempty bool [H,W]")

        import torch

        self._state = None
        self._shape = rgb.shape[:2]
        self._ids = tuple(masks)
        self.frame_index = -1
        try:
            with torch.inference_mode(), self._autocast():
                self._state = self._predictor.init_state(video_path=None)
                self._state["video_height"], self._state["video_width"] = self._shape
                self._append(rgb)
                for obj_id, mask in masks.items():
                    self._predictor.add_new_mask(
                        self._state, frame_idx=0, obj_id=obj_id, mask=mask,
                    )
                return self._output(self._predictor.infer_single_frame(self._state, 0))
        except BaseException:
            self._state = None
            raise

    def track(self, rgb):
        """Process only the supplied next frame; empty masks remain empty."""
        rgb = self._validate_rgb(rgb)
        if self._state is None:
            raise RuntimeError("Initialize named masks before tracking")
        if rgb.shape[:2] != self._shape:
            raise ValueError("Frame size changed; explicitly initialize a new sequence")

        import torch

        try:
            with torch.inference_mode(), self._autocast():
                self._append(rgb)
                result = self._output(self._predictor.infer_single_frame(
                    self._state, self.frame_index,
                ))
                self._prune()
                return result
        except BaseException:
            # A partial update cannot be replayed safely as a fresh next frame.
            self._state = None
            raise

    def close(self):
        """Release the sequence and model. Does not affect other CUDA models."""
        self._state = self._predictor = None
        self._ids = ()
        self._shape = None
        self._closed = True
        gc.collect()
