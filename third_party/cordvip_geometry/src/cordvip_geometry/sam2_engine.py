"""One subprocess-owned SAM2 model handles first-frame grounding and causal propagation."""

import gc
from pathlib import Path

import numpy as np

from .sam2_config import (
    validate_grounding_weights,
    validate_sam_checkpoint,
    validate_sam_source,
)


def prune_state(state, frame_index, history_frames):
    """Keep conditioning and all temporal memories/pointers needed by the next frame."""
    cutoff = frame_index - history_frames + 1
    for output in [
        state["output_dict"],
        *state["output_dict_per_obj"].values(),
        *state["temp_output_dict_per_obj"].values(),
    ]:
        for index in list(output["non_cond_frame_outputs"]):
            if index < cutoff:
                del output["non_cond_frame_outputs"][index]
    for index in list(state["frames_already_tracked"]):
        if index != 0 and index < cutoff:
            del state["frames_already_tracked"][index]
    state["consolidated_frame_inds"]["non_cond_frame_outputs"].intersection_update(
        range(max(1, cutoff), frame_index + 1)
    )


def normalize_prompt(text):
    if not isinstance(text, str) or not text.strip().rstrip(". "):
        raise ValueError("Provide a nonempty object description")
    text = text.strip().lower()
    return text if text.endswith(".") else text + "."


class SAM2Engine:
    def __init__(self, root, checkpoint, config, dino_weights):
        root = validate_sam_source(root)
        checkpoint = validate_sam_checkpoint(root, checkpoint, config)
        import sam2
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.utils.misc import process_stream_frame

        if not Path(sam2.__file__).resolve().is_relative_to(root):
            raise RuntimeError("A different sam2 package is loaded")
        if not torch.cuda.is_available():
            raise RuntimeError("Grounded-SAM2 requires CUDA; no automatic CPU fallback")
        self.device = torch.device("cuda:0")
        self.tracker = build_sam2_video_predictor(
            config,
            str(checkpoint),
            device=str(self.device),
            apply_postprocessing=False,
            hydra_overrides_extra=[
                "++model.non_overlap_masks=true",
                "++model.non_overlap_masks_for_mem_enc=true",
                "++model.fill_hole_area=0",
            ],
        )
        self.image_predictor = SAM2ImagePredictor(
            self.tracker, max_hole_area=0, max_sprinkle_area=0
        )
        self.preprocess = process_stream_frame
        if not callable(getattr(self.tracker, "infer_single_frame", None)):
            raise RuntimeError("SAM2 infer_single_frame API is missing")
        stride = int(self.tracker.memory_temporal_stride_for_eval)
        if stride < 1:
            raise ValueError("SAM2 memory stride must be positive")
        self.history_frames = max(
            1,
            (int(self.tracker.num_maskmem) - 1) * stride,
            int(self.tracker.max_obj_ptrs_in_encoder) - 1,
        )
        self.dino_weights = Path(dino_weights).expanduser().resolve()
        self.processor = self.detector = self._detection_rgb = None
        self.state = self.ids = self.shape = None
        self.index = -1
        self.provenance = dict(
            method="Grounding DINO + SAM2.1; fixed IDs; causal single-frame propagation",
            sam_root=str(root),
            checkpoint=str(checkpoint),
            sam_config=config,
            device=str(self.device),
            precision="SAM2 bfloat16 autocast",
            history_frames=self.history_frames,
            fill_hole_area=0,
            max_sprinkle_area=0,
            non_overlap_masks=True,
            non_overlap_masks_for_mem_enc=True,
            overlap_policy="initial overlap excluded; propagated SAM2 logit winner",
            compile_model=False,
            grounding=None,
            image_preprocessing="SAM2 image predictor; video: RGB PIL LANCZOS square resize",
        )

    def load_grounding(self):
        if self.detector is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.models.grounding_dino import modeling_grounding_dino

        directory = validate_grounding_weights(self.dino_weights)
        self.processor = AutoProcessor.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False
        )
        # Transformers 4.48 ignores disable_custom_kernels in the attention
        # constructor and attempts JIT compilation anyway. Suppress only that
        # loader during construction; forward still uses the official torch path.
        loader = modeling_grounding_dino.load_cuda_kernels
        try:
            modeling_grounding_dino.load_cuda_kernels = lambda: None
            self.detector = (
                AutoModelForZeroShotObjectDetection.from_pretrained(
                    str(directory),
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    disable_custom_kernels=True,
                )
                .to(self.device)
                .eval()
            )
        finally:
            modeling_grounding_dino.load_cuda_kernels = loader
        self.provenance["grounding"] = dict(
            directory=str(directory),
            precision="float32",
            disable_custom_kernels=True,
        )

    def predict(self, rgb, text, box_threshold, text_threshold):
        if self.state is not None:
            raise RuntimeError("Grounding is only allowed before fixed-ID tracking starts")
        text = normalize_prompt(text)
        import torch
        from PIL import Image

        self.load_grounding()
        inputs = self.processor(images=Image.fromarray(rgb), text=text, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.detector(**inputs)
            detections = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[rgb.shape[:2]],
            )[0]
        boxes = detections["boxes"].detach().float().cpu().numpy()
        scores = detections["scores"].detach().float().cpu().numpy()
        labels = detections["labels"]
        if boxes.shape != (len(scores), 4) or len(labels) != len(scores):
            raise RuntimeError("Grounding DINO returned inconsistent detections")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise RuntimeError("Grounding DINO returned nonfinite boxes/scores")
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, rgb.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, rgb.shape[0])
        valid = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        empty = dict(
            masks=np.empty((0, *rgb.shape[:2]), bool),
            boxes=np.empty((0, 4), np.float32),
            scores=np.empty(0, np.float32),
            mask_scores=np.empty(0, np.float32),
        )
        if not len(valid):
            return empty, []
        boxes = boxes[valid]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Repeated prompts on the same frozen image reuse its embedding.
            if self._detection_rgb is None or not np.array_equal(rgb, self._detection_rgb):
                self.image_predictor.set_image(np.ascontiguousarray(rgb))
                self._detection_rgb = rgb.copy()
            masks, mask_scores, _ = self.image_predictor.predict(
                point_coords=None, point_labels=None, box=boxes, multimask_output=False
            )
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        mask_scores = np.asarray(mask_scores).reshape(-1)
        if masks.shape != (len(valid), *rgb.shape[:2]) or mask_scores.shape != (len(valid),):
            raise RuntimeError("SAM2 returned invalid candidate shapes")
        if not np.isfinite(mask_scores).all() or not np.isfinite(masks).all():
            raise RuntimeError("SAM2 returned nonfinite candidate masks/scores")
        order = [i for i in np.argsort(-scores[valid], kind="stable") if masks[i].any()]
        return dict(
            masks=masks[order].astype(bool),
            boxes=boxes[order],
            scores=scores[valid][order],
            mask_scores=mask_scores[order],
        ), [str(labels[valid[i]]) for i in order]

    def append(self, rgb):
        tensor, height, width = self.preprocess(
            img_array=rgb,
            image_size=self.tracker.image_size,
            offload_to_cpu=False,
            compute_device=self.device,
        )
        if (height, width) != self.shape:
            raise ValueError("Frame dimensions changed; initialize a new session")
        self.index += 1
        # Absolute-index lookup preserves temporal positions without retaining all images.
        self.state["images"] = {self.index: tensor}
        self.state["cached_features"] = {}
        self.state["num_frames"] = self.index + 1

    def output(self, result):
        import torch

        index, ids, logits = result
        if (
            index != self.index
            or list(ids) != self.ids
            or tuple(logits.shape) != (len(self.ids), 1, *self.shape)
        ):
            raise RuntimeError("SAM2 returned mismatched frame, identities or mask dimensions")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("SAM2 returned nonfinite mask logits")
        return (logits[:, 0] > 0).detach().cpu().numpy()

    def initialize(self, rgb, ids, masks):
        import torch

        if self.state is not None:
            raise RuntimeError("Create a new SAM2 session for explicit reinitialization")
        # Boolean manual masks provide no evidence for assigning overlapping pixels.
        masks = masks & (masks.sum(axis=0) == 1)[None]
        if any(not mask.any() for mask in masks):
            raise ValueError(
                "Initial mask empty after overlap exclusion; select distinct instances"
            )
        self.detector = self.processor = self._detection_rgb = None
        self.image_predictor.reset_predictor()
        gc.collect()
        torch.cuda.empty_cache()
        self.ids, self.shape, self.index = list(ids), rgb.shape[:2], -1
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.state = self.tracker.init_state(video_path=None)
            self.state["video_height"], self.state["video_width"] = self.shape
            self.append(rgb)
            for name, mask in zip(ids, masks):
                self.tracker.add_new_mask(self.state, frame_idx=0, obj_id=name, mask=mask)
            return self.output(self.tracker.infer_single_frame(self.state, 0))

    def track(self, rgb):
        import torch

        if self.state is None:
            raise RuntimeError("Initialize masks before tracking")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.append(rgb)
            masks = self.output(self.tracker.infer_single_frame(self.state, self.index))
            prune_state(self.state, self.index, self.history_frames)
            return masks
