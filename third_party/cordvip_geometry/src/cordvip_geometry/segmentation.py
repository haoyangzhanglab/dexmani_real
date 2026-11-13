"""Grounded-SAM2 检测、单图分割与有界历史视频跟踪。"""

import sys
from pathlib import Path
from typing import Any

import numpy as np


def prune_state(state, frame_index, history_frames):
    """保留首帧 conditioning 和下一帧所需历史。"""
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


class SAM2Segmenter:
    def __init__(self, root, checkpoint, config, dino_weights):
        root = Path(root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        sys.path.insert(0, str(root))
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.utils.misc import process_stream_frame

        if not torch.cuda.is_available():
            raise RuntimeError("Grounded-SAM2 requires CUDA")
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
        stride = int(self.tracker.memory_temporal_stride_for_eval)
        self.history_frames = max(
            1,
            (int(self.tracker.num_maskmem) - 1) * stride,
            int(self.tracker.max_obj_ptrs_in_encoder) - 1,
        )
        self.dino_weights = Path(dino_weights).expanduser().resolve()
        self.processor: Any = None
        self.detector: Any = None

    def load_grounding(self):
        if self.detector is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.models.grounding_dino import modeling_grounding_dino

        directory = str(self.dino_weights)
        self.processor = AutoProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        # 禁用 Transformers 4.48 构造期的自定义 CUDA 编译。
        loader = modeling_grounding_dino.load_cuda_kernels
        try:
            modeling_grounding_dino.load_cuda_kernels = lambda: None
            self.detector = (
                AutoModelForZeroShotObjectDetection.from_pretrained(
                    directory,
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

    def detect(self, rgb, text, box_threshold=0.3, text_threshold=0.25):
        """返回同标签 NMS 去重后的检测框、分数和标签，按置信度降序排列。"""
        text = normalize_prompt(text)
        import torch
        from PIL import Image
        from torchvision.ops import batched_nms

        self.load_grounding()
        inputs = self.processor(
            images=Image.fromarray(rgb), text=text, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.detector(**inputs)
            detections = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[rgb.shape[:2]],
            )[0]

        boxes = detections["boxes"].detach().float().cpu()
        scores = detections["scores"].detach().float().cpu()
        boxes[:, 0::2].clamp_(0, rgb.shape[1])
        boxes[:, 1::2].clamp_(0, rgb.shape[0])
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        labels = [
            label for label, keep in zip(detections["labels"], valid.tolist()) if keep
        ]

        normalized = [label.strip().lower().rstrip(".") for label in labels]
        label_ids = {label: i for i, label in enumerate(dict.fromkeys(normalized))}
        class_ids = torch.tensor(
            [label_ids[label] for label in normalized], dtype=torch.int64
        )
        indices = batched_nms(boxes, scores, class_ids, iou_threshold=0.8)
        return [
            {
                "box_xyxy": boxes[i].numpy(),
                "score": float(scores[i]),
                "label": str(labels[i]),
            }
            for i in indices.tolist()
        ]

    def predict(self, rgb, text, box_threshold=0.3, text_threshold=0.25):
        """单图分割，在检测候选中加入 mask 并过滤空 mask。"""
        import torch

        candidates = self.detect(rgb, text, box_threshold, text_threshold)
        if not candidates:
            return []
        boxes = np.stack([item["box_xyxy"] for item in candidates])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.image_predictor.set_image(np.ascontiguousarray(rgb))
            masks, _, _ = self.image_predictor.predict(
                point_coords=None, point_labels=None, box=boxes, multimask_output=False
            )
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        return [
            {**item, "mask": mask.astype(bool)}
            for item, mask in zip(candidates, masks)
            if mask.any()
        ]

    def initialize(self, rgb, boxes):
        """以 {对象名: XYXY 检测框} 初始化新视频，共享首帧图像编码。"""
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.state = self.tracker.init_state(video_path=None)
            self.state["video_height"], self.state["video_width"] = rgb.shape[:2]
            self._append_frame(rgb)
            for name, box in boxes.items():
                self.tracker.add_new_points_or_box(
                    self.state, frame_idx=0, obj_id=name, box=box
                )
            return self._masks(self.tracker.infer_single_frame(self.state, 0))

    def track(self, rgb):
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            index = self._append_frame(rgb)
            masks = self._masks(self.tracker.infer_single_frame(self.state, index))
            prune_state(self.state, index, self.history_frames)
            return masks

    def _append_frame(self, rgb):
        tensor, _, _ = self.preprocess(
            img_array=rgb,
            image_size=self.tracker.image_size,
            offload_to_cpu=False,
            compute_device=self.device,
        )
        index = self.state["num_frames"]
        # 保留绝对帧号，仅保存当前图像。
        self.state["images"] = {index: tensor}
        self.state["cached_features"] = {}
        self.state["num_frames"] = index + 1
        return index

    @staticmethod
    def _masks(result):
        _, ids, logits = result
        return dict(zip(ids, (logits[:, 0] > 0).detach().cpu().numpy()))
