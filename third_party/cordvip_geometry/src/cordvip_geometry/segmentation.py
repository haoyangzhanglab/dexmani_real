"""Grounding DINO 检测与 SAM2 单图分割。"""

import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


def normalize_prompt(text):
    """统一检测与跟踪的描述格式，拒绝空白或纯标点输入。"""
    if not isinstance(text, str) or all(
        char.isspace() or unicodedata.category(char).startswith("P") for char in text
    ):
        raise ValueError("Prompts must contain nonempty object descriptions")
    return text.strip().lower().rstrip(". ") + "."


class SAM2Segmenter:
    def __init__(self, root, checkpoint, config, dino_weights):
        root = Path(root).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        sys.path.insert(0, str(root))
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not torch.cuda.is_available():
            raise RuntimeError("Grounded-SAM2 requires CUDA")
        self.device = torch.device("cuda:0")
        model = build_sam2(
            config,
            str(checkpoint),
            device=str(self.device),
            apply_postprocessing=False,
        )
        self.image_predictor = SAM2ImagePredictor(model)
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

        normalized = [label.strip().lower().rstrip(". ") for label in labels]
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
        candidates = self.detect(rgb, text, box_threshold, text_threshold)
        if not candidates:
            return []
        masks = self.segment_boxes(
            rgb, {i: item["box_xyxy"] for i, item in enumerate(candidates)}
        )
        return [
            {**item, "mask": masks[i]}
            for i, item in enumerate(candidates)
            if masks[i].any()
        ]

    def segment_boxes(self, rgb, boxes, *, reuse_image=False):
        """按对象名返回 mask。同一帧增加框提示时可复用已计算的图像特征。"""
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if not reuse_image:
                self.image_predictor.set_image(np.ascontiguousarray(rgb))
            masks, _, _ = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.stack(list(boxes.values())),
                multimask_output=False,
            )
        return {
            name: np.asarray(mask).reshape(rgb.shape[:2]).astype(bool)
            for name, mask in zip(boxes, masks)
        }
