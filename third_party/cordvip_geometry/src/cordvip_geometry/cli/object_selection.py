"""物体候选预览、显式确认与选择产物；不持有分割模型。"""

from __future__ import annotations

import os
import sys

from ..io import write_confirmed_selection


def display_available() -> None:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "没有图形会话；请使用 --headless --instance N，或 --list-only 查看候选后 select"
        )


def overlay(rgb, candidate, index):
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = rgb.copy()
    mask = candidate["mask"]
    canvas[mask] = (canvas[mask] * 0.4 + np.array([30, 220, 100]) * 0.6).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(float(x) for x in candidate["box_xyxy"]), outline=(255, 180, 0), width=2)
    caption = f"#{index} {candidate['label']}  detection={candidate['score']:.3f}"
    draw.rectangle((0, 0, image.width, 22), fill=(20, 20, 20))
    draw.text((5, 4), caption, fill=(255, 255, 255))
    return image


def save_candidates(output, rgb, candidates):
    import numpy as np
    from PIL import Image

    np.savez_compressed(
        output / "candidates.npz",
        masks=np.stack([c["mask"] for c in candidates]),
        boxes=np.stack([c["box_xyxy"] for c in candidates]),
        scores=np.asarray([c["score"] for c in candidates]),
        mask_scores=np.asarray([c["mask_score"] for c in candidates]),
        labels=np.asarray([c["label"] for c in candidates], dtype=str),
    )
    width = min(rgb.shape[1], 420)
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    columns = min(3, len(candidates))
    sheet = Image.new(
        "RGB", (columns * width, ((len(candidates) + columns - 1) // columns) * height)
    )
    for index, candidate in enumerate(candidates):
        thumb = overlay(rgb, candidate, index).resize((width, height))
        sheet.paste(thumb, ((index % columns) * width, (index // columns) * height))
        print(f"候选 {index}: {candidate['label']}，检测分数 {candidate['score']:.3f}", flush=True)
    sheet.save(output / "candidates.jpg", quality=95)


def choose(rgb, candidates, instance, *, headless):
    if instance is not None and not 0 <= instance < len(candidates):
        raise ValueError(f"--instance 必须位于 0..{len(candidates) - 1}；索引参见 candidates.jpg")
    if headless:
        if instance is None:
            raise ValueError("headless 必须显式指定 --instance N；不会按检测分数自动选物体")
        return instance
    display_available()
    import cv2
    import numpy as np

    index = 0 if instance is None else instance
    title = "Grounded-SAM2: n/p candidate; Enter confirm; Esc cancel"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(rgb.shape[1], 1280), min(rgb.shape[0], 850))
    try:
        while True:
            canvas = np.asarray(overlay(rgb, candidates[index], index))
            cv2.imshow(title, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(50) & 0xFF
            if key in (10, 13):
                return index
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("已取消候选确认；候选文件保留，可通过 select 继续")
            if key in (ord("n"), ord(" "), ord("]")):
                index = (index + 1) % len(candidates)
            elif key in (ord("p"), ord("[")):
                index = (index - 1) % len(candidates)
    finally:
        cv2.destroyWindow(title)


def save_selection(output, rgb, candidates, index, provenance):
    candidate = candidates[index]
    mask = candidate["mask"]
    if not mask.any() or mask.all():
        raise ValueError("生成输入必须包含非空物体和非空背景；请重新分割或拍摄")
    overlay(rgb, candidate, index).save(output / "overlay.png")
    write_confirmed_selection(
        output,
        mask,
        {
            **provenance,
            "selected_instance": index,
            "selection": (
                "explicit_headless_instance" if provenance["headless"] else "gui_confirmed"
            ),
            "label": candidate["label"],
            "box_xyxy": candidate["box_xyxy"].tolist(),
            "detection_score": candidate["score"],
            "score_kind": "grounding_dino",
            "mask_score": candidate["mask_score"],
            "mask_score_kind": "sam2_predicted_iou",
        },
        mask_file="mask.png",
        foreground_rgb=rgb,
    )
    print(f"已确认候选 {index}；全分辨率 mask、RGBA 与 overlay：{output}", flush=True)
