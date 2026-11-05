"""物体候选预览、显式确认与选择产物；不持有分割模型。"""

from ..io import write_npz, write_selection
from ..viewer import display_available


def overlay(rgb, candidate, index):
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = rgb.copy()
    mask = candidate["mask"]
    canvas[mask] = (canvas[mask] * 0.4 + np.array([30, 220, 100]) * 0.6).astype(
        np.uint8
    )
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        tuple(float(x) for x in candidate["box_xyxy"]), outline=(255, 180, 0), width=2
    )
    caption = f"#{index} {candidate['label']}  detection={candidate['score']:.3f}"
    draw.rectangle((0, 0, image.width, 22), fill=(20, 20, 20))
    draw.text((5, 4), caption, fill=(255, 255, 255))
    return image


def save_candidates(output, rgb, candidates):
    import numpy as np
    from PIL import Image

    write_npz(
        output / "candidates.npz",
        {
            "masks": np.stack([c["mask"] for c in candidates]),
            "boxes": np.stack([c["box_xyxy"] for c in candidates]),
            "scores": np.asarray([c["score"] for c in candidates]),
            "mask_scores": np.asarray([c["mask_score"] for c in candidates]),
            "labels": np.asarray([c["label"] for c in candidates], dtype=str),
        },
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
        print(
            f"候选 {index}: {candidate['label']}，检测分数 {candidate['score']:.3f}",
            flush=True,
        )
    sheet.save(output / "candidates.jpg", quality=95)


def choose(rgb, candidates, instance, *, headless, name="object"):
    if not candidates:
        raise RuntimeError(f"{name}: 没有检测到目标，请调整 prompt 后重试")
    if instance is not None and not 0 <= instance < len(candidates):
        raise ValueError(f"--instance 必须位于 0..{len(candidates) - 1}")
    if headless:
        if instance is not None:
            return instance
        if len(candidates) == 1:
            return 0
        raise ValueError(f"{name}: 多个候选需要明确 --instance（0..{len(candidates) - 1}）")
    display_available()
    import cv2
    import numpy as np

    index = 0 if instance is None else instance
    title = f"Select {name}: n/p, 1-9 or click; Enter confirm; Esc cancel"

    def mouse(event, x, y, flags, param):
        nonlocal index
        if event == cv2.EVENT_LBUTTONDOWN and (
            0 <= x < rgb.shape[1] and 0 <= y < rgb.shape[0]
        ):
            index = next(
                (
                    i
                    for i, candidate in enumerate(candidates)
                    if candidate["mask"][y, x]
                ),
                index,
            )

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(title, min(rgb.shape[1], 1280), min(rgb.shape[0], 850))
        cv2.setMouseCallback(title, mouse)
        while True:
            canvas = np.asarray(overlay(rgb, candidates[index], index))
            cv2.imshow(title, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(50) & 0xFF
            if key in (10, 13):
                return index
            if (
                key in (27, ord("q"))
                or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
            ):
                raise KeyboardInterrupt("已取消候选选择")
            if key in (ord("n"), ord(" "), ord("]")):
                index = (index + 1) % len(candidates)
            elif key in (ord("p"), ord("[")):
                index = (index - 1) % len(candidates)
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(candidates):
                index = key - ord("1")
    finally:
        cv2.destroyWindow(title)


def save_selection(output, rgb, candidates, index):
    candidate = candidates[index]
    mask = candidate["mask"]
    if not mask.any():
        raise ValueError("没有选中前景；请重新选择或分割")
    overlay(rgb, candidate, index).save(output / "overlay.png")
    write_selection(
        output,
        mask,
        {
            "selected_instance": index,
            "label": candidate["label"],
            "box_xyxy": candidate["box_xyxy"].tolist(),
            "detection_score": candidate["score"],
            "mask_score": candidate["mask_score"],
        },
        mask_file="mask.png",
        foreground_rgb=rgb,
    )
    print(f"已确认候选 {index}；全分辨率 mask、RGBA 与 overlay：{output}", flush=True)
