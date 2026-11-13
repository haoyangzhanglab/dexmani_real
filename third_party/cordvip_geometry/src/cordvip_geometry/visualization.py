"""网格预览与分割候选选择。"""

import os
import sys


def require_display():
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError("没有图形会话；请使用 --headless 或 --list-only")


def preview_mesh(mesh_path, *, units="unscaled"):
    require_display()
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("预览 mesh 为空或没有三角面")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.65, 0.78, 0.87])
    bounds = mesh.get_axis_aligned_bounding_box()
    bounds.color = [0.85, 0.65, 0.15]
    extent = bounds.get_extent()
    print(
        f"mesh 轴向尺寸 x/y/z ({units}): {extent.tolist()}；红 x、绿 y、蓝 z。",
        flush=True,
    )
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(extent.max()) * 0.35
    )
    o3d.visualization.draw_geometries(
        [mesh, bounds, frame],
        window_name=f"Object mesh ({units}); x=red y=green z=blue",
        mesh_show_back_face=True,
    )


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

    np.savez_compressed(
        output / "candidates.npz",
        masks=np.stack([c["mask"] for c in candidates]),
        boxes=np.stack([c["box_xyxy"] for c in candidates]),
        scores=np.asarray([c["score"] for c in candidates]),
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
        print(
            f"候选 {index}: {candidate['label']}，检测分数 {candidate['score']:.3f}",
            flush=True,
        )
    sheet.save(output / "candidates.jpg", quality=95)


def choose(rgb, candidates, instance, *, headless):
    if not candidates:
        raise RuntimeError("没有检测到目标，请调整 prompt 后重试")
    if instance is not None and not 0 <= instance < len(candidates):
        raise ValueError(f"--instance 必须位于 0..{len(candidates) - 1}")
    if headless:
        if instance is not None:
            return instance
        if len(candidates) == 1:
            return 0
        raise ValueError(f"多个候选需要明确 --instance（0..{len(candidates) - 1}）")
    require_display()
    import cv2
    import numpy as np

    index = 0 if instance is None else instance
    title = "Select object: n/p, 1-9 or click; Enter confirm; Esc cancel"

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
    import numpy as np
    from PIL import Image

    candidate = candidates[index]
    mask = candidate["mask"]
    if not mask.any():
        raise ValueError("没有选中前景；请重新选择或分割")
    overlay(rgb, candidate, index).save(output / "overlay.png")
    alpha = mask.astype(np.uint8) * 255
    Image.fromarray(alpha).save(output / "mask.png")
    Image.fromarray(np.dstack((rgb, alpha))).save(output / "foreground_rgba.png")
    print(f"已确认候选 {index}；全分辨率 mask、RGBA 与 overlay：{output}", flush=True)
