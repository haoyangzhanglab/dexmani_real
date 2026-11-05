"""读取离线追踪的逐行结果；不打开 Raw、相机或感知模型。"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path


def main(argv=None, *, workspace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result", type=Path, help="track 输出目录，含 scene.json 和 frames/"
    )
    parser.add_argument(
        "--no-3d", action="store_true", help="只看 RGB 叠加，不加载 Open3D"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="只读取 RGB 和可选来源说明，不加载 OpenCV/Open3D 或打开窗口",
    )
    args = parser.parse_args(argv)
    return view_results(args.result, show_3d=not args.no_3d, headless=args.headless)


def view_results(directory, *, show_3d=True, headless=False):
    import numpy as np

    directory = Path(directory).expanduser().resolve()
    paths = sorted((directory / "frames").glob("*.npz"))
    if not paths:
        raise ValueError("scene result contains no saved frames")
    viewer = None
    show_model = False
    model_templates = None
    count = 0
    if not headless:
        from ..viewer import OBJECT_COLORS, SceneViewer, scene_clouds, scene_display

        scene = json.loads((directory / "scene.json").read_text())
        names = [item["id"] for item in scene["objects"]]
        model_names = [item["id"] for item in scene["objects"] if item.get("asset")]
        colors = {name: OBJECT_COLORS[i % len(OBJECT_COLORS)] for i, name in enumerate(names)}
        viewer = SceneViewer(show_3d=show_3d)
    with viewer if viewer is not None else nullcontext() as viewer:
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                # 每个显示字段只解压一次；分析用大数组留在 NPZ 中。
                frame = {
                    key: saved[key]
                    for key in ("row_index", "timestamp_ns", "clock") if key in saved.files
                }
                frame["rgb"] = saved["rgb"]
                if viewer is not None:
                    frame["K"] = saved["K"]
                    for name in names:
                        prefix = name + "__"
                        for field in ("semantic_mask", "pose_valid", "state", "error"):
                            frame[prefix + field] = saved[prefix + field]
                        if bool(frame[prefix + "pose_valid"]):
                            frame[prefix + "T_camera_object"] = saved[
                                prefix + "T_camera_object"
                            ]
                        if show_3d:
                            frame[prefix + "observed_points_camera"] = saved[
                                prefix + "observed_points_camera"
                            ]
                if (
                    show_3d
                    and viewer is not None
                    and "hand_points_camera" in saved.files
                ):
                    frame["hand_points_camera"] = saved["hand_points_camera"]
            rgb = frame["rgb"]
            if (
                rgb.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[2] != 3
                or min(rgb.shape[:2]) == 0
            ):
                raise ValueError("scene result RGB must be nonempty uint8 [H,W,3]")
            details = " ".join(f"{key}={frame[key].item()}" for key in ("row_index", "timestamp_ns", "clock") if key in frame)
            print(f"{path.name}" + (f": {details}" if details else ""))
            if viewer is not None:
                viewer.update(scene_display(frame, names, colors))
                if show_3d:
                    points, point_colors = scene_clouds(
                        frame,
                        names,
                        colors,
                        model_templates=model_templates if show_model else None,
                    )
                    viewer.update_cloud(points, point_colors)
            while viewer is not None:
                key = viewer.poll()
                if key in (ord("q"), ord("Q"), 27):
                    return 0
                if key == ord("v"):
                    show_model = not show_model
                    if show_3d:
                        if show_model and model_templates is None:
                            model_templates = {}
                            for name in model_names:
                                with np.load(
                                    directory / "model_templates" / f"{name}.npz",
                                    allow_pickle=False,
                                ) as saved:
                                    points = saved["points_object"]
                                if (
                                    points.dtype != np.float32
                                    or points.ndim != 2
                                    or points.shape[1] != 3
                                    or not len(points)
                                    or not np.isfinite(points).all()
                                ):
                                    raise ValueError(
                                        f"{name}: template points must be nonempty finite float32 [N,3]"
                                    )
                                model_templates[name] = points
                        points, point_colors = scene_clouds(
                            frame,
                            names,
                            colors,
                            model_templates=model_templates if show_model else None,
                        )
                        viewer.update_cloud(points, point_colors)
                elif key in (ord("n"), ord(" "), 13):
                    break
            count += 1
    print(f"已读取 {count} 行场景结果：{directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
