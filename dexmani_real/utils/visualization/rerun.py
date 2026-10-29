"""Rerun 显示函数；采集、时间轴和数值处理由调用方负责。"""

from collections.abc import Iterable, Sequence

import numpy as np

from dexmani_real.geometry.camera import CameraIntrinsics


def episode_blueprint(
    series_groups: Iterable[tuple[str, Sequence[str]]], *, camera_in_base: bool = False,
):
    import rerun.blueprint as rrb

    spatial = ["/pcd/**", "/eef/**", "/fingertips/**"]
    if camera_in_base:
        spatial.append("/camera/**")
    tabs = [
        rrb.Vertical(
            contents=[rrb.TimeSeriesView(origin=path, name=path.rsplit("/", 1)[-1]) for path in paths],
            name=name,
        )
        for name, paths in series_groups
    ]
    return rrb.Blueprint(rrb.Horizontal(
        rrb.Vertical(
            rrb.Tabs(
                rrb.Spatial3DView(origin="/", contents=spatial, name="xArm-base"),
                rrb.Spatial3DView(origin="/", contents=["/pcd/**"], name="Point Cloud"),
                active_tab=0,
            ),
            rrb.Spatial2DView(origin="camera/color/rgb", name="RGB"),
            name="3D + RGB",
        ),
        rrb.Spatial2DView(origin="depth/image", name="Depth"),
        rrb.Tabs(contents=tabs, active_tab=0, name="Time Series"),
        column_shares=[2, 2, 3],
    ))


def log_camera_geometry(
    intrinsics: CameraIntrinsics, base_from_color: np.ndarray | None = None,
) -> None:
    import rerun as rr

    rr.log(
        "camera/color",
        rr.Pinhole(
            image_from_camera=intrinsics.matrix(),
            resolution=[intrinsics.width, intrinsics.height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.25,
        ),
        static=True,
    )
    if base_from_color is not None:
        rr.log(
            "camera/color",
            rr.Transform3D(
                translation=base_from_color[:3, 3], mat3x3=base_from_color[:3, :3],
                axis_length=0.15,
            ),
            static=True,
        )


def log_rgbd(
    rgb: np.ndarray, depth_raw: np.ndarray, *, depth_scale_m: float,
    depth_range_m: tuple[float, float] | None = None,
) -> None:
    import rerun as rr

    rr.log("camera/color/rgb", rr.Image(rgb, color_model="RGB"))
    rr.log(
        "depth/image",
        rr.DepthImage(
            depth_raw, meter=1.0 / depth_scale_m,
            depth_range=(
                np.asarray(depth_range_m) / depth_scale_m if depth_range_m is not None else None
            ),
        ),
    )


def log_points(path: str, points: np.ndarray | None, *, colors=None, radii=0.003) -> None:
    """记录调用方提供的有效 XYZ/XYZRGB；None 和空数组清除已有点。"""
    import rerun as rr

    if points is None or not len(points):
        rr.log(path, rr.Clear(recursive=False))
        return

    if points.shape[1] == 6:
        # XYZRGB 中的颜色已归一化，此处恢复为显示格式。
        colors = (points[:, 3:] * 255.0).astype(np.uint8)
    rr.log(path, rr.Points3D(points[:, :3], colors=colors, radii=radii))


def log_series(
    path: str, values: np.ndarray, *, labels: Sequence[str], steps: np.ndarray,
    timestamps_s: np.ndarray,
) -> None:
    """批量发送一组曲线；双时间轴由数据来源显式提供。"""
    import rerun as rr

    indexes = [rr.TimeSequenceColumn("step", steps), rr.TimeSecondsColumn("time", timestamps_s)]
    for index, label in enumerate(labels):
        entity = f"{path}/{index}"
        rr.log(entity, rr.SeriesLine(name=label), static=True)
        rr.send_columns(entity, indexes=indexes, columns=rr.Scalar.columns(scalar=values[:, index]))
