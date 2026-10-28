"""Rerun 显示函数；采集、时间轴和数值处理由调用方负责。"""

import numpy as np


def viewer_blueprint(series_groups=(), *, camera_in_base=False):
    import rerun.blueprint as rrb

    spatial = ["/pcd/**", "/eef/**", "/fingertips/**", "/workspace/**", "/base/**"]
    if camera_in_base:
        spatial.append("/camera/**")
    views: list[rrb.Container | rrb.View] = [
        rrb.Vertical(
            rrb.Spatial2DView(origin="camera/color/rgb", name="RGB"),
            rrb.Spatial2DView(origin="depth/image", name="Depth"),
            name="Camera",
        ),
        rrb.Spatial3DView(
            origin="/", contents=spatial, name="xArm-base", background=[0.12, 0.12, 0.14],
        ),
    ]
    if series_groups:
        tabs = []
        for name, paths in series_groups:
            curves = [
                rrb.TimeSeriesView(origin=path, name=path.rsplit("/", 1)[-1]) for path in paths
            ]
            tabs.append(rrb.Vertical(contents=curves, name=name))
        views.append(rrb.Tabs(contents=tabs, active_tab=0, name="Time Series"))
    return rrb.Blueprint(rrb.Horizontal(contents=views))


def log_camera_geometry(intrinsics, base_from_color=None):
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


def log_rgbd(rgb, depth_raw, *, depth_scale_m, depth_range_m=None):
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


def log_points(path, points, *, colors=None, radii=0.003):
    import rerun as rr

    if points is None or not len(points) or not np.isfinite(points).all():
        rr.log(path, rr.Clear(recursive=False))
        return
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        raise ValueError("display points must have shape (N, 3) or (N, 6)")

    if points.shape[1] == 6:
        # XYZRGB 中的颜色已归一化，此处恢复为显示格式。
        colors = (points[:, 3:] * 255.0).astype(np.uint8)
    rr.log(path, rr.Points3D(points[:, :3], colors=colors, radii=radii))


def log_series(path, values=None, *, labels=None):
    import rerun as rr

    if labels is not None:
        for index, label in enumerate(labels):
            rr.log(f"{path}/{index}", rr.SeriesLine(name=label), static=True)

    if values is not None:
        for index, value in enumerate(np.atleast_1d(values)):
            rr.log(f"{path}/{index}", rr.Scalar(float(value)))


def log_workspace(workspace):
    import rerun as rr

    lower, upper = np.asarray(workspace).reshape(2, 3)
    rr.log(
        "workspace", rr.Boxes3D(mins=lower, sizes=upper - lower, colors=[0, 255, 0]),
        static=True,
    )
    rr.log("base", rr.Transform3D(axis_length=0.2), static=True)
