"""物体重建使用的 Mesh 预览。"""

from __future__ import annotations

import os
import sys


def display_available():
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError("没有图形会话；请使用 --headless 或 --list-only")


def preview_mesh(mesh_path, *, units="unscaled"):
    display_available()
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
    geometries = [mesh, bounds, frame]
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Object mesh ({units}); x=red y=green z=blue",
        mesh_show_back_face=True,
    )
