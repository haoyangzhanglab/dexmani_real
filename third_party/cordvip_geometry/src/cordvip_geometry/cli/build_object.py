#!/usr/bin/env python3
"""单图分割、重建与显式定尺度；build 串行运行子进程，阶段结束释放 GPU。"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..io import existing_file, output_path, write_json
from .arguments import parse_object_args
from .object_selection import choose, display_available, save_candidates, save_selection

logger = logging.getLogger(__name__)


def load_rgb(args):
    import numpy as np
    from PIL import Image, ImageOps

    path = existing_file(args.image if args.image is not None else args.rgbd)
    info = {"source_file": str(path)}
    if args.image is not None:
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        info.update(input_type="image", exif_orientation="applied_to_saved_rgb")
    else:
        from ..rgbd import load_frame

        frame = load_frame(path)
        rgb = frame.rgb
        info.update(
            input_type="saved_rgbd",
            source_id=frame.source_id,
            timestamp_ns=frame.timestamp_ns,
            clock=frame.clock,
            depth_usage="preserved_only; no_automatic_scale_estimation",
            mask_grid="original_rgb_grid; not_automatically_rectified",
        )
    info["rgb_shape"] = list(rgb.shape)
    return rgb, path, info


def preview_mesh(mesh_path, *, units="unscaled", points=None, normals=None):
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
    print(f"mesh 轴向尺寸 x/y/z ({units}): {extent.tolist()}；红 x、绿 y、蓝 z。", flush=True)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(extent.max()) * 0.35)
    geometries = [mesh, bounds, frame]
    if points is not None:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.normals = o3d.utility.Vector3dVector(normals)
        cloud.paint_uniform_color([0.9, 0.2, 0.1])
        geometries.append(cloud)
    title = "Object mesh + surface points" if points is not None else "Object mesh"
    o3d.visualization.draw_geometries(
        geometries, window_name=f"{title} ({units}); x=red y=green z=blue", mesh_show_back_face=True
    )


class ObjectModelBuilder:
    """Own explicit segmentation, selection, reconstruction and asset preparation workflows."""

    def __init__(self, args):
        self.args = args

    def run(self):
        if self.args.command == "segment":
            self.segment()
        elif self.args.command == "select":
            self.select()
        elif self.args.command == "generate":
            self.generate()
        elif self.args.command == "build":
            self.build()
        else:
            preview_mesh(existing_file(self.args.mesh), units=self.args.units)

    def segment(self):
        if self.args.headless and self.args.instance is None and not self.args.list_only:
            raise ValueError("headless 须提供 --instance N；未知索引时先使用 --list-only")
        if not self.args.headless and not self.args.list_only:
            display_available()
        rgb, source, provenance = load_rgb(self.args)
        output = output_path(self.args.output_dir, workspace=self.args.workspace)
        output.mkdir(parents=True)
        from PIL import Image

        Image.fromarray(rgb).save(output / "rgb.png")
        source_name = (
            "source_rgbd.npz" if self.args.rgbd is not None else f"source{source.suffix.lower()}"
        )
        shutil.copyfile(source, output / source_name)
        provenance.update(
            method="Grounded-SAM2 text segmentation",
            prompt=self.args.text,
            sam_root=str(self.args.sam_root.expanduser().resolve()),
            sam_python=str(self.args.sam_python.expanduser().absolute()),
            sam_checkpoint=str(self.args.sam_checkpoint.expanduser().resolve()),
            sam_config=self.args.sam_config,
            dino_weights=str(self.args.dino_weights.expanduser().resolve()),
            box_threshold=self.args.box_threshold,
            text_threshold=self.args.text_threshold,
            device="cuda:0",
            headless=self.args.headless,
            saved_source=source_name,
            candidate_order="descending_grounding_dino_score; zero_based",
        )
        write_json(output / "segmentation_request.json", provenance)
        from ..segmentation import GroundedSAM2Segmenter

        self.segmenter = GroundedSAM2Segmenter(
            self.args.sam_python,
            self.args.sam_root,
            self.args.sam_checkpoint,
            sam_config=self.args.sam_config,
            dino_weights=self.args.dino_weights,
            log_path=output / "sam2_worker.log",
        )
        try:
            self.segmenter.start()
            candidates = self.segmenter.predict(
                rgb,
                self.args.text,
                box_threshold=self.args.box_threshold,
                text_threshold=self.args.text_threshold,
            )
            provenance.update(
                model=self.segmenter.provenance, performance=self.segmenter.last_metrics
            )
        except BaseException:
            try:
                self.segmenter.close()
            except BaseException:
                logger.exception("SAM2 cleanup also failed")
            raise
        self.segmenter.close()
        if not candidates:
            write_json(output / "segmentation_result.json", {**provenance, "num_candidates": 0})
            raise RuntimeError(f"没有检测到目标；原图和请求已保留在 {output}，可调整文本或阈值")
        save_candidates(output, rgb, candidates)
        provenance.update(num_candidates=len(candidates))
        write_json(output / "segmentation_result.json", provenance)
        if self.args.list_only:
            print(f"候选已保存：{output / 'candidates.jpg'}；使用 select 确认，无需重跑模型。")
            return
        index = choose(rgb, candidates, self.args.instance, headless=self.args.headless)
        save_selection(output, rgb, candidates, index, provenance)

    def select(self):
        import numpy as np
        from PIL import Image

        source = self.args.segmentation_dir.expanduser().resolve()
        output = output_path(
            self.args.output_dir, workspace=self.args.workspace, protected=(source,)
        )
        provenance = json.loads(existing_file(source / "segmentation_result.json").read_text())
        with Image.open(existing_file(source / "rgb.png")) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        with np.load(existing_file(source / "candidates.npz"), allow_pickle=False) as saved:
            candidates = [
                dict(
                    mask=mask,
                    box_xyxy=saved["boxes"][i],
                    score=float(saved["scores"][i]),
                    mask_score=float(saved["mask_scores"][i]),
                    label=str(saved["labels"][i]),
                )
                for i, mask in enumerate(saved["masks"])
            ]
        if not candidates or any(
            c["mask"].dtype != np.bool_ or c["mask"].shape != rgb.shape[:2] for c in candidates
        ):
            raise ValueError("保存的候选 mask 与 RGB 不一致")
        index = choose(rgb, candidates, self.args.instance, headless=self.args.headless)
        output.mkdir(parents=True)
        for name in (
            "rgb.png",
            "candidates.npz",
            "candidates.jpg",
            "segmentation_request.json",
            "segmentation_result.json",
            provenance["saved_source"],
        ):
            file = (source / name).resolve()
            if not file.is_relative_to(source):
                raise ValueError("候选元数据不能引用输入目录以外的文件")
            shutil.copyfile(existing_file(file), output / name)
        provenance.update(headless=self.args.headless, selection_source=str(source))
        save_selection(output, rgb, candidates, index, provenance)

    def generate(self):
        if self.args.preview:
            display_available()
        source = self.args.segmentation_dir.expanduser().resolve()
        metadata = json.loads(existing_file(source / "selection.json").read_text())
        if metadata["selection"] not in {"explicit_headless_instance", "gui_confirmed"}:
            raise ValueError("建模需要已明确确认的分割")
        image = existing_file(source / "foreground_rgba.png")
        output = output_path(
            self.args.output_dir, workspace=self.args.workspace, protected=(source,)
        )
        output.mkdir(parents=True)
        write_json(
            output / "generation_request.json",
            {
                "method": "Hunyuan3D-2.1 shape_only",
                "segmentation_directory": str(source),
                "selected_instance": metadata["selected_instance"],
                "source_root": str(self.args.hunyuan_root.expanduser().resolve()),
                "weights_dir": str(self.args.weights.expanduser().resolve()),
                "seed": self.args.seed,
                "steps": self.args.steps,
                "octree_resolution": self.args.octree_resolution,
                "num_chunks": self.args.num_chunks,
                "device": self.args.device,
                "units": "unscaled",
                "num_points": self.args.num_points,
            },
        )
        from ..reconstruction import generate_mesh

        mesh_path = output / "mesh_unscaled.obj"
        result = generate_mesh(
            image_rgba_path=image,
            source_root=self.args.hunyuan_root,
            weights_dir=self.args.weights,
            output_mesh_path=mesh_path,
            seed=self.args.seed,
            steps=self.args.steps,
            octree_resolution=self.args.octree_resolution,
            num_chunks=self.args.num_chunks,
            device=self.args.device,
        )
        existing_file(mesh_path)
        from ..geometry import load_metric_mesh, sample_mesh_surface, save_points

        # 保留生成模型原点、朝向和任意单位；单张 RGB 不能给出绝对米制尺度。
        mesh = load_metric_mesh(mesh_path, scale_to_meters=1.0)
        points, normals = sample_mesh_surface(mesh, self.args.num_points, self.args.seed)
        points_path, normals_path = output / "points_unscaled.npy", output / "normals.npy"
        save_points(points_path, points)
        save_points(normals_path, normals)
        write_json(
            output / "generation_result.json",
            {
                **result,
                "mesh_file": mesh_path.name,
                "units": "unscaled",
                "frame": "object",
                "points_file": points_path.name,
                "normals_file": normals_path.name,
                "num_points": self.args.num_points,
                "seed": self.args.seed,
                "sampling": "area_weighted_triangle_barycentric",
                "normals_method": "mesh_face",
                "origin": "source_mesh_origin_without_recentering",
            },
        )
        print(
            f"未定尺度 mesh：{mesh_path}；表面点云：{points_path}。准备追踪资产前必须提供实测尺寸或明确缩放系数。",
            flush=True,
        )
        if self.args.preview:
            preview_mesh(mesh_path, points=points, normals=normals)

    def build(self):
        if self.args.headless and self.args.preview:
            raise ValueError("--headless 与 --preview 不能同时使用")
        if (self.args.size_axis is None) != (self.args.size_meters is None):
            raise ValueError("--size-axis 与 --size-meters 必须一起提供")
        if self.args.scale_to_meters is not None and self.args.size_axis is not None:
            raise ValueError("--scale-to-meters 与 --size-axis 不能同时使用")
        if self.args.headless and self.args.instance is None:
            raise ValueError("headless 必须显式提供 --instance N")
        output = output_path(self.args.output_dir, workspace=self.args.workspace)
        output.mkdir(parents=True)
        # Fresh model processes need the source directory even when only parent sys.path was edited.
        environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[2])
        current = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source + (os.pathsep + current if current else "")
        command_prefix = [sys.executable, "-m", "cordvip_geometry", "object"]
        workspace_prefix = f"cd {shlex.quote(str(self.args.workspace))} && "
        segment_command = [*command_prefix, "segment", "--output-dir", str(output / "segmentation")]
        for name in (
            "image",
            "rgbd",
            "text",
            "sam_python",
            "sam_root",
            "sam_checkpoint",
            "sam_config",
            "dino_weights",
            "box_threshold",
            "text_threshold",
            "instance",
        ):
            value = getattr(self.args, name)
            if value is not None:
                if isinstance(value, Path):
                    value = (
                        value.expanduser().absolute()
                        if name == "sam_python"
                        else value.expanduser().resolve()
                    )
                segment_command.extend(["--" + name.replace("_", "-"), str(value)])
        if self.args.headless:
            segment_command.append("--headless")
        generate_command = [
            *command_prefix,
            "generate",
            "--segmentation-dir",
            str(output / "segmentation"),
            "--output-dir",
            str(output / "generation"),
        ]
        for name in (
            "hunyuan_root",
            "weights",
            "seed",
            "steps",
            "octree_resolution",
            "num_chunks",
            "num_points",
            "device",
        ):
            value = getattr(self.args, name)
            if isinstance(value, Path):
                value = value.expanduser().resolve()
            generate_command.extend(["--" + name.replace("_", "-"), str(value)])
        if self.args.preview:
            generate_command.append("--preview")
        for command in (segment_command, generate_command):
            print(workspace_prefix + shlex.join(command), flush=True)
            subprocess.run(command, check=True, env=environment, cwd=self.args.workspace)
        mesh = output / "generation" / "mesh_unscaled.obj"
        prepare_command = [
            sys.executable,
            "-m",
            "cordvip_geometry",
            "prepare",
            "--mesh",
            str(mesh),
            "--output-dir",
            str(output / "asset"),
            "--num-points",
            str(self.args.num_points),
            "--seed",
            str(self.args.seed),
        ]
        if self.args.scale_to_meters is not None:
            prepare_command.extend(["--scale-to-meters", str(self.args.scale_to_meters)])
        elif self.args.size_axis is not None:
            prepare_command.extend(
                ["--size-axis", self.args.size_axis, "--size-meters", str(self.args.size_meters)]
            )
        else:
            print("mesh 与表面点云已生成；尚未指定米制尺度，因此没有生成可用于追踪的资产。")
            print("检查 mesh 坐标轴后运行（替换 AXIS 与 MEASURED_METERS）：")
            print(
                workspace_prefix
                + shlex.join(prepare_command)
                + " --size-axis AXIS --size-meters MEASURED_METERS --preview"
            )
            return
        if self.args.preview:
            prepare_command.append("--preview")
        subprocess.run(prepare_command, check=True, env=environment, cwd=self.args.workspace)
        live = [
            sys.executable,
            "-m",
            "cordvip_geometry",
            "track",
            "--asset",
            str(output / "asset"),
            "--object-prompt",
            self.args.text,
        ]
        print("资产已准备。以下命令会连接 RealSense，确认相机可用后手动运行：", flush=True)
        print(workspace_prefix + shlex.join(live), flush=True)


def main(argv=None, *, workspace=None):
    args = parse_object_args(argv, workspace=workspace)
    ObjectModelBuilder(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
