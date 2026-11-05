"""固定 URDF 表面采样与关节 FK。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .geometry import sample_mesh_surface, validate_pose


def parse_urdf_vector(text, default):
    value = np.asarray(default if text is None else text.split(), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("URDF scale/origin must contain three finite numbers")
    return value


class HandCloud:
    def __init__(
        self,
        urdf_path,
        joint_names,
        excluded_links=(),
        num_points=1024,
        seed=0,
        *,
        T_camera_from_root,
    ):
        for name, value, minimum in (("num_points", num_points, 1), ("seed", seed, 0)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.T_camera_from_root = validate_pose(T_camera_from_root)
        import pinocchio as pin
        import trimesh

        self._pin = pin
        path = Path(urdf_path).expanduser().resolve()
        self.joint_names = tuple(joint_names)
        self._model = pin.buildModelFromUrdf(str(path))
        self._data = self._model.createData()
        active = {
            str(self._model.names[i]): joint
            for i, joint in enumerate(self._model.joints)
            if i > 0 and joint.nq > 0
        }
        if (
            not self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
            or set(self.joint_names) != set(active)
            or any(joint.nq != 1 for joint in active.values())
        ):
            raise ValueError(
                "joint_names must name every one-DoF URDF joint exactly once"
            )
        self._q_indices = np.asarray([active[name].idx_q for name in self.joint_names])
        root = ET.parse(path).getroot()
        excluded = set(excluded_links)
        unknown = excluded - {link.attrib["name"] for link in root.findall("link")}
        if unknown:
            raise ValueError(f"excluded links are absent from URDF: {sorted(unknown)}")
        meshes, names = [], []
        for link in root.findall("link"):
            name = link.attrib["name"]
            if name in excluded:
                continue
            parts = []
            for visual in link.findall("visual"):
                mesh_tag = visual.find("geometry/mesh")
                if mesh_tag is None:
                    raise ValueError(
                        f"{name}: only explicit visual triangle meshes are supported"
                    )
                filename = mesh_tag.get("filename", "")
                if not filename or "://" in filename:
                    raise ValueError(f"{name}: mesh must use a local filesystem path")
                mesh_path = (path.parent / filename).resolve()
                mesh = trimesh.load(str(mesh_path), process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise TypeError(f"{mesh_path}: expected a single triangle mesh")
                scale = parse_urdf_vector(mesh_tag.get("scale"), (1, 1, 1))
                if np.any(scale <= 0):
                    raise ValueError(f"{name}: mesh scale must be positive")
                mesh.apply_scale(scale)
                origin = visual.find("origin")
                transform = np.eye(4)
                if origin is not None:
                    transform[:3, 3] = parse_urdf_vector(origin.get("xyz"), (0, 0, 0))
                    transform[:3, :3] = pin.rpy.rpyToMatrix(
                        parse_urdf_vector(origin.get("rpy"), (0, 0, 0))
                    )
                mesh.apply_transform(transform)
                if (
                    not np.isfinite(mesh.vertices).all()
                    or not np.isfinite(mesh.area)
                    or mesh.area <= 0
                ):
                    raise ValueError(
                        f"{mesh_path}: mesh must have finite, positive surface area"
                    )
                parts.append(mesh)
            if parts:
                names.append(name)
                meshes.append(trimesh.util.concatenate(parts))
        if not meshes or num_points < len(meshes):
            raise ValueError(
                "num_points must cover every included visual link at least once"
            )
        self._frame_ids = [self._model.getFrameId(name) for name in names]
        areas = np.asarray([mesh.area for mesh in meshes])
        allocation = (int(num_points) - len(meshes)) * areas / areas.sum()
        counts = np.floor(allocation).astype(int) + 1
        remainder = int(num_points) - int(counts.sum())
        counts[
            np.argsort(-(allocation - np.floor(allocation)), kind="stable")[:remainder]
        ] += 1
        points, normals, link_ids = [], [], []
        for index, (mesh, count) in enumerate(zip(meshes, counts)):
            xyz, normal = sample_mesh_surface(mesh, int(count), seed=int(seed) + index)
            points.append(xyz)
            normals.append(normal)
            link_ids.append(np.full(int(count), index, dtype=np.int32))
        self.points = np.concatenate(points)
        self._normals = np.concatenate(normals)
        self.link_ids = np.concatenate(link_ids)
        self.link_ids.setflags(write=False)
        self.metadata = {
            "joint_names": list(self.joint_names),
            "joint_units": "rad",
            "link_names": names,
            "units": "m",
        }

    def transform(self, q):
        if np.iscomplexobj(q):
            raise ValueError("joint angles must be real")
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("joint angles must be finite and match joint_names")
        configuration = np.empty(self._model.nq)
        configuration[self._q_indices] = q
        self._pin.forwardKinematics(self._model, self._data, configuration)
        self._pin.updateFramePlacements(self._model, self._data)
        transforms = np.stack(
            [
                self.T_camera_from_root @ self._data.oMf[index].homogeneous
                for index in self._frame_ids
            ]
        )[self.link_ids]
        points = (
            np.einsum("nij,nj->ni", transforms[:, :3, :3], self.points)
            + transforms[:, :3, 3]
        )
        normals = np.einsum("nij,nj->ni", transforms[:, :3, :3], self._normals)
        return {
            "points_camera": points.astype(np.float32),
            "normals_camera": normals.astype(np.float32),
            "link_ids": self.link_ids,
        }
