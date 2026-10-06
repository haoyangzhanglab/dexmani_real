"""Resolve the current EEF-to-hand mount without changing model assets."""

import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.utils.geometry import validate_unit_quaternion_wxyz


def urdf_with_hand_mount(path, position, quaternion, *, mesh_directory=None):
    """Return XML with current custom_eef_link -> right_hand_link (m, wxyz)."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    joints = root.findall("./joint[@name='right_hand_mount_joint']")
    if len(joints) != 1:
        raise ValueError("URDF requires exactly one right_hand_mount_joint")
    joint = joints[0]
    if (
        joint.get("type") != "fixed"
        or joint.find("parent") is None
        or joint.find("child") is None
        or joint.find("parent").get("link") != "custom_eef_link"
        or joint.find("child").get("link") != "right_hand_link"
    ):
        raise ValueError("unexpected hand mount joint topology")
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("hand mount position must be finite (3,) meters")
    quaternion = validate_unit_quaternion_wxyz(quaternion)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected", category=UserWarning)
        rpy = Rotation.from_quat(np.roll(quaternion, -1)).as_euler("xyz")
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    for name, values in (("xyz", position), ("rpy", rpy)):
        origin.set(name, " ".join(format(float(x), ".17g") for x in values))
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute() and "://" not in filename:
            resolved = (path.parent / filename).resolve()
            # MPLib 0.2.1 prepends the URDF directory even to absolute filenames.
            mesh.set(
                "filename",
                str(resolved)
                if mesh_directory is None
                else os.path.relpath(resolved, mesh_directory),
            )
    return ET.tostring(root, encoding="unicode")
