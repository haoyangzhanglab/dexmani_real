"""将 GeoRT 的右手 XHand checkpoint 解码为 SDK 顺序的关节角。"""

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_RATED_QPOS_MAX_RAD,
    XHAND_RATED_QPOS_MIN_RAD,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.landmarks import FINGER_CHAINS

GEORT_COMMIT = "fb2075e2a3272be372016580c0b6ebd6120228f0"
XHAND_FINGER_JOINTS = tuple(
    XHAND_SDK_JOINT_NAMES[start:stop] for start, stop in ((0, 3), (3, 6), (6, 8), (8, 10), (10, 12))
)


def _parse_config(config):
    sdk_names = XHAND_SDK_JOINT_NAMES
    joint_order = config["joint_order"]
    if not isinstance(joint_order, list) or len(joint_order) != HAND_DOF or set(joint_order) != set(sdk_names):
        raise ValueError("joint_order must contain each right XHand SDK joint exactly once")
    sdk_indices = [joint_order.index(name) for name in sdk_names]

    bounds = np.asarray((config["joint"]["lower"], config["joint"]["upper"]), dtype=np.float64)
    if bounds.shape != (2, HAND_DOF) or not np.isfinite(bounds).all():
        raise ValueError("joint.lower and joint.upper must each contain 12 finite angles")
    lower, upper = bounds
    if np.any(lower >= upper):
        raise ValueError("joint.lower must be strictly below joint.upper")
    # GeoRT 训练配置中的额定限位可能经过一次 float32 转换。
    rounding = np.finfo(np.float32).eps
    if np.any(lower[sdk_indices] < np.asarray(XHAND_RATED_QPOS_MIN_RAD) - rounding) or np.any(
        upper[sdk_indices] > np.asarray(XHAND_RATED_QPOS_MAX_RAD) + rounding,
    ):
        raise ValueError("training joint limits exceed the rated XHand range in radians")

    expected = {
        tip_id: (link, set(names))
        for tip_id, link, names in zip(FINGER_CHAINS[:, -1], XHAND_FINGERTIP_LINK_NAMES, XHAND_FINGER_JOINTS)
    }
    fingertips = config["fingertip_link"]
    tip_ids = [entry["human_hand_id"] for entry in fingertips]
    if (
        len(tip_ids) != len(expected)
        or any(type(tip_id) is not int for tip_id in tip_ids)
        or set(tip_ids) != expected.keys()
    ):
        raise ValueError("fingertip_link must contain HTS tip IDs 4, 8, 12, 16, 20 exactly once")
    joint_ids = []
    for tip_id, entry in zip(tip_ids, fingertips):
        link, names = expected[tip_id]
        joints = entry["joint"]
        if entry["link"] != link or len(joints) != len(names) or set(joints) != names:
            raise ValueError(f"HTS tip {tip_id} must map to {link} and its SDK joints")
        joint_ids.append([joint_order.index(name) for name in joints])
    return np.asarray(tip_ids, dtype=np.intp), joint_ids, sdk_indices, lower, upper


class GeoRTModel:
    def __init__(self, checkpoint_dir: str | Path):
        checkpoint = Path(checkpoint_dir).expanduser()
        config_path = checkpoint / "config.json"
        config_bytes = config_path.read_bytes()
        try:
            tip_ids, joint_ids, sdk_indices, lower, upper = _parse_config(json.loads(config_bytes))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid GeoRT config {config_path}: {exc}") from exc

        try:
            import torch
            from geort.formatter import HandFormatter
            from geort.model import IKModel
        except ModuleNotFoundError as exc:
            if exc.name not in {"torch", "geort", "geort.formatter", "geort.model"}:
                raise
            raise RuntimeError(
                f"GeoRT inference requires PyTorch and the facebookresearch/GeoRT package at commit {GEORT_COMMIT}",
            ) from exc

        weights_path = checkpoint / "last.pth"
        weights = weights_path.read_bytes()
        # 旧版 PyTorch 会在 weights_only 路径外处理 legacy tar 和 TorchScript。
        try:
            with ZipFile(BytesIO(weights)) as archive:
                if not weights.startswith(b"PK\x03\x04") or any(
                    Path(name).name == "constants.pkl" for name in archive.namelist()
                ):
                    raise BadZipFile("expected a state_dict archive")
        except BadZipFile as exc:
            raise ValueError(f"GeoRT checkpoint {weights_path} must be a torch.save(state_dict) ZIP") from exc

        self._torch = torch
        self._tip_ids = tip_ids
        self._sdk_indices = sdk_indices
        self._model = IKModel(keypoint_joints=joint_ids).to(device="cpu", dtype=torch.float32)
        state_dict = torch.load(BytesIO(weights), map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict, strict=True)
        if any(not torch.isfinite(value).all().item() for value in self._model.state_dict().values()):
            raise ValueError(f"GeoRT checkpoint {weights_path} contains non-finite model parameters")
        self._model.eval()
        self._formatter = HandFormatter(lower, upper)
        self.weights_sha256 = sha256(weights).hexdigest()
        self.config_sha256 = sha256(config_bytes).hexdigest()

    def predict(self, canonical_landmarks: np.ndarray) -> np.ndarray:
        """输入米制 GeoRT 掌系的 21 点，返回训练区间内的 12 轴弧度。"""
        fingertips = self._torch.from_numpy(canonical_landmarks[self._tip_ids]).unsqueeze(0).float()
        with self._torch.inference_mode():
            normalized = self._model(fingertips).numpy()[0]
        return self._formatter.unnormalize(normalized)[self._sdk_indices]
