"""使用 HTS Raw 录制和本仓 XHand 运动学离线训练 GeoRT。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dexmani_real import ASSET_DIR
from dexmani_real.config.real import load_config
from dexmani_real.data.raw import RawReader, published_raw_path
from dexmani_real.geometry.kinematics import HandFK, kinematic_urdf
from dexmani_real.robot.model import (
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_RIGHT_URDF_PATH,
    XHAND_SDK_JOINT_NAMES,
)
from dexmani_real.teleop.retargeting.geort_model import GEORT_COMMIT, XHAND_FINGER_JOINTS
from dexmani_real.teleop.retargeting.landmarks import (
    FINGER_CHAINS,
    HandCalibration,
    HandGeometryRejected,
    hand_lengths,
    prepare_geort_landmarks,
)

# 参考 XHand 模型在各指尖 link 的局部坐标系中定义的跟踪点偏移，单位米。
FINGERTIP_OFFSETS_M = ((0, 0.02, 0),) + ((0, 0, 0.02),) * 4
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class TrainingHand:
    """提供上游训练器所需的关节限位和指尖 FK。"""

    def __init__(self, hand_config):
        self.fk = HandFK()
        # 上游 FK 网络和关节归一化使用 float32。
        self.limits = (
            np.asarray(hand_config.qpos_min_rad, dtype=np.float32),
            np.asarray(hand_config.qpos_max_rad, dtype=np.float32),
        )

    def get_joint_limit(self):
        return self.limits

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        self.names = keypoint_link_names
        self.frame_ids = [self.fk.model.getFrameId(name) for name in self.names]
        if any(frame_id >= self.fk.model.nframes for frame_id in self.frame_ids):
            raise ValueError("GeoRT XHand config has an unknown fingertip frame")
        self.offsets = np.asarray(keypoint_offsets)

    def keypoint_from_qpos(self, qpos):
        poses = self.fk.forward(qpos, self.frame_ids)
        points = poses[:, :3, 3] + (poses[:, :3, :3] @ self.offsets[..., None]).squeeze(-1)
        return dict(zip(self.names, points, strict=True))


def episode_dir(path: Path) -> Path:
    path = path.expanduser()
    resolved = path.resolve()
    if resolved.name == "data.h5" and resolved.is_file():
        path = resolved.parent
    return published_raw_path(path)


def load_episode(path):
    with RawReader(path) as reader:
        if reader.meta.get("collection_source") != "teleop":
            raise ValueError(f"{path}: a VR teleop Raw episode is required")
        parameters = json.loads(reader.meta["effective_parameters"])
        calibration = HandCalibration.from_recording(parameters["hand_retargeting_calibration"])
        tolerance = float(parameters["hand_retargeting"]["bone_length_tolerance"])
        if not 0 < tolerance < 1:
            raise ValueError(f"{path}: invalid hand bone length tolerance")
        frames = []
        rejected = 0
        for landmarks in reader["vr_landmarks"]:
            try:
                points = prepare_geort_landmarks(landmarks)
                calibration.validate_lengths(hand_lengths(points), tolerance=tolerance)
            except HandGeometryRejected:
                rejected += 1
            else:
                frames.append(points)
        if not frames:
            raise ValueError(f"{path}: no usable GeoRT landmark frames")
        points = np.asarray(frames)
        tips = points[:, FINGER_CHAINS[:, -1]]
        pinch_distances = np.linalg.norm(tips[:, 1:] - tips[:, :1], axis=-1)
        summary = {
            "path": str(reader.path),
            "frames": len(frames),
            "rejected_frames": rejected,
            "human_pinch_frames_below_20mm": dict(zip(FINGERS[1:], (pinch_distances < 0.02).sum(axis=0).tolist())),
        }
        return points, summary


def geort_config():
    return {
        "name": "xhand_right",
        "urdf_path": str(XHAND_RIGHT_URDF_PATH.relative_to(ASSET_DIR.parent)),
        "base_link": "right_hand_link",
        "joint_order": XHAND_SDK_JOINT_NAMES,
        "fingertip_link": [
            {
                "name": finger,
                "link": link,
                "joint": joints,
                "center_offset": offset,
                "human_hand_id": int(tip),
            }
            for finger, link, joints, offset, tip in zip(
                FINGERS, XHAND_FINGERTIP_LINK_NAMES, XHAND_FINGER_JOINTS,
                FINGERTIP_OFFSETS_M, FINGER_CHAINS[:, -1], strict=True,
            )
        ],
    }


def train(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GeoRT training requires a CUDA-enabled PyTorch environment")
    source = args.geort_source.expanduser().resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--", "geort"])
    if head != GEORT_COMMIT or changes:
        raise ValueError(f"GeoRT source must contain a clean geort package at {GEORT_COMMIT}")
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(source):
        raise ValueError("choose a checkpoint directory outside the GeoRT source checkout")

    train_paths = [episode_dir(path) for path in args.train_episode]
    validation_paths = [episode_dir(path) for path in args.validation_episode]
    paths = train_paths + validation_paths
    if len(set(paths)) != len(paths):
        raise ValueError("each Raw episode must be listed once across the train and validation sets")
    training = [load_episode(path) for path in train_paths]
    validation_summary = [load_episode(path)[1] for path in validation_paths]

    hand_config = load_config(args.config).hand
    hand_config.validate_limits()
    geometry = TrainingHand(hand_config)
    config = geort_config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".geort-", dir=output.parent) as workspace:
        work = Path(workspace)
        shutil.copytree(source / "geort", work / "geort", ignore=shutil.ignore_patterns("__pycache__", "mocap"))
        data_path = work / "hts.npy"
        np.save(data_path, np.concatenate([points for points, _ in training]), allow_pickle=False)
        previous_cwd = Path.cwd()
        sys.path.insert(0, str(work))
        try:
            os.chdir(work)
            from geort.trainer import GeoRTTrainer

            class XHandTrainer(GeoRTTrainer):
                def __init__(self):
                    self.config = config
                    self.hand = geometry

            XHandTrainer().train(data_path, epoch=args.epochs)
        finally:
            os.chdir(previous_cwd)
            sys.path.pop(0)

        checkpoint = work / "checkpoint" / f"{config['name']}_last"
        bundle = work / "bundle"
        bundle.mkdir()
        for name in ("config.json", "last.pth"):
            shutil.copyfile(checkpoint / name, bundle / name)
        summary = {
            "geort_commit": GEORT_COMMIT,
            "kinematic_urdf_sha256": hashlib.sha256(kinematic_urdf("right_hand_link").encode()).hexdigest(),
            "torch_version": torch.__version__,
            "epochs": args.epochs,
            "seed": args.seed,
            "training_episodes": [summary for _, summary in training],
            "validation_episodes": validation_summary,
        }
        (bundle / "training.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
        )
        bundle.rename(output)
    print(f"GeoRT checkpoint 已保存到 {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geort-source", type=Path, required=True, help=f"GeoRT checkout at {GEORT_COMMIT}")
    parser.add_argument(
        "--train-episode", type=Path, action="append", required=True, help="Raw episode directory or data.h5",
    )
    parser.add_argument(
        "--validation-episode", type=Path, action="append", required=True,
        help="Held-out Raw episode directory or data.h5",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New checkpoint directory")
    parser.add_argument("--config", type=Path, help="Experiment config used for XHand operating limits")
    parser.add_argument("--epochs", type=int, default=200, help="Number of IK training epochs")
    parser.add_argument("--seed", type=int, default=0, help="NumPy and PyTorch random seed")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    train(args)


if __name__ == "__main__":
    main()
