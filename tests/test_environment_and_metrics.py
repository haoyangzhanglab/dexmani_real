from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dexmani_real.config import defaults
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.planning.collision_model import CollisionModel
from dexmani_real.planning.preflight import (
    PreflightCertificate,
    create_preflight_certificate,
    hash_arrays,
    hash_files,
    verify_preflight_binding,
)
from dexmani_real.shm.shared_storage import (
    SharedStorage,
    SharedStorageConfig,
    publish_component_metrics,
    read_component_metrics,
)
from dexmani_real.utils.rate_manager import RateManager


class _ManualClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def __call__(self) -> float:
        return self.now_s

    def advance(self, duration_s: float) -> None:
        self.now_s += duration_s


def _box(name: str, center: tuple[float, float, float]) -> dict[str, object]:
    return {
        "name": name,
        "center_xyz_m": center,
        "size_xyz_m": (0.08, 0.08, 0.08),
        "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    }


def test_environment_config_is_validated_nested_immutable_and_hash_sensitive() -> None:
    first = _box("wall", (1.0, 2.0, 3.0))
    second = _box("fixture", (4.0, 5.0, 6.0))
    resolved = resolve_runtime_config(json_data={"environment": {"static_boxes": [first, second]}})

    assert resolved.environment.static_boxes[0].name == "wall"
    assert resolved.to_dict()["environment"]["static_boxes"][1]["name"] == "fixture"
    assert (
        resolve_runtime_config(json_data={"environment": {"static_boxes": [second, first]}}).sha256 != resolved.sha256
    )
    changed = _box("wall", (1.0, 2.0, 3.001))
    assert (
        resolve_runtime_config(json_data={"environment": {"static_boxes": [changed, second]}}).sha256 != resolved.sha256
    )
    assert resolve_runtime_config().environment.static_boxes == ()

    with pytest.raises((AttributeError, TypeError)):
        resolved.environment.static_boxes[0].name = "changed"
    with pytest.raises(ValueError, match="positive"):
        resolve_runtime_config(json_data={"environment": {"static_boxes": [{**first, "size_xyz_m": (1.0, 0.0, 1.0)}]}})
    with pytest.raises(ValueError, match="unit quaternion"):
        resolve_runtime_config(
            json_data={"environment": {"static_boxes": [{**first, "quat_wxyz": (2.0, 0.0, 0.0, 0.0)}]}}
        )
    with pytest.raises(ValueError, match="unique"):
        resolve_runtime_config(json_data={"environment": {"static_boxes": [first, first]}})
    with pytest.raises(ValueError, match="reserved"):
        resolve_runtime_config(json_data={"environment": {"static_boxes": [{**first, "name": "table"}]}})


def test_combined_collision_preserves_self_semantics_and_detects_scene_crossing() -> None:
    home = np.asarray(defaults.arm.home_qpos, dtype=np.float64)
    baseline = CollisionModel(hand_dof=False)
    far = CollisionModel(
        hand_dof=False,
        static_boxes=[
            {
                **_box("rotated_far", (10.0, 10.0, 10.0)),
                "size_xyz_m": (1.0, 2.0, 3.0),
                "quat_wxyz": (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
            }
        ],
    )
    assert baseline.check_self_collision(home) == far.check_self_collision(home)
    assert not far.check_environment_collision(home)

    crossing = CollisionModel(hand_dof=False, static_boxes=[_box("fixture", (0.402, 0.0, 0.191))])
    start = home.copy()
    end = home.copy()
    start[0] = -1.0
    end[0] = 1.0
    assert not crossing.check_environment_collision(start)
    assert crossing.check_environment_collision(home)
    assert not crossing.check_environment_collision(end)
    assert not crossing.check_combined_segment_collision_free(start, end, step_size=0.05)
    detail = crossing.check_environment_collision_details(home)
    assert detail and detail.collision_pairs[0].collision_type == "environment"
    assert detail.collision_pairs[0].object_name2 == "fixture"
    assert detail.sample_qpos_rad == pytest.approx(tuple(home))


def test_environment_collision_follows_hand_shape() -> None:
    model = CollisionModel(
        hand_dof=True,
        static_boxes=[
            {
                "name": "finger_fixture",
                "center_xyz_m": (0.61, 0.006, 0.181),
                "size_xyz_m": (0.02, 0.02, 0.02),
                "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
            }
        ],
    )
    arm_qpos = np.asarray(defaults.arm.home_qpos, dtype=np.float64)
    model.set_hand_qpos(np.asarray(defaults.hand.qpos_min_rad, dtype=np.float64))
    assert model.check_environment_collision(arm_qpos)
    model.set_hand_qpos(np.asarray(defaults.hand.qpos_max_rad, dtype=np.float64))
    assert not model.check_environment_collision(arm_qpos)


def test_preflight_v2_binds_scene_and_v1_only_accepts_empty_scene(tmp_path: Path) -> None:
    model_path = tmp_path / "model.urdf"
    model_path.write_text("robot", encoding="utf-8")
    arm = np.zeros((2, 7), dtype=np.float64)
    hand = np.zeros((2, 12), dtype=np.float64)
    workspace = np.array([[-1.0, 1.0]] * 3)
    scene = (_box("wall", (1.0, 0.0, 0.0)),)
    always = lambda *_args: True
    certificate = create_preflight_certificate(
        source_episode="episode.h5",
        arm_actions=arm,
        hand_actions=hand,
        collision_model_paths=(model_path,),
        workspace_bounds_m=workspace,
        resolved_config_sha256="a" * 64,
        transition_check=always,
        workspace_check=always,
        table_check=always,
        static_boxes=scene,
    )
    assert certificate.version == 2 and certificate.collision_scene_sha256 is not None
    verify_preflight_binding(
        certificate,
        source_episode="episode.h5",
        arm_actions=arm,
        hand_actions=hand,
        collision_model_paths=(model_path,),
        workspace_bounds_m=workspace,
        resolved_config_sha256="a" * 64,
        static_boxes=scene,
    )
    with pytest.raises(ValueError, match="collision_scene_sha256"):
        verify_preflight_binding(
            certificate,
            source_episode="episode.h5",
            arm_actions=arm,
            hand_actions=hand,
            collision_model_paths=(model_path,),
            workspace_bounds_m=workspace,
            resolved_config_sha256="a" * 64,
            static_boxes=(_box("wall", (1.1, 0.0, 0.0)),),
        )

    checks = ("shape_finite",)
    legacy_payload: dict[str, object] = {
        "version": 1,
        "trajectory_sha256": hash_arrays(arm, hand),
        "collision_model_sha256": hash_files((model_path,)),
        "workspace_sha256": hash_arrays(workspace),
        "resolved_config_sha256": "a" * 64,
        "source_episode": str(Path("episode.h5").resolve()),
        "frame_count": 2,
        "hand_enabled": True,
        "checks_run": list(checks),
        "created_utc": "2026-01-01T00:00:00+00:00",
    }
    canonical = json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    legacy_payload["certificate_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy = PreflightCertificate.read(legacy_path)
    verify_preflight_binding(
        legacy,
        source_episode="episode.h5",
        arm_actions=arm,
        hand_actions=hand,
        collision_model_paths=(model_path,),
        workspace_bounds_m=workspace,
        resolved_config_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="v1"):
        verify_preflight_binding(
            legacy,
            source_episode="episode.h5",
            arm_actions=arm,
            hand_actions=hand,
            collision_model_paths=(model_path,),
            workspace_bounds_m=workspace,
            resolved_config_sha256="a" * 64,
            static_boxes=scene,
        )


def test_rate_manager_counts_overruns_missed_slots_reanchors_and_reset() -> None:
    clock = _ManualClock()
    limiter = RateManager(10.0, clock=clock, sleep=clock.advance)
    clock.advance(0.05)
    limiter.wait()
    assert limiter.stats.loop_count == 1
    assert limiter.stats.last_work_duration_s == pytest.approx(0.05)
    assert limiter.stats.deadline_overrun_count == 0

    clock.advance(0.11)
    limiter.wait()
    assert limiter.stats.deadline_overrun_count == 1
    assert limiter.stats.missed_slot_count == 0

    clock.advance(0.31)
    limiter.wait()
    assert limiter.stats.missed_slot_count == 2
    clock.advance(1.2)
    limiter.wait()
    assert limiter.stats.long_block_reanchor_count == 1
    assert limiter.stats.max_work_duration_s == pytest.approx(1.2)

    counts = limiter.stats.deadline_overrun_count
    limiter.reset()
    assert limiter.stats.deadline_overrun_count == counts
    limiter.reset_statistics()
    assert limiter.stats.loop_count == 0
    assert limiter.stats.deadline_overrun_count == 0


def test_component_metrics_has_dedicated_ring_and_is_best_effort() -> None:
    import uuid

    shared = SharedStorage.create(
        prefix=f"metrics_{uuid.uuid4().hex}",
        config=SharedStorageConfig(camera_rgb_shape=(2, 2, 3), camera_depth_shape=(2, 2), camera_pc_shape=(1, 6)),
    )
    clock = _ManualClock()
    limiter = RateManager(20.0, clock=clock, sleep=clock.advance)
    try:
        limiter.wait()
        heartbeat_before = float(shared.arm_heartbeat_s.value)
        status_sequence_before = shared.component_status_ring.latest_sequence
        assert publish_component_metrics(shared, "arm", limiter, now_s=1.0)
        assert not publish_component_metrics(shared, "arm", limiter, now_s=1.5)
        metrics = read_component_metrics(shared, "arm")
        assert metrics is not None
        assert metrics["target_period_s"] == pytest.approx(0.05)
        assert metrics["loop_count"] == 1
        assert shared.component_status_ring.latest_sequence == status_sequence_before
        assert float(shared.arm_heartbeat_s.value) == heartbeat_before
        assert not publish_component_metrics(object(), "arm", limiter, interval_s=0.0, now_s=2.0)
    finally:
        shared.close()
