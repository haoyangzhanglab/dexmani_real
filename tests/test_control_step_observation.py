"""Visual deployment uses logical-grid state/contact with unchanged live gates."""

from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.deployment.inference import observation as obs
from dexmani_real.ipc.schema import (
    CAMERA_FRAME_HEADER_DTYPE,
    make_pointcloud_frame_dtype,
)
from test_deployment_eef_tactile import (
    _ANCHOR_NS,
    _REF_TICKS,
    _T0_NS,
    _FakeRing,
    _Field,
    _build,
    _fake_policy_spec,
    _fake_shared,
    _ref_ns,
)


def visual_fixture(visual: str, *, dense: bool):
    shared = _fake_shared()
    for ring in (
        shared.arm_state_ring,
        shared.hand_state_ring,
        shared.hand_tactile_ring,
    ):
        ring._records = ring._records[3:]
        for data, _, _ in ring._records:
            data["source_monotonic_ns"] -= np.uint64(9_000_000)
            if "publish_monotonic_ns" in data.dtype.names:
                data["publish_monotonic_ns"] -= np.uint64(9_000_000)
    headers = {}
    clouds = []
    for sequence, tick in enumerate(_REF_TICKS, start=1):
        source = _ref_ns(tick) - 25_000_000
        header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
        header["source_monotonic_ns"] = source
        header["receive_monotonic_ns"] = source + 1_000_000
        header["publish_monotonic_ns"] = source + 2_000_000
        header["camera_generation"] = 1
        headers[sequence] = header
        cloud = np.zeros(1, dtype=make_pointcloud_frame_dtype(1024))
        cloud["source_monotonic_ns"] = source
        cloud["camera_publish_monotonic_ns"] = source + 2_000_000
        cloud["publish_monotonic_ns"] = source + 3_000_000
        cloud["source_camera_sequence"] = sequence
        cloud["camera_generation"] = 1
        cloud["point_cloud"] = 0.25
        clouds.append((cloud, source + 3_000_000, sequence))
    shared.camera_ring = SimpleNamespace(
        maxlen=8,
        get_last_metadata=lambda k: [
            (h, int(h["publish_monotonic_ns"][0]), s) for s, h in headers.items()
        ],
        read_sequence=lambda sequence, **kwargs: {
            "header": headers[sequence],
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        },
    )
    shared.pointcloud_ring = _FakeRing(clouds)
    fields = [_Field("contact_force", (5, 3), "float32")]
    if visual in ("rgb", "rgb_pc"):
        fields.append(_Field("rgb", (2, 2, 3), "uint8"))
    if visual in ("pointcloud", "rgb_pc"):
        fields.append(_Field("point_cloud", (1024, 6), "float32"))
    if dense:
        fields.append(_Field("tactile_force", (5, 120, 3), "float32"))
    return shared, _fake_policy_spec(*fields), headers


@pytest.mark.parametrize("visual", ["rgb", "pointcloud", "rgb_pc"])
@pytest.mark.parametrize("dense", [False, True])
def test_d9_visual_state_contact_and_dense_use_logical_grid(visual, dense):
    shared, spec, _ = visual_fixture(visual, dense=dense)
    batch = _build(shared, spec)
    assert batch is not None
    refs = np.asarray([_ref_ns(tick) for tick in _REF_TICKS])
    expected_source = refs - 9_000_000
    windows = [batch.arm_history, batch.hand_history, batch.hand_tactile_sum_history]
    windows.append(
        batch.hand_tactile_force_history
        if dense
        else batch.hand_tactile_provenance_history
    )
    for window in windows:
        np.testing.assert_array_equal(window.source_monotonic_ns, expected_source)
        assert np.all(window.source_monotonic_ns > refs - 25_000_000)
        assert np.all(window.source_monotonic_ns <= refs)
    assert batch.logical_step_monotonic_ns == refs[-1]
    np.testing.assert_array_equal(
        batch.hand_tactile_sum_history.values[:, 0, 0], _REF_TICKS
    )


@pytest.mark.parametrize(
    "defect",
    [
        "too_old",
        "before_start",
        "health",
        "invalid_arm",
        "invalid_hand",
        "qpos_stale",
        "future_source",
        "future_publish",
        "uncalibrated",
        "unit",
        "source_mismatch",
        "dense_unfresh",
    ],
)
def test_d9_visual_runtime_failures_remain_closed(defect):
    shared, spec, headers = visual_fixture("rgb_pc", dense=True)
    if defect == "health":
        for header in headers.values():
            header["camera_health"] = 1
    elif defect in (
        "too_old",
        "before_start",
        "future_source",
        "future_publish",
        "invalid_arm",
    ):
        for data, _, _ in shared.arm_state_ring._records:
            if defect == "before_start":
                data["source_monotonic_ns"] = _T0_NS - 1
            elif defect == "too_old":
                data["source_monotonic_ns"] = _T0_NS + 1
            elif defect == "future_source":
                # Causal to the inference cut, but newer than the logical step.
                data["source_monotonic_ns"] = _ANCHOR_NS - 5_000_000
                data["publish_monotonic_ns"] = _ANCHOR_NS - 4_000_000
            elif defect == "future_publish":
                data["publish_monotonic_ns"] = _ANCHOR_NS + 1
            else:
                data["state_valid"] = False
    elif defect in ("invalid_hand", "qpos_stale"):
        for data, _, _ in shared.hand_state_ring._records:
            data["state_valid" if defect == "invalid_hand" else "qpos_stale"] = (
                defect == "qpos_stale"
            )
    else:
        for data, _, _ in shared.hand_tactile_ring._records:
            if defect == "uncalibrated":
                data["calibrated"] = False
            elif defect == "unit":
                data["unit_code"] = 255
            elif defect == "source_mismatch":
                data["source_monotonic_ns"] -= np.uint64(1)
            else:
                data["fresh"] = False
    assert _build(shared, spec) is None


def test_visual_contact_only_keeps_dense_freshness_optional():
    shared, spec, _ = visual_fixture("rgb_pc", dense=False)
    for data, _, _ in shared.hand_tactile_ring._records:
        data["fresh"] = False
        data["tactile_force"] = np.nan
    assert _build(shared, spec) is not None
