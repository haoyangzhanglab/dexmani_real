"""Existential excitation admission against an independent exhaustive pair oracle."""

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexmani_real.calibration.camera.solver import check_hand_eye_excitation


def axes_for(poses):
    rotations = Rotation.from_euler("xyz", poses)
    axes = []
    for i, j in itertools.combinations(range(len(poses)), 2):
        vector = (rotations[j] * rotations[i].inv()).as_rotvec()
        if np.degrees(np.linalg.norm(vector)) >= 5:
            axes.append(vector / np.linalg.norm(vector))
    return axes


def exhaustive(poses, threshold):
    return any(
        np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))) >= threshold
        for a, b in itertools.combinations(axes_for(poses), 2)
    )


def test_pair_not_involving_first_axis_is_admitted():
    poses = [
        [-0.8758092137841385, -1.0323129656543197, 0.2907847967992972],
        [0.5177411551935099, 0.8608946899214116, 0.33733889832116615],
        [-0.64638303474513, -0.42345311623321136, 0.773383740265998],
    ]
    axes = axes_for(poses)
    assert all(np.degrees(np.arccos(abs(axes[0] @ b))) < 60 for b in axes[1:])
    result = check_hand_eye_excitation(poses, min_axis_separation_deg=65)
    assert result["axis_separation_witness_deg"] >= 65
    assert "axis_separation_max_deg" not in result


def test_small_samples_match_exhaustive_admission():
    rng = np.random.default_rng(79)
    for count in (3, 4, 8):
        for _ in range(15):
            poses = rng.normal(size=(count, 3)) * 0.3
            threshold = float(rng.uniform(10, 89))
            if exhaustive(poses, threshold):
                assert (
                    check_hand_eye_excitation(poses, min_axis_separation_deg=threshold)[
                        "axis_separation_witness_deg"
                    ]
                    >= threshold
                )
            else:
                with pytest.raises(ValueError, match="excitation"):
                    check_hand_eye_excitation(poses, min_axis_separation_deg=threshold)
