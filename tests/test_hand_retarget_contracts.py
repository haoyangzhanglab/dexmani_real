from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from dexmani_real import ASSET_DIR
from dexmani_real.config.defaults import hand
from dexmani_real.teleop.hand_retarget import _OPERATOR2MANO_RIGHT, TAGHandRetargeter, _tag_config_with_urdf
from dexmani_real.teleop.tag_retargeting.pin_grad import PinGrad, validate_fingertip_frame_names


def test_operator_to_mano_transform_is_a_proper_rotation() -> None:
    transform = np.asarray(_OPERATOR2MANO_RIGHT, dtype=np.float64)
    np.testing.assert_allclose(transform.T @ transform, np.eye(3))
    assert np.linalg.det(transform) == pytest.approx(1.0)


def test_fingertip_contract_accepts_exact_semantic_order() -> None:
    assert validate_fingertip_frame_names(hand.fingertip_link_names) == tuple(hand.fingertip_link_names)


@pytest.mark.parametrize(
    "names, message",
    [
        (hand.fingertip_link_names[:-1], "exactly five"),
        (
            (*hand.fingertip_link_names[:-1], hand.fingertip_link_names[-2]),
            "unique",
        ),
        (
            (hand.fingertip_link_names[1], hand.fingertip_link_names[0], *hand.fingertip_link_names[2:]),
            "ordered thumb/index/mid/ring/pinky",
        ),
        (
            (*hand.fingertip_link_names[:-1], ""),
            "non-empty strings",
        ),
    ],
)
def test_fingertip_contract_rejects_missing_duplicate_or_misordered_names(names: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_fingertip_frame_names(names)


def test_pin_grad_requires_all_five_configured_frames_to_exist_in_the_urdf() -> None:
    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf")
    pin_grad = PinGrad(urdf_path, list(hand.fingertip_link_names))
    assert len(pin_grad.tip_frame_ids) == 5

    missing = (*hand.fingertip_link_names[:-1], "right_hand_pinky_missing_tip")
    with pytest.raises(ValueError, match="not found in URDF"):
        PinGrad(urdf_path, list(missing))


def test_tag_retargeter_uses_the_configured_urdf_for_model_and_optimizer() -> None:
    model_joint_names = [
        "right_hand_index_bend_joint",
        "right_hand_index_joint1",
        "right_hand_index_joint2",
        "right_hand_mid_joint1",
        "right_hand_mid_joint2",
        "right_hand_pinky_joint1",
        "right_hand_pinky_joint2",
        "right_hand_ring_joint1",
        "right_hand_ring_joint2",
        "right_hand_thumb_bend_joint",
        "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
    ]
    fake_model = SimpleNamespace(
        names=["universe", "root_joint", *model_joint_names],
        lowerPositionLimit=np.full(19, -3.0, dtype=np.float64),
        upperPositionLimit=np.full(19, 3.0, dtype=np.float64),
    )

    class _FakeOptimizer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    custom_urdf = "/tmp/custom_xhand_right.urdf"
    with (
        patch("dexmani_real.teleop.hand_retarget.pin_loading", return_value=fake_model) as load,
        patch("dexmani_real.teleop.tag_retargeting.optimizer.HandOptimizer", _FakeOptimizer),
    ):
        retargeter = TAGHandRetargeter(
            tag_config=_tag_config_with_urdf(None, custom_urdf),
            smoothing_alpha=1.0,
        )

    load.assert_called_once_with(custom_urdf)
    assert retargeter._optimizer.kwargs["urdf_path"] == custom_urdf  # noqa: SLF001
    assert retargeter._optimizer.kwargs["fingertip_frame_names"] == list(hand.fingertip_link_names)  # noqa: SLF001
