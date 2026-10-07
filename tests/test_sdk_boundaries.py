"""Required SDK fields cannot be replaced with apparently healthy defaults."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.robot.drivers.xarm7 import XArm7, _wait_controller_ready
from dexmani_real.robot.drivers.xhand import XHand


@pytest.mark.parametrize("missing", ["connected", "mode"])
def test_xarm_missing_report_field_cannot_pass_readiness(missing):
    api = NS(
        connected=True,
        mode=6,
        get_state=lambda: (0, 0),
        get_err_warn_code=lambda: (0, [0, 0]),
    )
    delattr(api, missing)
    with pytest.raises(AttributeError):
        _wait_controller_ready(api, expected_mode=6, timeout_s=0.1)


@pytest.mark.parametrize("name", ["axis", "error_code"])
@pytest.mark.parametrize("missing", [False, True])
def test_xarm_absent_report_value_is_not_zero(name, missing):
    arm = XArm7(ExperimentConfig().arm)
    arm._api = NS(**({} if missing else {name: None}))
    with pytest.raises((AttributeError, TypeError)):
        getattr(arm, name)


def hand_state():
    return NS(
        finger_state=[
            NS(
                id=i,
                position=i * 0.01,
                torque=i,
                commboard_err=i,
                jonitboard_err=i + 1,
                tipboard_err=i + 2,
            )
            for i in range(12)
        ]
    )


def test_xhand_vendor_fields_preserve_order_errors_and_missing_current():
    state = hand_state()
    del state.finger_state[3].torque
    state.finger_state.reverse()
    qpos, current, errors = XHand._parse_joints(state)
    np.testing.assert_allclose(qpos, np.arange(12) * 0.01)
    assert np.isnan(current[3])
    np.testing.assert_array_equal(errors["commboard_err"], np.arange(12))
    np.testing.assert_array_equal(errors["jointboard_err"], np.arange(12) + 1)
    np.testing.assert_array_equal(errors["tipboard_err"], np.arange(12) + 2)


@pytest.mark.parametrize("field", ["commboard_err", "jonitboard_err", "tipboard_err"])
def test_xhand_missing_board_status_is_not_healthy_feedback(field):
    state = hand_state()
    delattr(state.finger_state[3], field)
    state.finger_state[3].jointboard_err = 0
    hand = XHand(ExperimentConfig().hand)
    hand.connected_flag = True
    hand._control = NS(read_state=lambda *args: (NS(error_code=0), state))
    assert hand.get_state() is None
