"""Recording derives aggregate/dense tactile from one causal hand sample.

The recording boundary no longer searches backward for an older valid tactile
sample: qpos, current, aggregate, and dense all come from the single selected
hand-state frame, and each representation's validity bit decides independently
whether its payload is copied or persisted as NaN.
"""

import numpy as np

from dexmani_real.ipc.schema import HAND_STATE_DTYPE
from dexmani_real.recording.sample import build_episode_state


def _hand_state(*, aggregate, aggregate_valid, dense, dense_valid) -> np.ndarray:
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    hand["qpos"][0] = 0.1
    hand["current"][0] = 0.2
    hand["tactile_aggregate"][0] = aggregate
    hand["tactile_aggregate_valid"][0] = int(aggregate_valid)
    hand["tactile_dense"][0] = dense
    hand["tactile_dense_valid"][0] = int(dense_valid)
    return hand


def test_invalid_aggregate_is_nan_and_never_falls_back():
    # An invalid aggregate on the selected sample must become NaN+false, even
    # though a caller could in principle reach an older valid sample.
    hand = _hand_state(aggregate=3.0, aggregate_valid=False, dense=4.0, dense_valid=True)
    state = build_episode_state(None, hand)
    assert not state.hand_contact_valid
    assert np.isnan(state.hand_contact).all()
    assert state.hand_tactile_force_valid
    assert np.all(state.hand_tactile_force == 4.0)


def test_valid_zero_contact_stays_finite_and_valid():
    hand = _hand_state(aggregate=0.0, aggregate_valid=True, dense=0.0, dense_valid=True)
    state = build_episode_state(None, hand)
    assert state.hand_contact_valid
    assert np.all(state.hand_contact == 0.0)
    assert state.hand_tactile_force_valid
    assert np.all(state.hand_tactile_force == 0.0)


def test_aggregate_and_dense_validity_are_independent():
    hand = _hand_state(aggregate=3.0, aggregate_valid=True, dense=4.0, dense_valid=False)
    state = build_episode_state(None, hand)
    assert state.hand_contact_valid
    assert np.all(state.hand_contact == 3.0)
    assert not state.hand_tactile_force_valid
    assert np.isnan(state.hand_tactile_force).all()


def test_missing_hand_state_maps_everything_to_nan():
    state = build_episode_state(None, None)
    assert not state.hand_contact_valid
    assert np.isnan(state.hand_contact).all()
    assert not state.hand_tactile_force_valid
    assert np.isnan(state.hand_tactile_force).all()
