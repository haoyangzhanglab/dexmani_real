"""alpha_from_tau / tau_from_alpha conversion tests."""

from __future__ import annotations

import math

from dexmani_real.utils.signal_utils import alpha_from_tau, tau_from_alpha


def test_round_trip_50hz():
    """alpha=0.6 @ 50Hz → tau → back to the same alpha."""
    dt = 1.0 / 50.0
    alpha = 0.6
    tau = -dt / math.log(1.0 - alpha)
    assert abs(alpha_from_tau(tau, dt) - alpha) < 1e-12


def test_known_conversions_16hz():
    """Reference values from the 16Hz migration (tau preserved from 50Hz)."""
    dt50, dt16 = 1.0 / 50.0, 1.0 / 16.0
    for alpha50, expected16 in [(0.6, 0.943), (0.3, 0.672), (0.8, 0.994), (0.4, 0.797)]:
        tau = -dt50 / math.log(1.0 - alpha50)
        assert abs(alpha_from_tau(tau, dt16) - expected16) < 2e-3


def test_edge_cases():
    assert alpha_from_tau(0.0, 0.02) == 1.0  # no smoothing
    assert alpha_from_tau(-1.0, 0.02) == 1.0
    assert 0.0 < alpha_from_tau(10.0, 0.02) < 0.01  # huge tau → heavy smoothing


def test_tau_from_alpha_round_trip():
    """tau_from_alpha is the exact inverse of alpha_from_tau (production path)."""
    dt = 1.0 / 50.0
    for alpha in (0.05, 0.3, 0.6, 0.95):
        assert abs(alpha_from_tau(tau_from_alpha(alpha, dt), dt) - alpha) < 1e-12


def test_tau_from_alpha_known_value():
    # 0.6 @ 50Hz ↔ tau ≈ 21.8ms (independent oracle, not via alpha_from_tau)
    assert abs(tau_from_alpha(0.6, 1.0 / 50.0) - (-0.02 / math.log(0.4))) < 1e-15
    assert abs(tau_from_alpha(0.6, 1.0 / 50.0) - 0.02183) < 1e-4


def test_tau_from_alpha_boundaries():
    assert tau_from_alpha(1.0, 0.02) == 0.0  # pass-through → zero time constant
    assert tau_from_alpha(1.5, 0.02) == 0.0
    assert tau_from_alpha(0.0, 0.02) == float("inf")  # frozen → infinite tau
    assert tau_from_alpha(-0.2, 0.02) == float("inf")
