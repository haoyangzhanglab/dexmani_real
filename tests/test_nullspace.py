"""Unit tests for planning/nullspace.py — mathematical correctness."""

import numpy as np
from dexmani_real.planning.nullspace import (
    nullspace_projector,
    joint_limit_gradient,
    apply_nullspace_optimization,
)

# xArm7 joint limits: [low, high] in radians
LIMITS = np.array([
    [-np.pi, np.pi],      # J1: continuous
    [-2.059, 2.094],      # J2: -118 to 120 deg
    [-np.pi, np.pi],      # J3: continuous
    [-0.192, 3.927],      # J4: -11 to 225 deg
    [-np.pi, np.pi],      # J5: continuous
    [-1.693, 3.142],      # J6: -97 to 180 deg
    [-np.pi, np.pi],      # J7: continuous
])


def test_nullspace_projector():
    """N = I - J⁺J must satisfy: J @ N = 0, N = Nᵀ, N² = N."""
    np.random.seed(42)
    for _ in range(10):
        J = np.random.randn(6, 7)
        # Reconstruct full-rank Jacobian via SVD
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        J = U @ np.diag(S) @ Vt

        N = nullspace_projector(J)

        # J @ N @ v = 0 for any v (null-space property)
        v = np.random.randn(7)
        assert np.max(np.abs(J @ (N @ v))) < 1e-13, f"J@N@v != 0: {np.max(np.abs(J @ (N @ v)))}"

        # N symmetric
        assert np.allclose(N, N.T), "N not symmetric"

        # N idempotent
        assert np.allclose(N @ N, N), "N not idempotent"

        # Eigenvalues: 1 one (null-space), 6 zeros (range-space)
        eigvals = np.sort(np.linalg.eigvalsh(N))
        assert np.abs(eigvals[-1] - 1.0) < 1e-10, f"Largest eigenvalue != 1: {eigvals[-1]}"
        assert np.max(np.abs(eigvals[:-1])) < 1e-12, f"Non-null eigenvalues too large: {eigvals[:-1]}"

    print("✓ nullspace_projector: all properties verified")


def test_joint_limit_gradient_zero_at_mid():
    """Gradient must be zero when all joints are at mid-range."""
    q_mid = 0.5 * (LIMITS[:, 0] + LIMITS[:, 1])
    g = joint_limit_gradient(q_mid, LIMITS, margin_deg=15.0)
    assert np.all(g == 0.0), f"Non-zero gradient at mid-range: {g}"
    print("✓ joint_limit_gradient: zero at mid-range")


def test_joint_limit_gradient_near_limit():
    """Near J4 lower limit (-11°), gradient must push positive."""
    q = 0.5 * (LIMITS[:, 0] + LIMITS[:, 1])
    q[3] = LIMITS[3, 0] + np.deg2rad(5.0)  # 5° from -11° limit
    g = joint_limit_gradient(q, LIMITS, margin_deg=15.0)
    # J4 gradient should be positive (pushing away from lower limit)
    assert g[3] > 0, f"J4 gradient not positive: {g[3]}"
    # Other joints at mid-range → zero gradient
    for i in [0, 1, 2, 4, 5, 6]:
        assert g[i] == 0.0, f"Joint {i} gradient non-zero at mid-range: {g[i]}"
    print(f"✓ joint_limit_gradient: J4 near limit → gradient={g[3]:.4f} > 0")


def test_joint_limit_gradient_near_upper():
    """Near J6 upper limit (180°), gradient must push negative."""
    q = 0.5 * (LIMITS[:, 0] + LIMITS[:, 1])
    q[5] = LIMITS[5, 1] - np.deg2rad(5.0)  # 5° from 180° limit
    g = joint_limit_gradient(q, LIMITS, margin_deg=15.0)
    assert g[5] < 0, f"J6 gradient not negative: {g[5]}"
    print(f"✓ joint_limit_gradient: J6 near upper → gradient={g[5]:.4f} < 0")


def test_joint_limit_gradient_nan_safety():
    """NaN input must return zero gradient."""
    g = joint_limit_gradient(np.full(7, np.nan), LIMITS, margin_deg=15.0)
    assert np.all(g == 0.0), f"NaN gradient not zero: {g}"
    g2 = joint_limit_gradient(np.array([np.inf, 0, 0, 0, 0, 0, 0]), LIMITS)
    assert np.all(g2 == 0.0), f"Inf gradient not zero: {g2}"
    print("✓ joint_limit_gradient: NaN/Inf → zero gradient")


def test_apply_nullspace_no_change_at_mid():
    """At mid-range (zero gradient), qpos must be unchanged."""
    np.random.seed(42)
    J = np.random.randn(6, 7)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    J = U @ np.diag(S) @ Vt

    q = 0.5 * (LIMITS[:, 0] + LIMITS[:, 1])
    result = apply_nullspace_optimization(q, J, LIMITS, step_size_rad=np.deg2rad(1.0), margin_deg=15.0)
    assert np.allclose(result, q), f"qpos changed at mid-range: delta={np.max(np.abs(result - q))}"
    print("✓ apply_nullspace: no change at mid-range")


def test_apply_nullspace_step_bounded():
    """Max step must not exceed step_size_rad."""
    np.random.seed(42)
    J = np.random.randn(6, 7)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    J = U @ np.diag(S) @ Vt

    # Push J4 right to the limit
    q = 0.5 * (LIMITS[:, 0] + LIMITS[:, 1])
    q[3] = LIMITS[3, 0] + np.deg2rad(1.0)  # 1° from -11° limit

    step = np.deg2rad(1.0)
    result = apply_nullspace_optimization(q, J, LIMITS, step_size_rad=step, margin_deg=15.0)
    max_delta = np.max(np.abs(result - q))
    assert max_delta <= step + 1e-10, f"Step exceeded: {np.rad2deg(max_delta):.4f} deg > 1.0 deg"

    # J4 should move away from limit
    assert result[3] > q[3], f"J4 not pushed away: {np.rad2deg(result[3] - q[3]):.4f} deg"
    print(f"✓ apply_nullspace: J4 moved +{np.rad2deg(result[3] - q[3]):.4f} deg, max_step={np.rad2deg(max_delta):.4f} deg")


def test_c1_continuity():
    """Gradient must be C¹ continuous at the margin boundary."""
    q_mid = 0.5 * (LIMITS[:, 0] + LIMITS[:, 1])
    margin_rad = np.deg2rad(15.0)

    # Exactly at boundary
    q_at_boundary = q_mid.copy()
    q_at_boundary[3] = LIMITS[3, 0] + margin_rad
    g_at = joint_limit_gradient(q_at_boundary, LIMITS, margin_deg=15.0)
    assert g_at[3] == 0.0, f"Gradient not zero at margin boundary: {g_at[3]}"

    # Just inside
    q_inside = q_mid.copy()
    q_inside[3] = LIMITS[3, 0] + margin_rad - 1e-6
    g_inside = joint_limit_gradient(q_inside, LIMITS, margin_deg=15.0)
    assert g_inside[3] > 0, f"Gradient zero just inside margin: {g_inside[3]}"
    # Should be very small (near-zero at boundary)
    assert g_inside[3] < 1e-3, f"Gradient too large at boundary: {g_inside[3]}"

    print(f"✓ joint_limit_gradient: C¹ continuous (boundary={g_at[3]:.2e}, inside={g_inside[3]:.2e})")


if __name__ == "__main__":
    test_nullspace_projector()
    test_joint_limit_gradient_zero_at_mid()
    test_joint_limit_gradient_near_limit()
    test_joint_limit_gradient_near_upper()
    test_joint_limit_gradient_nan_safety()
    test_apply_nullspace_no_change_at_mid()
    test_apply_nullspace_step_bounded()
    test_c1_continuity()
    print("\n✅ All tests passed!")
