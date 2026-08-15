"""P11: DexMani Policy adapter (execution doc §86–§91).

Locks the integration's architecture gate and contract without the real model
repository installed:

  - importing the integration imports neither torch nor ``dexmani_policy`` (the
    lazy import lives inside ``DexManiPolicyBackend.load``, §66)
  - the three classes conform to the three deployment Protocols
  - ``encode``/``decode`` are self-contained: an ``ObservationBatch`` round-trips
    to a model-native dict and back to a ``JointActionChunk`` with strictly
    increasing targets anchored to the causal cut
  - ``load`` fails closed when ``dexmani_policy`` is absent, rejects an EE-action
    checkpoint (§90), and drives a joint agent end-to-end through a stubbed
    ``dexmani_policy.build_agent``

Real-checkpoint inference is skipped: ``dexmani_policy`` is not importable in
this environment, so the stub below is the only way to exercise ``load``'s
post-import logic.
"""

from __future__ import annotations

import sys
import types

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.contracts import (
    ActionAdapter,
    InferenceContext,
    JointActionChunk,
    ObservationAdapter,
    PolicyBackend,
)
from dexmani_real.deployment.observation import FrameWindow, ObservationBatch
from dexmani_real.integrations.dexmani_policy import (
    DexManiActionAdapter,
    DexManiObservationAdapter,
    DexManiPolicyBackend,
)


def _window(values: np.ndarray, start_ns: int) -> FrameWindow:
    t = values.shape[0]
    seq = np.arange(1, t + 1, dtype=np.uint64)
    src = np.arange(start_ns, start_ns + t, dtype=np.uint64)
    return FrameWindow(
        values=values,
        source_sequence=seq,
        source_monotonic_ns=src,
        publish_monotonic_ns=src + 1,
        valid_mask=np.ones(t, dtype=np.uint8),
    )


def _install_fake_dexmani_policy(action_space: str) -> None:
    """Install a stub ``dexmani_policy`` module exposing ``build_agent``."""
    mod = types.ModuleType("dexmani_policy")

    class _Agent:
        def __init__(self, space: str) -> None:
            self.action_space = space
            self.events: list[str] = []

        def predict_action(self, obs: dict) -> dict:
            self.events.append("predict")
            arm = np.asarray(obs["arm_qpos"], dtype=np.float64)
            arm_out = np.tile(arm[None, :], (4, 1))
            hand = obs.get("hand_qpos")
            hand_out = np.tile(hand[None, :], (4, 1)) if hand is not None else None
            return {"arm_qpos": arm_out, "hand_qpos": hand_out}

        def reset(self) -> None:
            self.events.append("reset")

        def close(self) -> None:
            self.events.append("close")

    def build_agent(model_config_path=None, checkpoint=None, device=None):
        return _Agent(action_space)

    mod.build_agent = build_agent
    sys.modules["dexmani_policy"] = mod


def main() -> int:
    # ── architecture gate: lazy import, no torch / no dexmani_policy yet ──
    assert "torch" not in sys.modules
    assert "dexmani_policy" not in sys.modules
    # Importing the integration must not pull in the model repository.
    import dexmani_real.integrations.dexmani_policy  # noqa: F401
    assert "torch" not in sys.modules
    assert "dexmani_policy" not in sys.modules

    # ── Protocol conformance ──
    assert isinstance(DexManiObservationAdapter(), ObservationAdapter)
    assert isinstance(DexManiPolicyBackend(), PolicyBackend)
    assert isinstance(DexManiActionAdapter(), ActionAdapter)

    cfg = DeploymentConfig(checkpoint="ckpt.pt", model_config_path="cfg.yaml", device="cpu")

    # ── encode/decode are self-contained (no dexmani_policy needed) ──
    arm_hist = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    hand_hist = np.zeros((1, 12))
    batch = ObservationBatch(
        observation_id=5,
        run_generation=2,
        anchor_monotonic_ns=1_000_000_000,
        arm_history=_window(arm_hist, start_ns=1_000_000),
        hand_history=_window(hand_hist, start_ns=2_000_000),
    )
    model_input = DexManiObservationAdapter().encode(batch)
    np.testing.assert_allclose(model_input["arm_qpos"], arm_hist[0])
    np.testing.assert_allclose(model_input["hand_qpos"], hand_hist[0])

    # absent hand window -> None; absent arm window -> zero vector (never crash)
    no_hand = ObservationBatch(
        observation_id=6, run_generation=2, anchor_monotonic_ns=1_000_000_001,
        arm_history=_window(arm_hist, start_ns=1_000_000),
    )
    no_hand_input = DexManiObservationAdapter().encode(no_hand)
    assert no_hand_input["hand_qpos"] is None
    empty = ObservationBatch(observation_id=7, run_generation=2, anchor_monotonic_ns=1_000_000_002)
    empty_input = DexManiObservationAdapter().encode(empty)
    np.testing.assert_allclose(empty_input["arm_qpos"], np.zeros(7))

    ctx = InferenceContext(
        run_generation=2,
        observation_id=5,
        observation_anchor_monotonic_ns=1_000_000_000,
        inference_started_monotonic_ns=1_000_000_100,
        inference_finished_monotonic_ns=1_000_001_000,
        step_dt_ns=100_000_000,
    )
    decoder = DexManiActionAdapter()
    chunk = decoder.decode(
        {"arm_qpos": np.tile(arm_hist[0][None, :], (4, 1)), "hand_qpos": np.zeros((4, 12))},
        context=ctx,
    )
    assert isinstance(chunk, JointActionChunk)
    assert chunk.arm_qpos.shape == (4, 7)
    assert chunk.hand_qpos is not None and chunk.hand_qpos.shape == (4, 12)
    expected = np.asarray(ctx.inference_finished_monotonic_ns, dtype=np.uint64) + (
        np.arange(1, 5, dtype=np.uint64) * np.uint64(ctx.step_dt_ns)
    )
    np.testing.assert_array_equal(chunk.target_monotonic_ns, expected)

    # EE / non-dict output fails closed in decode (defensive, §90)
    for bad in ({"ee_pose": np.zeros((4, 7))}, np.zeros((4, 7)), {"arm_qpos": np.zeros(7)}):
        try:
            decoder.decode(bad, context=ctx)
        except ValueError:
            pass
        else:
            raise AssertionError(f"decode({type(bad).__name__}) must raise ValueError")

    # ── load fails closed when dexmani_policy is absent ──
    sys.modules.pop("dexmani_policy", None)
    absent = DexManiPolicyBackend(config=cfg)
    try:
        absent.load()
    except ImportError:
        pass
    else:
        raise AssertionError("load() without dexmani_policy must raise ImportError")

    # ── EE-action checkpoint is a startup reject (§90) ──
    _install_fake_dexmani_policy("ee")
    ee_backend = DexManiPolicyBackend(config=cfg)
    try:
        ee_backend.load()
    except ValueError:
        pass
    else:
        raise AssertionError("EE-action checkpoint must raise ValueError on load")

    # ── joint agent: load -> infer -> decode -> reset -> close ──
    _install_fake_dexmani_policy("joint")
    backend = DexManiPolicyBackend(config=cfg)
    backend.load()
    raw = backend.infer({"arm_qpos": np.zeros(7), "hand_qpos": np.zeros(12)})
    chunk = decoder.decode(raw, context=ctx)
    assert chunk.arm_qpos.shape == (4, 7)
    assert chunk.hand_qpos is not None and chunk.hand_qpos.shape == (4, 12)
    backend.reset(run_generation=2)
    backend.close()
    assert backend._agent is None

    # ── infer before load fails closed ──
    try:
        DexManiPolicyBackend(config=cfg).infer({})
    except RuntimeError:
        pass
    else:
        raise AssertionError("infer before load must raise RuntimeError")

    print("check_dexmani_policy: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
