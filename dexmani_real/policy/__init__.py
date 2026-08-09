from __future__ import annotations

__all__ = ["PolicyConfig", "policy_loop"]


def __getattr__(name: str):
    """Keep policy package import free of planners, HDF5, SDKs and UI modules."""
    if name in __all__:
        from dexmani_real.policy.vr_teleop_policy import PolicyConfig, policy_loop

        return {"PolicyConfig": PolicyConfig, "policy_loop": policy_loop}[name]
    raise AttributeError(name)
