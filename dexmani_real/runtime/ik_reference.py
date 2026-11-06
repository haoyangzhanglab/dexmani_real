"""运行层拥有的提交参考；准备尝试不改变状态。"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.control.ik import IKSolver


@dataclass(frozen=True)
class IKReference:
    command: np.ndarray
    anchor: np.ndarray
    submitted_ns: int


def prepare_reference(
    solver: IKSolver,
    reference: IKReference | None,
    measured: np.ndarray,
    *,
    control_ns: int,
    period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """首帧给一个名义周期；后续按成功提交后的间隔分配，至多一个周期。"""
    measured = solver.measured_reference(measured)
    if reference is None:
        command = anchor = measured
        h = period_s
    else:
        command, anchor = reference.command, reference.anchor
        if control_ns < reference.submitted_ns:
            raise ValueError("IK monotonic clock moved backwards")
        h = min((control_ns - reference.submitted_ns) * 1e-9, period_s)
    return measured, command, anchor, h
