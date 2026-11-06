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
    feedback_ns: int,
    control_ns: int,
    period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    measured = solver.measured_reference(measured)
    command = measured if reference is None else reference.command
    anchor = measured if reference is None else reference.anchor
    stamp = feedback_ns if reference is None else reference.submitted_ns
    if control_ns < stamp:
        raise ValueError("IK monotonic clock moved backwards")
    h = min((control_ns - stamp) * 1e-9, period_s)
    return measured, command, anchor, h
