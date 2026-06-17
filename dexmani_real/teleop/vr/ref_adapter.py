"""LeFranX-style pinky reference adapter for XHand retargeting."""

from __future__ import annotations

__all__ = ["XHandRefAdapter"]

from typing import Any

import numpy as np


EPS: float = 1e-8
PINKY: tuple[int, int, int, int] = (17, 18, 19, 20)


def _scale_chain(
    src: np.ndarray,
    dst: np.ndarray,
    ids: tuple[int, int, int, int],
    scale: float,
) -> None:
    p0 = src[ids[0]]
    p1 = src[ids[1]]
    p2 = src[ids[2]]
    p3 = src[ids[3]]

    dst[ids[1]] = p0 + scale * (p1 - p0)
    dst[ids[2]] = dst[ids[1]] + scale * (p2 - p1)
    dst[ids[3]] = dst[ids[2]] + scale * (p3 - p2)


class XHandRefAdapter:
    """LeFranX-style pinky ref adapter for XHand + DexPilot/vector retargeting."""

    def __init__(
        self,
        enable: bool = True,
        pinky_extension_range: tuple[float, float] = (0.03, 0.07),
        pinky_scale: tuple[float, float] = (1.2, 2.2),
        pinky_blend: float = 1.0,
        debug: bool = False,
    ) -> None:
        self.enable = bool(enable)
        self.pinky_extension_range = pinky_extension_range
        self.pinky_scale = pinky_scale
        self.pinky_blend = float(pinky_blend)
        self.debug = bool(debug)
        self.last_debug = {}

    def apply(
        self,
        ref_value: np.ndarray,
        hand: np.ndarray | None,
        origin_indices: np.ndarray,
        task_indices: np.ndarray,
    ) -> np.ndarray:
        if not self.enable or hand is None:
            return ref_value

        hand = np.asarray(hand, dtype=float)
        raw_ref = np.asarray(ref_value, dtype=float)
        origin_indices = np.asarray(origin_indices, dtype=int)
        task_indices = np.asarray(task_indices, dtype=int)

        mapped_hand = hand.copy()
        pinky_debug = self.adapt_pinky(hand, mapped_hand)
        mapped_ref = mapped_hand[task_indices] - mapped_hand[origin_indices]

        out = raw_ref.copy()
        changed_rows = []

        for row, (origin, task) in enumerate(zip(origin_indices, task_indices)):
            if not self.use_pinky_row(int(origin), int(task)):
                continue

            out[row] = (1.0 - self.pinky_blend) * raw_ref[row] + self.pinky_blend * mapped_ref[row]

            if self.debug:
                changed_rows.append({
                    "row": int(row),
                    "origin": int(origin),
                    "task": int(task),
                    "raw_norm": float(np.linalg.norm(raw_ref[row])),
                    "mapped_norm": float(np.linalg.norm(mapped_ref[row])),
                })

        if self.debug:
            self.last_debug = {
                "pinky": pinky_debug,
                "changed_rows": changed_rows,
            }

        return out

    def adapt_pinky(self, hand: np.ndarray, mapped_hand: np.ndarray) -> dict[str, Any]:
        mcp, pip, dip, tip = PINKY
        extension = float(np.linalg.norm(hand[tip] - hand[mcp]))
        low, high = self.pinky_extension_range

        if high > low:
            ratio = float(np.clip((extension - low) / (high - low), 0.0, 1.0))
        else:
            ratio = 0.0

        scale = self.pinky_scale[0] + (self.pinky_scale[1] - self.pinky_scale[0]) * ratio
        _scale_chain(hand, mapped_hand, PINKY, scale)

        length = self.chain_length(hand, PINKY)
        straightness = extension / length if length > EPS else 0.0

        return {
            "extension": extension,
            "extension_ratio": ratio,
            "straightness": straightness,
            "scale": scale,
            "blend": self.pinky_blend,
        }

    def use_pinky_row(self, origin: int, task: int) -> bool:
        return origin in PINKY or task in PINKY

    def chain_length(self, points: np.ndarray, ids: tuple[int, ...]) -> float:
        return float(
            np.linalg.norm(points[ids[1]] - points[ids[0]])
            + np.linalg.norm(points[ids[2]] - points[ids[1]])
            + np.linalg.norm(points[ids[3]] - points[ids[2]])
        )