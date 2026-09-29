"""Results for measured, ARMED home operations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HomeResult:
    ok: bool
    reason: str = ""
    interrupted: bool = False
