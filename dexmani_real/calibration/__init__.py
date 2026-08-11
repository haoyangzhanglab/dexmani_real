"""Offline-testable calibration helpers."""

from dexmani_real.calibration.vr_heading import (
    DEFAULT_HEADING_QUALITY_GATE,
    HeadingEstimate,
    HeadingQualityGate,
    build_heading_config,
    estimate_heading,
    forward_from_quat_wxyz,
    reference_sample_from_vr_frame,
    timestamp_is_fresh,
    write_json_atomic_with_backup,
)

__all__ = [
    "DEFAULT_HEADING_QUALITY_GATE",
    "HeadingEstimate",
    "HeadingQualityGate",
    "build_heading_config",
    "estimate_heading",
    "forward_from_quat_wxyz",
    "reference_sample_from_vr_frame",
    "timestamp_is_fresh",
    "write_json_atomic_with_backup",
]
