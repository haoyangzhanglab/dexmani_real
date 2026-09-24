"""Offline calibration math and explicit calibration workflows."""

from pathlib import Path

CALIBRATION_STATE_DIR = Path(__file__).resolve().parent / "state"

CAMERAS_PATH = CALIBRATION_STATE_DIR / "cameras.json"
VR_TRANSFORM_PATH = CALIBRATION_STATE_DIR / "vr_transform.json"
