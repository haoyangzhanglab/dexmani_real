"""Offline check: B1 single-controller connect uses one XHandControl per attempt.

Verifies the B1 refactor of ``XHand._retry_open_device`` without the vendor SDK:

1. discovery + open happen on the SAME XHandControl instance (no throwaway
   discovery control);
2. a failed open closes its control before the retry, and the retry does NOT
   re-run discovery (retry delay / stale-OP recovery wait are preserved);
3. the config-named path opens without any discovery call;
4. discovery that finds no device fails closed and leaves no dangling control;
5. a discovery exception closes the control and propagates;
6. discovery re-runs on a fresh connect (the trigger is ``config.device_name``,
   not a cached ``self.device_name``).

Run from the repo root:
    conda run -n real_robot python checks/offline/check_b1_single_controller.py
"""

from __future__ import annotations

import types

import dexmani_real.robot.xhand as xhand_mod
from dexmani_real.robot.xhand import XHand, XHandConfig


class _Err:
    def __init__(self, code: int) -> None:
        self.error_code = code


# Recorded sleeps so the check can assert the retry/post-recovery delays are
# still present (the B1 contract preserves them byte-for-byte).
_SLEEPS: list[float] = []


def _spy_sleep(seconds: float) -> None:
    _SLEEPS.append(seconds)


def _reset_sleeps() -> None:
    _SLEEPS.clear()


class _RecordingControl:
    instances: list["_RecordingControl"] = []

    def __init__(self) -> None:
        _RecordingControl.instances.append(self)
        self.enumerated = False
        self.opened_name: str | None = None
        self.closed = False

    def enumerate_devices(self, comm_type: str):
        self.enumerated = True
        return ["fake-eth-device"]

    def open_ethercat(self, device_name: str):
        self.opened_name = device_name
        return _Err(0)

    def close_device(self) -> None:
        self.closed = True


class _FlakyControl:
    instances: list["_FlakyControl"] = []

    def __init__(self) -> None:
        _FlakyControl.instances.append(self)
        self.enumerated = False
        self.opened_name: str | None = None
        self.closed = False

    def enumerate_devices(self, comm_type: str):
        self.enumerated = True
        return ["fake-eth-device"]

    def open_ethercat(self, device_name: str):
        self.opened_name = device_name
        # First attempt fails, the second succeeds.
        return _Err(0) if len(_FlakyControl.instances) > 1 else _Err(1)

    def close_device(self) -> None:
        self.closed = True


class _RaisingDiscoveryControl:
    instances: list["_RaisingDiscoveryControl"] = []

    def __init__(self) -> None:
        _RaisingDiscoveryControl.instances.append(self)
        self.closed = False

    def enumerate_devices(self, comm_type: str):
        raise OSError("fake discovery failure")

    def close_device(self) -> None:
        self.closed = True


def _patch_xhc(control_cls) -> None:
    xhand_mod.xhc = types.SimpleNamespace(XHandControl=control_cls)


def _check_single_controller_success() -> None:
    _reset_sleeps()
    _RecordingControl.instances = []
    _patch_xhc(_RecordingControl)
    x = XHand(XHandConfig(device_name=None, comm_type="EtherCAT", open_ethercat_retries=1))
    assert x._retry_open_device("EtherCAT") is True
    assert len(_RecordingControl.instances) == 1, len(_RecordingControl.instances)
    ctrl = _RecordingControl.instances[0]
    assert ctrl.enumerated, "discovery must run on the single control"
    assert ctrl.opened_name == "fake-eth-device", ctrl.opened_name
    assert not ctrl.closed, "successful control must not be closed"
    assert x.control is ctrl, "successful control is the live control"
    assert x.device_name == "fake-eth-device", x.device_name
    assert _SLEEPS == [], f"first-attempt success must not sleep: {_SLEEPS}"


def _check_retry_no_rediscovery() -> None:
    _reset_sleeps()
    _FlakyControl.instances = []
    _patch_xhc(_FlakyControl)
    x = XHand(XHandConfig(device_name=None, comm_type="EtherCAT", open_ethercat_retries=2))
    assert x._retry_open_device("EtherCAT") is True
    assert len(_FlakyControl.instances) == 2, len(_FlakyControl.instances)
    first, second = _FlakyControl.instances
    assert first.enumerated and first.opened_name == "fake-eth-device"
    assert first.closed, "failed control must be closed before retry"
    assert not second.enumerated, "retry must not re-run discovery"
    assert second.opened_name == "fake-eth-device", second.opened_name
    assert x.control is second, "successful retry control is the live control"
    # Preserved retry-delay contract: the first EtherCAT failure sleeps the
    # stale-OP recovery wait, then a late success sleeps the 1.0s stabilisation.
    assert len(_SLEEPS) == 2, f"expected two sleeps: {_SLEEPS}"
    assert _SLEEPS[0] >= XHand._STALE_OP_RECOVERY_WAIT_S, f"stale-OP wait dropped: {_SLEEPS}"
    assert _SLEEPS[1] == 1.0, f"post-recovery stabilisation dropped: {_SLEEPS}"


def _check_config_named_no_discovery() -> None:
    _reset_sleeps()
    _RecordingControl.instances = []
    _patch_xhc(_RecordingControl)
    x = XHand(XHandConfig(device_name="cfg-eth0", comm_type="EtherCAT", open_ethercat_retries=1))
    assert x._retry_open_device("EtherCAT") is True
    assert len(_RecordingControl.instances) == 1, len(_RecordingControl.instances)
    ctrl = _RecordingControl.instances[0]
    assert not ctrl.enumerated, "config-named connect must skip discovery"
    assert ctrl.opened_name == "cfg-eth0", ctrl.opened_name
    assert _SLEEPS == [], f"config-named success must not sleep: {_SLEEPS}"


def _check_no_device_fail_closed() -> None:
    _reset_sleeps()

    class _EmptyControl:
        instances: list["_EmptyControl"] = []

        def __init__(self) -> None:
            _EmptyControl.instances.append(self)
            self.closed = False

        def enumerate_devices(self, comm_type: str):
            return []  # no devices found

        def close_device(self) -> None:
            self.closed = True

    _patch_xhc(_EmptyControl)
    x = XHand(XHandConfig(device_name=None, comm_type="EtherCAT", open_ethercat_retries=1))
    assert x._retry_open_device("EtherCAT") is False
    assert x.error_state is True
    assert x.last_error_code == -2, x.last_error_code
    assert len(_EmptyControl.instances) == 1, len(_EmptyControl.instances)
    assert _EmptyControl.instances[0].closed, "no-device control must be closed"
    assert x.control is None, "no dangling control after failed discovery"


def _check_discovery_raise_closes_and_propagates() -> None:
    _reset_sleeps()
    _RaisingDiscoveryControl.instances = []
    _patch_xhc(_RaisingDiscoveryControl)
    x = XHand(XHandConfig(device_name=None, comm_type="EtherCAT", open_ethercat_retries=1))
    try:
        x._retry_open_device("EtherCAT")
    except OSError:
        pass
    else:
        raise AssertionError("discovery exception must propagate")
    assert len(_RaisingDiscoveryControl.instances) == 1, len(_RaisingDiscoveryControl.instances)
    assert _RaisingDiscoveryControl.instances[0].closed, "discovery control must be closed on raise"
    assert x.control is None, "no dangling control after discovery raise"


def _check_rediscovery_on_reconnect() -> None:
    _reset_sleeps()
    _RecordingControl.instances = []
    _patch_xhc(_RecordingControl)
    x = XHand(XHandConfig(device_name=None, comm_type="EtherCAT", open_ethercat_retries=1))
    assert x._retry_open_device("EtherCAT") is True
    assert x.device_name == "fake-eth-device"
    # Reconnect: discovery must run again even though self.device_name is cached,
    # because the trigger is config.device_name is None, not self.device_name.
    x.connected_flag = False
    assert x._retry_open_device("EtherCAT") is True
    assert len(_RecordingControl.instances) == 2, len(_RecordingControl.instances)
    assert _RecordingControl.instances[1].enumerated, "reconnect must re-run discovery"


def main() -> int:
    orig_sleep = xhand_mod.time.sleep
    orig_xhc = xhand_mod.xhc
    xhand_mod.time.sleep = _spy_sleep
    try:
        _check_single_controller_success()
        _check_retry_no_rediscovery()
        _check_config_named_no_discovery()
        _check_no_device_fail_closed()
        _check_discovery_raise_closes_and_propagates()
        _check_rediscovery_on_reconnect()
    finally:
        xhand_mod.time.sleep = orig_sleep
        xhand_mod.xhc = orig_xhc
    print("OK: B1 single-controller lifecycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
