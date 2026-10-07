"""Real HTS parsing/assembly with scripted transports; no sockets or hardware."""

import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.config.hardware import VRParams
from dexmani_real.runtime.observation import read_observation
from dexmani_real.sensor.vr_worker import run_vr_worker


def wrist(side="Right", metadata=""):
    return f"{side} wrist{metadata}: 1,2,3,0,0,0,1"


def landmarks(side="Right", metadata=""):
    return f"{side} landmarks{metadata}: " + ",".join(["1,2,3"] * 21)


@pytest.fixture
def stream(monkeypatch):
    sdk = pytest.importorskip("hand_tracking_sdk")
    import hand_tracking_sdk.frame as frame

    clock = NS(now=1_000_000_000, idle=False)
    monkeypatch.setattr(frame, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr("time.monotonic_ns", lambda: clock.now)
    rows, steps, receivers = [], [], []
    shared = NS(
        is_running=NS(value=True),
        vr_ready=threading.Event(),
        vr_ring=NS(write=lambda row: rows.append(row.copy())),
    )

    class Receiver:
        def __init__(self, config):
            self.config, self.closed, self.reads, self.opens = config, False, 0, 0
            receivers.append(self)

        def open(self):
            self.closed = False
            self.opens += 1

        def close(self):
            self.closed = True

        def __enter__(self):
            self.open()
            return self

        def __exit__(self, *args):
            self.close()

        def recv_line(self):
            self.reads += 1
            if not shared.is_running.value:
                raise AssertionError("receiver read again after shutdown")
            if not steps:
                shared.is_running.value = False
                if clock.idle:
                    raise sdk.TransportTimeoutError("idle during shutdown")
                clock.now += 1
                return wrist()
            clock.now, value = steps.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

    for name in ("UDPLineReceiver", "TCPServerLineReceiver", "TCPClientLineReceiver"):
        monkeypatch.setattr(sdk, name, Receiver)
    return NS(sdk=sdk, clock=clock, shared=shared, rows=rows, steps=steps, receivers=receivers)


@pytest.mark.parametrize("advancing", [wrist, landmarks])
def test_one_advancing_component_cannot_refresh_stalled_component(stream, advancing):
    stream.steps[:] = [
        (1_000_000_000, wrist()),
        (1_000_000_100, landmarks()),
        (3_000_000_000, advancing()),
    ]
    run_vr_worker(stream.shared, VRParams())
    row = stream.rows[-1]
    assert int(row["recv_ts_ns"][0]) <= 1_000_000_100
    stream.shared.vr_ring.read_latest = lambda: (row, stream.clock.now, 2)
    robot = NS(read_state=lambda: NS(arm={"timestamp_ns": [stream.clock.now]}, hand=None))
    assert (
        read_observation(
            stream.shared, ExperimentConfig(), robot, require_hand=False, require_vr=True
        )
        is None
    )
    assert stream.receivers[0].closed and not stream.shared.vr_ready.is_set()


@pytest.mark.parametrize("transport", ["udp", "tcp_server", "tcp_client"])
def test_idle_shutdown_does_not_need_another_frame(stream, transport):
    stream.clock.idle = True
    run_vr_worker(stream.shared, VRParams(transport=transport))
    assert stream.receivers[0].reads == 1
    assert stream.receivers[0].closed and not stream.shared.vr_ready.is_set()


@pytest.mark.parametrize("boundary", ["write", "ready"])
@pytest.mark.parametrize("error", [ValueError, TypeError, AttributeError])
def test_ipc_failure_propagates_and_closes_receiver(stream, boundary, error):
    stream.steps[:] = [(100, wrist()), (200, landmarks())]

    def fail(*args):
        raise error("broken IPC")

    if boundary == "write":
        stream.shared.vr_ring.write = fail
    else:
        stream.shared.vr_ready.set = fail
    with pytest.raises(error, match="broken IPC"):
        run_vr_worker(stream.shared, VRParams())
    assert stream.receivers[0].closed and not stream.shared.vr_ready.is_set()


def test_conversion_and_bad_payload_recovery(stream):
    stream.steps[:] = [
        (10, "malformed input"),
        (20, "Head pose: 1,2,3,0,0,0,1"),
        (30, wrist("Left")),
        (40, landmarks("Left")),
        (50, wrist().replace("1,2,3", "nan,2,3")),
        (60, landmarks()),
        (70, wrist()),
    ]
    run_vr_worker(stream.shared, VRParams())
    assert len(stream.rows) == 1
    row = stream.rows[0][0]
    np.testing.assert_array_equal(row["wrist_pos"], [3, -1, 2])
    np.testing.assert_array_equal(row["wrist_quat_wxyz"], [1, 0, 0, 0])
    np.testing.assert_array_equal(row["head_pos"], [3, -1, 2])
    assert row["head_recv_ts_ns"] == 20 and row["recv_ts_ns"] == 60
    assert row["side"] == 0


@pytest.mark.parametrize("metadata", ["|t=-1", "|f=-1", f"|t={2**64}", f"|f={2**64}"])
def test_source_metadata_cannot_wrap_uint64(stream, metadata):
    stream.steps[:] = [(100, wrist(metadata=metadata)), (200, landmarks(metadata=metadata))]
    run_vr_worker(stream.shared, VRParams())
    assert stream.rows == []


@pytest.mark.parametrize("transport", ["udp", "tcp_server", "tcp_client"])
def test_timeout_then_fresh_data_remains_usable(stream, transport):
    stream.steps[:] = [
        (100, stream.sdk.TransportTimeoutError("no data yet")),
        (200, wrist()),
        (300, landmarks()),
    ]
    run_vr_worker(stream.shared, VRParams(transport=transport))
    row = stream.rows[-1]
    stream.shared.vr_ring.read_latest = lambda: (row, stream.clock.now, 1)
    robot = NS(read_state=lambda: NS(arm={"timestamp_ns": [stream.clock.now]}, hand=None))
    observation = read_observation(
        stream.shared, ExperimentConfig(), robot, require_hand=False, require_vr=True
    )
    assert observation is not None and observation.vr["recv_ts_ns"] == 200
    assert np.isnan(observation.vr["head_pos"]).all()
    assert stream.receivers[0].opens == 1 and stream.receivers[0].closed


@pytest.mark.parametrize("transport", ["tcp_server", "tcp_client"])
def test_disconnect_keeps_existing_transport_recovery(stream, monkeypatch, transport):
    delays = []
    monkeypatch.setattr("dexmani_real.sensor.vr_worker.time.sleep", delays.append)
    stream.steps[:] = [
        (100, stream.sdk.TransportDisconnectedError("disconnected")),
        (200, wrist()),
        (300, landmarks()),
    ]
    run_vr_worker(stream.shared, VRParams(transport=transport))
    assert len(stream.rows) == 1
    assert stream.receivers[0].opens == (2 if transport == "tcp_client" else 1)
    assert delays == ([0.25] if transport == "tcp_client" else [])
    assert stream.receivers[0].closed


def test_shutdown_during_reconnect_delay_does_not_reopen(stream, monkeypatch):
    stream.steps[:] = [(100, stream.sdk.TransportDisconnectedError("disconnected"))]
    monkeypatch.setattr(
        "dexmani_real.sensor.vr_worker.time.sleep",
        lambda _: setattr(stream.shared.is_running, "value", False),
    )
    run_vr_worker(stream.shared, VRParams(transport="tcp_client"))
    assert stream.receivers[0].opens == 1 and stream.receivers[0].closed


def test_transport_failure_after_readiness_clears_ready(stream):
    stream.steps[:] = [(100, wrist()), (200, landmarks()), (300, OSError("transport failed"))]
    with pytest.raises(OSError, match="transport failed"):
        run_vr_worker(stream.shared, VRParams())
    assert len(stream.rows) == 1
    assert stream.receivers[0].closed and not stream.shared.vr_ready.is_set()


@pytest.mark.parametrize("transport", ["udp", "tcp_server", "tcp_client"])
def test_shutdown_before_start_does_not_open_transport(stream, transport):
    stream.shared.is_running.value = False
    run_vr_worker(stream.shared, VRParams(transport=transport))
    assert stream.receivers[0].opens == 0
