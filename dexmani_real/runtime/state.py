"""Main-process motion state; only SensorChannels crosses the spawn boundary."""

from dataclasses import dataclass, field
from threading import RLock

from dexmani_real.ipc.channels import SensorChannels, SensorChannelsConfig
from dexmani_real.runtime.safety import RunEndReason, SafetyState, StopRequest


@dataclass
class RuntimeState:
    sensors: SensorChannels
    run_id: int = 1
    run_ended_id: int = 0
    run_ended_reason: RunEndReason = RunEndReason.NONE
    error_state: bool = False
    estop_request: bool = False
    quit_requested: bool = False
    start_request: bool = False
    stop_request: StopRequest = StopRequest.NONE
    safety_state: SafetyState = SafetyState.DISARMED
    # Never held across an SDK call; keyboard callbacks revoke from another thread.
    motion_lock: object = field(default_factory=RLock, repr=False)

    @classmethod
    def create(
        cls, prefix: str = "dexmani", *,
        config: SensorChannelsConfig | None = None, mp_context=None,
    ) -> "RuntimeState":
        return cls(SensorChannels.create(prefix, config=config, mp_context=mp_context))
