"""Main-process state; sensors and a narrow xArm authority signal cross spawn."""

from dataclasses import dataclass, field
from threading import RLock

from dexmani_real.ipc.channels import SensorChannels, SensorChannelsConfig
from dexmani_real.runtime.safety import RunEndReason, SafetyState, StopRequest


@dataclass
class RuntimeState:
    sensors: SensorChannels | None = None
    running: bool = True
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
    arm_authority: object = field(default=None, repr=False)
    arm_emergency: object = field(default=None, repr=False)

    def end_session(self) -> None:
        """End local authority and notify sensor workers when this session has any."""
        with self.motion_lock:
            self.running = False
            if self.arm_authority is not None:
                self.arm_authority.value = 0
            if self.sensors is not None:
                self.sensors.is_running.value = False

    @classmethod
    def create(
        cls, prefix: str = "dexmani", *,
        config: SensorChannelsConfig | None = None, mp_context=None,
    ) -> "RuntimeState":
        if config is None or not (config.camera or config.vr or config.pointcloud):
            return cls()
        return cls(SensorChannels.create(prefix, config=config, mp_context=mp_context))
