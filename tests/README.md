# Headless hardware-loop harness

Runs the production loop bodies (`arm_loop`, `teleop_loop`) against a real
`SharedStorage` with fakes injected only at the device/operator seams that would
otherwise need hardware.  This is the regression gate for the Phase-1.4
closure extraction: it pins the loops' observable behavior so that a mechanical
refactor can be verified as zero-behavior-change.

## Run

```bash
conda run -n real_robot python -m pytest tests/ -q
```

No hardware, display, or `examples/` entry point is used.

## How it works

- Each loop runs in a **daemon thread**; `SharedStorage` (POSIX shm + `mp`
  primitives) is process-local and cross-thread safe.
- `arm_loop` does `from xarm.wrapper import XArmAPI` inside its body, so
  `tests/fakes/xarm_sdk.py` installs a fake `xarm`/`xarm.wrapper` into
  `sys.modules` (`FakeXArmAPI`).  The fake models `state`/`mode`/`error_code`
  and joint feedback; `set_servo_angle` moves the modelled position
  instantaneously so homing converges in bounded dwell time.
- `teleop_loop` needs no SDK, but does need (a) a pynput keyboard, (b) an audio
  player, and (c) a `signal.signal` registration (main-thread-only).  All three
  are patched headless in `tests/conftest.py::teleop_fakes`.  The readiness
  events it waits on (`arm_ready`, `vr_ready`, …) are set directly, and the
  sensor/robot workers are replaced by ring writers (`tests/fakes/workers.py`).

## Coverage

| Loop | Scenario | Closures / paths exercised |
|---|---|---|
| arm | startup → state + `arm_ready` | connect, initial publish |
| arm | endpoint servo | Mode-6 endpoint apply + state feedback |
| arm | STOP then RESUME | decelerated State-6 + measured-hold resume |
| arm | HOME, single-point path | `_confirm_home_dwell` |
| arm | HOME, multi-milestone path | `_execute_mode0_milestones` |
| arm | C31 collision fault | `_latch_collision_fault` → sticky `error_state` |
| teleop | BEGIN → RUNNING | capability wait, retargeter init, `_transition_or_fault` |
| teleop | PAUSE | `_enter_measured_hold` + measured-hold publication |
| teleop | stale arm feedback | consecutive-error fault path |
| teleop | hand feedback unhealthy | `_enter_hand_feedback_pause` (`advance_run_generation`) |
| teleop | clean shutdown | `is_running` teardown |

## Known gaps (not yet covered)

- `_complete_reanchor` (fresh-feedback hold release) and `_handoff_control_hold_to_home`
  (H/SAVE-AND-HOME) require a live arm worker ACKing a hold through `last_cmd_seq`.
  The natural next step is an integration test that runs `arm_loop` and
  `teleop_loop` together against the fake SDK, so the coordinator's holds are
  actually applied and re-anchored.
- `_on_sigterm` is a one-line flag setter exercised only via the real SIGTERM
  handler; it is trivially verified by inspection.
