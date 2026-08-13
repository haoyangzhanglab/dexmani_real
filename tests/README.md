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
- `tests/test_integration.py` runs **both** loops on one `SharedStorage` so the
  coordinator's holds and HOME requests are actually consumed, applied, and
  acknowledged by the arm worker.

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
| both | HOME round-trip | `_handoff_control_hold_to_home` → `_do_configured_teleop_home` → `send_arm_home` → `_planned_homing` |
| both | vr_stale hold → re-anchor | `_enter_measured_hold` → `_complete_reanchor` |

## Bug the harness caught

`_enter_measured_hold` assigned `_hold_sent_at_s = time.monotonic()` without
declaring it `nonlocal`, so the outer `_hold_sent_at_s` stayed `None`.  That
made `ControlHold.observe_delivery(None, …)` never report `applied`, so every
measured hold (PAUSE/STOP/QUIT/vr_stale/hand-recovered) faulted after the
0.75 s apply timeout, and `_complete_reanchor` could never fire.  Fixed with a
one-line `nonlocal` (see git history).  The re-anchor integration test is the
regression guard for it.

## Not covered

- `_on_sigterm` is a one-line flag setter exercised only via the real SIGTERM
  handler; it is trivially verified by inspection.
- Real xArm7/XHand firmware behavior (Mode-6 tracking, collision recovery,
  homing convergence) is out of scope for this harness and requires hardware.
