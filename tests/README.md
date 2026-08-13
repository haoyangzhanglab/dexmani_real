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

The harness surfaced a pre-existing safety-critical bug in `_enter_measured_hold`:
it assigned the hold-send timestamp (`_hold_sent_at_s = time.monotonic()`) without
declaring it `nonlocal`, so the outer variable stayed `None`.  That made
`ControlHold.observe_delivery(None, …)` never report `applied`, so every
measured hold (PAUSE/STOP/QUIT/vr_stale/hand-recovered) faulted after the
0.75 s apply timeout, and `_complete_reanchor` could never fire.  It was fixed
with a one-line `nonlocal` (commit `7ce4813`); after the later `TeleopLoopState`
refactor the field lives as `ctx.hold_sent_at_s` (an attribute assignment,
which needs no `nonlocal`).  The re-anchor integration test
(`test_vr_stale_hold_releases_after_reanchor`) is the regression guard.

A second regression was caught by review, not the harness (the homing tests
only exercised the success path): `_execute_mode0_milestones_impl` returned a
bare `HomeResult` on every failure path but a `(HomeResult, np.ndarray)` tuple
on success, so the wrapper's tuple-unpack raised `TypeError` on any multi-
milestone homing failure.  Fixed, and `test_homing_failure_paths` now guards it.

## Not covered by the harness

- `_on_sigterm` is a one-line flag setter exercised only via the real SIGTERM
  handler; it is trivially verified by inspection.

## Real-hardware validation checklist

The harness reaches every behavior that is verifiable headless.  These require
the real xArm7/XHand and firmware, and are the terminal gate before trusting the
Phase-1.4 Tier B refactor (especially the `_hold_sent_at_s` fix, `7ce4813`):

1. **Measured-hold apply + re-anchor** — teleop, then PAUSE/STOP/vr-stale:
   the hold must log `applied` (not `delivery timed out`) and motion must resume
   after a fresh re-anchor.  This is the bug the harness caught.
2. **Mode-6 tracking** — fast wrist rotation produces no tracking-error warning
   spikes; the firmware is the velocity/acceleration backstop (C22/C24/C31).
3. **Collision recovery** — C24 → bounded measured-hold recovery; C22/C31 →
   sticky fault and safe stop.
4. **Homing** — return_home converges to canonical home through Mode-0
   milestones, settles (position + velocity dwell), and restores Mode 6.
5. **Hand** — contact/backlash/steady-state error is a valid outcome (not a
   freshness fault); tactile reset works; hand-home milestones ACK.
6. **Recording** — one full collect episode (v16) still reads back via
   `visualize_episode.py` and `replay_episode.py --dry-run`.

