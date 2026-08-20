# DexMani Real Golden Paths

This document defines the behavior that refactors must preserve. It is an
acceptance checklist, not a replacement for source code, configuration, or
hardware safety procedures.

## Safety boundary

Do not run a hardware path as part of ordinary development or CI. The commands
in the **hardware acceptance** column may connect to or move hardware and
require an operator's explicit approval, a clear workspace, and the normal
site safety procedure.

Every refactor must first run the applicable offline checks. Hardware
acceptance is required only when the changed path reaches the relevant device
or physical output.

## Invariants shared by every path

- Existing CLI options and their meanings remain stable unless a change is
  explicitly requested.
- Resolved runtime configuration remains the canonical source of operational
  defaults.
- `SafetyState`, e-stop, worker startup/shutdown, and device ownership retain
  their existing semantics.
- Robot command and observation shapes, units, coordinate frames, control
  rates, and episode schemas do not change incidentally.
- A refactor must not add import-time hardware, camera, worker, or thread
  startup.

## Paths

| Path | Entry point and main modules | Offline gate | Hardware acceptance (operator only) |
|---|---|---|---|
| XArm initialize, enable, and home | `examples/keyboard_teleop.py` → `robot.arm_loop` → `robot.xarm7`; home request → `robot.homing` | Compile and inspect the arm worker configuration and home command contract. | Start keyboard teleop with the hand secured as required by the CLI; verify readiness, explicit home, software disarm, and clean shutdown. |
| XHand initialize and control | `examples/xhand_control_example.py` (standalone vendor diagnostic), or `robot.hand_process` → `robot.xhand` in coupled workflows | Compile; do not invoke the standalone example because it always commands the hand. | Verify connection, state readback, configured home/preset behavior, stop, and disconnect with the workspace clear. |
| Keyboard teleoperation | `examples/keyboard_teleop.py` → `teleop.keyboard_session` / `teleop.keyboard` / `planning` / `policy.safety` → arm and optional hand workers | `python examples/keyboard_teleop.py --help` | Verify keyboard command mapping, stale-feedback rejection, e-stop, worker failure handling, and shutdown. |
| VR teleoperation and data collection | `examples/collect_teleop.py` → `teleop.session` / `teleop.loop` → arm/hand/VR/camera/recorder workers | `python examples/collect_teleop.py --help` and configuration-print path when configuration inputs change. | Verify readiness ordering, ARMED transition, record/pause/stop/discard, episode publication, and clean shutdown. |
| Episode replay | `examples/replay_episode.py` → `robot.episode_replay` → `recording.episode_reader` → arm/hand workers | `python examples/replay_episode.py --help`; use focused loader/preflight tests for data-path changes. The replay command itself always owns physical hardware execution. | Verify preflight rejection, command quiescence, replay, return-home prompt, e-stop, and retained partial result behavior. |
| Camera calibration | `examples/calibrate_camera.py` → `sensor.realsense` / `planning` → arm worker | `python examples/calibrate_camera.py --help` | Verify explicit operator procedure, arm safety gate, calibration output format, and failure cleanup. |
| Policy deployment | `examples/run_policy.py` → `deployment.lifecycle` → configured inference child and device workers | `python examples/run_policy.py --help` and `--print-config` with a valid deployment config. | Verify model/adapter loading, action validation, safety rejection, device worker lifecycle, and shutdown. |

## Baseline commands

Run these from the repository root in the configured offline environment:

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples tests
conda run -n real_robot python -m unittest tests.test_episode_replay_contract
conda run -n real_robot python examples/collect_teleop.py --help
conda run -n real_robot python examples/keyboard_teleop.py --help
conda run -n real_robot python examples/replay_episode.py --help
conda run -n real_robot python examples/calibrate_camera.py --help
conda run -n real_robot python examples/run_policy.py --help
```

For replay data-path changes, test the episode loader and preflight directly
against an approved fixture. Do not add a fixture containing real-robot data
without confirming that it is appropriate to store in the repository. Running
the replay CLI with an episode path is always a hardware operation.
