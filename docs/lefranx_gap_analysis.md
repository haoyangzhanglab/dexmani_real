# LeFranX -> DexMani: Unified Gap Analysis

> **This is the single authoritative report.** Supersedes: `lefranx_factcheck_report.md`, `lefranx_deep_dive_round2.md`, `lefranx_unified_gap_analysis.md`, `lefranx_round3_findings.md`.
> 2026-07-13 | 41 open gaps (2 shipped: F1, G7) | 7 rejected | ~10,500 LOC net-new

---

## Key Corrections from Earlier Reports

| # | Correction |
|---|-----------|
| 1 | **R1#6 (stats.json) -- CLOSED.** `norm_stats.json` auto-generated on every Zarr export (`export_hdf5_to_zarr.py:608-620`), unconditional. Not a gap. |
| 2 | **`validate_action()` is 4 checks, not 2.** CLAUDE.md description is stale. Code at `validate.py:38-57`. |
| 3 | **`limit_jerk()` exists but has zero call sites.** In `signal_utils.py:29-75`, exported via `utils/__init__.py`, never imported or called. |
| 4 | **controller.py docstring references deleted RECORDING state.** Actual enum has 4 states (IDLE/TELEOP/PAUSED/EMERGENCY_STOP). |
| 5 | **DataValidator runs 7 checks, not 5-6.** Docstring undercounts; `timestamp_monotonicity` is the 7th. |

---

## A. Motion Smoothing & Control Quality

### A1. Jerk Limiting -- P0 -- 30 LOC
**What:** xArm mode 6 limits velocity/acceleration only; no jerk constraint. DexMani's `limit_jerk()` (`signal_utils.py:29-75`) is fully implemented but wired to zero call sites.
**Why:** Without jerk limiting, acceleration changes are discontinuous, causing mechanical shock.
**LeFranX:** `franka_server.cpp:80-83` -- Ruckig enforces per-joint jerk/accel/vel triple limits at 1kHz.
**DexMani:** `inner_loop.py:60-61` -- scalar `joint_max_speed`/`joint_max_acc`, no jerk param.
**Target:** `robot/inner_loop.py` -- call `limit_jerk()` in `_send_target()` on velocity delta before dispatch.

### A2. Controlled Stop on Disconnect -- P0 -- 50 LOC
**What:** `emergency_stop()` calls `arm.set_state(4)` (hard stop). LeFranX uses Ruckig to ramp velocity to zero over ~1s.
**Why:** Hard stop stresses joints and may cause the arm to sag. `_hold_position()` already exists.
**LeFranX:** `franka_server.cpp:490-536` -- decreasing velocity until `max_vel < 0.001 rad/s` or 1s timeout.
**DexMani:** `xarm7.py:158-163` -- `stop()` = hard stop; `inner_loop.py:372-386` -- `_hold_position()` exists, called on timeout but NOT in normal disconnect path.
**Target:** `robot/inner_loop.py` -- add `_hold_position()` before `arm.disconnect()` in `_run()` finally block.

### A3. Post-IK Joint-Space EMA -- P1 -- 15 LOC
**What:** Cartesian EMA before IK cannot suppress IK solver outputting different joint configs between adjacent frames (redundant solution jumps).
**Why:** Joint-space smoothing suppresses redundant-joint-config jitter that Cartesian EMA misses.
**LeFranX:** `arm_ik_processor.py:360-363` -- `alpha*current + (1-alpha)*previous` on all 7 joints, alpha=0.3.
**DexMani:** `pipeline.py:131-148` -- `ema_smooth_pose()` on Cartesian pose before IK; IK result used raw.
**Target:** `teleop/core/pipeline.py` -- one line: `ema_smooth(ik_result.qpos, self._prev_qpos, alpha=0.6)` after IK.

### A4. Per-Joint Motion Constraints -- P2 -- 20 LOC
**What:** Arm velocity/acceleration limits are uniform scalars; hand delta clip (`max_delta_rad=0.3`) is also scalar. Base joints (J0-J2) have high inertia, wrist joints (J4-J6) are light -- uniform limits are either too tight or too loose.
**Why:** Per-joint limits allow tighter constraint on delicate wrist joints without over-constraining the base.
**LeFranX:** `franka_server.cpp:80-82` -- per-joint vel/acc/jerk arrays.
**DexMani:** `inner_loop.py:60-61` -- scalar limits; `xhand.py:524-527` -- scalar delta clip.
**Target:** `robot/inner_loop.py` (change `joint_max_speed`/`joint_max_acc` to `np.ndarray(7,)`), `robot/xhand/xhand.py` (accept `np.ndarray(12,)` for `max_delta_rad`).

### A5. VR Dead Zone -- P2 -- 40 LOC
**What:** No VR dead zone. Tiny VR tracking noise causes continuous micro-motion of the real arm.
**Why:** Suppresses tremor and tracking noise when the operator intends to hold still.
**LeFranX:** `config_franka_fer_vr.py:31-32` -- declared `position_deadzone=0.001`, `orientation_deadzone=0.03` (but never actually applied in code).
**DexMani:** `arm_mapper.py:38` -- `max_delta_rot_rad=1.0` (~57 deg) is a jump detector, not a dead zone.
**Target:** `teleop/vr/arm_mapper.py` -- add delta thresholding in `map()`: skip update if delta below 1mm / ~1deg.

---

## B. Safety Architecture

### B1. validate_action() Safety Expansion -- P0 -- 150 LOC
**What:** 4 basic checks (robot error, arm connection, joint-limit clipping). No torque/current/temperature/workspace/collision gating. CLAUDE.md calls this "a hard prerequisite before autonomous policy rollouts."
**Why:** Without these gates, autonomous policy commands cannot be safely executed without human supervision.
**LeFranX:** torque/current/temperature monitoring gating in control loop.
**DexMani:** `validate.py:38-57` -- 4 checks; params accepted but unused (`actual_arm_qpos`, `env_collision_check`); `_ARM_TORQUE_LIMIT_NM` (`types.py:23`) defined but never referenced.
**Target:** `robot/validate.py` -- add 5 gates: torque, current, temperature, workspace, collision.

### B2. TCP Health Probe -- P0 -- 141 LOC
**What:** `connect()` directly instantiates SDK objects with no pre-connection probing. SDK init timeouts are slow and uninformative.
**Why:** A 2s TCP SYN check fails faster and gives actionable diagnostics ("arm unreachable at 192.168.1.x:502").
**LeFranX:** pre-connection health checks before SDK init.
**DexMani:** `interface.py:103-107` -- direct SDK construction; `xarm7.py:79-108` -- single constructor, terminal on failure.
**Target:** `robot/interface.py`, `robot/xarm7/xarm7.py`, `robot/xhand/xhand.py` -- add `_probe_tcp(host, port, timeout=2.0)`.

### B3. Partial-Connect Cleanup + Hand Reconnect Guard -- P0 -- 133 LOC
**What:** (a) If arm connects but hand fails, arm stays connected in zombie state -- no rollback. (b) `disconnect()`/`emergency_stop()`/`clear_error()` chain arm-then-hand with no per-component try/except -- an exception in arm skips hand. (c) `XHand.connect()` has no re-entry guard -- repeated calls leak device handles.
**Why:** Resource leak on partial failure; emergency stop must attempt to stop everything.
**LeFranX:** per-component try/except in connect/disconnect; `DeviceAlreadyConnectedError` hard-fail.
**DexMani:** `interface.py:103-111` -- unconditional chain; `xhand.py:245-280` -- no re-entry guard; `xarm7.py:80-81` -- silent return on double-connect.
**Target:** `robot/interface.py` (rollback on partial connect, per-component try/except in disconnect/emergency_stop/clear_error), `robot/xhand/xhand.py` (re-entry guard).

### B4. ArmInnerLoop Auto-Reconnect -- P1 -- 200 LOC
**What:** ArmInnerLoop sets `_error_state = True` and exits thread on arm error. No retry -- caller must detect and restart the entire inner loop. XHand already has `reset_connection()` with circuit-breaker pattern.
**Why:** Transient errors (communication glitch) should trigger reconnect, not permanent thread death.
**LeFranX:** `franka_server.cpp` -- main loop disconnects and reconnects on control exceptions.
**DexMani:** `inner_loop.py:287-298` -- error -> `_error_state=True` + `break` (thread exit); `xhand.py:427-451` -- `reset_connection()` with retry.
**Target:** `robot/inner_loop.py` -- wrap main loop in retry (max 3 attempts, 2s cooldown).

### B5. Watchdog Process -- P2 -- 200 LOC
**What:** All safety timeouts run in-process. If the Python process crashes, daemon threads die too. xArm firmware mode 6 holds position on disconnect (hardware safety net) but no software watchdog exists.
**Why:** A separate watchdog process can call `arm.stop()` if the main process heartbeat stops.
**LeFranX:** C++ server independently holds position on Python crash.
**DexMani:** `inner_loop.py:64` -- `target_timeout_s=0.2` (in-process); no cross-process heartbeat.
**Target:** New `robot/watchdog.py` -- `multiprocessing.Process`, heartbeat Event, independent XArmAPI connection, stop on timeout >1s.

---

## C. Training Infrastructure

### C1. Gym Environment + Unified Observation Interface -- P0 -- 400 LOC
**What:** Zero `gym.Env` subclasses. No `observation_space`/`action_space`/`reset()`/`step()`. Observation keys scattered across `_OBS_KEYS` (export) and `_DEFAULT_OBS_KEYS` (replay_buffer) with no single authoritative source.
**Why:** A standard Gym env makes DexMani compatible with the entire RL ecosystem (SB3, RLlib, LeRobot).
**LeFranX:** `gym_manipulator.py` -- composable wrapper chain: `RobotEnv` -> `AddJointVelocity` -> `AddCurrent` -> `EEObs` -> `ImageCropResize` -> `RewardWrapper` -> `TimeLimit` -> `BatchCompatible`.
**DexMani:** global search `gym.Env`, `observation_space`, `action_space` -- zero results.
**Target:** New `training/env.py` (env wrapping RobotInterface, obs: arm_qpos(7)+arm_qvel(7)+eef_pos(3)+eef_rot6d(6)+hand_qpos(12)=35d, action: 19d), `policy/observation.py` (unified feature schema).

### C2. Training Pipeline -- P0 -- 2500 LOC
**What:** Zero `torch.nn.Module` subclasses, training loops, optimizers, or schedulers across all 98 .py files. Project is pure teleop + data collection.
**Why:** This is the single largest gap -- without it, DexMani cannot train policies on its own data.
**LeFranX:** `train_act_policy.py` (ACT: chunk_size=8, dim_model=512, ResNet18 VAE, 100k steps), `train_dp_policy.py` (Diffusion: DDPM, horizon=16, 100 timesteps).
**DexMani:** global search `import torch`, `nn.Module`, `DataLoader` -- zero results.
**Target:** New `training/train.py` (~800 LOC), `training/models/act.py` (~500 LOC), `training/models/diffusion.py` (~800 LOC), `training/config.py` (~150 LOC).

### C3. Training Resume/Checkpoint Lifecycle -- P0 -- 150 LOC
**What:** No checkpoint save/load infrastructure, no `--resume` flag, no training state serialization.
**Why:** Long training runs (~100k steps) must be resumable after interruption.
**LeFranX:** `train.py` -- explicit `cfg.resume` flag + `save_checkpoint()` with optimizer/scheduler state.
**DexMani:** zero checkpoint infrastructure.
**Target:** New `training/checkpoint.py` -- save/load model+optimizer+scheduler+step, resume flag.

### C4. PyTorch Dataset + Lazy Loading -- P1 -- 300 LOC
**What:** `ReplayBuffer` returns `list[np.ndarray]`, no `torch.utils.data.Dataset` subclass. Zarr export sets chunks but `ReplayBuffer.from_zarr()` eagerly materializes entire arrays into memory.
**Why:** DataLoader integration (num_workers, pin_memory, shuffle) and lazy loading are essential for training on large datasets.
**LeFranX:** LeRobotDataset with lazy loading from chunked storage.
**DexMani:** `replay_buffer.py:172` -- `np.asarray(root["data"]["obs"])` eager load; Zarr chunks set at `export_hdf5_to_zarr.py:454-462` but not exploited.
**Target:** New `training/dataset.py` (~150 LOC) + modify `recording/replay_buffer.py` (lazy mode, ~150 LOC).

### C5. Reward Functions -- P2 -- 200 LOC
**What:** No per-step reward signal. Only episode-level binary `success` in `/meta.attrs`.
**Why:** RL training requires dense or sparse per-step rewards. Episode-level success covers sparse-reward baseline only.
**LeFranX:** `RewardWrapper` -- learned `Classifier` model on camera images outputs success probability; prob > 0.7 -> reward=1.0 + terminated.
**DexMani:** `EpisodeRecorder.stop_episode(success=True/False)` stores binary flag. No per-step reward dataset.
**Target:** New `training/rewards.py` -- sparse reward from episode labels (scheme A, ~20 LOC) + optional learned classifier (scheme B, ~150 LOC).

### C6. Image Preprocessing Pipeline -- P2 -- 400 LOC
**What:** Three gaps: (a) no composable augmentation (C1: random crop, color jitter, normalization); (b) no ROI crop + resize with black-frame detection (C3); (c) Zarr export omits camera frames entirely (`_OBS_KEYS` is kinematic-only).
**Why:** Vision-based policies require augmentation to avoid overfitting; black-frame detection catches silent camera failures at runtime.
**LeFranX:** `crop_dataset_roi.py` (interactive ROI), `ImageCropResizeWrapper`, visual augmentation pipeline.
**DexMani:** `export_hdf5_to_zarr.py` -- `_OBS_KEYS` only `arm_qpos`, `arm_ee`, `hand_qpos`; no `torchvision` import anywhere.
**Target:** New `training/transforms.py` (~150 LOC), new `sensor/image_wrapper.py` (ROI + black-frame, ~250 LOC), modify `tools/export_hdf5_to_zarr.py` (include camera, ~50 LOC).

### C7. Joint Limit Auto-Discovery -- P1 -- 150 LOC
**What:** Joint limits are hardcoded. LeFranX has `find_joint_limits.py` that auto-scans actual hardware limits.
**Why:** Hardcoded limits may not match physical hardware after maintenance or calibration.
**LeFranX:** `find_joint_limits.py` -- automated hardware limit discovery routine.
**DexMani:** joint limits set in config, no discovery tool.
**Target:** New `tools/find_joint_limits.py`.

### C8. Coordinated Arm+Hand Homing -- P1 -- 100 LOC
**What:** DexMani's return-to-home is arm-only. LeFranX coordinates arm and hand as a unit.
**Why:** A coordinated homing sequence prevents the hand from colliding with the desk while the arm moves to home.
**LeFranX:** arm+hand homing as coordinated unit.
**DexMani:** `controller.py` -- arm-only two-phase home (EEF Cartesian path + joint-space alignment).
**Target:** `teleop/core/controller.py` -- add hand homing phase to return-to-home sequence.

### C9. Distributed Actor-Learner RL -- P1 -- 2500 LOC
**What:** LeFranX has complete gRPC-based distributed RL (HIL-SERL): learner (1215 LOC) + actor + gRPC service. DexMani has none.
**Why:** Required for scaling beyond single-robot training. Only needed when scaling up.
**LeFranX:** full gRPC distributed architecture with separate learner/actor processes.
**DexMani:** zero distributed training infrastructure.
**Target:** New `training/learner.py`, `training/actor.py`, `training/grpc_service.py`.

---

## D. Policy Deployment & Evaluation

### D1. Policy Deployment Suite -- P0 -- ~1200 LOC
**What:** No policy loading, inference, or rollout code. `SharedSyncPrimitives` (cross-process handshake) and `TeleopControllerConfig.synchronized` flag exist -- infrastructure for policy deployment is there, only the inference layer is missing. This entry merges: R1#5 (deploy scripts), R2#T3 (inference controller), R2#D1 (real replay), R2#I3 (action chunking).
**Why:** This is the bridge from data collection to autonomous execution. Without it, trained policies are only checkpoints on disk.
**LeFranX:** `dual_robot_deploy_act.py` (overlapping chunks, chunk_size=8, query_frequency=3, EMA alpha=0.5, safetensors loading, stats normalization), `dual_robot_deploy_dp.py` (obs_history window n_obs_steps=2, DDIM/DDPM), `dual_robot_replay.py` (~280 LOC, open-loop HDF5 replay on hardware).
**DexMani:** `shm/sync_primitives.py` -- `SharedSyncPrimitives` two-phase handshake exists; global search `action_chunk`, `temporal_ensemble`, `safetensors`, `load_state_dict` -- zero results.
**Target:** New `tools/deploy_act.py` (~200 LOC), `tools/deploy_dp.py` (~220 LOC), `examples/real/deploy_policy.py` (~60 LOC), `teleop/core/policy_controller.py` (~400 LOC), `tools/replay_on_robot.py` (~300 LOC); modify `teleop/core/controller.py` (~15 LOC).

### D2. Policy Evaluation Framework -- P2 -- 60 LOC
**What:** No automated policy evaluation loop (rollout runner, success rate computation). However, `SharedSyncPrimitives`, `ArmInnerLoop` synchronized mode, `TeleopController` synchronized mode, and Diffusion Policy Zarr export are all built and wired -- only the eval orchestration layer is missing.
**Why:** Systematic evaluation is essential for comparing policy checkpoints and detecting regressions.
**LeFranX:** `eval_policy.py` -- load checkpoint, build env, run N rollouts, compute success rate.
**DexMani:** `sync_primitives.py:1-47` -- two-phase handshake for policy integration; `controller.py:74-77` -- synchronized mode flag.
**Target:** New `training/eval.py` (~30 LOC scaffolding) + new `tools/eval_policy.py` (~30 LOC CLI).

### D3. Policy Server/Client Separation -- P2 -- 800 LOC
**What:** LeFranX separates GPU inference (policy_server.py) from robot control (robot_client.py) via TCP. DexMani has all inference in-process.
**Why:** GPU inference should run on a separate machine or at least a separate process to avoid blocking the realtime control loop.
**LeFranX:** `policy_server.py` (GPU inference server) + `robot_client.py` (TCP client on robot machine).
**DexMani:** no client/server separation for inference.
**Target:** New `policy/server.py`, `policy/client.py`.

---

## E. Data Infrastructure

### E1. Video-Based Image Storage -- P1 -- 100 LOC
**What:** `EpisodeRecorder` stores raw uint8 RGB per timestep in HDF5 with no compression filter. One frame per 50Hz grid slot. No mp4/ffmpeg encoding.
**Why:** Raw frame storage is ~100x larger than H.264 for equivalent visual quality. At scale this determines whether data fits on disk.
**LeFranX:** video-based storage with compression.
**DexMani:** `episode_recorder.py:258-264` -- `create_dataset(chunks=True)` with default `compression=None`. Zarr export (`export_hdf5_to_zarr.py:431`) already proves Blosc-zstd value downstream.
**Target:** `recording/episode_recorder.py` -- add `compression='gzip'` or `compression=32001` (Blosc) to HDF5 datasets; optionally add mp4 encoding path.

### E2. Per-Feature Normalization Strategy -- P2 -- 200 LOC
**What:** `compute_norm_stats()` returns only z-score params (mean/std). Joint angles are bounded [-pi, pi] and suit min-max; velocities are unbounded and suit z-score. Uniform strategy is suboptimal. Docstring falsely claims "Welford-style incremental" but uses batch `np.mean()`/`np.std()`.
**Why:** Different feature distributions need different normalization for optimal policy learning.
**LeFranX:** per-feature-type `NormalizationMode` (MIN_MAX vs Z_SCORE).
**DexMani:** `export_hdf5_to_zarr.py:378-404` -- z-score only; `replay_buffer.py:355-373` -- `(x-mean)/std` only.
**Target:** `tools/export_hdf5_to_zarr.py`, `recording/replay_buffer.py` -- add `NormalizationMode` enum, bounded features use min-max.

### E3. Dataset-Level Episode Filtering -- P2 -- 200 LOC
**What:** No mechanism to filter episodes by quality metrics at the dataset level.
**Why:** Low-quality episodes (high IK failure rate, excessive clamping) degrade policy training.
**LeFranX:** dataset-level quality filtering by episode metadata.
**DexMani:** `DataValidator` runs offline validation but no filtering tool; sidecar JSON has `held_ratio`, `classification` but no consumer.
**Target:** New `tools/filter_episodes.py`.

### E4. Dual Replay Buffer -- P3 -- 500 LOC
**What:** `ReplayBuffer` is "read-only by design" -- `from_hdf5()`/`from_zarr()` only. No `add()`/`push()`/`sample(batch_size)`. No dual-buffer (offline demo + online interaction).
**Why:** Online RL (HIL-SERL, IQL) interleaves offline demo data with online interaction data.
**LeFranX:** dual-buffer architecture with both offline and online data sources.
**DexMani:** `replay_buffer.py:1-4` -- docstring: "Read-only by design."
**Target:** New `training/dual_replay_buffer.py` or major extension of `recording/replay_buffer.py`.

---

## F. Observability & Diagnostics

### F1. File-Based Structured Logging -- P0 -- 50 LOC -- ✅ SHIPPED (O1)
**What (RESOLVED):** `get_logger()` now attaches both a `StreamHandler(sys.stdout)` and a shared `FileHandler`. Log dir from `$DEXMANI_LOG_DIR` (default `~/.dexmani/logs/`), file name `dexmani_{YYYYmmdd_HHMMSS}.log`; fail-safe (falls back to stdout-only on OSError).
**Why:** Console-only logging means field issues during autonomous operation are unrecoverable -- no persistent log for post-mortem.
**LeFranX:** per-session rotating file logs.
**DexMani:** `utils/log.py:26-47` -- `_get_file_handler()` creates the shared `FileHandler`; `get_logger()` (`:49-60`) attaches both handlers. 60-line file.
**Target:** DONE via `_get_file_handler()` (plain `FileHandler`, not rotating). Rotation still not implemented if that is desired.

### F2. Teleoperator Status API -- P0 -- 50 LOC
**What:** No unified `get_status()` returning a structured dict of all subsystem health. Individual methods exist (`QuestHandTracker.get_status()`, `MultiCameraManager.get_status()`) but are never combined. `RobotInterface.is_error()` is partial.
**Why:** Any dashboard, health-check script, or automated monitor needs a single entry point.
**LeFranX:** unified status aggregation.
**DexMani:** `controller.py:685-716` -- `_print_status()` is a logging method, not a programmatic API.
**Target:** `teleop/core/controller.py` (add public `get_status()`), `robot/interface.py` (add `get_status()`).

### F3. Latency Tracking with Breakdown -- P1 -- 50 LOC
**What:** `compute_action()` returns `(RobotAction, dict)` with only `ik_ok`/`retarget_ok` flags. Zero `time.perf_counter()` calls. Only whole-tick overrun detection exists.
**Why:** Without per-stage timing (VR read, IK, retarget, validate, send), performance regressions are invisible.
**LeFranX:** per-stage latency instrumentation.
**DexMani:** `pipeline.py:59-90` -- no timing; `controller.py:255,384-387` -- whole-tick overrun only.
**Target:** `teleop/core/pipeline.py`, `teleop/core/controller.py` -- instrument `compute_action()` + accumulate stats.

### F4. Loop Frequency Reporting -- P1 -- 50 LOC
**What:** Only throttled "over budget" warnings when tick exceeds target, plus a one-time startup message with configured Hz. No proactive per-second achieved-frequency reporting.
**Why:** Proactive Hz reporting catches slow degradation before it becomes an overrun.
**LeFranX:** regular frequency reporting.
**DexMani:** the control loop now uses `RateManager` -- `rate_manager.py:77-88` throttled overrun warning (~every 50 cycles); `RateManager.overdue_ratio` (`rate_manager.py:119`) exists but never called. (`RateLimiter` remains only in example scripts.)
**Target:** `teleop/core/controller.py`, `utils/rate_limiter.py`.

### F5. Periodic Performance Statistics (Timing) -- P1 -- 17 LOC
**What:** `_print_status()` already prints rich status every 2s (IK/retarget counts, jlimit%, manipulability, VR age). Only per-tick timing aggregation (avg/min/max ms) is missing.
**Why:** Adding avg/min/max tick-time to the existing status line makes runtime performance instantly visible.
**LeFranX:** periodic timing statistics.
**DexMani:** `controller.py:685-716` -- rich status output; tick timing measured at line 255/384 but not accumulated.
**Target:** `teleop/core/controller.py` -- accumulate tick times in a deque, print avg/min/max in `_print_status()`.

### F6. WandB Experiment Tracking -- P1 -- 200 LOC
**What:** Zero cloud experiment tracking. Global grep for `wandb`, `tensorboard`, `mlflow`, `SummaryWriter` across 98 .py files returned zero hits. This is a future-training concern (no training code exists yet).
**Why:** Without experiment tracking, comparing training runs requires manual log parsing. WandB is de-facto standard for robot learning.
**LeFranX:** `WandBConfig` + `WandBLogger` with full training metric logging.
**DexMani:** zero cloud-metrics infrastructure.
**Target:** New `training/wandb_logger.py`.

### F7. FPSTracker Abstraction -- P2 -- 200 LOC
**What:** Three timing abstractions exist (`RateLimiter`, `RateManager`, `StreamStats`) but all are rate-control-oriented, not rate-observation. No simple `FPSTracker` dataclass tracking cumulative count / elapsed time.
**Why:** A reusable FPS tracker simplifies monitoring in any loop (teleop, recording, future inference).
**LeFranX:** reusable FPS tracking abstraction.
**DexMani:** `RateManager` now instantiated by both controllers (`teleop/core/controller.py:153`, `robot/inner_loop.py:262`); `StreamStats` fully implemented and exported but still never instantiated.
**Target:** New `utils/fps_tracker.py`.

### F8. Matplotlib Visualization -- P2 -- 30 LOC
**What:** No matplotlib/pyplot-based plots. However, extensive analytics exist: `analyze_traj.py` (294 LOC, timing/tracking/correlation), `visualize_episode.py` (469 LOC, 3D+timeseries), sidecar JSON.
**Why:** Adding matplotlib enables static plot export for reports and papers from existing statistics.
**LeFranX:** matplotlib-based control quality plots.
**DexMani:** zero `matplotlib`/`plt`/`plotly`/`seaborn` imports.
**Target:** New `tools/plot_sessions.py` -- thin matplotlib wrapper around `analyze_traj.py` statistics.

### F9. Image Transform Debug Tool -- P2 -- 200 LOC
**What:** No tool to visualize the effect of image transforms (crop, resize, normalize) on actual recorded frames.
**Why:** Debugging image pipeline issues (wrong crop, black output) requires visual inspection.
**LeFranX:** `visualize_image_transforms.py` -- per-transform example output.
**DexMani:** no image transform debugging tool.
**Target:** New `tools/visualize_image_transforms.py`.

---

## G. Teleop Architecture & Workflow

### G1. In-Recording Retake -- P2 -- 50 LOC
**What:** Can only delete a written HDF5 file after exiting (Q key -> `discard_episode()` -> `unlink`). Cannot discard mid-recording and restart with the same episode number.
**Why:** Common workflow: operator makes a mistake 5s in, wants to restart immediately without incrementing the counter.
**LeFranX:** `dual_vr_record.py:319-332` -- 'r' key -> `clear_episode_buffer()` -> `continue` (counter unchanged).
**DexMani:** `controller.py` -- 6 keybindings (B/C/S/H/Q/ESC), no 'r'; `collection_loop.py:133-141` -- `discard_episode()` only post-write.
**Target:** `recording/collection_loop.py` (~20 LOC), `recording/episode_recorder.py` (~10 LOC), `teleop/control/keyboard_handler.py` (~5 LOC), `teleop/core/controller.py` (~15 LOC).

### G2. ADB Automation -- P1 -- 50 LOC
**What:** 5+ files tell users to manually run `adb reverse tcp:8000 tcp:8000`. No `subprocess.run(['adb', 'reverse', ...])` anywhere.
**Why:** Removes manual setup friction for VR connection. Graceful degradation on failure (log warning, not error).
**LeFranX:** `VRRouterManager` auto-runs `setup_adb_reverse()` on singleton init.
**DexMani:** global search `adb` -- only docstring comments and print statements.
**Target:** `teleop/vr/vr_tracker.py` or new `sensor/vr_setup.py`.

### G3. Teleoperator ABC -- P2 -- 300 LOC
**What:** No base class for teleoperation modes. Components (ArmWristMapper, XHandRetargeter, XArm7MotionPlanner) are assembled ad-hoc in entry-point scripts with no interface contract.
**Why:** An ABC makes teleoperation modes (VR, keyboard, gamepad, leader arm, policy inference) interchangeable without modifying entry points.
**LeFranX:** formal `Teleoperator` ABC with `config_class`, `name`, `get_action()`, `connect()`, `disconnect()`, `set_robot()`, `calibrate()`, `get_status()`, config-driven factory.
**DexMani:** zero ABCs in teleop module.
**Target:** New `teleop/core/teleoperator.py`.

### G4. Lazy IK Initialization -- P2 -- 120 LOC
**What:** `TeleopController.__init__()` requires all dependencies ready. Planner (with IK solver) must exist before controller. Cannot test VR path without hardware.
**Why:** Enables VR dry-run testing without robot connected.
**LeFranX:** separates `set_robot()`/`connect()` from construction; IK solver lazy-initialized on first `get_action()`.
**DexMani:** `controller.py:89-108` -- all dependencies are constructor params.
**Target:** `teleop/core/controller.py`, `teleop/core/pipeline.py`.

### G5. Recording Resume -- P3 -- 28 LOC
**What:** No explicit `--resume` flag or startup summary. However, `start_episode()` already auto-increments safely (scans existing files).
**Why:** Small UX improvement: print "Found N existing episodes, starting at N" on startup.
**LeFranX:** `get_existing_episode_count()` scans `episode_*.parquet`.
**DexMani:** `episode_recorder.py:88-89` -- `while exists: idx += 1` auto-skip already present.
**Target:** `recording/episode_recorder.py` (~8 LOC), `examples/real/vr_teleop_shm.py` (~15 LOC), `recording/collection_loop.py` (~5 LOC).

### G6. Config-Driven Factory + VR Router -- P3 -- 250 LOC
**What:** No `@register_subclass` decorator, no `make_teleoperator_from_config()`. Entry-point scripts manually construct each component. VR router is per-script with no reference-counting singleton.
**Why:** Declarative assembly simplifies adding new teleoperation modes; router singleton prevents port conflicts.
**LeFranX:** `@register_subclass` + `make_teleoperator_from_config()`; `VRRouterManager` reference-counted singleton.
**DexMani:** all construction is manual in entry-point scripts; SHM architecture may already be superior for VR routing.
**Target:** New `teleop/core/factory.py` (~150 LOC), `sensor/vr_router.py` (~100 LOC).

### G7. Control Mode Metadata -- P3 -- 8 LOC -- ✅ SHIPPED (R1)
**What (RESOLVED):** Recording now stores `control_mode`, `arm_mode`, `hand_mode` (plus `arm_delta_clip`/`hand_delta_clip`/`hand_ema_alpha`/`hand_low_pass_alpha`/`ema_alpha_pos`/`ema_alpha_rot`) in `/meta.attrs`.
**Why:** Future-proofing: downstream consumers need to know whether actions are position/velocity/torque.
**LeFranX:** records control_mode metadata.
**DexMani:** `episode_recorder.py:178` -- `control_mode`/`arm_mode`/`hand_mode` written to `/meta.attrs`.
**Target:** `robot/types.py` (add field to RobotAction), `recording/episode_recorder.py` (write to `/meta.attrs`).

### G8. IK Scoring Enhancement (Manipulability Nullspace) -- P3 -- 50 LOC
**What:** Position IK fallback already has manipulability + neutral pose distance + joint cost scoring -- more comprehensive than LeFranX. Only gap: differential IK main path does not use manipulability scoring.
**Why:** Adding manipulability gradient to nullspace optimization avoids singularities during teleop.
**LeFranX:** `weighted_ik.cpp` -- 3-term scoring.
**DexMani:** `ik_candidates.py:207-228` -- `score_ik_candidate()` with manipulability weight; `kinematics.py:118-129` -- `compute_manipulability()` (Yoshikawa).
**Target:** `planning/nullspace.py` -- add `manipulability_gradient()` from Jacobian SVD.

### G9. Docstring Cleanup -- P3 -- 10 LOC
**What:** Module docstring in `controller.py` references deleted RECORDING state; `DataValidator` docstring claims 5-6 checks but runs 7; CLAUDE.md says `validate_action()` is "2-check stub" but code has 4 checks.
**Why:** Stale documentation misleads developers about system behavior.
**Target:** `teleop/core/controller.py`, `recording/data_validator.py`, `CLAUDE.md`.

---

## Cross-Reference: Old Report IDs -> Unified IDs

| Old ID | Source | Unified ID | Notes |
|--------|--------|-----------|-------|
| R1#1 | Factcheck | A3 | Joint-space EMA |
| R1#2 | Factcheck | A1 | Jerk limiting (merged with ER-1) |
| R1#3 | Factcheck | A4 | Per-joint arm vel/acc (merged with S6) |
| R1#4 | Factcheck | A2 | Controlled stop (merged with ER-1) |
| R1#5 | Factcheck | D1 | Policy deployment (merged with T3, D1, I3) |
| R1#6 | Factcheck | -- | **CLOSED** -- stats.json already auto-generated |
| R1#7 | Factcheck | C1 | Unified obs interface (merged into Gym env) |
| R1#8 | Factcheck | G1 | In-recording retake |
| R1#9 | Factcheck | G5 | Recording resume |
| R1#10 | Factcheck | A5 | VR dead zone |
| R1#11 | Factcheck | -- | REJECTED (Appendix) |
| R1#12 | Factcheck | G8 | IK scoring enhancement |
| R1#13 | Factcheck | -- | REJECTED (Appendix) |
| R2#S1 | Deep Dive | B1 | validate_action() safety |
| R2#S2 | Deep Dive | B4 | ArmInnerLoop auto-reconnect |
| R2#S3 | Deep Dive | B3 | Hand reconnect guard (merged with S4, R3#B2) |
| R2#S4 | Deep Dive | B3 | Compound connect rollback (merged with S3, R3#B2) |
| R2#S5 | Deep Dive | B5 | Watchdog process |
| R2#S6 | Deep Dive | A4 | Per-joint hand delta (merged with R1#3) |
| R2#T1 | Deep Dive | C1 | Gym environment |
| R2#T2 | Deep Dive | C2 | Training pipeline |
| R2#T3 | Deep Dive | D1 | Inference controller (merged with R1#5) |
| R2#T4 | Deep Dive | C5 | Reward functions |
| R2#T5 | Deep Dive | C6 | Image preprocessing (merged with R3#C1, R3#C3) |
| R2#D1 | Deep Dive | D1 | Real replay (merged with R1#5) |
| R2#D2 | Deep Dive | D2 | Policy evaluation (merged with R3#D3) |
| R2#D3 | Deep Dive | D2 | Sim evaluation (merged into eval framework) |
| R2#I1 | Deep Dive | C4 | PyTorch Dataset (merged with R3#C4) |
| R2#I2 | Deep Dive | G7 | Control mode metadata |
| R2#I3 | Deep Dive | D1 | Action chunking (merged with R1#5) |
| R2#A1 | Deep Dive | G2 | ADB automation |
| R2#A2 | Deep Dive | G3 | Teleoperator ABC |
| R2#A3 | Deep Dive | G4 | Lazy IK initialization |
| R2#A4 | Deep Dive | G6 | Config-driven factory (merged with MG9) |
| R3#A1 | Round 3 | F1 | File-based logging |
| R3#A2 | Round 3 | F2 | Status API |
| R3#A3 | Round 3 | F3 | Latency tracking |
| R3#A4 | Round 3 | F5 | Periodic perf stats |
| R3#A5 | Round 3 | F7 | FPSTracker |
| R3#A6 | Round 3 | F4 | Loop frequency reporting |
| R3#B1 | Round 3 | B2 | TCP health probe |
| R3#B2 | Round 3 | B3 | Partial-connect cleanup (merged with S3, S4) |
| R3#C1 | Round 3 | C6 | Image augmentation (merged with T5, C3) |
| R3#C2 | Round 3 | E1 | Video-based storage |
| R3#C3 | Round 3 | C6 | ROI crop + black frame (merged with T5, C1) |
| R3#C4 | Round 3 | C4 | Chunked storage + lazy (merged with I1) |
| R3#C5 | Round 3 | E2 | Per-feature normalization |
| R3#D1 | Round 3 | -- | **REMOVED** (category error: training metrics in teleop project) |
| R3#D2 | Round 3 | F6 | WandB (same as MG1) |
| R3#D3 | Round 3 | D2 | Policy eval framework (merged with R2#D2) |
| R3#D4 | Round 3 | F8 | Matplotlib visualization |
| R3#E1 | Round 3 | E4 | Dual replay buffer |
| R3#E2 | Round 3 | -- | **REMOVED** (factually false: tracking error already computed in ik.py) |
| MG1 | Unified | F6 | WandB (same as R3#D2) |
| MG2 | Unified | C3 | Training resume/checkpoint |
| MG3 | Unified | C9 | Distributed Actor-Learner RL |
| MG4 | Unified | D3 | Policy server/client |
| MG5 | Unified | C7 | Joint limit auto-discovery |
| MG6 | Unified | F9 | Image transform debug |
| MG7 | Unified | E3 | Episode quality filtering |
| MG8 | Unified | C8 | Coordinated arm+hand homing |
| MG9 | Unified | G6 | VR router singleton (merged with A4) |

---

## Summary

| Priority | Count | LOC | Scope |
|----------|-------|-----|-------|
| **P0** | 10 | ~4,900 | Safety-critical + training hard prerequisites |
| **P1** | 13 | ~4,200 | Scale, production readiness, iteration speed |
| **P2** | 13 | ~2,900 | Workflow quality, observability polish, training readiness |
| **P3** | 5 | ~900 | Polish, future-proofing, nice-to-have |
| **Total** | **41** | **~10,500** | |

**New files:** ~25 | **Modified files:** ~20 | **Files touched:** ~45

### Recommended Implementation Order

```
Phase 1 -- Safety Foundations (~1 week, ~650 LOC):
  B1 (validate_action) -> B3 (connection safety) -> B2 (TCP probe) -> A2 (controlled stop) -> A1 (jerk)

Phase 2 -- Observability (~0.5 week, ~170 LOC):
  F1 (file logging) -> F2 (status API) -> F3+F4+F5 (latency + frequency + perf stats)

Phase 3 -- Training Prerequisites (~4 weeks, ~3,900 LOC):
  C1 (Gym env) -> C4 (PyTorch Dataset) -> C2 (training pipeline) -> C3 (checkpoint) -> C5 (rewards) -> C6 (image pipeline) -> F6 (WandB)

Phase 4 -- Deployment (~2 weeks, ~1,500 LOC):
  D1 (deployment suite) -> D2 (eval framework) -> G1 (retake) -> G2 (ADB)

Phase 5 -- Workflow & Polish (~2 weeks, ~2,500 LOC):
  A3+A4+A5 (motion) -> B4+B5 (auto-reconnect+watchdog) -> C7+C8 (joint limits+homing) -> E1+E2+E3 (data quality) -> G3+G4+G5+G6+G7 (teleop architecture)

Phase 6 -- Scale-Out (when needed, ~2,000+ LOC):
  C9 (distributed RL) -> D3 (policy server/client) -> E4 (dual buffer)
```

### Quick Wins (<50 LOC each, <1 hour each)

| ID | LOC | What |
|----|-----|------|
| A3 | 15 | Post-IK joint-space EMA -- one line in `pipeline.py` |
| A1 | 30 | Wire existing `limit_jerk()` into `inner_loop.py` |
| A4 | 20 | Scalar -> ndarray for per-joint vel/acc/delta |
| G9 | 10 | Fix stale docstrings in 3 files |
| ~~G7~~ | 8 | ~~Add `control_mode` to recording metadata~~ -- ✅ DONE |

---

## Appendix: Rejected / Not Applicable

| # | Title | Reason |
|---|-------|--------|
| R1#11 | LeRobot Dataset Format | Low ROI: DexMani already has Diffusion Policy-compatible Zarr export. HDF5+Blosc sufficient. A custom PyTorch Dataset reading HDF5 directly (~100 LOC) covers the need. Revisit only if HuggingFace Hub distribution becomes required. |
| R1#13 | GeoFIK Analytic IK Port | Not portable: GeoFIK DH parameters hardcoded for Franka kinematics (spherical shoulder + spherical wrist). xArm7 kinematic chain is different with no known analytic IK. MPlib-as-drop-in (~1-5ms vs ~2us) too slow for 50Hz teleop. A3/G8 nullspace optimization covers the benefit. |
| R3#D1 | MetricsTracker (AverageMeter, log_dict) | Category error: DexMani is a teleop/data-collection project with no training code by design. There is nothing to track training metrics against. This becomes relevant only after C2 (training pipeline) is implemented. |
| R3#E2 | Intervention Rate / Tracking Error | Factually false: `cmd_tracking_error_pos_m/rot_rad` is already computed per-frame (`ik.py:480`), `flag_ik_ok`/`flag_retarget_ok`/`flag_held` are recorded to HDF5, and `held_ratio` is aggregated in per-episode sidecar JSON. The "intervention" concept (human takeover from autonomous policy) is inapplicable to a pure teleop system. |
| R2#DexMani-Has-It (7 items) | SHM VR transport, cumulative VR timeout, DataValidator, Cartesian EMA, adaptive finger scaling, multi-layer collision, TimestampAlignedBuffer, structured state machine | DexMani already stronger than LeFranX in these areas. No action needed. |
| R2#Not-Applicable (2 items) | LeFranX server architecture, multi-platform factory | LeFranX's C++ server + Python client architecture and multi-robot-platform factory pattern are design choices not applicable to DexMani's single-platform Python architecture. |
