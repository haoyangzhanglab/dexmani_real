# Phase C — Learned-Policy Deployment Runtime

The fifth stable boundary of DexMani Real (§119). This is a generic
learned-policy deployment runtime built **on top of** the A/B Frozen Runtime
(`docs/ab_runtime_freeze_report.md`) without modifying any frozen contract.
Phase C adds exactly one new cross-process flow — *observations → model
proposal → robot command* — and reuses every existing safety, lifecycle, and
transport primitive.

Offline regression gate: `checks/offline/run_all.py` → **25/25 passed**.

---

## 1. Core invariant

> **Model output is a proposal, not a robot command** (§48).

The inference worker writes **only** `policy_plan_ring`. The coordinator is the
**sole** learned-policy robot-action producer: it adopts a plan, schedules the
due endpoint, and drives the shared publication boundary
(`SafetyGate.validate` → `send_command`). No model output ever reaches
`arm_action_q` / `hand_cmd_ring` directly, and no whole action chunk is ever
dumped into a robot transport (§73/§74).

```text
arm state ring ─┐
hand state ring ─┤── causal observation ──► inference worker ──► policy_plan_ring
                │                                    (encode → infer → decode)
                │                                             │
                │                              coordinator adopts plan, schedules
                │                              the single due endpoint (latest-wins)
                │                                             │
                └─────────────────────────────────────────────┤
                        build_action_candidate → SafetyGate → send_command
                                                              │
                                              arm_action_q (endpoint) / hand_cmd_ring
```

---

## 2. New modules (file map)

| Module | Role |
|---|---|
| `deployment/contracts.py` | `PolicyBackend` / `ObservationAdapter` / `ActionAdapter` Protocols + `JointActionChunk` / `InferenceContext` |
| `deployment/observation.py` | `ObservationBatch` / `FrameWindow` / `CameraWindow` (process-local, never in SharedStorage) |
| `deployment/config.py` | `DeploymentConfig` + `resolve_deployment_config` (CLI > file/data > defaults + SHA-256) |
| `deployment/loader.py` | lazy `module:symbol` backend/adapter loader (fail-closed, Protocol-checked) |
| `deployment/fake.py` | deterministic torch-free fake backend (architecture gate, swap fixture) |
| `deployment/worker.py` | `inference_loop` — observations → proposals → `policy_plan_ring` |
| `deployment/coordinator.py` | `coordinator_loop` — the sole robot-action producer + scheduler + silence watchdog |
| `deployment/lifecycle.py` | `build_policy_worker_specs` + `run_policy_deployment` (composes the frozen runtime) |
| `deployment/metrics.py` | counter/gauge registry + structured logging (§94) |
| `deployment/provenance.py` | one-time startup provenance log (§96) |
| `integrations/dexmani_policy.py` | the DexMani Policy model-repository adapter (§86–§91) |
| `examples/run_policy.py` | thin CLI (resolve → lifecycle → exit code) |
| `shm/causal_reader.py` | extracted causal observation reader (P2, reused by snapshot + worker) |

`deployment/*` never imports `integrations/*`; the dependency direction is
integration → deployment (§86).

---

## 3. Worker set and identities

| Worker | `name` / `ready_name` | Owns |
|---|---|---|
| `arm` | `arm` | xArm Mode 6 servo + FK state |
| `hand` (only when `deployment.hand_enabled`) | `hand` | XHand servo + state/tactile |
| `inference` | `inference` | encode → infer → decode → `policy_plan_ring` only |
| `policy` (coordinator) | `policy` | adopt plan → schedule → candidate → SafetyGate → send |

`inference` is a new worker identity, so it is registered in six places (§G-1):
`SharedStorage.HEARTBEAT_FIELDS`/`READY_FIELDS`, `_HEARTBEAT_SUBSYSTEMS`/
`_READINESS_SUBSYSTEMS`, and `SafetyParams.heartbeat_timeouts`/
`readiness_timeouts_s` (5.0 s / 120.0 s). The coordinator reuses the existing
`policy` control-source slot; no new identity is added for it.

Joint-only first version: no VR worker (only an adapter that declares VR, §85),
no camera worker (only RGB/pointcloud adapters), no recorder (Phase C leaves v16
recording to a separate migration, §95/§97).

---

## 4. Data contracts

### `policy_plan_ring` (new)

`POLICY_PLAN_DTYPE` (`utils/schema.py`) is a latest-wins seqlock
(`SharedMemoryRingBuffer`, maxlen 3) with `MAX_POLICY_CHUNK_STEPS = 32`. One
record carries `plan_id`, `run_generation`, `observation_id`, the observation
anchor and inference start/finish timestamps (all `<u8` monotonic ns),
`num_steps`, `arm_present`/`hand_present`, and the fixed arrays
`target_monotonic_ns[32]`, `arm_qpos[32,7]`, `hand_qpos[32,12]`, `valid_mask[32]`.

- An over-capacity chunk **fails**; it is never truncated (§61).
- The ring is the *only* channel for a whole chunk. `arm_action_q` and
  `hand_cmd_ring` receive single endpoints only (§73/§74).

### Protocols

`PolicyBackend.load/reset/infer/close`, `ObservationAdapter.encode`,
`ActionAdapter.decode` — model boundary; none import torch or SharedStorage.
The model-internal parameters (§93) never appear in `DeploymentConfig`.

---

## 5. Failure semantics

| Class | Examples | Result |
|---|---|---|
| Drop-only (§80.1) | generation mismatch, stale observation, superseded, expired | counted, continue |
| Abort policy run (§80.2) | NaN/Inf/shape/timestamp, SafetyGate reject, hand-delta violation, repeated no-valid-action | `advance_run_generation` + `RUNNING → ARMED` (**not** FAULT) |
| Process failure (§81) | crash, CUDA fatal, backend exception, heartbeat timeout | existing supervisor fault path |

- The coordinator's silence watchdog (§82): `RUNNING` with no valid policy
  command for `max_command_silence_s` → advance generation + `RUNNING → ARMED`.
- The coordinator enters `RUNNING` itself (it is the policy control source — no
  operator BEGIN): set ready while `DISARMED/ARMED`, wait for Main to arm, then
  `ARMED → RUNNING` and one `advance_run_generation`.
- Coupled-hand delta preflight runs controller-side before the arm endpoint is
  enqueued (reference = last *published* hand command, mirroring VR teleop), so a
  rejected hand command desyncs nothing (§74).

---

## 6. Metrics and provenance

- **Metrics (§94):** `deployment/metrics.py` defines the §94 counter/gauge names
  and a `Metrics` registry (`increment`/`observe`/`snapshot`/`flush` +
  `flush_every` throttle). No Prometheus/OpenTelemetry. The worker records
  `observations_built`, `inference_ms`, `inference_failures`, `plans_created`,
  `plans_generation_dropped`; the coordinator records `endpoints_due`,
  `endpoints_coalesced`, `endpoints_published`, `safety_rejections`,
  `policy_aborts`, `command_silence_abort`. The remaining §94 names
  (`observation_age_ms`, `observation_skew_ms`, `plan_age_ms`, `plans_stale`,
  `plans_superseded`) are pinned down during the H0 gate when freshness/ring
  semantics are observed on hardware.
- **Provenance (§96):** `log_deployment_provenance` logs the DexMani Real and
  model commits, the three targets, checkpoint path + hash, model-config hash,
  and the resolved runtime SHA-256 — one startup line, never a SharedStorage
  payload.

---

## 7. Backend swap boundary (§100)

Replacing the policy model is a **config-only** change: `backend_target` /
`observation_adapter_target` / `action_adapter_target` / `checkpoint` /
`device` / environment. It must **never** touch `robot/`, `sensor/`,
`policy/safety.py`, `shm/shared_storage.py`, or `deployment/coordinator.py`.
`check_backend_swap.py` locks this with a static guard plus a two-backend
run through the identical core.

---

## 8. Hardware gates H0–H6 (documented manual checklist)

These cannot run offline; each is a manual hardware checklist (§101–§107).
The metrics above are the observation instrument for H0.

- **H0 — No Command:** camera/state + inference + coordinator active, command
  publication disabled. Observe observation, model input, inference latency,
  plan, scheduled endpoint, candidate, SafetyGate result. **No motion.**
- **H1 — Connected Dry Run:** connect arm + hand, candidate publication dry-run.
  Confirm planned command = SafetyGate command = expected transport.
- **H2 — Arm Only Restricted:** hand disabled, small motion, restricted
  workspace, operator e-stop ready. First real policy motion, arm only.
- **H3 — Arm + Hand:** after H2 is stable. Check shared `action_id`,
  hand latest-wins, no command backlog, no silent clip.
- **H4 — Pause During Inference:** inference running → pause → `generation++`
  → old CUDA inference returns. Confirm the old plan **never executes**.
- **H5 — Fault / Worker Death:** terminate the inference process (safely).
  Confirm supervisor → fault path → verified shutdown. Do **not** induce a
  dangerous actuator fault on hardware.
- **H6 — Soak:** long inference, plan replacement, pause/restart, arm+hand,
  optional recording. Observe memory, shared-memory leaks, queue pressure,
  plan lag, CUDA memory growth, heartbeat, command silence.

---

## 9. Reject list (§109)

Any of the following is a `REJECT`:

- Inference worker writes `arm_action_q` / `hand_cmd_ring`, owns `SafetyState`,
  or imports the robot SDK.
- Model adapter holds `SharedStorage`.
- Model-specific branch in the coordinator.
- Core package imports torch at import time (child lazy import only).
- Whole action chunk dumped into robot transport; application-side arm
  interpolation; parallel process watchdog; parallel recording framework; new
  global plugin registry; JSON/object dtype in high-frequency IPC.
- XHand lifecycle modified by the policy deployment.
