# A/B Runtime Freeze Report

Phase C must treat every contract in this document as **FROZEN**. Schema/format
changes require an independent migration; contract changes require a new phase.

Generated 2026-08-15 at the end of Phase B (B3/B4/B5/B6 subtraction). Offline
regression gate: `checks/offline/run_all.py` → **12/12 passed**.

---

## 1. Base / final SHAs

| Boundary | SHA | Meaning |
|---|---|---|
| A base | `34ddc70` | Phase A audit baseline (named in `docs/phase-a-audit-evidence-2026-08-15.md`) |
| A final | `4561e05` | A0–A18 fixes landed + execution doc committed (`2b731d3` is the last A-fix; `4561e05` is its marker child) |
| B base | `4cb75ae` | Parent of the first Phase B commit (`2834fb6`) |
| B final | `a3e7c6e` | Current HEAD after B3/B4/B5/B6 subtraction |

**Commit topology note.** Phase B spans two non-contiguous ranges because the
hardware-gated experiments ran before the Phase A audit:

1. **B hardware experiments (B1/B2)** — `2834fb6` … `38010d1` (2026-08-14),
   *before* the A audit.
2. **A audit + fixes** — `61b9564` … `2b731d3` (2026-08-15).
3. **B subtraction (B3/B4/B5/B6)** — `6bf11e7`, `fbce8b8`, `13f680e`, `a3e7c6e`
   (2026-08-15), *after* the A audit.

"B final" (`a3e7c6e`) is the single frozen runtime HEAD.

### Evidence chain

| Commit | Meaning |
|---|---|
| `789418e` | B1 GO — single-controller connect (source `b70eeca`, cherry-picked) |
| `f2e2a99` | B soak evidence record (baseline/B1/B2 jsonl) |
| `38010d1` | B finalization — B1 landed/go, B2 rejected/no-go |
| `66472e3` | Removed soak `jsonl` + hardware harness from the tree |

Per decision, the soak `jsonl` files are **not restored**; the summary numbers in
§3 are the evidence, with commits as the reference chain.

---

## 2. xArm Mode & recovery contract

- **Mode 6** is the normal servo mode. Firmware is the final collision/current
  safety backstop. Application-side interpolation of arm targets is **unsafe**;
  one Mode 6 endpoint is sent per grid tick.
- **C22 / C31** are immediate faults. **C24** has bounded measured-hold recovery;
  a second C24 inside 2 s becomes a sticky fault.
- **Homing** uses a separately validated Mode 0 milestone path
  (`plan_joint_home_path` / `plan_band_alignment_path`); homing restore is
  fail-closed (live controller error read → non-zero → no Mode 6 restore).
- **Live-error contract (A2):** control decisions (setter-failure
  classification, homing restore, homing milestone check) read the live
  `get_err_warn_code()` via `_read_live_error_code`, **never** the cached
  `arm.error_code`; a live-read failure fails closed. Steady-state telemetry may
  still report the cached value.
- **Cleanup:** arm cleanup confirms physical stop (state 4) with
  `require_error_clear=False`, because a fault exit leaves a latched non-zero
  controller error.
- **Feedback validity:** arm feedback is valid only when both the SDK state read
  and URDF FK succeed; FK failure publishes NaN EEF with `state_valid=0`.

---

## 3. XHand connect / disconnect contract

### Connect (B1 GO)

- Discovery and open share **ONE `XHandControl` per attempt** — no throwaway
  discovery controller followed by a fresh open controller. SDK ownership stays
  inside the spawned hand worker.
- Soak (100 cycles): **100/100 connect, 0 `write sdo failed`, 0 open retry,
  exit 0** (baseline had 1 retry @ cycle 77 + teardown segfault).

### Disconnect (B2 NO-GO)

- `disconnect()` = `_request_slave_init()` (slave → `_EC_STATE_INIT`)
  + `control.close_device()` + `time.sleep(_POST_DISCONNECT_WATCHDOG_WAIT_S = 2.0)`.
- The **INIT transition + 2 s watchdog wait is load-bearing**. B2 (close-only)
  soak was the only run to reproduce `write sdo failed` (16×, cycle 3–4) plus a
  14.25 s stale-slave open retry. Rejected.
- `_EC_STATE_INIT = 1` is the only AL-state constant used. `_EC_STATE_PRE_OP/
  SAFE_OP/OP` were dead and removed in B3.
- B2 caveat (n=1, honest not over-asserted): the 16 SDO failures cluster in the
  cold-start region (self-healing transient, not a hard wedge), and B2 removed
  both INIT-request and sleep in one patch, so the causal claim is directional,
  not statistically significant.

---

## 4. arm queue / hand ring semantics

| Transport | Direction | Semantics |
|---|---|---|
| `arm_action_q` | controller → arm | Ordered `mp.Queue(maxsize=2)`; fixed endpoints + correlated HOME requests; endpoint backpressure intentional |
| `hand_cmd_ring` | controller → hand | Seqlock, latest-wins servo target |

- XHand is a 12-DoF EtherCAT position servo; the command ring is latest-wins.
- Hand velocity is **command-to-command**, never target-to-measured; workers
  reject stale-generation, expired, operational-limit, and rated
  mechanical-limit violations **without changing an endpoint**. Runtime config
  may narrow, but cannot widen, the rated mechanical envelope.

---

## 5. SharedStorage ring commit semantics

- All rings use the documented seqlock API. `get_last_k(k)` returns verified
  frames oldest-first, may be shorter than `k`, and raises for `k > maxlen`.
- **Source freshness and publish freshness are distinct.**
- **Commit order (A9):** a ring writer stamps its publish timestamp **after**
  the payload commit (`begin_write → payload → stamp_timestamp → end_write`),
  mirroring `camera_ring`.
- `record_control_ring` is latest immutable fixed-field START/STOP; a START is
  refused until the prior STOP terminal is harvested. `record_sample_ring` is a
  fixed-grid sample + transient control generation; overflow aborts the episode.
- Flags and heartbeats use `time.monotonic()`.

---

## 6. SafetyGate contract

- `SafetyGate` (`policy/safety.py`) is the **single validation boundary**:
  well-formed → joint limits → workspace.
- Velocity-envelope checking was **removed** (2026-08-12); xArm Mode 6 firmware
  is the final velocity/acceleration/collision backstop.
- Collision and transition-geometry checks were **removed** from SafetyGate
  (2026-08-12). Collision-free homing paths are planned independently through
  `plan_joint_home_path` / `plan_band_alignment_path`.
- Coupled hand paths run a controller-side preflight
  (`validate_hand_command_delta`) on the rated mechanical envelope and the
  command-to-command delta **before** the arm endpoint is enqueued, so a rejected
  hand command desyncs nothing.

---

## 7. run_generation semantics

- `run_generation` tags commands and candidates. Begin, pause, home, feedback
  fault, and camera re-warm advance it.
- Workers reject queued/ring commands from an older generation; this cannot
  retract an endpoint already accepted by firmware.
- **Ordinary pause** publishes no replacement endpoint — it is command
  quiescence: advance `run_generation`, stop publishing, let Mode 6 finish the
  last accepted endpoint. Keyboard idle rebuilds joint/Cartesian baselines from
  feedback; VR resume accepts only feedback newer than pause entry and spends its
  first grid re-anchoring.
- Every explicit VR BEGIN opens a distinct generation and supersedes an earlier
  STOP/DISCARD/max-duration boundary.

---

## 8. Recording grid semantics

- One sample is emitted per `control_hz` grid tick (normally 16 Hz), **not** per
  sensor arrival.
- A command-silent pause is **not** a sampled grid interval; no pause-time hold
  action is synthesized.
- The first sample from a new `run_generation` re-anchors the recorder's next
  contiguous storage slot; its wall-time jump is retained.
- Writer failure, stream mismatch, overflow, codec failure, or ENOSPC aborts the
  episode rather than silently publishing partial data.
- `min_record_duration_s` is a quality label, not a publication gate
  (`min_frames_met=False` keeps schema-v16 validity).

---

## 9. Episode schema version

- **HDF5 schema v16** is the only runtime episode format. Readers, visualization,
  and replay accept only a published v16 directory.
- Reserved v16 fields (`hand_qpos_stale`, `hand_current`) are **retained** even
  though the runtime currently fills them false/optional; they are persisted and
  must not be deleted without an independent migration.

---

## 10. Not run / deferred

- **No real xArm/XHand motion, collect, replay-live, or homing** was run in this
  phase (hardware boundary). The B1/B2 hardware numbers in §3 are from the
  2026-08-14 soak runs; B3/B4/B5/B6 are offline-only and were validated by
  `compileall` + `checks/offline/run_all.py` (12/12).
- Manual hardware checklist before Phase C live validation:
  1. One cold-start xArm connect → verify Mode 6 ready + `get_joint_states` fresh.
  2. One XHand connect → disconnect → reconnect cycle (confirm INIT + 2 s watchdog
     path, no `write sdo failed`).
  3. One arm-only coupled ACK and one arm+hand coupled ACK (`last_cmd_seq`
     confirmation).
