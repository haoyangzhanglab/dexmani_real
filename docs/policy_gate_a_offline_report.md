# Policy Promotion Gate A — Offline Qualification

## Scope

NO HARDWARE qualification only. Live shadow, H4, homing, rollout, device connection, and command publication were not run.

## Frozen Real baseline

- Real main / Gate base: `effe745c68847a4b32ed1e4680041a350da4f4fe`
- R2 merge: guarded fast-forward from `fd8195f757f341f99a50f232bf59820a0fb15ec6` to `effe745c68847a4b32ed1e4680041a350da4f4fe`
- Remote main verification: PASS

## Policy current docs-only state

- Current Policy main: `5994037db9d3a56d7bd0f807bd6df2cbca3adfcc`
- Semantic handoff: `aa4a0a39dd5a69e3a4ad85ea8190d6889610d175`
- Diff since handoff: one documentation file only

## Current fixture qualification

CURRENT FIXTURE EVIDENCE uses current Policy main as producer.

- Producer: `5994037db9d3a56d7bd0f807bd6df2cbca3adfcc`
- Checkpoint SHA-256: `6fa0d49e044a281eec66133a140bdeee534986f2ebda4468d6bcd28bff720218`
- Sidecar SHA-256: `b10defb759e2e06e6d2a44078a3fac23c4bc68bc3f02eafbb1d1f0861709ebf8`
- Inspect is a real `examples/run_policy.py --print-config` CLI subprocess check (exit 0).
- Inspect allocation: O2 / A8 / H16 / required 15 / control 19D
- Inspect / direct restore / isolated preflight: PASS / PASS / PASS

## Representative artifact qualification

REPRESENTATIVE ARTIFACT EVIDENCE uses the artifact's exact historical producer.

- Experiment: `2026-08-28_13-59_42`
- Checkpoint: `epoch=1126-step=00080000-pr3-fc6b7df-deployment-v2.pt`
- Checkpoint SHA-256: `28ff79a6ca5d5b746bbde877ff96abbb88543539f4c73ef554348184f446effc`
- Sidecar/index SHA-256: `52683587a024a18d9251eb073a6290c1cc123d966edbaa7dc282097c38040b06`
- Producer: `fc6b7dfb45748f4187f2e82b5425721ed02b028e`
- Exact clean detached producer checkout: YES
- Direct restore / isolated preflight: PASS / PASS

## Recorded replay

RECORDED REPLAY uses recorded source-relative timing rebased onto a fresh monotonic epoch plus measured offline model-path latency. It is not a live latency measurement.

- Episode: `episode_20260827_224527` (schema v24, VALID, task `pick_place_toy`)
- Window: compact index 0, 2 consecutive observations
- Gate A representative inference baseline seed: 1066
- Actual representative Policy inference: PASS
- Model latency: 182.212 ms
- Prediction / control shapes: `[1, 16, 19]` / `[8, 19]`
- First deliverable / transport-valid / usable: 4 / 4 / 4
- ActionBuffer: PASS; target grid was not retimed
- SafetyGate publication boundary: `SHADOW_VALIDATED`

## Multiprocess shadow

MULTIPROCESS SHADOW is a hardware-free shared-memory/process integration using only inference, policy coordinator, and replay feeder processes.

- Inference / coordinator ready: YES / YES
- Policy plan ring advanced: YES (sequence 1)
- Shadow validated count: 1
- Replay feeder stayed within one persisted source segment: [0, 140); fed [0, 20) (20 frames)
- Multiprocess representative inference seed: 1066
- Mandatory negative cross-process checks: PASS

## Timing evidence

- Basis: `recorded_relative_rebased`
- Immutable target grid, strict finish+lead lower bound, independent deadline, strict target&lt;deadline upper bound, and `0*1*` transport topology: PASS
- Deadline relative to recorded latest source: 750000000 ns

## No-write proof

- Direct replay coupled sequence: 0 → 0
- Multiprocess coupled sequence: 0 → 0 (delta 0)

## No-hardware proof

- Arm / hand / camera / pointcloud workers started: NO / NO / NO / NO
- xArm / XHand / RealSense connection: NO / NO / NO
- Forbidden hardware SDK/owner module audit: PASS

## Failures and skips

- Mandatory offline checks: no failures, no skips
- Live shadow: NOT RUN — OPERATOR REQUIRED
- Fresh H4: NOT RUN — OPERATOR REQUIRED

## Remaining live gates

Offline evidence cannot establish live sensor freshness under current load, physical tracking, device acknowledgement, or physical collision clearance. Those remain operator-owned live gates.

## Operator-only next steps

DO NOT RUN FROM CODEX — OPERATOR ONLY — HARDWARE SIDE EFFECTS POSSIBLE

Offline replay, multiprocess shadow, live shadow handoff, and fresh H4 handoff share representative inference seed `1066`.

Exact artifact producer `fc6b7dfb45748f4187f2e82b5425721ed02b028e`, Real `effe745c68847a4b32ed1e4680041a350da4f4fe`, device `cuda:0`, seed `1066`, checkpoint SHA `28ff79a6ca5d5b746bbde877ff96abbb88543539f4c73ef554348184f446effc`.

Live shadow:

```bash
git -C "$REAL_ROOT" switch --detach effe745c68847a4b32ed1e4680041a350da4f4fe && git -C "$POLICY_ROOT" switch --detach fc6b7dfb45748f4187f2e82b5425721ed02b028e && python "$REAL_ROOT/examples/run_policy.py" --experiment-dir "$POLICY_ROOT/experiments/dp3/pick_place_toy/2026-08-28_13-59_42" --device cuda:0 --inference-seed 1066 --hand --execution-mode shadow --max-running-seconds 10
```

Fresh H4, one endpoint:

```bash
git -C "$REAL_ROOT" switch --detach effe745c68847a4b32ed1e4680041a350da4f4fe && git -C "$POLICY_ROOT" switch --detach fc6b7dfb45748f4187f2e82b5425721ed02b028e && python "$REAL_ROOT/examples/run_policy.py" --experiment-dir "$POLICY_ROOT/experiments/dp3/pick_place_toy/2026-08-28_13-59_42" --device cuda:0 --inference-seed 1066 --hand --execution-mode execute --max-running-seconds 10 --execute-max-published-endpoints 1 --execute-ack-timeout-seconds 2 --execute-expected-checkpoint-sha256 28ff79a6ca5d5b746bbde877ff96abbb88543539f4c73ef554348184f446effc
```

## Final decision

`READY_FOR_LIVE_GATE_A`
