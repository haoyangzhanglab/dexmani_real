# CODEX TASK — Processed Episode Viewer Progress + Processing Decision Summary

> This is a temporary implementation task file for Codex. Read `AGENTS.md` first and obey it throughout the task. After the implementation is complete and all acceptance checks pass, delete `CODEX_TASK.md` before final handoff. If the task is incomplete or validation fails, keep this file so the work can be resumed.

## 1. Goal

Implement a small, coherent improvement around processed-episode inspection:

1. Add a terminal progress bar while `examples/visualize_episode_processed.py` logs frames to Rerun.
2. Persist a small, versioned processing-decision report alongside a successfully published processed batch.
3. At the end of `visualize_episode_processed.py`, print a concise batch summary that identifies episodes that were skipped/rejected and the reason, while separately identifying user-excluded episodes.
4. Preserve the existing processing safety model: technical/source-corruption/programming failures still fail the whole batch and must **not** be converted into per-episode rejections.
5. Preserve atomic publication: the report and processed HDF5 files must appear together or not at all.

Do not broaden the task into unrelated dataset, calibration, point-cloud, Rerun-layout, or processing-policy changes.

---

## 2. Current behavior that must be understood before editing

Trace the real code before making changes:

```text
examples/process_episodes.py
    -> dexmani_real.dataset.processing.process_episode_root()
        -> analyze_episode()
        -> EpisodeDecision
        -> _write_processed_episode()
        -> validate_processed_hdf5()
        -> atomic_publish()

examples/visualize_episode_processed.py
    -> ProcessedEpisodeVisualizer
        -> log_step()
        -> Rerun
```

Important existing semantics:

- `examples/visualize_episode_processed.py` accepts one already-published processed `.h5` file. It cannot infer historical rejections from that file alone because rejected episodes have no processed `.h5`.
- `process_episode_root()` already owns the authoritative admission decisions.
- The canonical CLI calls `process_episode_root(..., skip_rejected_unannotated=True, ...)`, so a successfully published batch may contain accepted episodes plus skipped behavior-level rejections such as `persistent IK failure`.
- Annotation `include: false` produces `excluded by annotation` and is operator-owned exclusion, not the same category as an ordinary skipped/rejected episode.
- Technical/source-corruption/programming failures intentionally raise and fail the whole batch. Examples include corrupt shape/dtype, invalid timestamps, bad camera geometry, RGB decode failures, invalid task identity, point-cloud derivation failure, and final processed validation failure. Do not weaken this behavior.
- The target processed directory is currently published transactionally through a staging directory and `atomic_publish()`.
- `examples/export_policy_zarr.py` already uses `tqdm` with progress output sent to `stderr`; follow that terminal style rather than inventing a custom progress implementation.
- `AGENTS.md` explicitly says not to create permanent one-off planning documents and not to invent test infrastructure merely to satisfy a checklist. This file is temporary and must be removed after successful acceptance.

---

## 3. Required architecture

### 3.1 One persisted-report owner

Prefer a small dedicated module:

```text
dexmani_real/dataset/processing_report.py
```

This module should own the persisted processing-report contract so the producer and consumer do not independently interpret YAML.

Suggested responsibilities:

- report filename constant;
- report schema name/version constants;
- construction of the minimal persisted payload from authoritative processing decisions;
- strict validation of an in-memory report payload;
- YAML write helper;
- YAML load + validation helper.

Do **not** dump the complete internal `process_episode_root()` return dictionary verbatim. The internal report is an implementation return value; the persisted report is a stable cross-boundary artifact and should contain only information needed for provenance and human summary.

Suggested constants:

```python
PROCESSING_REPORT_FILENAME = "processing_report.yaml"
PROCESSING_REPORT_SCHEMA_NAME = "dexmani-real-processing-report"
PROCESSING_REPORT_SCHEMA_VERSION = 1
```

Names may be adjusted if there is a stronger repository convention, but keep the report schema independent from `PROCESSED_SCHEMA_VERSION`.

### 3.2 Suggested persisted schema

Keep it minimal and self-describing. A suitable shape is:

```yaml
report_schema_name: dexmani-real-processing-report
report_schema_version: 1
processed_schema_name: dexmani-real-processed-hdf5
processed_schema_version: <current processed schema version>
task_name: <resolved batch task name>
source_episode_count: 20
accepted_episode_count: 17
skipped_episode_count: 2
excluded_episode_count: 1
episodes:
  - source_episode: episode_001
    status: accepted
    reason: null
    source_frames: 1830
    processed_frames: 1830
  - source_episode: episode_010
    status: skipped
    reason: persistent IK failure
    source_frames: 742
    processed_frames: 0
  - source_episode: episode_019
    status: excluded
    reason: excluded by annotation
    source_frames: 0
    processed_frames: 0
```

Required status vocabulary:

```text
accepted
skipped
excluded
```

The persisted report may derive `excluded` from the existing explicit `excluded by annotation` decision, but there must be one canonical derivation in the report-owner module. Do not duplicate magic-string classification in the viewer.

The report validator should at minimum enforce:

- root is a mapping;
- exact supported report schema name/version;
- processed schema identity fields are present and well-formed;
- non-empty valid `task_name` using the repository's existing task-name validator;
- counts are non-negative integers, not booleans;
- `episodes` is a list;
- each `source_episode` is a non-empty string;
- no duplicate `source_episode` entries;
- `status` is one of the allowed values;
- accepted entries have `reason: null` and positive/consistent processed-frame semantics;
- skipped/excluded entries have a non-empty reason and zero processed frames;
- aggregate counts equal the episode-entry counts;
- total episode count equals the number of entries.

Do not add speculative metadata that is not needed by this task.

---

## 4. Processing-side implementation

Modify `dexmani_real/dataset/processing.py` only as required to publish the new report.

### 4.1 Source of truth

The report must be generated from the same `EpisodeDecision` objects already used by `process_episode_root()`. Do not re-open/re-analyze episodes just to build the report.

Do not change `analyze_episode()` rejection rules unless an implementation necessity is discovered and clearly justified. This task is not a rejection-policy change.

### 4.2 Publication ordering

The successful non-dry-run transaction must remain conceptually:

```text
analyze source episodes
    -> decide accepted / skipped / excluded
    -> write accepted processed HDF5 files into staging
    -> validate all accepted processed HDF5 files
    -> build + validate + write processing_report.yaml into staging
    -> atomic_publish(staging, target)
```

The report must be written **after** accepted processed files pass `validate_processed_hdf5()` and **before** `atomic_publish()`.

If report construction or writing fails, the staging directory must be cleaned by the existing failure path and no target batch should be published.

Do not write the report after `atomic_publish()`.

### 4.3 Dry-run behavior

`--dry-run` must remain non-publishing. It must not create `processing_report.yaml` on disk.

The existing in-memory report returned by `process_episode_root()` may stay as-is unless a small adjustment is required for clean implementation. Do not force callers to consume the new YAML artifact in dry-run mode.

### 4.4 Fatal failures remain fatal

Do not catch technical failures and convert them to `EpisodeDecision(..., rejected_reason=...)` merely to make them appear in the report.

Preserve the existing policy:

```text
behavior/admission rejection -> may be skipped and reported
operator annotation exclusion -> reported as excluded
technical/source/programming failure -> raises, whole batch fails, no published report
```

This invariant is mandatory.

---

## 5. Viewer-side implementation

Modify `examples/visualize_episode_processed.py`.

### 5.1 Progress bar

Replace the existing periodic frame logger:

```python
if step % 500 == 0:
    ...
```

with `tqdm` around the actual `viz.log_step(step)` loop.

Requirements:

- total must be `viz.num_steps`, so `--max-frames` is reflected correctly;
- unit should be `frame`;
- output goes to `sys.stderr`;
- description should identify the current episode, e.g. `log <episode_stem>`;
- remove the old every-500-frame progress log so it does not corrupt the progress display;
- do not claim the GUI rendered a frame; the bar represents frames logged/sent to Rerun.

Example UX:

```text
log episode_20260914_221356:  67%|███████████▍     | 1241/1842 [00:04<00:02, 298 frame/s]
```

`--info` must not show a progress bar.

### 5.2 Dependency declaration

`examples/export_policy_zarr.py` already imports `tqdm`. Verify whether `tqdm` is declared in the repository's install/dependency contract.

- If it is already declared elsewhere in the canonical install path, do not duplicate it.
- If it is not declared and this repository expects `pyproject.toml` to describe runtime Python dependencies, add the smallest appropriate `tqdm` dependency entry to `pyproject.toml`.
- Do not otherwise refactor dependency management.

### 5.3 Load the sibling report

For a processed file:

```text
episodes_processed/<task>/episode_xxx.h5
```

look for:

```text
episodes_processed/<task>/processing_report.yaml
```

Use the canonical loader from `dexmani_real.dataset.processing_report`; do not hand-parse a second copy of the schema in the example script.

### 5.4 Cross-check report relevance

Before presenting a loaded report as belonging to the current HDF5, cross-check enough identity to avoid showing an unrelated/copied report:

- report `task_name` must match the processed HDF5 `task_name`;
- report processed-schema identity must be compatible with the HDF5 schema identity;
- the HDF5 `source_episode` must appear in the report as an `accepted` entry.

A mismatch makes the batch summary unavailable, but must **not** prevent visualization of an otherwise valid processed HDF5.

### 5.5 Summary failure behavior

Batch-summary loading is auxiliary to visualization.

- Missing report (legacy batch): visualization succeeds; print one concise note such as:

```text
Batch summary unavailable: processing_report.yaml not found (legacy batch).
```

- Malformed/unsupported/mismatched report: visualization succeeds; print one concise warning explaining the summary is unavailable.
- Do not emit a traceback for an optional summary failure unless normal repository logging policy already requires one.
- A malformed processed HDF5 remains a real viewer failure; do not weaken its current validation.

### 5.6 End-of-run summary

After all requested frames are successfully logged, print a concise completion line and the batch summary.

Preferred UX with skipped/excluded episodes:

```text
Rerun logging complete: 1842 frames.
Batch: 17 accepted, 2 skipped, 1 user-excluded
Skipped:
  episode_20260914_215103 — persistent IK failure
  episode_20260914_221540 — persistent IK failure
Excluded:
  episode_20260914_223012 — excluded by annotation
```

When there are no skipped/excluded episodes:

```text
Rerun logging complete: 1842 frames.
Batch: 20 accepted, no skipped episodes.
```

Requirements:

- do not list all accepted episode names;
- preserve the stored reason text rather than inventing new explanations;
- keep `skipped` and `user-excluded` separate;
- only print `Rerun logging complete` after successful completion of the logging loop;
- if `viz.log_step()` raises, let the visualization failure remain visible and do not print a misleading success line.

### 5.7 `--info`

`--info` should continue to print the existing HDF5 structure summary and exit without opening Rerun.

Also print the same batch decision summary when available, but no progress bar.

Do not turn missing/invalid report metadata into a fatal `--info` error.

---

## 6. Keep class responsibilities clean

Do not put batch-report ownership into `ProcessedEpisodeVisualizer` unless there is a compelling reason discovered during implementation.

Preferred separation:

```text
ProcessedEpisodeVisualizer
    owns HDF5 -> Rerun behavior

CLI helpers in visualize_episode_processed.py
    own terminal presentation and current-file/report cross-check

processing_report.py
    owns persisted report schema + serialization/validation

processing.py
    owns episode admission + transactional batch publication
```

Avoid unnecessary abstractions beyond this real persisted-data boundary.

---

## 7. Non-goals

Do not use this task to change any of the following:

- processed HDF5 modality schema or tensor shapes;
- point-cloud derivation/filtering;
- camera/table calibration semantics;
- `EpisodeDecision` admission thresholds;
- IK failure threshold;
- annotation semantics;
- Rerun 2D/3D blueprint layout;
- tactile/contact visualization;
- raw episode visualization;
- export-to-Zarr semantics;
- unrelated logging cleanup;
- hardware behavior.

Do not execute hardware-affecting programs.

---

## 8. Implementation order

Use this order to minimize rework:

1. Read `AGENTS.md` and inspect `git status --short`; preserve unrelated user changes.
2. Re-read the current implementations of:
   - `dexmani_real/dataset/contracts.py` (`EpisodeDecision`),
   - `dexmani_real/dataset/processing.py`,
   - `dexmani_real/dataset/processed.py`,
   - `examples/process_episodes.py`,
   - `examples/visualize_episode_processed.py`,
   - `examples/export_policy_zarr.py`,
   - `pyproject.toml`.
3. Implement the canonical `processing_report.py` contract and keep it pure/offline.
4. Integrate report writing into the existing staging transaction in `processing.py`.
5. Add viewer report loading/cross-check/summary helpers.
6. Replace periodic viewer progress logging with `tqdm`.
7. Resolve `tqdm` dependency declaration only if needed.
8. Run focused offline validation and repository low-cost checks.
9. Inspect the final diff for scope expansion and duplicate logic.
10. Clean temporary artifacts.
11. If and only if all acceptance criteria pass, delete `CODEX_TASK.md` and re-run final low-cost checks.

---

## 9. Acceptance criteria

All of the following must hold.

### A. Persisted report contract

- A successful non-dry-run processed batch contains `processing_report.yaml` in the final processed task directory.
- The report is versioned independently from the processed HDF5 schema.
- The report is minimal and does not serialize the whole internal processing return dictionary.
- The report accurately distinguishes accepted, skipped, and annotation-excluded episodes.
- Stored counts exactly match stored episode entries.
- Reason strings come from authoritative processing decisions.

### B. Atomicity

- Report creation occurs in staging before `atomic_publish()`.
- Any report-write/validation failure prevents publication.
- Any processed-HDF5 validation failure prevents publication and leaves no final report.
- Dry-run writes no report.

### C. Existing safety semantics

- Technical/source-corruption/programming failures still fail the entire batch.
- No technical exception is silently converted to a skipped episode.
- Existing acceptance/rejection thresholds are unchanged.

### D. Viewer progress

- Normal visualization shows one clean frame progress bar on `stderr`.
- Progress total equals the effective `viz.num_steps` after `--max-frames`.
- Old every-500-frame progress logging is removed.
- `--info` shows no progress bar.

### E. Viewer summary

- Valid report: concise batch counts plus skipped/excluded episode names and reasons.
- No skipped/excluded episodes: concise no-skip summary; do not list accepted episodes individually.
- Missing legacy report: current HDF5 still visualizes; one concise summary-unavailable note.
- Invalid or mismatched report: current HDF5 still visualizes; one concise warning/note.
- Current HDF5/report task identity and source-episode membership are cross-checked.
- A `viz.log_step()` failure does not print a false success message.

### F. Scope

- No hardware behavior is executed or changed.
- No unrelated refactors or cleanup.
- No duplicated report schema/parsing logic between producer and viewer.

---

## 10. Validation strategy

Follow `AGENTS.md`: use the smallest safe offline checks that genuinely validate this change. Do not run hardware examples and do not create a new testing framework solely for this task.

### 10.1 Required repository checks

Run:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

All must pass/produce an understood clean result.

### 10.2 Focused pure/offline checks

Exercise the report module without robot/camera hardware. Use a temporary directory outside the repository or `tempfile.TemporaryDirectory()`.

At minimum verify:

1. valid synthetic accepted/skipped/excluded decisions -> report build -> YAML write -> YAML load -> exact semantic round-trip;
2. duplicate episode name is rejected by report validation;
3. count mismatch is rejected;
4. invalid status is rejected;
5. accepted entry with a non-null reason is rejected;
6. skipped/excluded entry with missing reason is rejected;
7. unsupported report schema version is rejected.

Do not leave these temporary files in the repository.

### 10.3 Transactional reasoning/check

Inspect the final `process_episode_root()` control flow and confirm in the diff that the order is exactly:

```text
write accepted files -> validate accepted files -> write validated report -> atomic_publish
```

If a safe existing offline fixture/workflow already exists in the repository for `process_episode_root()`, use it. If none exists, do not invent a hardware-like integration fixture just for this task.

### 10.4 Viewer presentation checks

Do not run an example program as a generic test just to satisfy a checklist. Prefer testing pure formatting/loading helpers directly if they are factored so that this is possible.

Verify at least these cases from pure helper calls or existing safe fixtures:

- valid report with skipped + excluded;
- valid report with no skipped/excluded;
- missing report;
- malformed report;
- report task mismatch;
- current `source_episode` absent/not accepted in report.

### 10.5 Dependency check

If `pyproject.toml` is changed for `tqdm`, verify the syntax remains valid and explain why the dependency change is necessary. Do not alter unrelated dependency versions.

---

## 11. Review checklist before handoff

Perform an explicit self-review of the final diff:

- Is there exactly one owner for report schema validation?
- Is any admission/rejection logic duplicated in the viewer?
- Can a report be published without all accepted HDF5 files validating? It must not.
- Can a fatal technical error be mislabeled as a skipped episode? It must not.
- Can a stale/copied report be shown for an unrelated HDF5 without identity checks? It must not.
- Does missing report break legacy visualization? It must not.
- Does `--max-frames` correctly control tqdm total?
- Does any success message run from `finally` after a logging exception? It must not.
- Are progress output and summary concise enough for terminal use?
- Did the change touch any file not necessary for this vertical feature?
- Are comments/docstrings describing current behavior rather than this task history?

---

## 12. Cleanup

Before final handoff:

1. Remove all temporary YAML/HDF5/test fixture files created during validation.
2. Remove any temporary processed directories created only for this task.
3. Do not delete or modify real user episode data.
4. Inspect `git status --short`; only intentional source changes should remain.
5. Remove stale comments/debug prints introduced during development.
6. Do not commit generated caches, logs, Rerun artifacts, or bytecode.
7. After all acceptance checks pass, delete this `CODEX_TASK.md` file because it is a one-off task artifact prohibited as permanent repository documentation by `AGENTS.md`.
8. After deleting `CODEX_TASK.md`, run again:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

If the task is not fully complete or validation is not passing, **do not delete this file**.

---

## 13. Final handoff format

Codex final response should be concise and factual:

```text
Changed:
- <report contract/persistence>
- <viewer progress>
- <viewer summary/backward compatibility>

Validated:
- python -m compileall -q dexmani_real examples
- git diff --check
- <focused pure/offline checks actually run>

Not validated:
- hardware: not exercised
- <anything genuinely unverified>

Final worktree:
- <brief git status summary>
```

Never claim a check passed if it was skipped or failed. Never claim hardware validation for this task.
