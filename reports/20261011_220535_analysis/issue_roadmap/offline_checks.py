"""Reproduce two planning-relevant behaviors without importing a device/runtime owner.

Run from the repository root with the existing real_robot Python environment.
This is a diagnostic for the reviewed source revision, not a hardware safety test.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
REPORT = Path(__file__).resolve().parent
PREVIOUS = REPORT.parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_hand_range():
    # Configuration and model constants are offline-only imports.
    from dexmani_real.config.hardware import HandParams

    baseline = HandParams()
    baseline.validate()
    rows = []
    for value in (-1, 0, 1, 300, 360, 400, 401, 1200, True, 300.0):
        candidate = replace(baseline, tor_max_ma=(value,) * 12)
        try:
            candidate.validate()
        except ValueError as exc:
            accepted, reason = False, str(exc)
        else:
            accepted, reason = True, None
        rows.append(dict(value=value, python_type=type(value).__name__,
                         accepted=accepted, reason=reason))
    return dict(
        default_values=list(baseline.tor_max_ma),
        cases=rows,
        interpretation=("Current validation accepts integers above the documented 0..400 "
                        "Mode-3 command range. Bool acceptance concerns direct dataclass "
                        "construction; the YAML loader has separate type validation. "
                        "No command was constructed or sent."),
    )


def check_handoff():
    path = ROOT / "dexmani_real/deployment/runner.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in ast.walk(tree)
                if isinstance(item, ast.FunctionDef) and item.name == "_handoff")
    # Execute the actual method body, with explicit fake clock/authority/freshness.
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    results = []
    cases = (
        dict(name="completed_before_nominal", completed_ms=624, expected="installed"),
        dict(name="completed_after_nominal_but_before_consumption", completed_ms=630,
             expected="handoff_model_deadline_exceeded"),
        dict(name="result_not_ready", ready=False, expected="handoff_result_not_ready"),
        dict(name="authority_revoked", authority=False, expected="handoff_authority_revoked"),
        dict(name="logical_slot_missed", slot=11, now_ms=700, expected="handoff_slot_missed"),
        dict(name="observation_invalid", observation_issue="observation_stale",
             expected="observation_stale"),
        dict(name="decision_stale", fresh=False, expected="handoff_decision_expired"),
        dict(name="previous_run", query_run_id=6, expected="ignored"),
        dict(name="bootstrap_head_retained", bootstrap=True, completed_ms=630,
             expected="installed"),
        dict(name="consumption_after_slot_end_method_only", now_ms=695,
             expected="installed"),
    )
    for case in cases:
        now_ns = int(case.get("now_ms", 637) * 1e6)
        namespace = dict(
            time=SimpleNamespace(monotonic_ns=lambda: now_ns),
            decision_is_fresh=lambda *args: case.get("fresh", True),
            Plan=SimpleNamespace,
        )
        exec(compile(module, str(path), "exec"), namespace)
        query = SimpleNamespace(
            run_id=case.get("query_run_id", 7), handoff_slot=10,
            result=tuple(range(15)) if case.get("ready", True) else None,
            completed_ns=(int(case.get("completed_ms", 624) * 1e6)
                          if case.get("ready", True) else None),
            bootstrap=case.get("bootstrap", False), slot=9,
            sources={"arm": (500_000_000, 562_500_000)}, query_id=2,
        )
        runner = SimpleNamespace(
            query=query, run_id=7, started_ns=0, dt_ns=62_500_000,
            plan=None, issue=None,
            execution=SimpleNamespace(max_decision_age_s=1.0, action_steps=8),
            _has_motion_authority=lambda: case.get("authority", True),
            _observation_issue=lambda row: case.get("observation_issue"),
        )

        def finish(issue, *, row):
            runner.issue = issue
            runner.run_id = None

        runner._finish_episode = finish
        slot = case.get("slot", 10)
        namespace["_handoff"](runner, slot, None)
        outcome = runner.issue or ("installed" if runner.plan else "ignored")
        assert outcome == case["expected"], (case, outcome)
        results.append(dict(
            case=case["name"], nominal_ms=625, slot_end_ms=687.5,
            completed_ms=(query.completed_ns / 1e6 if query.completed_ns is not None else None),
            consumed_ms=now_ns / 1e6, logical_slot=slot, outcome=outcome,
            first_prediction_index=(slot - runner.plan.anchor_slot if runner.plan else None),
        ))
    return dict(
        source_sha256=digest(path), source_line=node.lineno, cases=results,
        interpretation=("Isolated current _handoff behavior, not a full runner integration "
                        "test. The late-completion case is rejected even while the current "
                        "slot still has time. The last case motivates explicitly defining "
                        "the consumption window before relaxing the nominal deadline; it "
                        "does not establish that a late dispatch occurred in the Raw session. "
                        "No implementation fix has been applied."),
    )


def main():
    previous = json.loads((PREVIOUS / "next_experiments/preparation.json").read_text())
    metrics = json.loads((PREVIOUS / "metrics.json").read_text())
    source_paths = sorted(set(previous["source_hashes"]) | {
        "dexmani_real/dataset/export.py", "dexmani_real/utils/episode_results.py",
        "examples/export_policy_zarr.py", "dexmani_real/utils/log.py",
    })
    sources = {name: digest(ROOT / name) for name in source_paths}
    raw_root = Path(previous["original_session"])
    raw = {name: dict(sha256=digest(raw_root / name), expected_sha256=entry["sha256"])
           for name, entry in metrics["source_files"].items()}
    for entry in raw.values():
        entry["unchanged"] = entry["sha256"] == entry["expected_sha256"]
    result = dict(
        scope="offline diagnostic and experiment planning only",
        production_code_modified=False, hardware_connected=False, model_loaded=False,
        hand_config=check_hand_range(), handoff=check_handoff(),
        source_hashes=sources,
        reviewed_sources_unchanged=all(sources[name] == expected
            for name, expected in previous["source_hashes"].items()),
        raw_source_files=raw,
        raw_files_unchanged=all(entry["unchanged"] for entry in raw.values()),
        diagnostic_source_sha256=digest(Path(__file__)),
    )
    (REPORT / "offline_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(
        raw_files_unchanged=result["raw_files_unchanged"],
        reviewed_sources_unchanged=result["reviewed_sources_unchanged"],
        hand_range_cases=result["hand_config"]["cases"],
        handoff_cases=result["handoff"]["cases"],
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
