# Formal Eval follow-up guide — superseded

> **SUPERSEDED — do not use for implementation.**

The former version of this document described a second Formal Eval transaction
with initial evidence barriers, per-command evidence admission, pending
termination, acceptance-fenced invalidation, and action-step truncation. Those
mechanisms were removed in commit `267ca0d8c73300ea69d49b8dc07598793642583f`.

Use [`policy_rollout_simplification_guide.md`](policy_rollout_simplification_guide.md)
and the current source as the only implementation references. The current
contract is:

```text
Recorder START / RECORDING ACK
    → one RUNNING generation
    → causal Policy observation
    → predict_action_chunk() → timestamped full chunk
    → decode / IK / safety
    → publish or reject
    → record the resulting raw-v24 row
    → S/C/D/Q outcome or timeout
    → motion fence
    → Recorder STOP(save=...)
    → result.json
```

Recording observes the control result and never controls action admission or
waits for arm/hand acceptance before invalidating a rollout. `save` describes
raw storage commitment; task outcome is recorded separately. `run` and `eval`
share this physical path, while `shadow` validates the path without actuator
publication. Safety, generation fences, worker supervision, and SDK-boundary
checks remain active.
