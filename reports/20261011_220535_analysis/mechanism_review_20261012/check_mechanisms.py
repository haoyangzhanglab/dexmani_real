"""Small offline counterexamples for the reviewed scheduler and SDK evidence.

No hardware objects are constructed. Actual scheduler method bodies are executed
against a fake clock and fake robot, to distinguish a logic issue from a claim
about a measured hardware event. SDK analysis only reads the installed ELF.
"""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import struct
from types import MethodType, SimpleNamespace as NS


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def scheduler_cases():
    path = ROOT / "dexmani_real/deployment/runner.py"
    tree = ast.parse(path.read_text())
    names = {"step", "_handoff", "_execute_slot", "_dispatch_deadline_ns"}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    clock = NS(now=625_000_000)
    env = dict(time=NS(monotonic_ns=lambda: clock.now), Plan=NS,
               decision_is_fresh=lambda sources, now, seconds: all(
                   0 <= now - stamp <= seconds * 1e9
                   for stamps in sources.values() for stamp in stamps),
               feedback_deadline_ns=lambda row, runtime: min(row.arm_ns, row.hand_ns) + 133_333_334,
               DispatchError=type("DispatchError", (Exception,), {}),
               DispatchInterrupted=type("DispatchInterrupted", (Exception,), {}),
               DispatchStatus=NS(NOT_CALLED=0, ACCEPTED=1, CRC_UNCONFIRMED=2))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), env)

    def runner(query=None, slow=False):
        r = NS(run_id=7, query=query, plan=None, started_ns=0, dt_ns=62_500_000,
               last_slot=9, next_step_ns=625_000_000, recorder=None, results=None,
               runtime=None, execution=NS(max_decision_age_s=1., action_steps=8, execution_mode="async"),
               shared=NS(quit_requested=False), history=deque(maxlen=2), issues=[], sends=[],
               execute=True, model=NS(future=None), wait_started_ns=None)
        for n in names:
            setattr(r, n, MethodType(env[n], r))
        r._has_motion_authority = lambda: True
        r._expired_budget = lambda: None
        r._budget_limit = lambda: None
        r._poll_model = lambda *args: None
        r._prefetch = lambda *args: None
        r._observation_issue = lambda row: None
        r._command_for_slot = lambda row, slot: NS(arm_qpos=(0.,) * 7)
        r._record = lambda *args: None

        def finish(issue, **kwargs):
            r.issues.append(issue)
            r.run_id = None

        def read():
            if slow:
                clock.now += 70_000_000
            return NS(arm_ns=625_200_000, hand_ns=clock.now)

        def send(command, *, valid_until_ns):
            r.sends.append(dict(send_ms=clock.now / 1e6,
                                valid_until_ms=valid_until_ns / 1e6,
                                current_logical_slot=r.last_slot))
            return NS(arm=1, hand=1, continued=True)

        r._finish_episode = finish
        r._read_observation = read
        r.robot = NS(check=lambda: None, send_action=send)
        return r

    query = NS(run_id=7, handoff_slot=10, result=(0.,) * 15, completed_ns=630_000_000,
               bootstrap=False, slot=9, sources={"joint_state": (562_500_000,)}, query_id=2)
    clock.now = 637_000_000
    r = runner(query)
    r._handoff(10, NS())
    assert r.issues == ["handoff_model_deadline_exceeded"]
    late_model = dict(nominal_ms=625, ready_ms=630, consumed_ms=637,
                      slot_end_ms=687.5, outcome=r.issues[0])

    clock.now = 625_000_000
    r = runner(slow=True)
    r.plan = NS(query_id=1, anchor_slot=2, segment_end=14,
                sources={"joint_state": (500_000_000,)})
    r.step()
    assert not r.issues and len(r.sends) == 1
    assert r.sends[0]["send_ms"] > 687.5
    return dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                early_cutoff=late_model,
                slow_read=dict(slot_start_ms=625, slot_end_ms=687.5,
                               observation_block_ms=70, sends=r.sends,
                               outcome="previous slot's command sent after slot end"),
                caveat="Fake I/O exercises the actual step, handoff, execute and deadline method bodies. Other collaborators are explicit stubs. No late send of this kind was found in the four recorded trials.")


def sdk_evidence():
    path = Path("/usr/lib/libxhand_control.so")
    b = path.read_bytes()
    # In this ELF, the reviewed .rodata virtual addresses equal file offsets.
    def string_at(address):
        return b[address:b.index(b"\0", address)].decode()
    overcurrent = string_at(0x99e88)
    crc = string_at(0x9a5f3)
    assert "overcurrent" in overcurrent and "500ms" in overcurrent
    assert "CRC" in crc
    pair = b.find(struct.pack("<ii", 307, 1501070))
    assert pair == 0x9af38
    disassembly = Path("/tmp/dexmani_xhand_warning_research_20261011/libxhand_control_disassembly.txt")
    lines = disassembly.read_text().splitlines()
    selected = []
    for lo, hi in [(0x2bb02, 0x2bb0d), (0x2be1c, 0x2be2e),
                   (0x56fb0, 0x57024), (0x5d04a, 0x5d0c2),
                   (0x5d0fe, 0x5d171)]:
        selected.append(f"\n# Instructions {lo:#x}..{hi:#x}")
        for line in lines:
            try:
                address = int(line.split(":", 1)[0].strip(), 16)
            except ValueError:
                continue
            if lo <= address <= hi:
                selected.append(line)
    (OUT / "sdk_disassembly_excerpts.txt").write_text("\n".join(selected) + "\n")
    return dict(library=str(path), sha256=hashlib.sha256(b).hexdigest(),
                method="Static ELF reads and disassembly, no SDK invocation",
                sdk_1501035=dict(message=overcurrent, string_address="0x99e88", mapping_instruction="0x2be23"),
                sdk_1501070=dict(message=crc, string_address="0x9a5f3", mapping_instruction="0x2bb02"),
                hardware_307=dict(custom_code=1501070, table_pair_address=hex(pair)),
                read_state="force_update calls the request method, then copies internal HandState storage. Complete finite values are not in themselves proof of fresh per-board samples.",
                crc_limit="The existing CRC sample is frame 0, with several board 307 values. It has no previous captured frame to establish whether joints are cached. Do not claim all CRC joint data are stale, or always fresh.",
                reset_evidence=json.loads(Path("/tmp/dexmani_xhand_warning_research_20261011/sdk_warning_evidence.json").read_text()))


def main():
    result = dict(scheduler=scheduler_cases(), xhand_sdk=sdk_evidence())
    (OUT / "mechanism_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
