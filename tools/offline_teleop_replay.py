#!/usr/bin/env python3
"""Offline teleop replay harness — evaluate smoothing/mapping configs through REAL IK.

Loads a recorded trajectory (wrist + raw mapped target), replays the raw target
stream through {EMA config} -> solve_teleop_ik -> joint command chain, and measures
joint-space chatter / jerk / tracking-lag / IK-success for each config.

Idealized-tracking assumption: the arm perfectly follows each command within one
tick, so IK seed = previous command. This isolates the smoothing+redundancy chain
(what actually differs between configs) from firmware servo dynamics.
"""
from __future__ import annotations
import sys, time, glob, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)

import numpy as np
from numpy.linalg import norm

from dexmani_real import ASSET_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.collision_config import CollisionConfig
from dexmani_real.utils.signal_utils import ema_smooth, ema_smooth_pose

TRAJ = sorted(glob.glob("trajectories/*.npz"))[-1]
d = np.load(TRAJ)
t = d["t"]; wp = d["wrist_pos"]; wq = d["wrist_quat_wxyz"]
tp = d["target_pos"]; tq = d["target_quat_wxyz"]          # raw mapped (pre-EMA in old code)
ae = d["actual_eef_pos"]; q0 = d["arm_qpos_actual"][0]
N = len(t)
print(f"traj={TRAJ}  N={N}  dur={t[-1]-t[0]:.1f}s")

# ── build planner (offline, no hardware) ──
COLL = CollisionConfig(table_z_world=0.0, hand_extension_below_eef=0.076, hand_safe_margin=0.03)
planner = XArm7MotionPlanner(
    XArm7PlannerConfig(
        urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
        srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf"),
        base_pose_world=Pose(p=np.array([0.,0.,0.]), q=np.array([np.cos(np.pi/12),0.,0.,np.sin(np.pi/12)])),
        collision=COLL,
    ),
    planning_profile=PlanningProfile(),
    teleop_profile=TeleopProfile(teleop_dt=0.02, use_position_ik=True,
                                 max_pose_error_pos_m=0.02, max_pose_error_rot_rad=np.deg2rad(5.0)),
)

def fk_pose(qp):
    P = planner.compute_eef_pose_world(qp)
    return P.p, P.q

def quat_ang_deg(a, b):
    a = a/norm(a); b = b/norm(b)
    return np.rad2deg(2*np.arccos(min(1.0, abs(np.dot(a, b)))))

# ── metrics on a joint command chain (lag measured vs RAW wrist intent) ──
def metrics(qcmd, ik_ok):
    qcmd = np.asarray(qcmd)
    qv = np.gradient(qcmd, t, axis=0)
    qa = np.gradient(qv, t, axis=0)
    qj = np.gradient(qa, t, axis=0)
    rev = 0
    for jt in (4,5,6):                       # wrist joints 5,6,7 chatter
        s = np.sign(qv[:,jt]); rev += int(np.sum(np.abs(np.diff(s))>0))
    ok = np.array(ik_ok, bool)
    fkp = np.zeros((len(qcmd),3)); fkq = np.zeros((len(qcmd),4))
    for i in range(len(qcmd)):
        fkp[i], fkq[i] = fk_pose(qcmd[i])
    perr = norm(fkp[ok] - tp[ok], axis=1)                          # vs RAW target pos
    rerr = np.array([quat_ang_deg(fkq[i], tq[i]) for i in range(len(qcmd)) if ok[i]])
    return dict(
        jerk_p95=np.rad2deg(np.percentile(norm(qj,axis=1),95)),
        wrist_reversals=rev,
        ik_ok=int(ok.sum()),
        pos_lag_rms_mm=float(np.sqrt(np.mean(perr**2))*1000),
        rot_lag_p95_deg=float(np.percentile(rerr,95)),
    )

# ── replay one config ──
def replay(mode, ap=None, ar=None, aj=None):
    prev = q0.copy()
    epos = eq = None
    qout = []; okout = []
    for i in range(N):
        P, Q = tp[i].copy(), tq[i].copy()
        if mode == "cart":
            if epos is not None:
                P, Q = ema_smooth_pose(P, Q, epos, eq, ap, ar)
            epos, eq = P.copy(), Q.copy()
        res = planner.solve_teleop_ik(Pose(p=P, q=Q), prev, prev)
        if res.success and res.qpos is not None:
            cmd = np.asarray(res.qpos, float)
            if mode == "joint":
                cmd = ema_smooth(cmd, prev, aj)
            ok = True
        else:
            cmd = prev.copy(); ok = False
        qout.append(cmd); okout.append(ok); prev = cmd.copy()
    return metrics(np.array(qout), okout)

configs = [
    ("baseline none",        dict(mode="none")),
    ("OLD joint-EMA 0.6",    dict(mode="joint", aj=0.6)),
    ("cart 0.4/0.2 (chosen)",dict(mode="cart", ap=0.4, ar=0.2)),
    ("cart 0.3/0.15",        dict(mode="cart", ap=0.3, ar=0.15)),
    ("cart 0.5/0.15 asym",   dict(mode="cart", ap=0.5, ar=0.15)),
    ("cart 0.5/0.12 asym",   dict(mode="cart", ap=0.5, ar=0.12)),
    ("cart 0.6/0.12 asym",   dict(mode="cart", ap=0.6, ar=0.12)),
    ("cart 0.6/0.10 asym",   dict(mode="cart", ap=0.6, ar=0.10)),
]
print(f"\n{'config':24s} {'jerk_p95':>9s} {'wristRev':>9s} {'posLag':>8s} {'rotLag':>8s}")
print(f"{'':24s} {'(deg/s3)':>9s} {'(j5-7)':>9s} {'rms mm':>8s} {'p95 deg':>8s}")
t0 = time.time()
for name, kw in configs:
    m = replay(**kw)
    print(f"{name:24s} {m['jerk_p95']:9.0f} {m['wrist_reversals']:9d} "
          f"{m['pos_lag_rms_mm']:8.1f} {m['rot_lag_p95_deg']:8.2f}")
print(f"\n(replay took {time.time()-t0:.0f}s)")

# ══════════════════════════════════════════ MAPPING ANALYSIS ═══════════════
print("\n" + "="*60 + "\n MAPPING ANALYSIS (wrist -> target)\n" + "="*60)
# 1. recover affine target_pos = A @ wrist_pos + b ; check A is a clean rotation
X = np.hstack([wp, np.ones((N,1))])          # (N,4)
M, *_ = np.linalg.lstsq(X, tp, rcond=None)   # (4,3)
A = M[:3].T; b = M[3]
U,S,Vt = np.linalg.svd(A)
resid = norm(tp - (wp@A.T + b), axis=1)
print(f" affine fit residual: rms={resid.mean()*1000:.2f}mm max={resid.max()*1000:.1f}mm")
print(f" A singular values (should be ~[1,1,1] for rigid 1:1): {np.round(S,4)}")
print(f" A orthogonality err |A A^T - I|max: {np.abs(A@A.T-np.eye(3)).max():.4f}")

# 2. position vs rotation TREMOR (high-freq energy, 1-pole HPF of the raw signal)
def hf_energy(sig_rate):    # mean |x[n]-x[n-1]| = per-tick jitter
    return np.mean(np.abs(np.diff(sig_rate)))
pv = np.gradient(tp, t, axis=0)                              # target pos vel
pj = norm(np.diff(pv,axis=0),axis=1)                         # pos vel increment (tremor)
rav = np.array([quat_ang_deg(tq[i-1],tq[i])/(t[i]-t[i-1]) for i in range(1,N)])
raj = np.abs(np.diff(rav))                                   # rot vel increment (tremor)
print(f"\n position tremor (|Δvel|): mean={pj.mean()*1000:.1f} mm/s/tick   p95={np.percentile(pj,95)*1000:.1f}")
print(f" rotation tremor (|Δvel|): mean={raj.mean():.1f} deg/s/tick  p95={np.percentile(raj,95):.1f}")
print(f" -> rotation channel tremor is {np.percentile(raj,95)/ (np.percentile(pj,95)*1000):.1f}x the position channel"
      f" (per-unit) => justifies alpha_rot << alpha_pos")

