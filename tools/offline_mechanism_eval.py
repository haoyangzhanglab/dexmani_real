#!/usr/bin/env python3
"""Offline mechanism evaluation — test adaptive filters vs fixed EMA through REAL IK.

Compares:
  - fixed cartesian EMA (current mechanism, 0.5/0.15)
  - One-Euro adaptive filter (speed-adaptive cutoff: slow->heavy, fast->light)
  - rotation deadband (freeze sub-threshold orientation tremor when holding still)

Same idealized-tracking replay + joint-space metrics as offline_teleop_replay.py.
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
from dexmani_real.utils.signal_utils import ema_smooth_pose

d = np.load(sorted(glob.glob("trajectories/*.npz"))[-1])
t = d["t"]; tp = d["target_pos"]; tq = d["target_quat_wxyz"]; q0 = d["arm_qpos_actual"][0]
N = len(t)
DT = float(np.median(np.diff(t)))

COLL = CollisionConfig(table_z_world=0.0, hand_extension_below_eef=0.076, hand_safe_margin=0.03)
planner = XArm7MotionPlanner(
    XArm7PlannerConfig(
        urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
        srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf"),
        base_pose_world=Pose(p=np.array([0.,0.,0.]), q=np.array([np.cos(np.pi/12),0.,0.,np.sin(np.pi/12)])),
        collision=COLL),
    planning_profile=PlanningProfile(),
    teleop_profile=TeleopProfile(teleop_dt=0.02, use_position_ik=True,
                                 max_pose_error_pos_m=0.02, max_pose_error_rot_rad=np.deg2rad(5.0)))

def fk(qp):
    P = planner.compute_eef_pose_world(qp); return P.p, P.q
def geo_deg(a, b):
    a=a/norm(a); b=b/norm(b); return np.rad2deg(2*np.arccos(min(1.0,abs(np.dot(a,b)))))

# ── One-Euro building blocks ──
def _alpha(cutoff, dt):
    tau = 1.0/(2*np.pi*max(cutoff,1e-6)); return 1.0/(1.0+tau/dt)

class OneEuroVec:
    def __init__(self, fmin, beta, dcut=1.0):
        self.fmin=fmin; self.beta=beta; self.dcut=dcut; self.xp=None; self.dxp=None
    def __call__(self, x, dt):
        x=np.asarray(x,float)
        if self.xp is None:
            self.xp=x.copy(); self.dxp=np.zeros_like(x); return x
        dx=(x-self.xp)/dt
        ad=_alpha(self.dcut,dt); dxf=ad*dx+(1-ad)*self.dxp
        cut=self.fmin+self.beta*np.abs(dxf)
        a=np.array([_alpha(c,dt) for c in cut])
        xf=a*x+(1-a)*self.xp
        self.xp=xf; self.dxp=dxf; return xf

class OneEuroRot:
    def __init__(self, fmin, beta, dcut=1.0):
        self.fmin=fmin; self.beta=beta; self.dcut=dcut; self.qf=None; self.sf=0.0
    def __call__(self, q, dt):
        q=np.asarray(q,float); q=q/norm(q)
        if self.qf is None: self.qf=q.copy(); return q
        ang=2*np.arccos(min(1.0,abs(np.dot(self.qf/norm(self.qf),q)))); spd=ang/dt
        ad=_alpha(self.dcut,dt); self.sf=ad*spd+(1-ad)*self.sf
        a=_alpha(self.fmin+self.beta*self.sf, dt)
        _,qf=ema_smooth_pose(np.zeros(3),q,np.zeros(3),self.qf,1.0,a)
        self.qf=qf.copy(); return qf

# ── metrics ──
def metrics(qcmd, ok):
    qcmd=np.asarray(qcmd); ok=np.array(ok,bool)
    qv=np.gradient(qcmd,t,axis=0); qj=np.gradient(np.gradient(qv,t,axis=0),t,axis=0)
    rev=sum(int(np.sum(np.abs(np.diff(np.sign(qv[:,jt])))>0)) for jt in (4,5,6))
    fkp=np.zeros((N,3)); fkq=np.zeros((N,4))
    for i in range(N): fkp[i],fkq[i]=fk(qcmd[i])
    perr=norm(fkp[ok]-tp[ok],axis=1)
    rerr=np.array([geo_deg(fkq[i],tq[i]) for i in range(N) if ok[i]])
    return (np.rad2deg(np.percentile(norm(qj,axis=1),95)), rev,
            np.sqrt(np.mean(perr**2))*1000, np.percentile(rerr,95))

# ── replay with a pluggable filter(pos,quat,dt,state)->(pos,quat) ──
def replay(filt):
    prev=q0.copy(); st={}; qout=[]; okout=[]
    for i in range(N):
        P,Q=filt(tp[i].copy(), tq[i].copy(), DT, st)
        res=planner.solve_teleop_ik(Pose(p=P,q=Q), prev, prev)
        if res.success and res.qpos is not None: cmd=np.asarray(res.qpos,float); ok=True
        else: cmd=prev.copy(); ok=False
        qout.append(cmd); okout.append(ok); prev=cmd.copy()
    return metrics(qout, okout)

# fixed EMA reference
def make_fixed(ap, ar):
    def f(P,Q,dt,st):
        if "p" in st: P,Q=ema_smooth_pose(P,Q,st["p"],st["q"],ap,ar)
        st["p"]=P.copy(); st["q"]=Q.copy(); return P,Q
    return f
# One-Euro
def make_oneeuro(pmin,pbeta,rmin,rbeta):
    fp=OneEuroVec(pmin,pbeta); fr=OneEuroRot(rmin,rbeta)
    def f(P,Q,dt,st): return fp(P,dt), fr(Q,dt)
    return f
# fixed EMA + rotation deadband (freeze sub-threshold orientation)
def make_ema_db(ap,ar,db_deg):
    def f(P,Q,dt,st):
        qc=st.get("qc")
        if qc is not None and geo_deg(Q,qc)<db_deg: Q=qc.copy()      # hold committed orient
        st["qc"]=Q.copy()
        if "p" in st: P,Q=ema_smooth_pose(P,Q,st["p"],st["q"],ap,ar)
        st["p"]=P.copy(); st["q"]=Q.copy(); return P,Q
    return f

tests=[
    ("fixed EMA 0.5/0.15 (ref)", make_fixed(0.5,0.15)),
    ("OneEuro p(1.5,25) r(1.0,3)", make_oneeuro(1.5,25,1.0,3.0)),
    ("OneEuro p(2.0,40) r(1.2,5)", make_oneeuro(2.0,40,1.2,5.0)),
    ("OneEuro p(1.2,30) r(0.8,4)", make_oneeuro(1.2,30,0.8,4.0)),
    ("EMA 0.5/0.15 + rot db 2deg", make_ema_db(0.5,0.15,2.0)),
    ("EMA 0.5/0.15 + rot db 3deg", make_ema_db(0.5,0.15,3.0)),
    ("EMA 0.6/0.2 + rot db 3deg",  make_ema_db(0.6,0.20,3.0)),
]
print(f"N={N} dt={DT*1000:.1f}ms\n")
print(f"{'mechanism':30s} {'jerk_p95':>9s} {'wristRev':>9s} {'posLag':>8s} {'rotLag':>8s}")
print(f"{'':30s} {'(deg/s3)':>9s} {'(j5-7)':>9s} {'rms mm':>8s} {'p95deg':>8s}")
t0=time.time()
for name,filt in tests:
    j,rev,pl,rl=replay(filt)
    print(f"{name:30s} {j:9.0f} {rev:9d} {pl:8.1f} {rl:8.2f}")
print(f"\n(took {time.time()-t0:.0f}s)")
