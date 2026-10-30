
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from training.e2e_phase10_lidar_v7_curriculum.train_stage7_basic_maneuver_m0_m1 import BasicManeuverEnv, evaluate
from training.e2e_phase10_lidar_v7_curriculum.train_stage7_random_spawn_r0_reachable import ReachableRandomSpawnR0Env
from training.e2e_phase10_lidar_v7_curriculum.test_stage7_full_random_v2_reachable_fix import FullRandomV2ReachableEnv
from training.e2e_phase10_lidar_v7_curriculum.diagnose_stage7_m1r_left_right_asymmetry import install_goal

ROOT=Path("experiments/e2e_phase10_lidar_v7_curriculum")
BASE_MODEL=ROOT/"stage7_basic_maneuver_m0_m1/candidate_model.zip"
OUT=ROOT/"stage7_v5_closed_loop_residual_sac_to_m2"
OUT.mkdir(parents=True,exist_ok=True)
STATE=OUT/"state.json"; HB=OUT/"heartbeat.json"
ANCHORA=OUT/"anchor_A.json"; ANCHORB=OUT/"anchor_B.json"
BESTM=OUT/"best_residual_sac.zip"; BESTP=OUT/"best_residual_replay.pkl"; BESTR=OUT/"best_stage7_report.json"
FINAL7=OUT/"stage7_final_report.json"; M2READY=OUT/"M2_READY.json"; M2REPORT=OUT/"m2_report.json"

STEER_SCALE=.36; SPEED_SCALE=.06
A={"straight":1510000000,"left":1511000000,"right":1512000000,"r0":1513000000,"full":1514000000}
B={"straight":1520000000,"left":1521000000,"right":1522000000,"r0":1523000000,"full":1524000000}
F={"straight":1530000000,"left":1531000000,"right":1532000000,"r0":1533000000,"full":1534000000}

def savej(p,d):
    t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8"); t.replace(p)
def loadj(p):
    try:return json.loads(p.read_text(encoding="utf-8"))
    except:return None
def hb(phase,**kw): savej(HB,{"wall_time":time.time(),"phase":phase,**kw})

def task(name,seed):
    if name=="straight": return BasicManeuverEnv(seed=seed,fixed_skill="straight")
    if name=="left": return BasicManeuverEnv(seed=seed,fixed_skill="left")
    if name=="right": return BasicManeuverEnv(seed=seed,fixed_skill="right")
    if name=="r0": return ReachableRandomSpawnR0Env(seed=seed)
    if name=="full": return FullRandomV2ReachableEnv(seed=seed)
    if name=="m2": return M2Env(seed)
    raise ValueError(name)

class ResidualEnv(gym.Wrapper):
    def __init__(self,env,base):
        super().__init__(env); self.base=base; self.last=None
        self.action_space=spaces.Box(-1,1,(2,),dtype=np.float32)
    def reset(self,**kw):
        o,i=self.env.reset(**kw); self.last=np.asarray(o,np.float32); return o,i
    def step(self,a):
        ba,_=self.base.predict(self.last,deterministic=True)
        ba=np.asarray(ba,np.float32); a=np.asarray(a,np.float32)
        full=ba.copy()
        full[0]=np.clip(full[0]+SPEED_SCALE*a[0],-1,1)
        full[1]=np.clip(full[1]+STEER_SCALE*a[1],-1,1)
        o,r,te,tr,info=self.env.step(full)
        r=float(r)-.001*float(a[0]**2)-.0025*float(a[1]**2)
        self.last=np.asarray(o,np.float32)
        return o,r,te,tr,info

class Composite:
    def __init__(self,base,res): self.base=base; self.res=res
    def predict(self,obs,state=None,episode_start=None,deterministic=True):
        ba,_=self.base.predict(obs,deterministic=True)
        ra,ns=self.res.predict(obs,state=state,episode_start=episode_start,deterministic=deterministic)
        x=np.asarray(ba,np.float32).copy(); ra=np.asarray(ra,np.float32)
        if x.ndim==1:
            x[0]=np.clip(x[0]+SPEED_SCALE*ra[0],-1,1); x[1]=np.clip(x[1]+STEER_SCALE*ra[1],-1,1)
        else:
            x[...,0]=np.clip(x[...,0]+SPEED_SCALE*ra[...,0],-1,1); x[...,1]=np.clip(x[...,1]+STEER_SCALE*ra[...,1],-1,1)
        return x,ns

class M2Env(BasicManeuverEnv):
    def __init__(self,seed):
        super().__init__(seed=seed,fixed_skill="straight"); self.rng=np.random.default_rng(seed+77); self.k=0
    def reset(self,*,seed=None,options=None):
        self.k+=1; b=int(seed) if seed is not None else 1700000000+2*self.k
        last=None
        for j in range(32):
            o,i=super().reset(seed=b+2*j,options=options); last=(o,i)
            s=-1 if ((b//2+j)%2) else 1
            ang=s*float(self.rng.uniform(9,15)); dist=float(self.rng.uniform(2.0,2.8))
            z=install_goal(self,dist,ang)
            if z is not None:
                i=dict(i); i["m2"]=True; i["m2_angle_deg"]=ang; i["m2_distance_m"]=dist
                return np.asarray(z,np.float32),i
        return last

def trainenv(name,seed,base): return Monitor(ResidualEnv(task(name,seed),base))

def newres(base,device):
    e=trainenv("right",1560000000,base)
    m=SAC("MlpPolicy",e,learning_rate=8e-5,buffer_size=300000,learning_starts=600,batch_size=256,
          tau=.005,gamma=.99,train_freq=1,gradient_steps=1,ent_coef="auto_0.03",
          policy_kwargs={"net_arch":[128,128]},verbose=0,device=device)
    with torch.no_grad():
        nn.init.zeros_(m.actor.mu.weight); nn.init.zeros_(m.actor.mu.bias)
        if isinstance(getattr(m.actor,"log_std",None),nn.Linear):
            nn.init.zeros_(m.actor.log_std.weight); nn.init.constant_(m.actor.log_std.bias,-2.5)
    e.close(); return m

def suite(pol,seeds,tag,se=24,re=24,fe=16):
    hb("eval",tag=tag)
    return {
      "straight":evaluate(pol,task("straight",seeds["straight"]+9999),se,seeds["straight"],f"{tag}_STRAIGHT_{se}"),
      "left":evaluate(pol,task("left",seeds["left"]+9999),se,seeds["left"],f"{tag}_LEFT_{se}"),
      "right":evaluate(pol,task("right",seeds["right"]+9999),se,seeds["right"],f"{tag}_RIGHT_{se}"),
      "r0":evaluate(pol,task("r0",seeds["r0"]+9999),re,seeds["r0"],f"{tag}_R0_{re}"),
      "full":evaluate(pol,task("full",seeds["full"]+9999),fe,seeds["full"],f"{tag}_FULL_{fe}")}

def score(r):
    l=r["left"]["success_rate"]; q=r["right"]["success_rate"]
    cs=r["left"]["collision_rate"]+r["right"]["collision_rate"]+r["r0"]["collision_rate"]
    return 3*min(l,q)+1.5*(l+q)/2+r["r0"]["success_rate"]+.45*r["straight"]["success_rate"]+.25*r["full"]["success_rate"]-2*cs

def gate(anchor,cur,cand,se=24,re=24,fe=16):
    ol,orr=cur["left"]["success_rate"],cur["right"]["success_rate"]
    nl,nr=cand["left"]["success_rate"],cand["right"]["success_rate"]
    checks={
      "left_no_collapse":nl>=ol-1/se-1e-12,
      "right_no_collapse":nr>=orr-1/se-1e-12,
      "straight_keep":cand["straight"]["success_rate"]>=anchor["straight"]["success_rate"]-2/se-1e-12,
      "r0_keep":cand["r0"]["success_rate"]>=anchor["r0"]["success_rate"]-2/re-1e-12,
      "r0_best_keep":cand["r0"]["success_rate"]>=cur["r0"]["success_rate"]-2/re-1e-12,
      "full_keep":cand["full"]["success_rate"]>=anchor["full"]["success_rate"]-2/fe-1e-12,
      "straight_col":cand["straight"]["collision_rate"]<=.0834,
      "left_col":cand["left"]["collision_rate"]<=.0834,
      "right_col":cand["right"]["collision_rate"]<=.0834,
      "r0_col":cand["r0"]["collision_rate"]<=.0834,
      "full_col":cand["full"]["collision_rate"]<=.125}
    useful=min(nl,nr)>=min(ol,orr)+1/se-1e-12 or (nl+nr)/2>=(ol+orr)/2+2/se-1e-12 or score(cand)>=score(cur)+.12
    return {"passed":bool(all(checks.values()) and useful),"useful":useful,"checks":checks,
            "old_turn_floor":min(ol,orr),"new_turn_floor":min(nl,nr),"old_score":score(cur),"new_score":score(cand)}

def qfoundation(a,b):
    for r in (a,b):
        if not (r["straight"]["success_rate"]>=.75 and r["left"]["success_rate"]>=.65 and r["right"]["success_rate"]>=.65
                and r["straight"]["collision_rate"]<=.0834 and r["left"]["collision_rate"]<=.0834 and r["right"]["collision_rate"]<=.0834
                and r["r0"]["success_rate"]>=.65 and r["r0"]["collision_rate"]<=.0834): return False
    return True

def formalok(r):
    return bool(r["straight"]["success_rate"]>=.80 and r["left"]["success_rate"]>=.70 and r["right"]["success_rate"]>=.70
        and r["straight"]["collision_rate"]<=.05 and r["left"]["collision_rate"]<=.05 and r["right"]["collision_rate"]<=.05
        and r["r0"]["success_rate"]>=.70 and r["r0"]["collision_rate"]<=.08)

def trainblock(model,base,name,steps,seed,cycle):
    e=trainenv(name,seed,base); model.set_env(e); hb("train",cycle=cycle,task=name,steps=steps)
    try:model.learn(total_timesteps=steps,reset_num_timesteps=False,log_interval=8,progress_bar=False)
    finally:e.close()

def formal(base,device):
    m=SAC.load(str(BESTM),device=device); p=Composite(base,m)
    r=suite(p,F,"STAGE7_FINAL",60,64,80); r["foundation_complete"]=formalok(r); savej(FINAL7,r); return r

def init(base,device):
    m=newres(base,device); m.save(str(BESTM)); m.save_replay_buffer(str(BESTP)); p=Composite(base,m)
    a=suite(p,A,"ANCHOR_A"); b=suite(p,B,"ANCHOR_B"); savej(ANCHORA,a); savej(ANCHORB,b)
    savej(BESTR,{"eval_A":a,"eval_B":b,"score_A":score(a),"score_B":score(b)})
    st={"status":"RUNNING","phase":"STAGE7_SEARCH","cycles_completed":0,"promotions":0,"stage7_formal_pass":False,"m2_started":False}
    savej(STATE,st); return st,a,b,a,b

def loadstate():
    st=loadj(STATE); a=loadj(ANCHORA); b=loadj(ANCHORB); br=loadj(BESTR)
    return st,a,b,br["eval_A"],br["eval_B"]

def search(base,device,deadline,st,aa,ab,ca,cb):
    cutoff=deadline-70*60
    for cyc in range(int(st.get("cycles_completed",0))+1,50):
        if time.time()>=cutoff: break
        print(f"\n========== STAGE7 V5 CYCLE {cyc:02d} ==========")
        m=SAC.load(str(BESTM),device=device); m.load_replay_buffer(str(BESTP))
        blocks=[("right",4200),("left",3400),("r0",3600),("straight",1400),("full",1400),("right",2200),("left",1800)]
        for j,(name,steps) in enumerate(blocks):
            if time.time()>=cutoff: break
            trainblock(m,base,name,steps,1600000000+cyc*100000+j*1000,cyc)
        mp=OUT/f"cycle_{cyc:02d}.zip"; rp=OUT/f"cycle_{cyc:02d}_replay.pkl"; m.save(str(mp)); m.save_replay_buffer(str(rp))
        p=Composite(base,m); xa=suite(p,A,f"C{cyc:02d}_A"); ga=gate(aa,ca,xa); print("GATE_A =",ga)
        passed=False; xb=None; gb=None
        if ga["passed"]:
            xb=suite(p,B,f"C{cyc:02d}_B"); gb=gate(ab,cb,xb); print("GATE_B =",gb); passed=gb["passed"]
        if passed:
            shutil.copy2(mp,BESTM); shutil.copy2(rp,BESTP); ca,cb=xa,xb
            st["promotions"]=int(st.get("promotions",0))+1
            savej(BESTR,{"eval_A":ca,"eval_B":cb,"gate_A":ga,"gate_B":gb,"cycle":cyc,"score_A":score(ca),"score_B":score(cb)})
            print("PROMOTED = True")
        else: print("PROMOTED = False")
        st["cycles_completed"]=cyc; savej(STATE,st); hb("cycle_done",cycle=cyc,promotions=st["promotions"])
        if qfoundation(ca,cb):
            fr=formal(base,device)
            if fr["foundation_complete"]:
                st["stage7_formal_pass"]=True; st["phase"]="M2_READY"; savej(STATE,st); return st,True
    fr=formal(base,device); st["stage7_formal_pass"]=bool(fr["foundation_complete"]); st["phase"]="M2_READY" if st["stage7_formal_pass"] else "STAGE7_FINAL_FAILED"; savej(STATE,st)
    return st,st["stage7_formal_pass"]

def m2eval(pol,seed,n,tag): return evaluate(pol,M2Env(seed+9999),n,seed,tag)

def m2train(base,device,deadline,st):
    savej(M2READY,{"m2":"turn-then-straight","stage7_formal_pass":True,"time":time.time()})
    st["phase"]="M2_TRAINING"; st["m2_started"]=True; savej(STATE,st)
    cutoff=deadline-15*60; m=SAC.load(str(BESTM),device=device); m.load_replay_buffer(str(BESTP)); p=Composite(base,m)
    anc=m2eval(p,1800000000,32,"M2_ANCHOR_32"); best=anc; promos=0; block=0
    while time.time()<cutoff:
        block+=1; m=SAC.load(str(BESTM),device=device); m.load_replay_buffer(str(BESTP))
        trainblock(m,base,"m2",7000,1810000000+block*2000,100+block)
        trainblock(m,base,"right",1800,1820000000+block*2000,100+block)
        trainblock(m,base,"left",1800,1830000000+block*2000,100+block)
        p=Composite(base,m); c=m2eval(p,1800000000,32,f"M2_B{block:02d}_32")
        retain=suite(p,A,f"M2_KEEP_{block:02d}",16,16,8)
        rok=(retain["left"]["success_rate"]>=.625 and retain["right"]["success_rate"]>=.625 and retain["r0"]["success_rate"]>=.625
             and retain["left"]["collision_rate"]<=.125 and retain["right"]["collision_rate"]<=.125 and retain["r0"]["collision_rate"]<=.125)
        improved=c["success_rate"]>=best["success_rate"]+1/32-1e-12 and c["collision_rate"]<=.10
        print("M2_DECISION =",{"improved":improved,"retention_ok":rok,"best":best["success_rate"],"cand":c["success_rate"]})
        if improved and rok:
            m.save(str(BESTM)); m.save_replay_buffer(str(BESTP)); best=c; promos+=1; print("M2_PROMOTED = True")
        if best["success_rate"]>=.60: break
    fm=SAC.load(str(BESTM),device=device); fp=Composite(base,fm); final=m2eval(fp,1900000000,60,"M2_FINAL_60")
    savej(M2REPORT,{"anchor":anc,"best_fixed_seed":best,"final_60":final,"promotions":promos})
    st["phase"]="M2_COMPLETE"; st["m2_promotions"]=promos; st["m2_final_success"]=final["success_rate"]; st["m2_final_collision"]=final["collision_rate"]; savej(STATE,st)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--deadline-epoch",type=float,required=True); ap.add_argument("--resume",action="store_true"); a=ap.parse_args()
    deadline=float(a.deadline_epoch); device="cuda" if torch.cuda.is_available() else "cpu"
    print("STAGE = STAGE7_V5_CLOSED_LOOP_RESIDUAL_SAC_TO_M2"); print("DEVICE =",device)
    print("OLD_MAGNITUDE_RATIO_GATE = REMOVED"); print("REAL_EPISODE_GATE = True"); print("TWO_SEED_CONFIRMATION = True")
    print("BASE_M0M1_FROZEN = True"); print("AUTO_M2_AFTER_STAGE7_PASS = True"); print("M2 = TURN_THEN_STRAIGHT")
    if not BASE_MODEL.is_file(): raise FileNotFoundError(BASE_MODEL)
    base=SAC.load(str(BASE_MODEL),device=device)
    for p in base.policy.parameters(): p.requires_grad_(False)
    can=a.resume and all(x.is_file() for x in (STATE,ANCHORA,ANCHORB,BESTM,BESTP,BESTR))
    if can: st,aa,ab,ca,cb=loadstate()
    else: st,aa,ab,ca,cb=init(base,device)
    if st.get("phase")=="STAGE7_SEARCH": st,passed=search(base,device,deadline,st,aa,ab,ca,cb)
    else: passed=bool(st.get("stage7_formal_pass",False))
    if passed:
        print("STAGE7_FORMAL_PROMOTION = True")
        if time.time()<deadline-20*60: m2train(base,device,deadline,st)
        else: savej(M2READY,{"m2":"turn-then-straight","stage7_formal_pass":True,"training_deferred_due_to_deadline":True})
    else: print("STAGE7_FORMAL_PROMOTION = False")
    st=loadj(STATE) or st
    if st.get("phase")=="M2_COMPLETE": status="M2_COMPLETE"
    elif st.get("stage7_formal_pass"): status="M2_READY"
    else: status="STAGE7_NOT_PROMOTED"
    st["status"]=status; st["finished_wall_time"]=time.time(); savej(STATE,st); hb("done",status=status)
    print("STATUS =",status); print("STAGE7_PROMOTIONS =",st.get("promotions",0)); print("STAGE7_FORMAL_PASS =",st.get("stage7_formal_pass",False))
    print("M2_STARTED =",st.get("m2_started",False)); print("DONE = True")
if __name__=="__main__": main()
