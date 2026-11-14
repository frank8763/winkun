"""Two fixed training-only reactive laws. No maps, trajectories or rollouts."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t544_terminal_feedback_repair'));import t544_runtime as Old
np=Old.np;E=Old.E;F=Old.F
NORMAL=Old.BASE_PAYLOAD
ANGLES=np.linspace(-.75*math.pi,.75*math.pi,48,dtype=np.float32).astype(np.float64)
def control(kind,pose,goal,scan,parameters):
 speed_gain,maxwheel,wheelbase,yawgain,maxsteer=map(float,parameters);assert kind in ['APF','GAP'] and np.asarray(scan).shape==(48,) and np.isfinite(scan).all()
 delta=np.asarray(goal)-np.asarray(pose)[:2];d=float(np.linalg.norm(delta));e=math.atan2(math.sin(math.atan2(delta[1],delta[0])-pose[2]),math.cos(math.atan2(delta[1],delta[0])-pose[2]));ranges=np.clip(np.asarray(scan,float)*6,.001,6)
 pts=np.c_[.48+ranges*np.cos(ANGLES),ranges*np.sin(ANGLES)];rad=np.linalg.norm(pts,axis=1);directions=np.arctan2(pts[:,1],pts[:,0]);forward=np.abs(directions)<=math.pi/3
 clearance=float(rad[forward].min()-.60);attractive=np.asarray([math.cos(e),math.sin(e)])
 if kind=='APF':
  weights=np.maximum(0.,1/np.maximum(rad,.30)-1/1.8)/np.maximum(rad,.30)**2
  repel=-(weights[:,None]*pts/np.maximum(rad[:,None],.3)).sum(0);mag=float(np.linalg.norm(repel));repel=repel*min(1.,1.5/max(mag,1e-12));vector=attractive+repel;direction=math.atan2(vector[1],vector[0]) if np.linalg.norm(vector)>1e-12 else e
 else:
  candidates=ANGLES[(np.abs(ANGLES)<1.45)];valid=np.ones(len(candidates),bool)
  for r,a in zip(rad,directions):
   if r<1.8:valid &= np.abs(np.arctan2(np.sin(candidates-a),np.cos(candidates-a)))>math.asin(min(1.,.60/max(r,.001)))
  if valid.any():
   # Largest continuous free angular interval; goal alignment breaks equal widths.
   starts=np.flatnonzero(valid & ~np.r_[False,valid[:-1]]);ends=np.flatnonzero(valid & ~np.r_[valid[1:],False]);groups=[np.arange(i,j+1) for i,j in zip(starts,ends)];group=max(groups,key=lambda g:(len(g),float(np.cos(candidates[g]-e).max()),-int(g[0])));direction=float(candidates[group[np.argmax(np.cos(candidates[group]-e))]])
  else:direction=float(candidates[np.argmax(np.interp(candidates,ANGLES,ranges))])
 lookahead=min(1.2,max(.2,d));steer=float(np.clip(math.atan(.6*2*math.sin(direction)/(lookahead*.7112683393)),-maxsteer,maxsteer))
 desired=min(.14,speed_gain*maxwheel)*min(1.,d/.5)*max(.25,math.cos(direction))*min(1.,max(.10,clearance/.9));vmax=speed_gain*maxwheel
 u=np.asarray([np.clip(2*math.sqrt(max(0.,desired)/vmax)-1,-1,1),steer/maxsteer],np.float32);assert np.isfinite(u).all() and np.abs(u).max()<=1
 return u
class Session(F.FieldSession):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.teacher_actions=[];self.teacher_scans=[];self.teacher_parameters=np.asarray([self.env.speed_gain,self.env.max_wheel,self.env.wheelbase,self.env.yaw_gain,self.env.max_steer],np.float64)
 def step(self,forced=None,owner=0,actor_active=True):
  assert forced is None and owner==0;before=E.N.K.R.integration(self.env).copy();scan=self.env._lidar48().copy();u=control(self.q.p['teacher_kind'],self.env._pose()[:3],self.goal,scan,self.teacher_parameters);assert np.array_equal(before,E.N.K.R.integration(self.env));self.teacher_actions.append(u.copy());self.teacher_scans.append(scan);return super().step(u,owner=1,actor_active=True)
def payload(s,tape=None):
 a=NORMAL(s,tape);n=len(a['states']);a['teacher_actions']=np.asarray(s.teacher_actions[:n],np.float32);a['teacher_scans']=np.asarray(s.teacher_scans[:n],np.float32);a['teacher_vehicle_parameters']=s.teacher_parameters.copy();return a
def audit(p,r,a):
 assert a['ownership'].all() and not a['stochastic'].any();assert len(a['teacher_scans'])==len(a['states'])
 for pose,scan,u in zip(a['poses'],a['teacher_scans'],a['teacher_actions']):assert np.array_equal(control(p['teacher_kind'],pose,a['task_goal_xy'],scan,a['teacher_vehicle_parameters']),u)
 assert np.array_equal(a['actions'],a['teacher_actions']);F.M.audit_full_history(p,r,a)
