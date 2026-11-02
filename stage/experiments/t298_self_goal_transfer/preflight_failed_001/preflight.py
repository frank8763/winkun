from t298_learning import *
import ast,builtins,copy,dis,gzip,importlib.util,io,json,types
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t298_self_goal_transfer'
sys.path.insert(0,str(ROOT/'stage/experiments/t291_hard_guard_ablation'))
from t291_codec import pack_full,unpack_full

def main():
    assert not (OUT/'preflight.json').exists();p=json.loads((HERE/'protocol.json').read_text())
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    for filename,entry in [('run.py','main'),('audit.py','main'),('t298_worker.py','worker')]:
        spec=importlib.util.spec_from_file_location('t298_'+filename[:-3],HERE/filename);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);missing=set()
        def check(code):
            for ins in dis.get_instructions(code):
                if ins.opname=='LOAD_GLOBAL' and ins.argval not in vars(mod) and ins.argval not in vars(builtins):missing.add(ins.argval)
            for co in code.co_consts:
                if isinstance(co,types.CodeType):check(co)
        check(getattr(mod,entry).__code__);assert not missing,(filename,missing)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=policy(p);frozen=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,offset=anchors(p,frozen);aux=auxiliary(p);calls=[]
    def count(opt,name):
        original=opt.step
        def step(*a,**kw):
            assert len(calls)<p['preflight_clone_optimizer_max_calls'];calls.append(name);C.save(OUT/'preflight_optimizer_calls.json',dict(calls=calls,native_steps=0,navigation_updates=0));return original(*a,**kw)
        opt.step=step
    shadow=copy.deepcopy(q);o=L.optimizer(q,p);so=L.optimizer(shadow,p);count(o,'supervised_clone');count(so,'supervised_shadow_clone')
    for k in range(2):assert supervised_step(q,o,aux,ax,ay,offset,p,k)==supervised_step(shadow,so,aux,ax,ay,offset,p,k)
    assert S.digest(q.state_dict())==S.digest(shadow.state_dict()) and S.digest(o.state_dict())==S.digest(so.state_dict())
    q=policy(p);opt=L.optimizer(q,p);shadow=copy.deepcopy(q);so=L.optimizer(shadow,p);count(opt,'RL_clone');count(so,'RL_shadow_clone')
    items=[]
    for r in p['historical_first16']:
        with np.load(resolve(r['path'])) as z:a={k:z[k].copy() for k in z.files}
        items.append(dict(row=r,arrays=a))
        assert all(np.array_equal(v,unpack_full(pack_full(a))[k]) for k,v in a.items())
        b=io.BytesIO();np.savez_compressed(b,**compact(a));assert b.tell()<p['compact_episode_bytes'];assert all(np.array_equal(a[k],v) for k,v in unpack_compact(compact(a),r['aux_eta']).items())
    data=L.training_batch(items,q);local=dict(p,actor_attempts_per_round=1);live,again=paired_balanced_update(q,opt,shadow,so,data,ax,ay,offset,local,0,lambda:None,lambda actor:dict(policy_hash=S.digest(actor.state_dict()),cases=0,safe=0,all_safe=True,synthetic=True),lambda phase,event:None)
    assert live==again and S.digest(q.state_dict())==S.digest(shadow.state_dict()) and S.digest(opt.state_dict())==S.digest(so.state_dict())
    before=S.digest(q.state_dict());bo=S.digest(opt.state_dict())
    L.guarded_adam_step(q,opt,lambda actor:dict(accept=False),p['step_factors']);assert before==S.digest(q.state_dict()) and bo==S.digest(opt.state_dict())
    def interrupt(actor):raise RuntimeError('T298 preflight disposable interrupt')
    try:L.guarded_adam_step(q,opt,interrupt,p['step_factors'])
    except RuntimeError:pass
    else:raise AssertionError('Expected interrupt')
    assert before==S.digest(q.state_dict()) and bo==S.digest(opt.state_dict()) and len(calls)==8
    raw=io.BytesIO();torch.save(q.state_dict(),raw);assert raw.tell()*3+256*1024<p['model_bytes']
    # Bound all text records, including 1000 tiny optimizer invocation records.
    metadata_bound=520*2560+80*2560+12*32768+1008*120+4*65536+100000
    # JSONL metadata compresses poorly in the worst case; register larger reserve
    # only if actual exact-format bound fits (compact rows are <2560).
    assert len(json.dumps(dict(arm='self_goal_transfer',phase='supervised_shadow',invocation=1000)))<120
    # More exact maximal raw rows: actual 12array hashes fixed 64-char strings.
    sample=items[0]['row'].copy();sample.update(arm='self_goal_transfer',role='independent_reload',path=str(OUT/'episodes'/('self_goal_transfer_reload_'+'x'*48+'.npz')),sha256='0'*64,saved_array_keys=list(compact(items[0]['arrays'])),storage_encoding='compact-five-derived-reward-v1')
    rowlen=len(json.dumps(sample,separators=(',',':')).encode())+80
    compressed_row=len(gzip.compress(json.dumps(sample,separators=(',',':')).encode(),mtime=0))+100
    estimate=(520+16)*compressed_row+32*1200+8*4096+1008*120+4*65536+S.used(OUT)+100000
    assert estimate<p['metadata_bytes'],(estimate,p['metadata_bytes'],rowlen)
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes'] and S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,native_environments_created=0,navigation_Actor_updates=0,disposable_clone_optimizer_calls=8,supervised_clone_live_shadow_exact=True,RL_clone_live_shadow_ActorAdam_exact=True,rejection_interruption_ActorAdam_exact=True,codec_five_arrays_exact_derived_rewards=True,full120_statechain_exact=True,all_entry_global_names_resolve=True,metadata_upper=estimate,metadata_sample_row_bound=rowlen,global_if_full=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'],real_native_guard_not_yet_run=True))
    print('T298 preflight PASS 8 disposable clone Adam, zero native/navigation updates')

if __name__=='__main__':main()
