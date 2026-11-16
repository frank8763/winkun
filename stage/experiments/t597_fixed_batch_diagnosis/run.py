"""Read-only T596 diagnosis: fixed histories and six component gradients.

No simulator, optimizer, parameter intervention, new seeds or navigation tests.
"""
from pathlib import Path
import os, sys, time, json, math, hashlib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'stage/experiments/t596_recurrent_joint_online'))
import t596_learning as L

torch, np = L.torch, L.np
OUT = ROOT/'stage/runs/t597_fixed_batch_diagnosis'
DOC = ROOT/'stage/docs/day_20261116/t597'
SOURCE = ROOT/'stage/runs/t596_recurrent_joint_online'


def write(path, data):
    blob = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    assert len(blob.encode()) < 256*1024
    with path.open('x') as f:
        f.write(blob)


def register():
    assert not OUT.exists()
    assert L.read(SOURCE/'audit.json')['state'] == 'PASS'
    sealed = L.read(SOURCE/'SEALED.json')
    assert all(not Path('/proc', str(pid)).exists() for pid in sealed['scientific_pids'])
    total = sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
    assert total+256*1024 < 2**30
    import shutil, datetime
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    end = datetime.datetime.fromisoformat('2026-11-16T20:00:00+08:00').timestamp()
    assert time.time()+900 < end
    DOC.mkdir(parents=True, exist_ok=True)
    p = L.read(SOURCE/'protocol.json'); s = L.read(SOURCE/'summary.json')
    inputs = {str(f): L.sha(f) for f in [SOURCE/'protocol.json', SOURCE/'summary.json', SOURCE/'audit.json', Path(__file__)]}
    for row in p['keep_rows']+p['recovery_rows']+s['result']['training_rows']:
        assert L.sha(row['path']) == row['sha256']
        inputs[row['path']] = row['sha256']
    for row in [p['base_model'], s['result']['proposal']]:
        assert L.sha(row['path']) == row['sha256']; inputs[row['path']] = row['sha256']
    # Lock all imported Stage Python dependencies; no imports initialize physics.
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):
            inputs[str(Path(f).resolve())] = L.sha(f)
    protocol = dict(id='T597_FIXED_BATCH_READ_ONLY_COMPONENT_DIAGNOSIS',
        inputs=inputs, new_native=0, new_optimizer=0, backward_components=6,
        seconds=900, own_bytes=256*1024, cpu=[14], GPU=False, all_stop=end,
        feature_rows='all39 keep histories; initial versus final on identical observations',
        gradients='initial and final x RL/weighted keep/weighted recovery; original fixed8 advantages',
        limitations='Saved-history measurement only, not closed-loop causality or a new trained policy.',
        authorization='User asked to continue thinking through the next step after the explicit offline diagnosis proposal; no new training or navigation budget consumed.')
    OUT.mkdir(); write(OUT/'protocol.json', protocol); write(DOC/'PLAN.json', protocol)
    print('REGISTERED_ZERO_NATIVE_ZERO_OPTIMIZER', flush=True)


def main():
    protocol = L.read(OUT/'protocol.json')
    assert not (OUT/'launch.json').exists()
    for f, h in protocol['inputs'].items(): assert L.sha(f) == h, f
    start = time.time()
    def guard():
        assert time.time()-start < protocol['seconds'] and time.time() < protocol['all_stop']
    write(OUT/'launch.json', dict(pid=os.getpid(), epoch=start, protocol_sha256=L.sha(OUT/'protocol.json')))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    assert set(os.sched_getaffinity(0)) == {14}
    p = L.read(SOURCE/'protocol.json'); s = L.read(SOURCE/'summary.json')['result']
    before = L.fresh(p); after = L.frozen(p, s['proposal']['path'])
    hashes = [L.digest(q.state_dict()) for q in [before, after]]
    data = L.references(p)
    feature_rows = []
    basis = L.Parent.Prior.Old.basis
    with torch.no_grad():
        for g in data[:39]:
            guard(); features=[]; cmds=[]; activations=[]; corrections=[]
            for q in [before, after]:
                n=q.normalized(g['x']); h,_=q.core(q.encoder(n)[:,None]); f=torch.cat([h[:,0],n],-1)
                features.append(f); cmds.append(q.sequence(g['x']))
                specs=[(q.repair_centers,q.repair_radii,q.repair_weights),
                       (q.added_centers,q.added_radii,q.added_weights)]
                specs += [(z.centers,z.radii,z.weights) for z in q.layers]
                flags=[]; contrib=[]
                for centers,radii,weights in specs:
                    phi=torch.cat([basis(f[k:k+16],centers,radii) for k in range(0,len(f),16)])
                    flags.append((phi>0).any(-1)); contrib.append(phi@weights)
                activations.append(torch.stack(flags,-1)); corrections.append(torch.stack(contrib,-1))
            delta=(cmds[1].clamp(-1,1).float().double()-cmds[0].clamp(-1,1).float().double())
            changed=(activations[0]!=activations[1]).any(-1)
            feature_rows.append(dict(id=g['id'],decisions=len(g['x']),
                lost=g['id'] in s['check']['lost_initial'],
                hidden_max=float((features[1][:,:64]-features[0][:,:64]).abs().max()),
                basis_active_decisions_before=int(activations[0].any(-1).sum()),
                basis_active_decisions_after=int(activations[1].any(-1).sum()),
                decisions_with_changed_layer_active_status=int(changed.sum()),
                first_changed_decision=int(torch.nonzero(changed)[0]) if changed.any() else None,
                correction_max_change=float((corrections[1]-corrections[0]).abs().max()),
                command_RMS=float(delta.square().mean().sqrt()),command_max=float(delta.abs().max())))
    arrays=[L.G.E.N.S.arrays(r['path']) for r in s['training_rows']]
    groups=[L.Credit.peer_batch((before,None),arrays[k:k+4],p)[0] for k in [0,4]]
    gradient_reports=[]
    for label,q in [('initial',before),('final',after)]:
        q.requires_grad_(False)
        for v in L.variables(q): v.requires_grad_(True)
        components={}; values={}
        for kind in ['RL','old','recovery']:
            guard();q.zero_grad(set_to_none=True);value=0.
            if kind=='RL':
                for items in groups:
                    for b in items:
                        loc=q.sequence(b['x']);ratio=(L.P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
                        loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/8
                        loss.backward();value+=float(loss.detach())
            else:
                for g in data:
                    if g['kind']!=kind: continue
                    z=q.sequence(g['x'])[g['mask']];target=g['target'][g['mask']]
                    if kind=='old':
                        per=.5*((z-target)/z.new_tensor(p['raw_sigma'])).square().mean(-1); weight=p['keep_weight']/39
                    else:per=L.P.action_loss(z,target).mean(-1);weight=p['recovery_weight']/3
                    loss=(.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean())*weight
                    loss.backward();value+=float(loss.detach())
            vector=torch.cat([(v.grad if v.grad is not None else torch.zeros_like(v)).flatten() for v in L.variables(q)])
            assert torch.isfinite(vector).all();components[kind]=vector;values[kind]=value
        cos={}
        for a,b in [('RL','old'),('RL','recovery'),('old','recovery')]:
            denom=float(components[a].norm()*components[b].norm())
            cos[a+'__'+b]=float(torch.dot(components[a],components[b]))/denom if denom else None
        total=sum(components.values())
        gradient_reports.append(dict(checkpoint=label,losses=values,
            norms={k:float(v.norm()) for k,v in components.items()},cosines=cos,
            sum_gradient_norm=float(total.norm()),before_global_norm_clip=1.,
            component_dot_total={k:float(torch.dot(v,total)) for k,v in components.items()}))
        q.zero_grad(set_to_none=True)
    assert hashes==[L.digest(q.state_dict()) for q in [before,after]]
    assert not torch.cuda.is_initialized()
    write(OUT/'result.json',dict(state='COMPLETE_READ_ONLY_MEASUREMENT',new_native=0,new_optimizer=0,
        backward_components=6,feature_rows=feature_rows,gradient_reports=gradient_reports,
        model_states_unchanged=True,elapsed_s=time.time()-start,
        evidence_scope='Same-input saved histories and fixed8 training batch only; not causal navigation interventions.'))
    print('T597_COMPLETE',time.time()-start,flush=True)


if __name__=='__main__': globals()[sys.argv[1]]()
