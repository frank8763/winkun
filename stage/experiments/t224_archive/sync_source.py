"""Content-exact GitHub API source snapshot while preserving local history."""
from pathlib import Path
import sys,json,subprocess,datetime,hashlib
from archive import api,ROOT,DOC,OUT,write

def main():
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),'Commit tracked changes first'
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    remote=api('GET','git/ref/heads/main')['object']['sha'];meta=api('GET','git/commits/'+remote)
    if meta['tree']['sha']==target:print(json.dumps(dict(state='ALREADY_CONTENT_IDENTICAL',remote_commit=remote)));return
    # All local commits must build on current remote; don't overwrite divergent edits.
    subprocess.run(['git','cat-file','-e',remote+'^{commit}'],cwd=ROOT,check=True)
    subprocess.run(['git','merge-base','--is-ancestor',remote,local],cwd=ROOT,check=True)
    names=subprocess.check_output(['git','diff','--name-only',remote,local],cwd=ROOT,text=True).splitlines();tree=[]
    for p in names:
        assert not p.startswith('stage/runs/') and Path(p).suffix not in ('.pt','.npz','.gz','.mp4','.zip'),p
        b=subprocess.check_output(['git','show',local+':'+p],cwd=ROOT);assert len(b)<2*1024**2
        mode=subprocess.check_output(['git','ls-tree',local,'--',p],cwd=ROOT,text=True).split()[0]
        tree.append(dict(path=p,mode=mode,type='blob',content=b.decode()))
    obj=api('POST','git/trees',dict(base_tree=meta['tree']['sha'],tree=tree));assert obj['sha']==target
    commit=api('POST','git/commits',dict(message='Synchronize verified Stage source and bounded experiment outcomes\n\nContent-exact local '+local+'. Raw models/trajectories live in Release, not Git. Preserve all failure evidence; no Phase research or runtime planner.',tree=target,parents=[remote]))
    now=api('GET','git/ref/heads/main')['object']['sha'];assert now==remote,'Remote moved; inspect before updating'
    api('PATCH','git/refs/heads/main',dict(sha=commit['sha'],force=False))
    ref=api('GET','git/ref/heads/main');final=api('GET','git/commits/'+commit['sha']);assert ref['object']['sha']==final['sha'] and final['tree']['sha']==target
    # Rebuild only exact remote commit bytes to reconcile local histories.
    tzs=['+0800','+0000'];matched=False
    for tz in tzs:
        def person(p):
            stamp=int(datetime.datetime.fromisoformat(p['date'].replace('Z','+00:00')).timestamp());return f"{p['name']} <{p['email']}> {stamp} {tz}"
        raw='tree '+target+'\n'+''.join('parent '+p['sha']+'\n' for p in final['parents'])+'author '+person(final['author'])+'\ncommitter '+person(final['committer'])+'\n\n'+final['message']
        for suffix in ('','\n'):
            b=(raw+suffix).encode();sha=hashlib.sha1(b'commit '+str(len(b)).encode()+b'\0'+b).hexdigest()
            if sha==final['sha']:
                subprocess.run(['git','hash-object','-w','-t','commit','--stdin'],cwd=ROOT,input=b,check=True,stdout=subprocess.DEVNULL)
                subprocess.run(['git','update-ref','refs/remotes/origin/main',sha],cwd=ROOT,check=True)
                subprocess.run(['git','merge','--no-edit','-m','Preserve local Stage history alongside verified API source snapshot',sha],cwd=ROOT,check=True);matched=True;break
        if matched:break
    assert matched,'Remote source verified but local commit encoding unrecovered; inspect before any further sync'
    write(OUT/'latest_source_sync_receipt.json',dict(state='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED',local_source_commit=local,remote_commit=final['sha'],root_tree=target,changed_files=len(names),url=final['html_url'],raw_binary_in_git=False))
    print(json.dumps(dict(state='SOURCE_SYNC_VERIFIED',url=final['html_url'],files=len(names))))
if __name__=='__main__':main()
