from run import *


def main_audit():
    dest=ROOT/'stage/docs/presentation_20261102/VIDEO_AUDIT.json';assert not dest.exists()
    p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE' and len(s['recordings'])==2
    for pid in [json.loads((OUT/'launch.json').read_text())['pid'],json.loads((OUT/'workers.json').read_text())['parent'],*json.loads((OUT/'workers.json').read_text())['encoders']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Still live or reused PID',pid))
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert s['commands']==2 and s['physics_steps']==sum(r['native']['physics_steps'] for r in s['recordings'])<=p['max_physics']
    for r in s['recordings']:
        row=r['native'];ref=p['references'][r['side']];assert row['array_hashes']==ref['array_hashes'] and row['integration_sha256']==ref['integration_sha256']
        blob=row['payload'];assert C.sha(blob['path'])==blob['sha256']
        with np.load(blob['path']) as z:assert len(z.files)==10 and all(S.digest(z[k])==row['array_hashes'][k] for k in z.files) and int(z['native_counts'].astype(np.int64).sum())==row['physics_steps']
        movie=r['movie'];assert C.sha(movie['path'])==movie['sha256'] and verify_movie(movie['path'],movie['decoded_frames'])=={k:movie[k] for k in ['width','height','fps','decoded_frames','duration_s','codec','all_frames_decoded']}
        for poster in movie['posters'].values():assert C.sha(poster['path'])==poster['sha256']
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(dest,dict(state='PASS',physics_steps_added=0,optimizer_calls_added=0,all_native_and_encoder_processes_absent=True,all10_arrays_and_integration_reference_exact=True,
        two1080p25fps_H264_movies_all_frames_decoded=True,all_input_model_movie_payload_poster_SHA_verified=True,physics_steps=s['physics_steps'],commands=2,summary_sha256=C.sha(OUT/'summary.json')))
    print('T288 independent video/raw/hash audit PASS')


if __name__=='__main__':main_audit()
