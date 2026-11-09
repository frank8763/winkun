"""Stdlib-only closeout of the last incomplete attempt; never resume science."""
from pathlib import Path
import hashlib,json,time,os,gzip,shutil,sys
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t440_final_functional_constraint';DOC=ROOT/'stage/docs/review_20261109'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def process_check():
    found=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            args=(d/'cmdline').read_bytes().split(b'\0')
            if any(a.endswith(b'/t440_final_functional_constraint/run.py') or a.endswith(b'/t440_final_functional_constraint/driver.py') for a in args):found.append(int(d.name))
        except OSError:pass
    assert not found,found
    return found
def finish():
    assert not (OUT/'summary.json').exists();p=read(OUT/'protocol.json')
    assert time.time()<p['deadline_all'];process_check()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert not any((OUT/name).exists() for name in ['native_started.json','inner_optimizer_calls.jsonl','dataset_proof.json','fit_summary.json','proposed.pt.gz','initial.pt.gz','launch.json','driver_started.json'])
    # solve() is called only after load_data() returns and dataset_proof is
    # saved. Its event writes BEFORE every optimizer step; neither file exists.
    before={str(f.relative_to(ROOT)):sha(f) for f in OUT.rglob('*') if f.is_file()}
    s=dict(state='LAST_ATTEMPT_INCOMPLETE_LINE_STOPPED_PER_USER',pid=os.getpid(),epoch=time.time(),
        reason='External120second preprocessing/fit timeout sentTERM thenKILL; observed exit137. No dataset_proof, optimizer event, model or native execution was reached.',
        actual_execution_command='timeout --signal=TERM --kill-after=2 120 taskset -c0 ./run_stage.sh stage/experiments/t440_final_functional_constraint/run.py fit',
        observed_exit_code=137,offline_feasibility='NOT_COMPLETED',algorithm_effect='NOT_EVALUATED',
        physics=0,native_commands=0,inner_live_Adam=0,inner_shadow_Adam=0,new_online_RL=0,
        partial_Actor_forward=True,completed_cached_PG_evaluation=None,
        wide54=None,old275=None,fresh16=None,reload78=None,six_gate=False,eight_started=False,
        initial_and_all_historical_models_unmodified=True,all_scientific_processes_stopped=True,
        line_stopped_by_explicit_user_condition=True,no_further_experiment_without_new_explicit_user_authorization=True,
        no_claim_method_infeasible_or_all_RL_failed=True,automatic_driver_never_launched=True,
        preflight_physics=0,preflight_optimizer=0)
    save(OUT/'summary.json',s)
    cold=read(ROOT/'stage/docs/overnight_20261108_relay/RESERVE_CANDIDATES.json');restored=[]
    for row in cold['rows']:
        encoded=ROOT/row['encoded_path'];assert sha(encoded)==row['encoded_sha256']
        raw=gzip.decompress(encoded.read_bytes());assert len(raw)==row['original_bytes'] and hashlib.sha256(raw).hexdigest()==row['original_sha256']
        f=ROOT/row['original_path']
        if f.exists():assert sha(f)==row['original_sha256'];restored.append(row['original_path'])
    assert len(restored)==5
    model_bytes={path:h for path,h in p['inputs'].items() if path.endswith(('.pt','.pt.gz','.pth'))}
    assert all(sha(f)==h for f,h in model_bytes.items())
    size=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());free=shutil.disk_usage(ROOT).free
    assert size<=2**30 and free>=40*2**30
    save(OUT/'closeout_audit.json',dict(state='PASS_ZERO_NEW_SCIENCE_CLOSEOUT_ONLY',
        scientific_method_or_candidate_audit='NOT_AVAILABLE_NO_CANDIDATE_PRODUCED',
        all_scientific_processes_absent=process_check(),all_original_registered_input_SHA_exact=True,
        original_preflight_and_error_bytes_unchanged=all(sha(ROOT/f)==h for f,h in before.items()),
        no_native_or_optimizer_stage_reached=True,all_original_model_SHA_verified=True,
        cold_original_bytes_independently_verified=194,restored_bare_JSON=restored,
        logical_bytes=size,free_bytes=free,new_native=0,new_optimizer=0))
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,epoch=time.time(),line_stopped=True))
    save(DOC/'T440_FINAL_REPORT.json',dict(summary=s,closeout='PASS_ZERO_NEW_SCIENCE_ONLY',remote='PENDING'))
    text='''# Stage最后尝试与停止交付\n\n**2026-11-09，最后尝试未完成，按用户条件停止Stage实验线。没有完整六米通过，也没有论文完成或真机成功可交付。**\n\nT440拟做一个实际函数变化受限的策略梯度/恢复辅助提案。真实历史的特征公式预检通过；随后离线前置重算在外部120秒硬上限后被TERM/KILL（exit137）。dataset_proof、任何优化器事件、候选模型均未产生，所以0原生导航、0新在线RL、0live/shadow内层Adam，完整开发/留出/重载全未执行。Actor历史前向发生过，不冒称所有计算为零。没有算法有效/无效的实验结论。\n\n原完整3300秒启动准入因剩余3296.32秒被拒，0进程；随后采用更严外部fit120/native2400/audit300/transport180的阶段预算，科学步骤/阈值不变。该安排未能覆盖前置历史重算，是执行与时间预算安排失败。不能用它掩饰成算法被科学否定，也不借新编号补跑。\n\n预检两次依赖缺失均发生0native/0Adam；原记录保留。已无损编码194个旧JSON（29,365,945B payload释放），初始化递归读取的5个原件已按旧SHA精确恢复共1,079,658B，原gzip也保留。194原件全部可本地解码且旧完整远端原始字节仍在，独立收尾核验通过；所有源模型/历史模型/轨迹/失败不删除。\n\n## 真实最后进度\n\n- 五米安全T255历史旧275/275，视频与模型保留。\n- 宽六米历史候选T425 A54/54但旧264/275；最近T437宽52/54旧271/275、修5丢4、回退。它不是稳定联合门。\n- T439开放表示控制0提案保留、完整329未执行；不能填写全六米新成功率。\n- 未消费16留出、78独立原生模型重载和八米均没有通过证据；全图与实体机尚未完成。\n- 当前没有可投稿完整论文或成立的新算法贡献。PAPER_OUTLINE.md只作提纲，FULL_RETROSPECTIVE_AND_PAPER_GAPS.md列全缺口。\n\n## 停止与恢复\n\n停止包含不再安排新机制、追加轮数/seed、启动通宵或新编号续预算。只做本次备份和交付收尾；日后继续须用户明确新授权。safeT255与历史失败全部保留，Phase完全隔离。代码/报告Git、原始数据Release；完整索引见HANDOFF.md及long_distance_20261101的逐组receipt/manifest。T440本次收据生成前remote明确PENDING，完成后单独补收据，不把收尾audit说成方法验证PASS。\n'''
    (DOC/'T440_FINAL_REPORT.md').write_text(text)
    status=ROOT/'stage/docs/RESEARCH_STATUS.md'
    entry='2026-11-09用户最后尝试T440未完成，现按明确止损条件停止Stage实验线，不再自动新实验/通宵/追加分支；只有必要备份交付。T440离线前置重算被外部120秒TERM/KILL(exit137)，0候选/0native/0内层Adam/0新在线RL，方法有效性未测，所有验收null，不能称算法已否定。完整3300秒准入被拒后更严分阶段执行未覆盖预处理，属于执行预算失败。所有进程退出、原输入与模型SHA/194冷原字节收尾核验PASS（非科学audit），5嵌套原JSON已精确恢复，safeT255/失败全保留，Phase隔离。详review_20261109/T440_FINAL_REPORT.md及HANDOFF；原始备份待收据，未经用户新明确授权不得恢复研究。\n\n'
    status.write_text(entry+status.read_text());print('STAGE_LINE_STOPPED_NO_FURTHER_SCIENCE',flush=True)
def backup():
    process_check();s=read(OUT/'summary.json');assert s['line_stopped_by_explicit_user_condition']
    sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
    import transport as T
    T.install()
    def stopped(folder):
        if folder.name=='t435_relay_reserve':
            assert read(folder/'summary.json')['state']=='COMPLETE' and read(folder/'audit.json')['state']=='PASS'
            assert not Path('/proc',str(read(folder/'summary.json')['pid'])).exists()
        else:
            assert folder==OUT and read(folder/'SEALED.json')['line_stopped'] and read(folder/'closeout_audit.json')['state']=='PASS_ZERO_NEW_SCIENCE_CLOSEOUT_ONLY'
            assert not Path('/proc',str(read(folder/'summary.json')['pid'])).exists();process_check()
    T.B.GROUPS['T440']=['t440_final_functional_constraint','t435_relay_reserve'];T.B.require_stopped=stopped
    import small_parts
    small_parts.main('T440')
if __name__=='__main__':globals()[sys.argv[1]]()
