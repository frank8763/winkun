"""Real new HTTP handler and inert fake app: zero native/human actions."""
import json,threading,types,urllib.request,urllib.error
import recorder as M
def main():
    dest=M.OUT/'http_contract.json';assert not dest.exists()
    app=types.SimpleNamespace(p={'port':0,'max_attempts':3},active=True,closed=False,attempt_number=1,
        session=types.SimpleNamespace(rows=[]),accepted_requests=set(),failure=None,jpg=b'dummy',status=lambda:{'dummy':True})
    def step(data):app.accepted_requests.add(data['request_id']);app.session.rows.append(data)
    app.step_request=step
    server=M.R.HTTPServer(('127.0.0.1',0),M.Handler);app.p['port']=server.server_address[1];server.app=app
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));base='http://127.0.0.1:'+str(app.p['port'])
    def request(path,data=None,origin=None):
        headers={'X-Stage-Control':'human-ui','Content-Type':'application/json','Origin':origin or base}
        req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers=headers)
        try:
            with opener.open(req,timeout=5) as r:return r.status,r.read()
        except urllib.error.HTTPError as e:return e.code,e.read()
    try:
        assert request('/status')[0]==200 and request('/control.js')[0]==200
        action={'action':[.8,.2],'expected_decision':0,'request_id':'test-1'}
        assert request('/step',action,'http://invalid.local')[0]==400 and len(app.session.rows)==0
        assert request('/step',dict(action,action=[1.,0.]))[0]==400 and len(app.session.rows)==0
        assert request('/step',action)[0]==200 and len(app.session.rows)==1
        assert request('/step',action)[0]==409 and len(app.session.rows)==1
        assert request('/step',dict(action,request_id='test-2'))[0]==409 and len(app.session.rows)==1
        M.K.C.save(dest,dict(state='PASS_DUMMY_HTTP_ONLY',real_native_commands=0,physics_steps=0,human_actions=0,
            original_Origin_domain_checks=True,unique_request_id_and_expected_decision_enforced=True,duplicate_and_stale_requests_do_not_execute=True))
    finally:server.shutdown();thread.join(5);server.server_close()
    print('T364 dummy HTTP PASS')
if __name__=='__main__':main()
