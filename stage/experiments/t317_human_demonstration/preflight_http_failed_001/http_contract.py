"""Actual HTTP handler with a deliberately non-native dummy, zero physics."""
import recorder as M
import threading,urllib.request,urllib.error,json,types

def main():
    target=M.OUT/'preflight_http';target.mkdir(exist_ok=False);old=M.OUT;M.OUT=target
    server=M.HTTPServer(('127.0.0.1',0),M.Handler);port=server.server_address[1]
    def forbidden_native(value):raise RuntimeError('PREFLIGHT_DUMMY_EXECUTION_ERROR_NO_NATIVE')
    fake=types.SimpleNamespace(p={'port':port},active=True,attempt_number=0,closed=False,failure=None,meter=types.SimpleNamespace(physics=0),
        checkpoint=lambda:None,step=forbidden_native,status=lambda:{'preflight_dummy':True,'physics_steps':0})
    server.app=fake;thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();results=[]
    def request(path,payload=None,origin=None):
        headers={'Content-Type':'application/json','X-Stage-Control':'human-ui'}
        if origin:headers['Origin']=origin
        data=None if payload is None else json.dumps(payload).encode();req=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=data,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=5) as r:return r.status,json.loads(r.read())
        except urllib.error.HTTPError as error:return error.code,json.loads(error.read())
    try:
        status,body=request('/status');assert status==200 and body['preflight_dummy'];results.append('GET status200')
        status,_=request('/step',{'action':[0,0]},'http://invalid.local');assert status==400 and fake.failure is None;results.append('wrong Origin400 before execution')
        origin='http://127.0.0.1:'+str(port)
        status,_=request('/step',{'action':[1,0]},origin);assert status==400 and fake.failure is None;results.append('endpoint action400 before execution')
        status,_=request('/step',{'action':[0,0]},origin);assert status==500 and fake.closed and 'PREFLIGHT_DUMMY_EXECUTION_ERROR' in fake.failure;results.append('dummy execution error500 closes collector')
        M.K.C.save(target/'contract.json',dict(state='PASS_DUMMY_HTTP_ONLY',checks=results,physics_steps=0,Adam_calls=0,gradient_calls=0,real_human_actions=0,no_actual_App_or_Session_created=True))
    finally:server.shutdown();thread.join(5);server.server_close();M.OUT=old
    print('T317 actual HTTP handler preflight PASS, dummy only0physics/Adam/human actions')
if __name__=='__main__':main()
