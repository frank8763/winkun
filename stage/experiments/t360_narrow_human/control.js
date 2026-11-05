(function(global){
  'use strict';
  // One in-flight request. Stop invalidates the loop immediately; finish waits
  // for that request instead of silently disappearing while a step is busy.
  class HumanControl {
    constructor({request,readAction,display,error,pace=()=>250,delay=ms=>new Promise(r=>setTimeout(r,ms))}) {
      Object.assign(this,{request,readAction,display,error,pace,delay});
      this.state=null;this.pending=null;this.running=false;this.epoch=0;this.uncertain=false;this.counter=0;
    }
    stop(){this.running=false;this.epoch++;this.display(this.state);}
    async send(path,data={}){
      if(this.pending)return false;
      this.pending=(async()=>{
        try{const s=await this.request(path,data);this.state=s;this.display(s);if(!s.active)this.stop();return true;}
        catch(e){this.uncertain=true;this.stop();this.error(e.message);return false;}
      })();
      try{return await this.pending;}finally{this.pending=null;this.display(this.state);}
    }
    async step(){
      if(this.pending||this.uncertain||!this.state?.active)return false;
      return this.send('/step',{action:this.readAction(),expected_decision:this.state.decision_index,
        request_id:'human-'+Date.now()+'-'+(++this.counter)});
    }
    async run(){
      if(this.running||this.uncertain||!this.state?.active)return;
      this.running=true;const epoch=++this.epoch;this.display(this.state);
      if(this.pending)await this.pending;
      while(this.running&&epoch===this.epoch&&this.state?.active&&!this.uncertain){
        if(!await this.step())break;
        if(!this.running||epoch!==this.epoch)break;
        await this.delay(Number(this.pace()));
      }
    }
    async end(path){this.stop();if(this.pending)await this.pending;if(this.uncertain)return false;return this.send(path);}
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={HumanControl};
  global.HumanControl=HumanControl;
  if(typeof document==='undefined')return;
  const $=id=>document.getElementById(id);let control,frameKey='';
  const clamp=x=>Math.max(-.999,Math.min(.999,x));
  function input(id,value){$(id).value=clamp(value);$(id==='drive'?'dv':'sv').textContent=Number($(id).value).toFixed(3);}
  function display(s){
    if(!s)return;const busy=!!control?.pending,active=!!s.active;
    const names={WAITING_FOR_HUMAN:'待开始',HUMAN_RECORDING:'窄场景示范采集',POLICY_PREFIX:'策略运行至接管点',EXACT_NATIVE_ACTION_REPLAY:'保存与核验中',ATTEMPT_SAVED:'本次已保存'};
    $('badge').textContent=control?.uncertain?'通信状态待核对':busy?'处理中':control?.running?'连续推进':active?'已暂停':names[s.state]||s.state;
    $('time').textContent=Math.max(0,80-s.sim_time_s).toFixed(2)+' s';$('progress').value=s.sim_time_s;
    $('distance').textContent=s.remaining_m==null?'— m':s.remaining_m.toFixed(3)+' m';
    $('speed').textContent=s.speed_mps==null?'— m/s':s.speed_mps.toFixed(3)+' m/s';
    $('angle').textContent=s.front_steer_rad==null?'— °':(s.front_steer_rad*180/Math.PI).toFixed(1)+' °';
    $('detail').textContent=`尝试 ${s.attempt} / ${s.max_attempts} · 接管后到达 ${s.human_successes} 次\n${names[s.state]||s.state} · 仿真已用 ${s.sim_time_s.toFixed(3)} / 80 秒\n本次从原起点由你全程控制；没有策略前缀。\n采集窗口截止 ${new Date(s.deadline_epoch*1000).toLocaleTimeString()}；等待仍计墙钟时间。`;
    $('start').disabled=busy||active||s.closed||s.attempt>=s.max_attempts;
    for(const id of ['once','run'])$(id).disabled=busy||!active||control?.uncertain;
    $('abort').disabled=!active||control?.uncertain;$('finish').disabled=!!s.closed||control?.uncertain;
    const key=s.attempt+':'+s.physics_steps+':'+s.state;
    if(key!==frameKey){frameKey=key;$('view').src='/frame.jpg?t='+Date.now();}
  }
  async function request(path,payload){
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Stage-Control':'human-ui'},body:JSON.stringify(payload)});
    const data=await response.json();if(!response.ok)throw Error(data.error||'请求失败，已暂停；不要重复发送动作，请回聊天检查。');return data;
  }
  control=new HumanControl({request,readAction:()=>[Number($('drive').value),Number($('steer').value)],display,
    error:message=>{$('error').textContent=message;},pace:()=>Number($('pace').value)});
  $('start').onclick=()=>control.send('/start');$('once').onclick=()=>control.step();$('pause').onclick=()=>control.stop();
  $('abort').onclick=()=>control.end('/abort');$('finish').onclick=()=>control.end('/finish');
  $('run').onpointerdown=e=>{e.preventDefault();control.run();};
  window.addEventListener('pointerup',()=>control.stop());window.addEventListener('pointercancel',()=>control.stop());
  window.addEventListener('blur',()=>control.stop());document.addEventListener('visibilitychange',()=>{if(document.hidden)control.stop();});
  for(const id of ['drive','steer'])$(id).oninput=()=>input(id,Number($(id).value));
  $('left').onclick=()=>input('steer',Number($('steer').value)+.05);$('right').onclick=()=>input('steer',Number($('steer').value)-.05);
  $('center').onclick=()=>input('steer',0);$('maxdrive').onclick=()=>input('drive',.999);
  for(const b of document.querySelectorAll('[data-drive]'))b.onclick=()=>input('drive',Number($('drive').value)+Number(b.dataset.drive));
  window.addEventListener('keydown',e=>{
    if(e.code==='Space'){e.preventDefault();if(!e.repeat)control.run();return;}
    const map={KeyA:['steer',.05],KeyD:['steer',-.05],KeyW:['drive',.05],KeyX:['drive',-.05]};
    if(e.code==='KeyS'){e.preventDefault();input('steer',0);}else if(map[e.code]){e.preventDefault();const[id,delta]=map[e.code];input(id,Number($(id).value)+delta);}
  });
  window.addEventListener('keyup',e=>{if(e.code==='Space')control.stop();});
  async function refresh(){if(control.pending)return;try{const r=await fetch('/status');if(!r.ok)throw Error('状态连接失败');control.state=await r.json();display(control.state);}catch(e){control.stop();$('error').textContent=e.message;}}
  refresh();setInterval(refresh,1000);
})(typeof window==='undefined'?globalThis:window);
