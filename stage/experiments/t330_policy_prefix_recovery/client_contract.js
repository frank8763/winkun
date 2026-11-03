'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {HumanControl}=require('./control.js');
(async()=>{
  const calls=[];let resolveStep;
  const control=new HumanControl({request:async(path,data)=>{
    calls.push([path,data]);
    if(path==='/step')return new Promise(resolve=>{resolveStep=resolve;});
    return {active:false,decision_index:1};
  },readAction:()=>[.8,.2],display:()=>{},error:e=>{throw Error(e);},delay:async()=>{}});
  control.state={active:true,decision_index:0};
  const running=control.run();assert.equal(calls.length,1);
  control.stop();resolveStep({active:true,decision_index:1});await running;
  assert.equal(calls.length,1,'release must not schedule another physical step');
  const step=control.step();assert.equal(calls.length,2);
  await control.step();assert.equal(calls.length,2,'double click must not duplicate in-flight step');
  const ending=control.end('/finish');assert.equal(calls.length,2,'finish must wait for active step');
  resolveStep({active:true,decision_index:2});await step;await ending;
  assert.equal(calls.length,3);assert.equal(calls[2][0],'/finish');
  assert.deepEqual(calls[1][1].action,[.8,.2]);assert.equal(calls[1][1].expected_decision,1);
  const proof={state:'PASS',release_stops_next_request:true,double_step_blocked:true,finish_waits_then_saves:true,
    explicit_action_and_expected_decision_forwarded:true,native_commands:0,physics_steps:0,human_actions:0,optimizer_calls:0};
  fs.writeFileSync(process.argv[2],JSON.stringify(proof,null,2)+'\n');console.log(JSON.stringify(proof));
})().catch(e=>{console.error(e);process.exitCode=1;});
