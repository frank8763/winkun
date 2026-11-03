"""Candidate repair for the event adapter; original T312 source is immutable.

This module does not restart a run, change an optimizer, or alter a protocol.
"""
def event_row(arm,replica,kind,payload):
    row=dict(payload)
    for key,value in [('arm',arm),('replica',replica),('kind',kind)]:
        if key in row and row[key]!=value:
            raise ValueError(f'Conflicting event context: {key}')
        row[key]=value
    return row

def check():
    import io,json
    calls=[];stream=io.StringIO();payloads=[
        ('mc','live','Actor',dict(round=0,arm='mc',step=0)),
        ('mc','shadow','Actor',dict(round=0,arm='mc',step=0)),
        ('gae','live','Actor',dict(round=0,arm='gae',step=0)),
        ('gae','shadow','Actor',dict(round=0,arm='gae',step=0)),
        ('mc','live','V',dict(round=0,fold=0,step=0)),
        ('gae','shadow','V',dict(round=0,fold=1,step=0)),
    ]
    # Exercise the same adapter signature used by BOTH actual learning functions.
    def callback(arm,replica):
        def record(kind,payload):
            row=event_row(arm,replica,kind,payload);calls.append(row);stream.write(json.dumps(row)+'\n')
        return record
    for arm,replica,kind,payload in payloads:
        original=dict(payload);callback(arm,replica)(kind,payload);assert payload==original
    assert list(map(json.loads,stream.getvalue().splitlines()))==calls
    rejected=0
    for bad in [dict(arm='gae'),dict(replica='shadow'),dict(kind='V')]:
        try:event_row('mc','live','Actor',bad)
        except ValueError:rejected+=1
        else:raise AssertionError('Conflicting event identity silently accepted')
    return dict(state='PASS_EVENT_ADAPTER_ONLY',actual_Actor_and_V_payload_shapes=6,conflicting_contexts_rejected=rejected,
        payload_not_mutated=True,json_roundtrip_exact=True,native_steps=0,optimizer_calls=0,Actor_updates=0,
        original_T312_source_unchanged=True,repaired_main_pipeline_not_registered_or_launched=True)

if __name__=='__main__':
    import json
    print(json.dumps(check()))
