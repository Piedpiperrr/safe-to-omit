"""Three measured scheduling arms; timestamps come from execution, never component sums."""
import math
import time

ARMS=('synchronous','early_return','reference')


def arm_order(pair_index,dataset_index,batch_size_index,batch_index):
    offset=(pair_index+dataset_index+batch_size_index+batch_index)%3
    return list(ARMS[offset:]+ARMS[:offset])


def execute_batch(rows, mode, threshold, probe, helper, answer, parse, sync, clock=time.perf_counter, on_complete=lambda rows:None):
    assert mode in ARMS
    sync(); start=clock()
    probes=probe(rows) if mode!='reference' else [None]*len(rows)
    skip=[math.isfinite(p['score']) and p['score']<=threshold for p in probes] if mode!='reference' else [False]*len(rows)
    sync(); phase={'probe_end_ms':(clock()-start)*1000}
    done=[None]*len(rows)

    def finish(indices, messages):
        answers=answer([rows[i] for i in indices],messages)
        assert len(answers)==len(indices)
        parsed=[parse(a['text'],rows[i]['choice_labels']) for i,a in zip(indices,answers)]
        sync(); completed=(clock()-start)*1000
        for i,a,p,h in zip(indices,answers,parsed,messages):
            done[i]={'id':rows[i]['id'],'skip':skip[i],'probe':probes[i],'helper':h,
                     'answer':a,'parsed':p,'latency_ms':completed}
        # Deliver ready requests before invoking the waiting branch. The replay's
        # consumer is in-memory; disk/network serialization is outside measurement.
        on_complete([done[i] for i in indices])
        return completed

    omitted=[i for i,s in enumerate(skip) if s]
    waiting=[i for i,s in enumerate(skip) if not s]
    if mode=='synchronous':
        generated=helper([rows[i] for i in waiting]) if waiting else []
        assert len(generated)==len(waiting)
        messages=dict(zip(waiting,generated))
        sync();phase['helper_end_ms']=(clock()-start)*1000
        phase['batch_complete_ms']=finish(list(range(len(rows))),[messages.get(i) for i in range(len(rows))])
    elif omitted:
        phase['skip_complete_ms']=finish(omitted,[None]*len(omitted))
    if waiting and mode!='synchronous':
        messages=helper([rows[i] for i in waiting])
        assert len(messages)==len(waiting)
        sync();phase['helper_end_ms']=(clock()-start)*1000
        phase['wait_complete_ms']=finish(waiting,messages)
    sync(); elapsed=(clock()-start)*1000
    assert all(r is not None and 0<=r['latency_ms']<=elapsed for r in done)
    return done,elapsed,phase
