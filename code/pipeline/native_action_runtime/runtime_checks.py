"""stage AC semantics, dimension/index parameters mechanically frozen for stage.

Production hooks only capture/inject; diagnostic reductions run after the outer
request timer. W placement and .to(device,dtype) ordering match stage/stage.
"""
import json, os, time, random
from pathlib import Path
import numpy as np
import torch
from protocol_min import receiver_prompt_tensors, generate_receiver
from runtime_common import save, readrows, sha, utc, ROOT

class StopHelper(Exception): pass

@torch.inference_mode()
def vector_at(model,tensors,k):
    captured=[]
    def capture(module,args,output):
        captured.append(output[0][:,-1,:].detach().clone())
        raise StopHelper()
    hook=model.model.layers[k].register_forward_hook(capture)
    try:
        try: model(**tensors,use_cache=False,return_dict=True,logits_to_keep=1)
        except StopHelper: pass
    finally: hook.remove()
    assert len(captured)==1
    return captured[0]

@torch.inference_mode()
def ac_request(runner,row,W,layers):
    _,_,hi=receiver_prompt_tensors(runner.helper_tok,row,runner.helper.device)
    _,_,ri=receiver_prompt_tensors(runner.receiver_tok,row,runner.receiver.device)
    a=vector_at(runner.helper,hi,layers['helper']['block_index'])
    # Keep the original combined device/dtype conversion operation intact.
    x=a.to(device=W.weight.device,dtype=torch.float32)
    mapped=W(x)
    v=mapped.to(device=runner.receiver.device,dtype=torch.bfloat16)
    length=ri['input_ids'].shape[1];state={'injections':0,'calls':0,'block_index':layers['receiver']['block_index'],'prompt_position':length-1}
    def inject(module,args,kwargs,output):
        h=output[0];call=state['calls'];state['calls']+=1
        if call: return output
        assert h.shape==(1,length,runner.receiver.config.hidden_size)
        modified=h.clone();modified[:,-1,:]=v;state['injections']+=1
        return (modified,)+output[1:]
    hook=runner.receiver.model.layers[layers['receiver']['block_index']].register_forward_hook(inject,with_kwargs=True)
    try: text,ids=generate_receiver(runner.receiver,runner.receiver_tok,ri)
    finally: hook.remove()
    return {'raw_answer':text,'generated_token_ids':ids.cpu().tolist(),'receiver_input_tokens':length,'helper_input_tokens':hi['input_ids'].shape[1],'audit':state,'transfer':{'operation':'a.to(device=cuda:1,dtype=float32)','path':'cuda:0 -> cuda:1; W FP32 cuda:1; mapped cast BF16 on cuda:1','source_bytes':a.numel()*a.element_size(),'destination_bytes':x.numel()*x.element_size(),'logical_hidden_payload_bytes':a.numel()*a.element_size()},'_ac_diagnostics':(a,mapped,v)}

def validate_result(result,action):
    t=time.perf_counter()
    p=result.pop('_projected',None)
    if p is not None: assert p.dtype==torch.float32 and torch.isfinite(p).all().item()
    d=result.pop('_ac_diagnostics',None)
    if d is not None:
        assert all(torch.isfinite(x).all().item() for x in d)
        assert result['audit']['injections']==1 and result['audit']['calls']==len(result['generated_token_ids'])
        assert d[1].dtype==torch.float32 and d[2].dtype==torch.bfloat16
    return (time.perf_counter()-t)*1000

def atomic_torch(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp);tmp.replace(path)

def large_w(runner,cfg,deadline):
    out=ROOT/'results/large/W';out.mkdir(parents=True,exist_ok=True)
    final=out/'W_final.pt';checkpoint=out/'training_state.pt'
    ds=cfg['large_W']['shape'][1];dr=cfg['large_W']['shape'][0]
    W=torch.nn.Linear(ds,dr,bias=False,device='cuda:1',dtype=torch.float32)
    if final.exists():
        s=torch.load(final,map_location='cpu',weights_only=False);assert s['steps']==960
        W.load_state_dict(s['state_dict'],strict=True)
        if not (out/'final_receipt.json').exists():save(out/'final_receipt.json',{'utc':utc(),'steps':960,'sha256':sha(final),'config_sha256':sha(ROOT/'frozen_config.json'),'metrics':json.loads((out/'training_metrics.json').read_text()),'recovered_receipt':True})
        return W.eval().requires_grad_(False)
    values={};extraction_start=time.perf_counter()
    for split in ['train','heldout']:
        rr=readrows(cfg['large_W']['c4_data'][split]['path']);vectors={'a':[],'b':[]}
        for start in range(0,len(rr),32):
            f=out/f'c4_{split}_{start:04d}.pt'
            if not f.exists():
                if time.monotonic()>deadline-180: return None
                v={'a':[],'b':[]}
                for r in rr[start:start+32]:
                    for role,key in [('helper','a'),('receiver','b')]:
                        model=getattr(runner,role);tok=getattr(runner,role+'_tok')
                        ids=tok(r['text'],add_special_tokens=True,truncation=False)['input_ids'];assert 1<=len(ids)<=256
                        x=torch.tensor([ids],device=model.device,dtype=torch.long)
                        h=vector_at(model,{'input_ids':x,'attention_mask':torch.ones_like(x)},cfg['ac_layers'][role]['large']['block_index'])
                        assert torch.isfinite(h).all().item()
                        v[key].append(h[0].cpu())
                atomic_torch(f,{k:torch.stack(vv) for k,vv in v.items()})
            v=torch.load(f,map_location='cpu',weights_only=True)
            for k in vectors:vectors[k].append(v[k])
            if (start+32)%256==0:print(json.dumps({'phase':'C4','split':split,'n':start+32,'utc':utc()}),flush=True)
        values[split]={k:torch.cat(v).to('cuda:1',dtype=torch.float32).clone() for k,v in vectors.items()}
    train=values['train'];hold=values['heldout']
    optimizer=torch.optim.Adam(W.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    generator=torch.Generator(device='cpu').manual_seed(0)
    def mse(split):
        with torch.no_grad():return torch.nn.functional.mse_loss(W(values[split]['a']),values[split]['b']).item()
    def state_save(step,epoch,cursor,order,metrics):
        atomic_torch(checkpoint,{'state_dict':W.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'generator_rng':generator.get_state(),'steps':step,'epoch':epoch,'cursor':cursor,'order':order,'metrics':metrics,'config_sha256':sha(ROOT/'frozen_config.json')})
    if checkpoint.exists():
        s=torch.load(checkpoint,map_location='cpu',weights_only=False);assert s['config_sha256']==sha(ROOT/'frozen_config.json')
        W.load_state_dict(s['state_dict'],strict=True);optimizer.load_state_dict(s['optimizer'])
        torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng']);generator.set_state(s['generator_rng'])
        step,epoch,cursor,order,metrics=(s[k] for k in ['steps','epoch','cursor','order','metrics'])
    else:
        torch.manual_seed(0);torch.nn.init.xavier_uniform_(W.weight,gain=1.)
        with torch.no_grad():mean_hold=((hold['b']-train['b'].mean(0))**2).mean().item()
        metrics={'initial_train_mse':mse('train'),'initial_heldout_mse':mse('heldout'),'train_receiver_mean_heldout_mse':mean_hold,'epochs':[],'steps':0,'parameters':ds*dr,'extraction_this_job_seconds':time.perf_counter()-extraction_start,'seed':0,'activation_dtype':'BF16 cast FP32','W_dtype':'FP32'}
        step=0;epoch=1;cursor=0;order=torch.randperm(3072,generator=generator)
        state_save(step,epoch,cursor,order,metrics)
    W.train();started=time.perf_counter()
    while step<960:
        if time.monotonic()>deadline-120:return None
        ids=order[cursor:cursor+32].to('cuda:1');optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.mse_loss(W(train['a'][ids]),train['b'][ids]);assert torch.isfinite(loss).item()
        loss.backward();optimizer.step();step+=1;cursor+=32
        if cursor==3072:
            metrics['epochs'].append({'epoch':epoch,'train_mse':mse('train'),'heldout_mse':mse('heldout')})
            print(json.dumps({'phase':'W_TRAIN','step':step,**metrics['epochs'][-1]}),flush=True)
            epoch+=1;cursor=0
            if step<960:order=torch.randperm(3072,generator=generator)
        metrics['steps']=step
        # Atomic state after every step: recovery never initializes a second trajectory.
        state_save(step,epoch,cursor,order,metrics)
    assert step==960 and torch.isfinite(W.weight).all().item()
    metrics.update(final_train_mse=mse('train'),final_heldout_mse=mse('heldout'),training_this_job_seconds=time.perf_counter()-started)
    save(out/'training_metrics.json',metrics)
    atomic_torch(final,{'state_dict':{'weight':W.weight.detach().cpu()},'steps':960,'epoch':10,'shape':[dr,ds],'bias':False,'config_sha256':sha(ROOT/'frozen_config.json')})
    save(out/'final_receipt.json',{'utc':utc(),'steps':960,'sha256':sha(final),'config_sha256':sha(ROOT/'frozen_config.json'),'metrics':metrics,'job_id':os.environ['PBS_JOBID']})
    return W.eval().requires_grad_(False)
