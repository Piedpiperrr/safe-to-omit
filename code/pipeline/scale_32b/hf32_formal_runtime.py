"""Formal E2b Appendix J log-space score. The bab40ab smoke runtime is unchanged."""
import hashlib
import math
from hf32_runtime import Runtime32 as SmokeRuntime32, torch


def log_space_score(label_logits):
    assert label_logits.ndim==1 and label_logits.numel()==4, 'OBQA normalizes A-D only'
    logp=torch.log_softmax(label_logits.float(),dim=0)
    winner=int(torch.argmax(logp).item())
    rest=torch.cat([logp[:winner],logp[winner+1:]])
    u=torch.logsumexp(rest,dim=0)
    diagnostic=1-logp.exp().max()
    return u,diagnostic,logp


class Runtime32(SmokeRuntime32):
    @torch.inference_mode()
    def probe(self,row):
        model=self.models['receiver']
        inputs=self.tokens_for('receiver',[{'role':'user','content':self.prompt.receiver_prompt(row)}])
        prefix=torch.tensor([self.cfg['prefix_ids']],dtype=torch.long,device=inputs['input_ids'].device)
        ids=torch.cat([inputs['input_ids'],prefix],dim=1)
        hidden=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True,
                           output_hidden_states=False,output_attentions=False)
        assert hidden.past_key_values is None
        last=hidden.last_hidden_state[:,-1:,:].to(model.lm_head.weight.device)
        logits=model.lm_head(last)[0,0,:].float()
        labels=row['choice_labels']
        assert labels==['A','B','C','D'], 'OBQA requires exactly the four displayed labels'
        label_logits=torch.stack([torch.logsumexp(logits[self.label_ix[l]],dim=0) for l in labels])
        score,diagnostic,logp=log_space_score(label_logits)
        u=float(score.item());d=float(diagnostic.item())
        return {'score':u if math.isfinite(u) else None,'score_finite':math.isfinite(u),
                'score_name':'logsumexp_nonmax_label_logprob','ProbeMax_diagnostic':d if math.isfinite(d) else None,
                'input_tokens':int(ids.shape[1]),'probe_ids_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                'cache_reused':False}

    def replay(self,row,deployment):
        def policy():
            mode=deployment['mode']
            # q=1 still probes: non-finite scores must take the Text reference path.
            probe=self.probe(row) if mode!='fixed_Text' else None
            skip=bool(probe and probe['score_finite'] and probe['score']<=float(deployment['threshold']))
            helper=None if skip else self.helper(row)
            return {'skip':skip,'P':probe,'helper':helper,'answer':self.answer(row,helper)}
        def reference():
            helper=self.helper(row)
            return {'skip':False,'P':None,'helper':helper,'answer':self.answer(row,helper)}
        def receiver_alone():
            return {'skip':None,'P':None,'helper':None,'answer':self.answer(row,None)}
        return {'policy':policy,'reference':reference,'receiver_alone':receiver_alone}
