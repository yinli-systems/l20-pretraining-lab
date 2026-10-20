"""Independent native Transformers FP32 CPU loss and language-sample audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import datetime,json,re,time
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from common import ROOT,CACHE,SEED,tokenizer,write_json


def main():
    torch.set_num_threads(8); tok=tokenizer(); rng=np.random.default_rng(SEED+912)
    ids=np.memmap(CACHE/'packed/valid.ids.u16',mode='r',dtype=np.uint16).reshape(-1,2048)
    masks=np.memmap(CACHE/'packed/valid.mask.u8',mode='r',dtype=np.uint8).reshape(-1,2048)
    indices=[0,1]
    x=torch.tensor(np.array(ids[indices,:512],dtype=np.int64)); y=x.clone(); y[~torch.tensor(np.array(masks[indices,:512],dtype=np.bool_))]=-100
    report={'cpu_only':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sample':'first two validation blocks, first 512 tokens each; assistant labels only; diagnostic subset, not full validation loss','native_losses':{}}
    for label,path in [('base',CACHE/'base'),('sft2000',ROOT/'models/best')]:
        model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,attn_implementation='sdpa').eval()
        if model.config.vocab_size<len(tok):
            old=model.config.vocab_size; model.resize_token_embeddings(len(tok),pad_to_multiple_of=64,mean_resizing=False)
            with torch.no_grad():
                model.get_input_embeddings().weight[old:].copy_(model.get_input_embeddings().weight[:old].mean(0))
                model.get_output_embeddings().weight[old:].copy_(model.get_output_embeddings().weight[:old].mean(0))
        with torch.inference_mode():
            out=model(input_ids=x,labels=y,use_cache=False)
            target=y[:,1:]; logits=out.logits[:,:-1]; keep=target!=-100
            manual=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
            report['native_losses'][label]={'hf_loss':float(out.loss),'manual_shifted_ce':float(manual),'supervised_tokens':int(keep.sum()),'teacher_forced_token_accuracy':float((logits.argmax(-1)[keep]==target[keep]).float().mean()),'finite':bool(torch.isfinite(out.loss))}
        del model,out
    language={}
    for source in ['math','code','numina','general','r1-math','r1-code']:
        p=CACHE/'prepared'/source; off=np.load(p/'offsets.npy'); data=np.memmap(p/'ids.u16',mode='r',dtype=np.uint16)
        selected=rng.choice(len(off)-1,min(512,len(off)-1),replace=False); cjk=0; substantial=0
        for i in selected:
            text=tok.decode(data[int(off[i]):int(off[i+1])].tolist(),skip_special_tokens=True)
            count=len(re.findall(r'[\u3400-\u9fff]',text)); cjk+=int(count>0)
            substantial+=int(count>=20 and count/max(1,len(text))>=0.02)
        language[source]={'sampled_examples':len(selected),'any_cjk':cjk,'substantial_cjk':substantial,'sampling_seed':SEED+912}
    report['language_samples']=language
    write_json(ROOT/'audit-native-loss.json',report); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
