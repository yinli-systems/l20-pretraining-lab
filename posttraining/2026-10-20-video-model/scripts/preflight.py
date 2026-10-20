import concurrent.futures,json,multiprocessing,subprocess,sys
import numpy as np
from common import ROOT,CACHE,tokenizer,encode_messages,write_json,log
from evaluate import code_correct,math_correct

def math_check(x): return math_correct(*x)

def main():
    tok=tokenizer(); msgs=[{'role':'user','content':'What is 6 times 7?'},{'role':'assistant','content':'Multiply 6 by 7 to get \\boxed{42}.'}]
    ids,labels=encode_messages(tok,msgs)
    assert ids==tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False)
    trained=tok.decode([y for y in labels if y!=-100],skip_special_tokens=False)
    assert 'Multiply 6' in trained and 'What is' not in trained and '<|im_end|>' in trained
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as ex:
        math_results=list(ex.map(math_check,[(r'Work leads to \\boxed{42}'.replace('\\\\','\\'),'42'),(r'\\boxed{41}'.replace('\\\\','\\'),'42'),(r'\\boxed{\\frac{1}{2}}'.replace('\\\\','\\'),'0.5')]))
    assert math_results==[True,False,True],math_results
    assert code_correct('def f(x): return x+1','def check(f): assert f(2)==3','f')
    assert not code_correct('def f(x): return x-1','def check(f): assert f(2)==3','f')
    assert code_correct('import socket\ndef f():\n try:\n  socket.socket()\n except (OSError,ImportError):\n  return True\n return False','assert f()') is False # socket import is unavailable inside the sealed worker
    manifest=json.loads((ROOT/'data-manifest.json').read_text())
    checks={}
    for split in ['train','valid']:
        ids=np.memmap(CACHE/'packed'/f'{split}.ids.u16',dtype=np.uint16,mode='r').reshape(-1,2048)
        masks=np.memmap(CACHE/'packed'/f'{split}.mask.u8',dtype=np.uint8,mode='r').reshape(-1,2048)
        assert len(ids)>0 and ids.max()<len(tok) and masks.max()==1
        assert (masks.sum(1)>0).all()
        checks[split]={'blocks':len(ids),'supervision_fraction':float(masks.mean()),'packing_efficiency':manifest['splits'][split]['packing_efficiency']}
    write_json(ROOT/'preflight.json',{'passed':True,'math_verifier':math_results,'chat_and_loss_masks':True,'execution_sandbox_positive_and_negative_tests':True,'packed_data':checks})
    log('preflight_passed',checks=checks)

if __name__=='__main__': main()
