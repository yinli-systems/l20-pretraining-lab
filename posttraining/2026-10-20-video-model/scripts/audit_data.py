"""Independent CPU audit of prompt partitions and sampled teacher/verifier agreement."""
import concurrent.futures,hashlib,json,multiprocessing,random
import numpy as np
from common import ROOT,CACHE,SEED,write_json,log
from evaluate import code_correct,extract_code,math_correct

def grade_math(r): return math_correct(r['reference'],r['answer'])

def main():
    seen={}; duplicates=0; leaked=[]; totals={}
    for source in ['math','code','numina','general']:
        p=CACHE/'prepared'/source
        ids=np.memmap(p/'ids.u16',mode='r',dtype=np.uint16); masks=np.memmap(p/'mask.u8',mode='r',dtype=np.uint8)
        offsets=np.load(p/'offsets.npy'); val=np.load(p/'validation.npy'); totals[source]=len(val)
        for i,isval in enumerate(val):
            a,b=int(offsets[i]),int(offsets[i+1]); h=hashlib.sha256(ids[a:b][masks[a:b]==0].tobytes()).digest()
            if h in seen:
                duplicates+=1
                if seen[h][0]!=bool(isval): leaked.append({'first_source':seen[h][1],'source':source,'row':i})
            else: seen[h]=(bool(isval),source)
    rng=random.Random(SEED); samples={}
    for source in ['math','code']:
        rows=[json.loads(s) for s in (CACHE/'prepared'/source/'pool.jsonl').read_text().splitlines()]
        samples[source]=rng.sample(rows,min(100,len(rows)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as ex:
        math_results=list(ex.map(grade_math,samples['math']))
    def grade_code(r):
        tests=json.loads(r['tests']) if isinstance(r['tests'],str) else r['tests']
        return code_correct(extract_code(r['reference']),'\n'.join(tests))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: code_results=list(ex.map(grade_code,samples['code']))
    report={'source_examples':totals,'exact_token_prompt_duplicates_across_sources_and_within_sources':duplicates,'train_validation_prompt_token_overlap':len(leaked),'overlap_details':leaked[:20],'sample_seed':SEED,'math_reference_verifier_agreement':{'passed':sum(math_results),'n':len(math_results)},'code_reference_sandbox_tests':{'passed':sum(code_results),'n':len(code_results)},'scope':'small random audit sample; code sandbox intentionally excludes filesystem/network/uncached imports; correctness agreement is not a proof for every example'}
    write_json(ROOT/'data-audit.json',report); log('data_audit_complete',**report)
    if leaked: raise RuntimeError('Prompt partition overlap detected')

if __name__=='__main__': main()
