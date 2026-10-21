from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
jobs=[("AliceYin/L20-1B-20B-Base","model",None),
("HuggingFaceTB/smoltalk","dataset",["data/all/*"]),
("nvidia/OpenMathInstruct-2","dataset",["data/train_1M*"]),
("OpenCoder-LLM/opc-sft-stage2","dataset",None),
("allenai/tulu-3-sft-personas-instruction-following","dataset",None),
("HuggingFaceH4/ultrafeedback_binarized","dataset",None)]
def f(j):
    r,t,pat=j
    try: return r, snapshot_download(r,repo_type=t,allow_patterns=pat)
    except Exception as e: return r, "ERR "+repr(e)
with ThreadPoolExecutor(6) as ex:
    for r in ex.map(f,jobs): print(r,flush=True)
