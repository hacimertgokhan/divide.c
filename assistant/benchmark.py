"""Alternate inference runs after training; median numeric inference timings."""
from pathlib import Path
import json,re,statistics,subprocess
root=Path(__file__).resolve().parent.parent
runs={'divide':[],'dense':[]}
for repeat in range(10):
    for mode in ('divide','dense'):
        weight='basic' if mode=='divide' else 'dense'
        out=subprocess.check_output([str(root/'divide-chat'),'--load',str(root/f'assistant/{weight}.weights'),'--eval',str(root/'assistant/eval.tsv')],text=True)
        ms=float(re.search(r'ms/token=([\d.]+)',out)[1])
        if repeat:runs[mode].append(ms)
result={'scope':'Numeric inference; teacher-forced answer tokens on synthetic evaluation contexts; excludes checkpoint/process startup. Nine runs after one warm-up, alternating modes.', 'runs_ms_per_token':runs,'median_ms_per_token':{k:statistics.median(v) for k,v in runs.items()}}
result['divide_latency_reduction']=1-result['median_ms_per_token']['divide']/result['median_ms_per_token']['dense']
(root/'assistant/benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
