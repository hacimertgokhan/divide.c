#!/usr/bin/env python3
"""Compare previous shipped engine vs current; use identical unseen evaluation data."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

here=Path(__file__).resolve().parent
root=here.parent.parent
subprocess.run(['cc','-O3','-std=c11','-DPREVIOUS_ENGINE','-DSOURCE="previous.c"',str(here/'benchmark.c'),'-lm','-o',str(here/'bench-previous')],check=True)
subprocess.run(['cc','-O3','-std=c11',str(here/'benchmark.c'),'-lm','-o',str(here/'bench-new')],check=True)
checkpoints={'previous':here/'ordered-mixed-long-42.compat.weights','new':here/'identity-product-mixed-42.divide.weights'}
runs={k:[] for k in checkpoints}
for repeat in range(2):
    for name in ('previous','new'):
        binary=here/('bench-previous' if name=='previous' else 'bench-new')
        output=subprocess.check_output([str(binary),str(checkpoints[name]),'64','3','700001'],text=True)
        row=json.loads(output);runs[name].append(row)
        print(name,json.dumps(row),flush=True)
old=statistics.median(x['median_ms'] for x in runs['previous'])
new=statistics.median(x['median_ms'] for x in runs['new'])
result={'context':64,'recent':8,'task':'mixed','evaluation_seed':700001,'evaluation_examples':4000,
        'training_steps_each':100000,'warmup_passes':1,'timed_passes_per_process':9,'processes_per_variant':2,
        'runs':runs,'previous_median_ms':old,'new_median_ms':new,'latency_reduction_percent':100*(1-new/old),
        'speed_gate_passed':new<=old,'accuracy_gate_passed':runs['new'][0]['accuracy']>runs['previous'][0]['accuracy']}
(here/'speed-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
sys.exit(0 if result['speed_gate_passed'] and result['accuracy_gate_passed'] else 1)
