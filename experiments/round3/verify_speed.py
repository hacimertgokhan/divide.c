#!/usr/bin/env python3
"""Sequential old-engine, invariant optimization, new-training comparison."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

here=Path(__file__).resolve().parent
subprocess.run(['cc','-O3','-std=c11','-DSOURCE="previous.c"',str(here/'benchmark.c'),'-lm','-o',str(here/'bench-previous')],check=True)
subprocess.run(['cc','-O3','-std=c11',str(here/'benchmark.c'),'-lm','-o',str(here/'bench-new')],check=True)
variants={
    'previous':('bench-previous','previous-recommended.weights'),
    'optimized_same_weights':('bench-new','previous-recommended.weights'),
    'new_training':('bench-new','distinct-42.divide.weights')}
runs={name:[] for name in variants}
for repeat in range(2):
    for name,(binary,checkpoint) in variants.items():
        output=subprocess.check_output([str(here/binary),str(here/checkpoint)],text=True)
        row=json.loads(output);runs[name].append(row);print(name,json.dumps(row),flush=True)
medians={name:statistics.median(r['median_ms'] for r in rows) for name,rows in runs.items()}
a=runs['previous'][0];b=runs['optimized_same_weights'][0];c=runs['new_training'][0]
result={'context':64,'recent':8,'task':'mixed','evaluation_seed':700003,'examples':4000,
        'training_examples_each':100000,'warmup_passes':1,'timed_passes':9,'processes_per_variant':2,
        'runs':runs,'median_ms':medians,
        'latency_reduction_percent':100*(1-medians['new_training']/medians['previous']),
        'invariance_gate_passed':a['accuracy']==b['accuracy'] and abs(a['nll']-b['nll'])<1e-9,
        'speed_gate_passed':medians['new_training']<=medians['previous'],
        'accuracy_gate_passed':c['accuracy']>a['accuracy']}
(here/'speed-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
sys.exit(0 if all(result[k] for k in ['invariance_gate_passed','speed_gate_passed','accuracy_gate_passed']) else 1)
