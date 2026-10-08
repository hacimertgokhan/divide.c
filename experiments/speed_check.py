#!/usr/bin/env python3
"""Sequential, warmed original/new comparison; no concurrent benchmark workers."""
import json
import pathlib
import re
import statistics
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent.parent
exp = root / 'experiments'
subprocess.run(['cc', '-O3', '-std=c11', '-DSOURCE="original.c"', str(exp/'benchmark.c'), '-lm', '-o', str(exp/'bench-original')], check=True)
subprocess.run(['cc', '-O3', '-std=c11', str(exp/'benchmark.c'), '-lm', '-o', str(exp/'bench-current')], check=True)
weights = {'original': root/'model.divide.weights', 'current': exp/'gated-ordered-42.divide.weights'}
runs = {k: [] for k in weights}
for repeat in range(2):
    for name in ('original', 'current'):
        exe = exp / ('bench-original' if name == 'original' else 'bench-current')
        output = subprocess.check_output([str(exe), str(weights[name])], text=True)
        print(output.strip(), flush=True)
        runs[name].append({k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', output)})
old = statistics.median(r['median_ms'] for r in runs['original'])
new = statistics.median(r['median_ms'] for r in runs['current'])
result = {'context': 64, 'recent': 8, 'evaluation_examples': 4000,
          'warmup_passes_per_process': 1, 'timed_passes_per_process': 9,
          'processes_per_variant': 2, 'runs': runs,
          'original_median_ms': old, 'current_median_ms': new,
          'latency_reduction_percent': 100*(1-new/old),
          'speed_gate_passed': new <= old,
          'accuracy_gate_passed': runs['current'][0]['accuracy'] > runs['original'][0]['accuracy']}
(exp/'speed-results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
sys.exit(0 if result['speed_gate_passed'] and result['accuracy_gate_passed'] else 1)
