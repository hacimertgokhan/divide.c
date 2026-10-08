#!/usr/bin/env python3
"""Reproduce method trials sequentially; write fresh artifacts in reproduced/."""
import argparse
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--full',action='store_true')
args=parser.parse_args()
here=Path(__file__).resolve().parent
root=here.parent.parent
out=here/'reproduced'
out.mkdir(exist_ok=True)
subprocess.run(['make','-C',str(root)],check=True)

def train(name,flags):
    with (out/(name+'.log')).open('w') as stdout,(out/(name+'-training.log')).open('w') as stderr:
        subprocess.run([str(root/'divide'),'--only','divide','--tests','4000','--csv',str(out/(name+'.csv')),'--save',str(out/name),*flags],stdout=stdout,stderr=stderr,check=True)
    print('completed',name,flush=True)

for method in ['ordered','residual','local','binding','selective']:
    train(method+'-42',['--compressor',method,'--steps','20000','--task','fixed','--readout','gated','--init','random','--schedule','constant','--seed','42'])
for method in ['associative','associative-selective']:
    train(method+'-mixed-42',['--compressor',method,'--steps','50000','--task','mixed','--readout','gated','--init','random','--schedule','cosine','--seed','42'])
if args.full:
    for seed in [42,43,44]:
        for method in ['ordered','associative-selective']:
            flags=['--compressor',method,'--steps','100000','--task','mixed','--schedule','cosine','--seed',str(seed)]
            flags+=['--readout','gated','--init','random'] if method=='ordered' else ['--readout','product','--init','identity']
            train(method+'-long-'+str(seed),flags)
