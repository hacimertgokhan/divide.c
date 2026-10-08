#!/usr/bin/env python3
"""Reproduce selected training experiments in a fresh directory, sequentially."""
import argparse
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--full',action='store_true',help='also reproduce three initial seeds and alternative training')
args=parser.parse_args()
here=Path(__file__).resolve().parent
root=here.parent.parent
out=here/'reproduced'
out.mkdir(exist_ok=True)
subprocess.run(['make','-C',str(root)],check=True)

def train(name,flags):
    base=['--only','divide','--steps','100000','--tests','4000','--task','mixed','--readout','product','--init','identity','--schedule','cosine','--binding','additive','--batch','1','--lr','0.003','--distinct','no','--score-cache','no']
    with (out/(name+'.log')).open('w') as stdout,(out/(name+'-training.log')).open('w') as stderr:
        subprocess.run([str(root/'divide'),*base,'--save',str(out/name),'--csv',str(out/(name+'.csv')),*flags],stdout=stdout,stderr=stderr,check=True)
    print('completed',name,flush=True)

train('distinct-42',['--seed','42','--distinct','yes'])
if args.full:
    for seed in [42,43,44]:
        train('control-'+str(seed),['--seed',str(seed)])
        if seed!=42:train('distinct-'+str(seed),['--seed',str(seed),'--distinct','yes'])
        train('early-'+str(seed),['--seed',str(seed),'--distinct','early'])
    for method in ['mass-pool','sigmoid-pool']:
        train(method+'-42',['--compressor',method,'--seed','42'])
