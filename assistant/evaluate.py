"""Evaluate whole generated records, not teacher-forced token accuracy."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--language',choices=['en','tr'],default='tr');p.add_argument('--weights',default='assistant/basic.weights');p.add_argument('--output',default='assistant/results.json');args=p.parse_args()
results={}
suffix='-en' if args.language=='en' else ''
training_inputs={line.split('\t')[0] for line in (ROOT/f'assistant/train{suffix}.tsv').read_text().splitlines()}
for split in (['eval','manual','holdout'] if args.language=='en' else ['eval','manual']):
    examples=[];start=time.perf_counter()
    for line in (ROOT/f'assistant/{split}{suffix}.tsv').read_text().splitlines():
        question,expected=line.split('\t');run=subprocess.run([str(ROOT/'divide-chat'),'--load',str(ROOT/args.weights),'--prompt',question],capture_output=True,text=True,check=True)
        answer=run.stdout.strip();examples.append({'input':question,'expected':expected,'generated':answer,'correct':answer==expected,'unknown_words':bool(run.stderr.strip()),'seen_in_training':question in training_inputs})
    results[split]={'examples':len(examples),'exact_record_accuracy':sum(x['correct'] for x in examples)/len(examples),'end_to_end_ms_per_record':1000*(time.perf_counter()-start)/len(examples),'details':examples}
for r in results.values():
    novel=[x for x in r['details'] if not x['seen_in_training']]
    r['novel_examples']=len(novel)
    r['novel_exact_record_accuracy']=sum(x['correct'] for x in novel)/len(novel) if novel else None
Path(ROOT/args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
for split,r in results.items():print(split,{k:v for k,v in r.items() if k!='details'})
