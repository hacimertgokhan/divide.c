"""Author-created English inbox triage corpus; fixed held-out templates."""
import itertools,random
from pathlib import Path
root=Path(__file__).resolve().parent
rng=random.Random(712)
tasks={'email':['send an email','send mail'],'report':['send the report','prepare the report'],'meeting':['schedule a meeting','plan a meeting'],'shopping':['go shopping','do grocery shopping'],'payment':['pay the bill','make a payment']}
times={'today':['today'],'tomorrow':['tomorrow'],'week':['this week'],'unspecified':['']}
priorities={'urgent':['urgent','important'],'normal':['','not urgent']}
train=[];test=[]
for (task,verbs),(date,dates),(priority,qualifiers) in itertools.product(tasks.items(),times.items(),priorities.items()):
    answer=f'{task}_{date}_{priority}'
    for verb,when,qual in itertools.product(verbs,dates,qualifiers):
        for template in ['{when} {verb} {qual}','please {when} {verb} {qual}','{verb} {when} {qual}','remind me to {when} {verb} {qual}','{qual} {when} {verb}','{qual} {verb} {when}','{qual} please {when} {verb}', '{qual} {verb} remind me {when}', '{when} please {verb} remind me {qual}', 'please remind me to {qual} {verb} {when}', '{verb} {when} {qual} remind me please']:
            train.append((' '.join(template.format(when=when,verb=verb,qual=qual).split()),answer))
        test.append((' '.join(f'{qual} please {verb} remind me {when}'.split()),answer))
for q in ['hello','how are you','thanks','good night','nice weather','okay','goodbye','nice weather today','see you tomorrow','a nice week','an important topic','nothing urgent']:
    train.append((q,'none_unspecified_normal'))
base=list(train)
for q,a in base[::3]:train.append(('hello how are you nice weather thanks okay goodbye '*6+q,a))
for q,a in list(test)[::4]:test.append(('okay thanks hello nice weather how are you goodbye '*8+q,a))
for q in ['do not send the report','do not send mail','do not schedule a meeting','do not go shopping','do not pay the bill','do not make a payment']:
    for prefix in ['', 'today ', 'tomorrow ', 'please ', 'urgent ']:train.append((prefix+q,'none_unspecified_normal'))
train=list(dict.fromkeys(train));known={q for q,a in train};test=[(q,a) for q,a in dict.fromkeys(test) if q not in known]
rng.shuffle(train);rng.shuffle(test)
for name,data in [('train-en.tsv',train),('eval-en.tsv',test)]: (root/name).write_text(''.join(q+'\t'+a+'\n' for q,a in data))
print(f'train={len(train)} held-out-template={len(test)}')
