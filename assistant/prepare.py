"""Author-created synthetic Turkish inbox triage corpus, no external data."""
import itertools, random
from pathlib import Path
root = Path(__file__).resolve().parent
rng = random.Random(712)
tasks = {'e-posta': ['e-posta gönder', 'mail gönder'], 'rapor': ['raporu gönder', 'raporu hazırla'], 'toplantı': ['toplantı ayarla', 'toplantı planla'], 'alışveriş': ['alışveriş yap', 'market alışverişi yap'], 'ödeme': ['faturayı öde', 'ödeme yap']}
times = {'bugün':['bugün'], 'yarın':['yarın'], 'hafta':['bu hafta'], 'belirsiz':['']}
priorities = {'acil':['acil', 'önemli'], 'normal':['', 'acele değil']}
train=[]; test=[]
for (task,verbs),(date,dates),(priority,qualifiers) in itertools.product(tasks.items(),times.items(),priorities.items()):
    answer=f'{task}_{date}_{priority}'
    for verb,when,qual in itertools.product(verbs,dates,qualifiers):
        for template in ['{when} {verb} {qual}', 'lütfen {when} {verb} {qual}', '{verb} {when} {qual}', 'bana {when} {verb} hatırlat {qual}', '{qual} {when} {verb}', '{qual} {verb} {when}', '{qual} lütfen {when} {verb}']:
            question=' '.join(template.format(when=when,verb=verb,qual=qual).split())
            train.append((question,answer))
        question=' '.join(f'{qual} lütfen {verb} hatırlat {when}'.split())
        test.append((question,answer))
# Non-action messages must not be turned into invented tasks.
for q in ['merhaba','nasılsın','teşekkürler','iyi geceler','hava güzel','tamam','görüşürüz','bugün hava güzel','yarın görüşürüz','bu hafta güzel','önemli bir konu','acil bir şey yok']:
    train.append((q,'yok_belirsiz_normal'))
# Long distractor prefixes exercise compression, with different train/eval wording.
base=list(train)
for q,a in base[::3]:
    train.append(('merhaba nasılsın hava güzel teşekkürler tamam görüşürüz ' * 6 + q,a))
for q,a in list(test)[::4]:
    test.append(('tamam teşekkürler merhaba hava güzel nasılsın görüşürüz ' * 8 + q,a))
for q in ['raporu gönderme','mail gönderme','toplantı ayarlama','alışveriş yapma','faturayı ödeme','ödeme yapma']:
    for prefix in ['', 'bugün ', 'yarın ', 'lütfen ', 'acil ']:
        train.append((prefix+q,'yok_belirsiz_normal'))
# All file splits are fixed, and evaluation sentences are absent from training.
train=list(dict.fromkeys(train)); known={q for q,a in train};test=[(q,a) for q,a in dict.fromkeys(test) if q not in known]
rng.shuffle(train);rng.shuffle(test)
for name,data in [('train.tsv',train),('eval.tsv',test)]:
    (root/name).write_text(''.join(q+'\t'+a+'\n' for q,a in data))
print(f'train={len(train)} held-out-template={len(test)}')
