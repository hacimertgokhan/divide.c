"""Local bilingual inbox demo. All outputs are suggestions requiring review."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--text', help='Classify one message and exit.')
parser.add_argument('--language', choices=['en', 'tr'], default='en')
args = parser.parse_args()
TASKS = {'e-posta': 'email', 'rapor': 'report', 'toplantı': 'meeting',
         'alışveriş': 'shopping', 'ödeme': 'payment', 'yok': 'none'}
TIMES = {'bugün': 'today', 'yarın': 'tomorrow', 'hafta': 'week', 'belirsiz': 'unspecified'}
PRIORITIES = {'acil': 'urgent', 'normal': 'normal'}

def classify(text):
    weights = 'english' if args.language == 'en' else 'basic'
    run = subprocess.run([
        str(ROOT / 'divide-chat'), '--load', str(ROOT / f'assistant/{weights}.weights'),
        '--prompt', text,
    ], capture_output=True, text=True, check=True)
    parts = run.stdout.strip().split('_')
    if args.language == 'tr' and len(parts) == 3:
        parts = [TASKS.get(parts[0], ''), TIMES.get(parts[1], ''), PRIORITIES.get(parts[2], '')]
    valid = (len(parts) == 3 and parts[0] in set(TASKS.values())
             and parts[1] in set(TIMES.values()) and parts[2] in set(PRIORITIES.values()))
    result = {'message': text, 'language': args.language,
              'status': 'suggestion' if valid and not run.stderr else 'needs_review'}
    if valid and not run.stderr:
        result.update(task=parts[0], time=parts[1], priority=parts[2])
    else:
        result['raw_output'] = run.stdout.strip()
    if run.stderr:
        result['note'] = run.stderr.strip()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if args.text:
    classify(args.text)
else:
    print(f'Divide Inbox ({args.language}): /quit to exit. Messages are classified independently.')
    while True:
        try:
            text = input('message> ')
        except (EOFError, KeyboardInterrupt):
            break
        if text == '/quit':
            break
        if text.strip():
            classify(text)
