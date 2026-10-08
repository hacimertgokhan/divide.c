# Divide Inbox: lightweight English and Turkish message triage

A local CPU model that suggests **task type, time category, and priority** for short English or Turkish messages. English is the default UI language; each input language uses its own trained checkpoint. It uses Divide's learned binary compression of older context. It does not create reminders, send messages, resolve calendar dates, or extract arbitrary task descriptions.

Supported task categories: email, report, meeting, shopping, payment, and no task. Time categories: today, tomorrow, this week, and unspecified. Priority: urgent or normal.

## Run

From the repository root:

```sh
make divide-chat
python3 assistant/run.py --text 'send the report tomorrow urgent'
python3 assistant/run.py --language tr --text 'yarın raporu gönder acil'
python3 assistant/run.py
```

The wrapper returns JSON with the original message, a suggested task/time/priority, and a status. Unknown words or malformed outputs are marked `needs_review`. A valid schema is **not** a guarantee that the meaning was understood correctly. Classification is independent for each message, unlike the conversational CLI.

The C model can be called directly:

```sh
./divide-chat --load assistant/english.weights --prompt 'send the report tomorrow urgent'
```

It produces a compact label such as `report_tomorrow_urgent`. The Python wrapper only splits this label and validates its schema; the labels themselves are predicted by the model, not keyword rules in the wrapper.

## Training and evaluation

```sh
python3 assistant/prepare.py
./divide-chat --train assistant/train.tsv --steps 150000 \
  --save assistant/basic.weights --eval assistant/eval.tsv
python3 assistant/evaluate.py
make chat-check
```

The original Turkish data is author-created synthetic text, generated from task phrases and message templates. Evaluation holds out a sentence template and includes longer distractor prefixes. This measures generalization within this synthetic domain, not general Turkish comprehension. `manual.tsv` includes separately written sentences, negation, an unsupported task, and a change of intent. It is a small diagnostic set, not a representative user benchmark.

Training uses answer-token cross entropy, Adam, gradient clipping, and a cosine learning-rate schedule. The fixed model has 20,832 parameters, 32-dimensional vectors, a 256-slot word vocabulary, a 64-token context, and an 8-token detailed recent window. Inference uses direct numeric C operations without an autodiff graph. The checkpoint embeds its vocabulary, architecture settings, and text weights. It has no optimizer state.

The first attempted version generated three fields as a sequence. It obtained only 14.375% complete-record accuracy on 160 held-out-template examples despite high teacher-forced accuracy. Those results are kept in `initial-results.json`. The revised model predicts one joint task/time/priority token and trains on more word orders and cancellation phrases. The datasets differ between these attempts, so their scores are not a controlled architecture comparison.

## Limits

This is a restricted demonstration, not a production personal assistant. New vocabulary, multiple tasks in one sentence, negation outside trained forms, and changes of intent can cause errors. Confidence calibration is not implemented. A schema-valid output can still be wrong. No automatic external action is performed.

The 64-token window truncates earlier history, and fixed-size block summaries are lossy. Long-prefix tests exercise compression but do not establish reliable document understanding or long-term memory. Performance against an optimized Transformer is unproven.

## Recorded Turkish result (single initialization seed)

The revised checkpoint was trained from scratch on **1,337 synthetic messages**, with seed 42 and 150,000 sampled answer-token updates. On **200 unseen-template messages**, including 40 long distractor-prefix examples, complete-label accuracy was **100%**. These examples use the same restricted vocabulary and task distribution as training.

The 12-message manual diagnostic scored **10/12** overall. Six of those messages were already present in training; accuracy on the six novel messages was **4/6**. The failures were an unsupported doctor appointment and the phrase “acil değil” (“not urgent”), which was incorrectly labeled urgent. The wrapper flags unknown vocabulary in the appointment example, but does not detect every semantic error. The manual set has now been inspected and must not be treated as a fresh final test in future iterations.

See `results.json` for every input, expected label, generated label, and overlap marker. The complete-label evaluation generates the answer from scratch rather than supplying correct previous answer tokens.

This is evidence of a small restricted classifier working on its defined task. It is not evidence of general conversation, dependable reasoning, or a requirement to use compression. A simpler classifier remains an important untested baseline.

To reproduce the same-budget dense comparison and numeric timing measurement:

```sh
./divide-chat --train assistant/train.tsv --steps 150000 --dense \
  --save assistant/dense.weights --eval assistant/eval.tsv
python3 assistant/evaluate.py --weights assistant/dense.weights --output assistant/dense-results.json
python3 assistant/benchmark.py
```

The benchmark alternates the two engines after training finishes, with one warm-up and nine measured runs. Its `ms/token` timing excludes loading the checkpoint and launching a process. The Python UI starts a fresh process for each message, so its end-to-end latency is higher. Results apply to this CPU, vocabulary, context length, and synthetic dataset.

The same-budget dense model also achieved 200/200 on the synthetic split and 4/6 on novel manual diagnostics. Recorded median numeric latency was **0.0093 ms/token for Divide** versus **0.0127 ms/token for dense attention**, a **26.8% reduction**. These timings include both the joint-label and end-of-answer predictions; they are not timings for a large language model. The independently launched Python/C classification interface measured about **4.1 ms/message** for Divide on this run. See `benchmark.json` and `dense-results.json` for the measurements.

## English release checkpoint

`english.weights` is trained separately from scratch on English messages. It is not a translation layer around Turkish inference. The Python interface defaults to English and returns English JSON labels for both input languages. Use `--language tr` to retain Turkish input support.

```sh
python3 assistant/prepare_en.py
./divide-chat --train assistant/train-en.tsv --steps 150000 \
  --save assistant/english.weights --eval assistant/eval-en.tsv
python3 assistant/evaluate.py --language en --weights assistant/english.weights \
  --output assistant/english-results.json
```

The English corpus follows the same restricted task categories, uses 2,190 training messages. The 190-example template split was inspected during development; a separate 80-example wording split is provided in `holdout-en.tsv`. Long English prefixes can exceed the 64-token window, so earlier words are truncated. The English manual diagnostics include examples that also occur in training; overlap is explicitly recorded. The Turkish timing figures above have not been established for the English checkpoint.

### Example

Input: `send the report tomorrow urgent`

```json
{
  "message": "send the report tomorrow urgent",
  "language": "en",
  "status": "suggestion",
  "task": "report",
  "time": "tomorrow",
  "priority": "urgent"
}
```

`time` is a category, not a resolved calendar date. `suggestion` is a valid-schema prediction, not calibrated confidence. Messages are not routed to external services.

The first English attempt achieved only 15.5% complete-label accuracy on its 200-example template split. This prompted adding more word orders to training. The failed attempt is preserved in `english-initial-results.json` and `english-initial-training.log`. The updated development split differs because exact training overlaps are removed; it is not an independent final evaluation.

The final English checkpoint achieved **179/190 (94.2%)** complete-label accuracy on the inspected development template split and **80/80** on the separate held-out wording split. Its manual diagnostic scored **10/12 overall**, including **4/6 on novel sentences**. These are small synthetic/diagnostic results for seed 42, not evidence of production reliability. English end-to-end classification measured about 3.8 ms/message in this run, including process startup and checkpoint loading. Full per-example results are in `english-results.json`.
