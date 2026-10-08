# divide.c

**A dependency-free C11 research prototype for learned hierarchical context compression.**

Keep recent tokens in detail, compress older tokens into binary blocks of 2, 4, 8, 16… tokens, and let a new query attend to the resulting summaries. Each block carries its starting position and length.

The goal is to investigate how much useful information a small model can retain while reducing the number of attention entries. The implementation includes training, scalar reverse-mode automatic differentiation, a dense-attention baseline, checkpoints, and reproducible experiments.

> **Research status:** This is a small synthetic key–value retrieval experiment, not a general-purpose language model. The latest gains are relative to an earlier Divide implementation. An advantage over an optimized Transformer on real language data has **not** been established.

[Turkish documentation](README.tr.md)

## Quick start

You need a C11 compiler and Make. There are no external C library dependencies beyond the standard library and the math library. Python 3 is needed only for experiment scripts.

```sh
git clone https://github.com/hacimertgokhan/divide.c.git
cd divide.c
make
make check

# Try the included checkpoint: retrieve the value associated with a.
./divide --load recommended.divide.weights --predict 'a3;b7;c1;d9;xyzxyz?a='

# Evaluate it on independently generated examples.
./divide --load recommended.divide.weights \
  --eval-task mixed --eval-seed 700003 --tests 4000
```

The supported alphabet is `abcd0123456789;=?xyz`. Prediction prompts must end in `?a=`, `?b=`, `?c=`, or `?d=`. Predictions are learned and can be incorrect.

## How it works

For a 64-token context with an 8-token recent window:

```text
                         older context                    recent context
Original:      [                  56 tokens             ][8 individual tokens]
Compressed:    [32-token summary][16-token summary][8-token summary][8 tokens]

Attention entries: 64 → 11
```

1. **Encode each token.** A learned embedding passes through a learned leaf transformation. The default encoder adds a causal interaction with the preceding token:
   `encoded_current = current + current * rotate(previous)`.
2. **Merge older context.** Equal-sized neighboring blocks merge in a binary-carry hierarchy. The default `associative-selective` compressor uses learned salience scores and a two-child softmax to combine their vectors.
3. **Represent block location.** Start positions and lengths contribute learned features to attention keys and values.
4. **Read the summaries.** The query attends to compressed blocks and individual recent tokens. The default readout combines context and query by elementwise multiplication before predicting the next character.
5. **Share repeated work.** An optional score cache reuses salience calculations within a forward pass. It adds no trainable parameters and is checked against the uncached outputs and gradients.

The default model has a representation dimension of **12** and **1,112 trainable parameters**. The dense baseline uses the same token encoder and readout settings, but retains individual past-token representations instead of merging them.

Reducing attention entries does not make the complete implementation logarithmic: block construction still takes linear work. The current program rebuilds its summaries for each example and does not implement a persistent streaming cache.

## Train and compare

```sh
# Train only the compressed model.
./divide --only divide --steps 100000 --tests 4000 \
  --save my-model --csv my-results.csv

# Train both compressed and dense attention models.
./divide --only both --steps 100000 --tests 4000 \
  --save comparison --csv comparison.csv

# Reproduce the training settings of the included recommended checkpoint.
./divide --only divide --distinct yes --score-cache no \
  --steps 100000 --seed 42 --save retrained --csv retrained.csv
./divide --load retrained.divide.weights --score-cache yes \
  --eval-task mixed --eval-seed 700003 --tests 4000

./divide --help
```

`--only both` is the default. Use `--baseline` when evaluating or predicting with a saved dense-attention checkpoint. Checkpoints are readable text files; versions `DIVIDE1` through `DIVIDE7` are supported. They store weights and architecture settings, but not optimizer state, so loading is for evaluation and prediction rather than resuming training.

### Synthetic task

Each example contains four key–digit assignments, filler characters, and a final query. For example, `a3;b7;c1;d9;xyzxyz?a=` asks the model to predict `3`. Training supervises **only the final answer position**, rather than every token in a language-model corpus.

| Task | Assignment placement |
|---|---|
| `fixed` | Keys appear at the beginning in a, b, c, d order. |
| `shuffled` | Assignment order varies. |
| `scattered` | Assignment order and intervening filler lengths vary. |
| `mixed` | A mixture of all three; the default. |

The default `--distinct yes` uses four different digits during training. Evaluation always allows repeated digits. This training change reduces an easy digit-frequency shortcut; it is not a rule-based answer mechanism at inference time. `--distinct no` restores independent training digits; `early` applies distinct digits only in the first half of training.

### Main controls

| Option | Default | Purpose |
|---|---|---|
| `--context` / `--recent` | `64` / `8` | Context length and detailed recent window. |
| `--steps` / `--tests` | `100000` / `4000` | Training examples and evaluation examples. |
| `--seed` | `42` | Weight initialization seed. |
| `--data-seed` / `--eval-seed` | `12345` / `987654321` | Training and evaluation data seeds. |
| `--task` / `--eval-task` | `mixed` / training task | Training and evaluation layouts. |
| `--lr` / `--schedule` | `0.003` / `cosine` | Learning rate; cosine ends at 10% of its starting value. |
| `--batch` | `1` | Average gradients over 1–64 examples per update. |
| `--init` | `identity` | Identity leaf initialization; `random` is also available. |
| `--score-cache` | `yes` | Reuse mathematically equivalent salience calculations. |
| `--binding` | `additive` | Retain the individual token alongside its pair interaction. |
| `--curriculum` | `no` | Optionally grow training contexts from 16 tokens. |

`--steps` counts examples, not optimizer updates. With `--steps 100000 --batch 4`, there are 25,000 updates. A final partial batch is also processed. Dense and compressed comparisons share initialization and data seeds; keep the remaining settings equal for an architecture comparison.

Experimental compressors are `legacy`, `ordered`, `residual`, `local`, `binding`, `selective`, `associative`, `associative-selective` (default), `mass-pool`, and `sigmoid-pool`. Readouts are `legacy`, `gated`, `product` (default), `tied`, and `tied-mask`. Alternative modes remain available for ablations; they are not all improvements.

## Recorded results

The latest held-out comparison uses evaluation seed `700003`, 4,000 examples per layout, and three initialization seeds: 42, 43, and 44. Both training regimes use 100,000 examples and batch size 1. Training distributions differ; evaluation examples are identical between the compared models.

| Evaluation layout | Previous Divide training | Distinct-digit training |
|---|---:|---:|
| Fixed | 66.900% | 74.967% |
| Shuffled | 66.967% | 74.058% |
| Scattered | 71.775% | 81.308% |
| Mixed | 69.183% | 78.025% |

The mixed-task gain is **8.84 percentage points**, averaged across the three seeds. It comes from changing training data, not from establishing a better compression architecture. These are retrieval accuracies, not language-model accuracies.

The recorded CPU timing comparison uses seed 42, mixed evaluation, 64-token contexts, `-O3`, one warm-up pass, nine timed passes, and two processes per variant:

| Engine / checkpoint | Median ms per example | Accuracy | NLL | Used autodiff tape bytes |
|---|---:|---:|---:|---:|
| Previous engine / previous weights | 0.027485 | 69.875% | 1.027683 | 636,384 |
| Optimized engine / identical previous weights | 0.024877 | 69.875% | 1.027683 | 576,480 |
| Optimized engine / new training | 0.025073 | 83.650% | 0.710864 | 576,480 |

The new checkpoint takes about **8.8% less time** than the previous implementation in this recorded run. Reusing calculations with the same previous weights preserves the reported predictions and loss. Timings depend on hardware and system load; they are not a universal speed guarantee.

Evidence and unsuccessful alternatives are retained in the [round 2 report](experiments/round2/REPORT.md), [round 3 report](experiments/round3/REPORT.md), and [timing results](experiments/round3/speed-results.json). The historical reports are in Turkish; the main findings are summarized here in English.

### What the measurements mean

- `train_seconds` includes data generation, forward/backward computation, and optimizer updates.
- `eval_ms_per_example` includes rebuilding summaries and constructing the scalar autodiff graph. It is not a GPU or production inference benchmark.
- `nll` is the average negative log likelihood over the 20-character output vocabulary; lower is better.
- `representation_bytes` estimates block vectors, position/length fields, and enabled cached salience scalars. Some historical CSVs predate accounting for that scalar.
- `autodiff_peak_bytes` measures used graph nodes, not total RAM, allocated arena capacity, optimizer memory, or all working caches.

## Validation and reproduction

```sh
make check

# Optional address and undefined-behavior checks.
cc -O1 -g -std=c11 -fsanitize=address,undefined test.c -lm -o test-sanitize
./test-sanitize

# Compare the recorded previous engine with the current engine.
python3 experiments/round3/verify_speed.py

# Rerun experiments into separate reproduced/ directories.
python3 experiments/round2/run_experiments.py
python3 experiments/round3/run_experiments.py
# Add --full for the longer experiment sets.
```

Tests cover finite-difference gradients across compressor/readout/binding combinations, dense and compressed paths, cached-score equivalence, block properties, generated-data consistency, and checkpoint prediction smoke tests. The speed script fails if its configured accuracy and timing gates are not met; that gate applies only to its particular experiment.

## Limitations and next experiments

- Fixed-size summaries can discard exact names, numbers, and ordering information. Compression is not lossless.
- There is no real-text corpus, full next-token language-model training, GPU kernel, production KV cache, or persistent streaming memory.
- Results on a small synthetic distribution do not establish generalization to arbitrary text or larger models.
- The included dense baseline is a controlled small-model comparison, not an optimized Transformer implementation.
- The latest reported improvement compares two Divide versions. It does not demonstrate a win over dense attention under the latest training regime.
- The fixed-layout `specialized.divide.weights` checkpoint performs poorly when assignments move. Use `recommended.divide.weights` for the mixed-layout experiment.

The next useful evaluations are longer contexts, distant exact-value retrieval, real-text prediction, and equal-budget comparisons against optimized dense and hierarchical baselines. Training time, inference latency, accuracy, and total memory should be measured separately.

## Related research

The broad idea of hierarchical or compressed context is established research. This repository does not claim to have invented it.

- [BP-Transformer (2019)](https://arxiv.org/abs/1911.04070): binary partitioning and fine-to-coarse attention over multi-scale spans; particularly close to the binary-block motivation.
- [Compressive Transformers (2019)](https://arxiv.org/abs/1911.05507): compression of older memories for long-range sequence modeling.
- [H-Transformer-1D (2021)](https://arxiv.org/abs/2107.11906): efficient hierarchical attention.
- [Infini-attention (2024)](https://arxiv.org/abs/2404.07143): local attention combined with compressive long-term memory.

These are conceptual precedents, not implementations reproduced by this code. Any claim of a distinct research contribution would require a more thorough comparison.

## Repository layout

- `divide.c`: model, autodiff engine, optimizer, CLI, checkpoint handling, and measurements.
- `data.h`: synthetic data generator.
- `test.c`: numerical and structural checks.
- `recommended.divide.weights`: the seed-42 mixed-layout checkpoint described above.
- `experiments/`: historical source snapshots, experiment scripts, checkpoints, CSVs, and reports.
- `README.tr.md`: detailed Turkish documentation.

## Lightweight English and Turkish inbox demo

A separate task-specific model now learns task, time-category, and priority labels from English and Turkish messages, using separately trained checkpoints. It keeps the binary context compression and uses numeric C inference without an autodiff graph. See [the inbox demo](assistant/README.md) for training data, evaluation limits, and usage:

```sh
make divide-chat
python3 assistant/run.py --text 'send the report tomorrow urgent'
python3 assistant/run.py --language tr --text 'yarın raporu gönder acil'
```

The earlier experimental conversational model is also available through `./divide-chat --load chat/basic.weights --chat`. Its small training set does not support reliable general conversation.
