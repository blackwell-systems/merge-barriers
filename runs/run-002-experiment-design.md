# Run 002: Controlled Merge Barrier Experiment

## Hypothesis

A BPE tokenizer with merge barriers on delimiter characters produces a model that comprehends structured data better than an identical model trained with a standard BPE tokenizer (no barriers), given the same training data, architecture, and hyperparameters.

## Experimental Design

Train two models that differ ONLY in the tokenizer:

| | Model A (structok) | Model B (baseline) |
|---|---|---|
| Architecture | GPT-NeoX 410M | GPT-NeoX 410M |
| Tokenizer | structok-64k (merge barriers) | standard-64k (no barriers) |
| Vocab size | 65,539 | ~65,536 |
| Training data | Same source corpus | Same source corpus |
| tokens.bin | Different (different tokenizer) | Different (different tokenizer) |
| Batch size | 8 per GPU x 4 = 32 | 8 per GPU x 4 = 32 |
| Learning rate | 3e-4 with warmup + cosine decay | 3e-4 with warmup + cosine decay |
| Steps | 20,000 | 20,000 |
| Hardware | Same instance, same GPUs | Same instance, same GPUs |

## Tokenizer Training

### Model A: structok-64k
Already trained. Use existing `structok-64k.json`.

### Model B: standard-64k
Train a NEW tokenizer on the SAME corpus with standard HuggingFace BPE:
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # standard, no barriers
trainer = trainers.BpeTrainer(vocab_size=65536, special_tokens=["<pad>", "<eos>"])
tokenizer.train(files=[...], trainer=trainer)
tokenizer.save("standard-64k.json")
```

No merge barriers. Standard ByteLevel pre-tokenizer. Same vocab size. Same training corpus.

## Corpus (shared)

Rebalanced from run-001:

| Source | Size | % | Change from run-001 |
|--------|------|---|---------------------|
| FineWeb | 2.0 GB | 33% | Cut from 73% |
| Code (Go, Python, TS, JS, Rust) | 800 MB | 13% | Same |
| JSON | 850 MB | 14% | Same |
| GCF | 500 MB | 8% | 6x increase (from 1.3%) |
| YAML/CSV/TOML | 200 MB | 3% | 2x increase |
| Natural language (Wikipedia) | 200 MB | 3% | Same |
| **Total** | **~4.5 GB** | | Smaller but denser |

GCF data: 60K+ batches across all 7 data shapes, heavy on graph syntax (symbols, edges, sections).

## Pre-tokenization

Each tokenizer produces its own tokens.bin from the same source corpus:
- `tokens-structok-64k.bin`
- `tokens-standard-64k.bin`

Both archived to R2 under `archive/run-002-*/`.

## Training

Both models trained on the SAME hardware in the SAME session:
1. Train Model A for 20K steps
2. Save checkpoint, archive to R2
3. Train Model B for 20K steps on same GPUs
4. Save checkpoint, archive to R2

Same number of steps, same batch size, same LR schedule. The only variable is the tokenizer.

## Evaluation

### Held-out test data (NOT from training corpus)
Generate fresh test data not seen during training:
- 10 JSON/GCF payloads at sizes 5, 10, 20, 50, 100 records
- Both tabular (orders, users, logs) and graph (symbols, edges)
- Real-world patterns (API responses, MCP tool outputs)

### Metrics

For each model, on each test payload:

1. **Per-format PPL**: JSON PPL, GCF PPL, YAML PPL (lower = better comprehension)
2. **Next-token accuracy**: % of correctly predicted next tokens
3. **Per-category PPL**: Separate PPL for structured data vs natural language
4. **Scale curve**: How PPL changes from 5 to 100 records (does the gap widen?)

### Comparison table (the deliverable)

| Metric | Model A (structok) | Model B (standard) | Difference |
|--------|-------------------|-------------------|------------|
| JSON PPL (20 records) | ? | ? | ?x |
| GCF PPL (20 records) | ? | ? | ?x |
| JSON PPL (100 records) | ? | ? | ?x |
| GCF PPL (100 records) | ? | ? | ?x |
| Natural language PPL | ? | ? | ?x |
| Training loss (final) | ? | ? | ? |

If Model A has lower PPL on structured data than Model B at the same step count, the difference is attributable to merge barriers. If Model B has lower PPL on natural language, that's expected (compression advantage of standard BPE on prose).

## Estimated Cost

- Tokenizer training (standard-64k): minutes on CPU, free
- Pre-tokenization (both): ~30 min on GPU instance, ~$0.80
- Training Model A (20K steps): ~6 hours at $1.60/hr = $9.60
- Training Model B (20K steps): ~6 hours at $1.60/hr = $9.60
- Eval: ~$0.20
- **Total: ~$20-25**

## Success Criteria

The experiment succeeds if:
1. Model A (structok) has measurably lower PPL on structured data than Model B
2. The advantage is consistent across JSON, GCF, and YAML
3. The advantage scales with payload size (bigger payloads = bigger gap)
4. Model B is comparable or better on natural language (confirming the tradeoff)

## Failure Criteria

The experiment fails to prove the thesis if:
1. Both models have similar PPL on structured data (merge barriers don't help)
2. Model B is better on structured data (standard BPE's training familiarity wins)
3. The difference is within noise (< 5% PPL difference)
