# Research Background

structok is built on findings from the GCF tokenizer analysis, the most comprehensive tokenizer boundary study published for any wire format. This document connects each design decision to the data that motivated it.

## The study

43 tokenizers from 20 providers (OpenAI, Anthropic, Meta, Google, Mistral, DeepSeek, Qwen, Microsoft, TII, 01.AI, BigCode, NVIDIA, AI21, Stability AI, EleutherAI, Snowflake, AllenAI, and more). Every major model family in production.

Full analysis: [gcformat.com/guide/tokenizer-analysis](https://gcformat.com/guide/tokenizer-analysis)

Source code: [github.com/blackwell-systems/gcf/tree/main/eval](https://github.com/blackwell-systems/gcf/tree/main/eval)

## Why merge barriers?

### Finding 1: Delimiter merge rates vary wildly by character

We measured how often each delimiter fuses with adjacent content across 43 tokenizers:

| Delimiter | Merge rate | Checks |
|-----------|-----------|--------|
| Pipe (\|) | 0.47% | 135/29,025 |
| Quote (") | 8.17% | 158/1,935 |
| Tab (\t) | 32.91% | 283/860 |

The pipe has the lowest merge rate of any common delimiter. The tab has the highest. The difference is 70x. This is not a property of the characters themselves; it's a property of what appeared in the tokenizer training data. JSON's `"name` pattern appeared billions of times, so tokenizers learned it as a merge. GCF's `|name` pattern didn't, so they didn't.

structok's insight: instead of hoping a delimiter doesn't appear in training data, enforce it as a pre-tokenization rule. Zero merges by construction, not by accident.

### Finding 2: Merged entries are hardcoded in the vocabulary

BPE is deterministic. If `"name` is vocabulary entry #32586, the tokenizer WILL always select it. There is no context-dependence, no probability. The merge is a dictionary lookup.

We exhaustively scanned every vocabulary in all 43 tokenizers:

| Delimiter | Unique words that can ever merge | 
|-----------|--------------------------------|
| Pipe (\|) | 24 |
| Quote (") | 193 |
| Tab (\t) | 1,238 |

GPT-4 cl100k has 2,116 total merged delimiter entries. GPT-4o o200k has 1,892. These cannot be fixed without retraining the model from scratch. The vocabulary is frozen.

structok has 0 merged delimiter entries. The pre-tokenization barrier prevents any merge from ever being created.

### Finding 3: JSON grammar fuses into multi-operation tokens

92.5% of JSON's quote-containing tokens pack multiple grammar operations into a single integer. This happens on all 43 tokenizers:

| Token | Grammar operations fused |
|-------|------------------------|
| `":"` | Close string + colon + open string |
| `","` | Close string + comma + open string |
| `{"` | Open object + open string |

The model must learn to decompose these multi-operation tokens as emergent behavior. structok prevents this by ensuring each grammar character is always its own token.

### Finding 4: Structural equivalence breaks across tokenizers

We tokenized identical data on all 43 tokenizers and compared structural boundaries:

| Format | Grammar isolation rate |
|--------|---------------------|
| GCF (pipe, @, <) | 99.5% |
| JSON (quote) | 7.5% (92.5% fused) |

GCF's grammar is deterministic: every model sees the same structural boundaries. JSON's grammar is ambiguous: boundaries differ per tokenizer.

structok extends this property to ALL delimiters, not just GCF's.

## Why these 16 characters?

Each barrier character was chosen because it serves as a structural delimiter in at least one major data format:

| Character | Formats that use it structurally |
|-----------|--------------------------------|
| `\|` | GCF, markdown tables, shell pipes, SQL |
| `@` | GCF graph IDs, email, decorators (Python/Java/TS) |
| `<` `>` | HTML/XML tags, GCF edges, comparisons, shell redirect |
| `"` `'` | JSON, YAML, most programming languages |
| `:` | JSON key-value, YAML, Python dict, URL scheme |
| `,` | JSON arrays, CSV, function arguments |
| `;` | Statement terminator (C/Java/JS), CSS, CSV alternate |
| `\t` | TSV, TOON format, Makefile, indentation |
| `\n` | Line separator (GCF rows, code, log files) |
| `{` `}` | JSON objects, code blocks, template literals |
| `[` `]` | JSON arrays, indexing, regex character classes |
| `(` `)` | Function calls, grouping, SQL, regex |

We tested all 94 printable ASCII characters (codes 33-126) across tokenizers. 74 are "safe" (never merge). 20 are "unsafe" (merge with adjacent content). All 16 barrier characters are drawn from the safe set, meaning they naturally don't merge even without barriers. The barrier is a guarantee, not a workaround.

## The attention mechanism

We ran a separate experiment using PyTorch with Pythia 410M and Gemma 2B to measure how attention patterns change at scale.

### Entropy crossover (Pythia 410M)

| Orders | GCF entropy | JSON entropy |
|--------|------------|-------------|
| 5 | 3.03 | 2.87 (JSON lower, model knows JSON) |
| 10 | 3.32 | 3.01 |
| 20 | 3.66 | 3.16 |
| 50 | 3.99 | **4.50** (JSON crosses over) |

At small scale, JSON entropy is lower because the model has been trained on JSON. At 50 orders, JSON entropy exceeds GCF by 13%. The repeated field names overwhelm the model's learned parsing.

### Grammar attention collapse (Gemma 2B)

| Orders | JSON grammar attention | JSON payload attention |
|--------|----------------------|----------------------|
| 5 | 29.8% | 67.7% |
| 20 | 30.4% | 67.4% |
| 50 | **8.6%** | **86.3%** |
| 100 | **8.6%** | **86.3%** |

At 50+ orders, JSON's grammar attention collapses from 30% to 8.6%. The model stops attending to structural tokens and distributes attention uniformly across payload. This is the mechanism behind comprehension failure: the model can no longer distinguish structure from data.

GCF's payload attention increases steadily from 46% to 63% at 100 orders. The model progressively focuses more on data as the payload grows, because the grammar tokens are clean and don't compete for attention.

### Implication for structok

A tokenizer with merge barriers eliminates the training-data dependency entirely. The model doesn't need to learn which tokens contain hidden boundaries because there are no hidden boundaries. Every delimiter is its own token, every structural operation is explicit, and attention entropy stays bounded.

## Reproducing

All scripts and data are in the GCF eval directory:

| Script | What it measures |
|--------|-----------------|
| `eval/hf-tokenizer-analysis.py` | 43-tokenizer merge rates, savings, vocabulary analysis |
| `eval/structural-equivalence-proof.py` | Grammar isolation across all tokenizers |
| `eval/adversarial-vocab-dump.py` | Exhaustive vocabulary scan (24 vs 193 vs 1,238) |
| `eval/attention-analysis.py` | Entropy and attention distribution (Pythia 410M, Gemma 2B) |

```bash
cd gcf/eval
source .venv/bin/activate
pip install tokenizers huggingface_hub tiktoken torch transformers

python hf-tokenizer-analysis.py
python structural-equivalence-proof.py
python adversarial-vocab-dump.py
python attention-analysis.py
```
