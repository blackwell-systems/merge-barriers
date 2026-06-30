# Research Background

This document connects each finding to the data that motivated it, from the initial 43-tokenizer analysis through three controlled experiments and 30 ablation phases.

Paper: [Merge Barriers in BPE Tokenization: How Tokenizer Design Causally Determines Attention Head Specialization](paper/merge-barriers-in-bpe-tokenization.pdf)

DOI: [10.5281/zenodo.20925910](https://doi.org/10.5281/zenodo.20925910)

## The tokenizer study

43 tokenizers from 20 providers (OpenAI, Anthropic, Meta, Google, Mistral, DeepSeek, Qwen, Microsoft, TII, 01.AI, BigCode, NVIDIA, AI21, Stability AI, EleutherAI, Snowflake, AllenAI, and more). Every major model family in production. Vocabulary sizes from 32K to 262K.

### Finding 1: Delimiter merging is universal

BPE tokenizers fuse delimiter characters with adjacent content. This is not occasional; it is universal across all 43 tokenizers tested.

| Delimiter | Merge rate | Checks |
|-----------|-----------|--------|
| Pipe (\|) | 0.47% | 135/29,025 |
| Quote (") | 8.17% | 158/1,935 |
| Tab (\t) | 32.91% | 283/860 |

JSON's `"name` is token #32586 in GPT-4's vocabulary. The tokenizer always selects it. This is a dictionary lookup, not a probabilistic decision. The opening quote fuses with the field name, hiding the structural boundary inside the embedding.

### Finding 2: The adversarial surface is large

We decoded every entry in all 43 vocabularies and counted unique words that fuse with each delimiter:

| Delimiter | Unique mergeable words |
|-----------|----------------------|
| Pipe (\|) | 24 (all TypeScript union keywords) |
| Quote (") | 193 |
| Colon (:) | 232 |
| Comma (,) | 282 |
| Tab (\t) | 1,238 |

JSON's total adversarial surface across all 7 grammar characters: 1,939 words (81x the pipe's 24). These are hardcoded vocabulary entries that cannot be fixed without retraining the model from scratch.

### Finding 3: JSON grammar fuses into multi-operation tokens

92.5% of JSON's quote-containing tokens pack multiple grammar operations into a single integer. This happens on all 43 tokenizers:

| Token | Grammar operations fused | Present in |
|-------|------------------------|------------|
| `":"` | Close string + colon + open string | 43/43 |
| `","` | Close string + comma + open string | 43/43 |
| `{"` | Open object + open string | 43/43 |
| `":{"` | Close string + colon + open object + open string | 43/43 |

The model receives one integer where there should be four grammar decisions.

### Finding 4: Structural equivalence breaks across tokenizers

| Format | Grammar isolation rate |
|--------|---------------------|
| GCF (pipe, @, <) | 99.5% |
| JSON (quote) | 7.5% (92.5% fused) |

GCF's grammar is deterministic: every model sees the same structural boundaries. JSON's grammar is ambiguous: boundaries differ per tokenizer. The same JSON object `{"orderId":"ORD-001","value":"shipped"}` produces 4 different token counts (12, 13, 14, 15) depending on the model.

## The fix: merge barriers

16 delimiter characters are forbidden from participating in any BPE merge operation during tokenizer training. The BPE algorithm is unchanged; the constraint is a pre-tokenization rule (16 `Split` calls in a `Sequence` pre-tokenizer). See [Appendix A of the paper](paper/merge-barriers-in-bpe-tokenization.pdf) for the full configuration.

The resulting tokenizer (structok-64k, 65,539 vocabulary) has zero merged delimiter entries and zero adversarial surface. The barrier characters:

```
|  @  <  >  "  '  :  ,  ;  \t  {  }  [  ]  (  )
```

Each was chosen because it serves as a structural delimiter in at least one major data format. This is not format-specific: it fixes JSON, YAML, CSV, TOON, GCF, and any format that uses delimiter characters.

## The attention mechanism (pre-existing models)

Before the controlled experiments, we established the transformer-level mechanism using pre-existing models (Pythia 410M, Gemma 2B).

### Entropy crossover

At small scale (5-20 records), JSON entropy is lower than GCF because the model has been trained on JSON. At 50 records, JSON entropy exceeds GCF by 13%. The repeated merged tokens overwhelm the model's learned parsing.

### Grammar attention collapse

At 50+ records, JSON's grammar attention collapses from 30% to 8.6%. The model stops attending to structural tokens and distributes attention uniformly across payload. It can no longer distinguish structure from content.

This is the mechanism behind the comprehension gap observed on production models: GCF 91.6% accuracy vs JSON 54.6% on 500-record payloads across 12 frontier models.

## Controlled experiments

Three runs, each building on the previous one.

### Run-001: Preliminary feasibility

Two GPT-NeoX 410M models, early corpus. Confirmed the effect (2-9x better GCF perplexity) and justified the controlled follow-up.

### Run-002: GPT-NeoX 410M controlled experiment

Two identical models, same corpus (4.5 GB), same hyperparameters, same hardware (4x A100 40GB). Only difference: the tokenizer.

| Metric | Model A (merge barriers) | Model B (standard BPE) |
|--------|-------------------------|----------------------|
| Final overall PPL | 19.4 | 19.5 |
| GCF PPL (100 records) | 9,719 | 33,703 (3.5x worse) |
| Code PPL (Python) | 543 | 2,686 (4.9x worse) |
| Wikipedia PPL | 1,029 | 1,033 (identical) |
| Delimiter-majority heads (raw >50%) | 105 / 384 (27%) | 23 / 384 (6%) |
| Delimiter heads (excess-score 0.15) | 50 | 3 (non-functional) |
| Per-token delimiter loss | 6.10 | 14.81 (2.4x harder) |

Model A wins 11/11 format categories, 8/8 sizes, 5/5 adversarial tests. The advantage scales monotonically from 2.1x at 3 records to 5.3x at 100 records.

### Run-003: Llama 410M architecture independence

Same design on a different architecture: Llama 410M (RoPE, GQA 4:1, SwiGLU, RMSNorm). 40,000 steps (vs 20,000 for NeoX).

| Metric | Llama A (merge barriers) | Llama B (standard BPE) |
|--------|-------------------------|----------------------|
| GCF PPL | 15,166 | 152,264 (10x worse) |
| Code PPL | 341 | 2,652 (7.8x worse) |
| Delimiter heads (excess-score 0.15) | 66 | 35 (functional) |

The B0 finding: standard-BPE Llama develops 35 *functional* delimiter heads (ablation confirms 53.8% GCF PPL drop), while standard-BPE NeoX develops only 3 non-functional ones. GQA's shared KV projections provide implicit structural priors that enable partial delimiter specialization even without merge barriers. This is the most surprising result of the study.

## The causal proof

### Head identification: excess scores

A head is "delimiter-specialized" if its excess delimiter attention (raw attention minus base rate) exceeds threshold 0.15, averaged across 4 probing texts (GCF generic, GCF graph, JSON, YAML). This corrects for the base-rate problem: JSON has 76% delimiter positions, so a uniform-attention head scores 0.76 raw. Without correction, head counts are inflated (168 vs corrected 50-66).

### 18-phase ablation (NeoX) + 12-phase ablation (Llama)

Zero-ablation: deep copy the model, zero the output projection weights for selected heads, measure per-format PPL, discard the copy. Every delimiter ablation paired with a random-head control.

### The causal hierarchy

The experiments reveal a four-layer hierarchy:

**Layer 1 (tokenizer, root cause).** Clean delimiters vs corrupted. The only variable in the controlled experiment. Everything else flows from this.

**Layer 2 (whole model, first-order effect).** Better embeddings (69% more cohesive delimiter clusters), better per-token prediction (2.1-2.4x delimiter advantage), lower attention entropy. These are properties of the entire model. Ablating 50-66 heads does not change them.

**Layer 3 (specialized heads, second-order effect).** 50-66 delimiter heads emerge and are causally necessary for format-level comprehension:

| Test | Key result |
|------|-----------|
| Necessity | Ablating delimiter heads hurts GCF +59%; ablating random heads helps -36%. Opposite directions. |
| Sufficiency | 50 delimiter heads alone (13% of model) beat the full 384-head model on structured data. |
| Layer-wise | Late layers causal on NeoX (+63%), middle layers on Llama (+20%). Not pattern matching; structural reasoning. |
| Format-adversarial | JSON *improves* when delimiter heads are removed (-37%). The heads trust structural boundaries, which is harmful when boundaries are corrupted. |

**Layer 4 (cross-format transfer, third-order effect).** Delimiter heads generalize to 8 of 9 unseen formats. Average degradation when removed: +44.7% on NeoX, +21.4% on Llama. Transfer spans every delimiter style: commas (CSV), parentheses (SQL), braces (Protobuf), pipes (Markdown tables), equals signs (INI).

### GQA effects

Every difference between NeoX and Llama results traces to GQA's shared KV projections: smaller A/B baseline ratio (10x vs 46x), earlier layer distribution (middle vs late), weaker per-head ablation effects, B0 developing functional heads. None indicate the mechanism fails; they indicate the ablation methodology needs adaptation for GQA. KV-group ablation (zeroing all 4 query heads per KV head) was developed as the GQA-equivalent methodology.

## What this means

### For tokenizer designers

Merge barriers are a zero-cost improvement. Identical natural language performance, 3-46x better structured data and code comprehension, and a model that develops concentrated structural attention heads. No measured downside. Architecture-independent.

### For model providers

Every model retrain is an opportunity to adopt merge barriers. The tokenizer change is 16 lines of pre-tokenization config. Some providers may already be doing this inadvertently: Claude's tokenizer has 3 quote+letter entries, Gemma has 2.

### For mechanistic interpretability researchers

This is the first controlled experiment connecting a pre-training condition to post-training internal organization. The methodology (excess-score identification, paired random controls, KV-group ablation for GQA) is applicable to studying other training conditions that might influence head specialization.

## Reproducing

Everything needed to verify the paper's claims is public:

| Resource | Location |
|----------|----------|
| Paper (PDF + Markdown) | [github.com/blackwell-systems/merge-barriers/paper](https://github.com/blackwell-systems/merge-barriers/tree/main/paper) |
| 23 eval/ablation scripts | [github.com/blackwell-systems/merge-barriers](https://github.com/blackwell-systems/merge-barriers) |
| 86 result files (JSON + logs) | [github.com/blackwell-systems/merge-barriers/runs](https://github.com/blackwell-systems/merge-barriers/tree/main/runs) |
| 4 model checkpoints + 2 tokenizers | [huggingface.co/blackwell-systems/merge-barriers](https://huggingface.co/blackwell-systems/merge-barriers) |
| 39 charts + 5 generator scripts | [github.com/blackwell-systems/merge-barriers/charts](https://github.com/blackwell-systems/merge-barriers/tree/main/charts) |
| 43-tokenizer analysis scripts | [github.com/blackwell-systems/gcf/eval](https://github.com/blackwell-systems/gcf/tree/main/eval) |

The training pipeline (corpus construction, training scripts, infrastructure) is not public.

## Next steps

- **Run-004 (Llama 1.3B):** Confirm the mechanism scales. Same corpus, same tokenizers, larger model. Single RTX 6000 Ada, ~$40-55.
- **Run-005 (3B):** Three scale points (410M, 1.3B, 3B) produce a scaling law for head specialization.
- **Vocabulary size experiment:** Test at 32K and 128K in addition to 64K.
- **Direct comprehension eval:** Run the GCF comprehension eval on the merge-barrier models themselves (requires 1.3B+ to answer questions).
