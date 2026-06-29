# Run 003: Llama-Style Architecture at Scale

## Hypothesis

The merge barrier mechanism is architecture-independent and scale-independent. A Llama-style model with merge barriers will develop the same delimiter head specialization observed in the GPT-NeoX 410M (run-002), and at 1.3B+ parameters the model will be large enough to demonstrate direct comprehension gains (not just PPL), closing the PPL-to-comprehension gap.

## What This Addresses

Run-002 proved the mechanism at 410M scale on GPT-NeoX. Five specific limitations remain open:

1. **Architecture** (run-002 limitation): "GPT-NeoX was chosen for simplicity and reproducibility. Production models use Llama-style architectures (RoPE, GQA, SwiGLU, RMSNorm). The merge barrier mechanism is architecture-independent (it operates at the tokenizer level), but the specific head specialization patterns may differ on modern architectures. GQA changes the head dynamics: Llama 3.1 has 32 query heads but only 8 KV heads per layer."

2. **Context window** (run-002 limitation): "The 2048-token context window truncates JSON payloads at 50+ records. GCF fits approximately 110 records in the same window. This means the large-scale PPL advantages may be understated, and the scaling curve (2.1x to 5.3x) is partially confounded by JSON truncation at the high end."

3. **Model scale** (run-002 limitation): "410M parameters is sufficient for proving the mechanism but insufficient for production deployment. Whether the 3x PPL advantage and head specialization scale to larger models is an open question."

4. **Vocabulary size** (run-002 limitation): "Only tested at 64K vocabulary. Production models use 32K (Llama), 128K (GPT-4), or 150K (Qwen). The 16 barrier characters are fixed, but merge dynamics change with vocabulary size."

5. **PPL-to-comprehension gap** (run-002 limitation): "PPL on a 410M model measures the tokenizer's effect on learning dynamics. Comprehension scores on production models (91.2% GCF vs 53.4% JSON) measure downstream task accuracy. The connection is supported by correlation but has not been directly measured on structok-trained models at production scale."

Additionally, the generation ablation (run-002 Phase 20) was inconclusive because the 410M model at 20K steps could not generate coherent GCF in either condition (baseline validity 13%, ablated 0%). A larger model should clear this noise floor.

## Phased Rollout

Run-003 is staged to maximize value per dollar. Each stage is independently valuable, and later stages only run if earlier stages succeed.

### Stage 1: Architecture Independence (~$50-100, few hours)

Train a Llama-style 410M model. Same scale, same corpus (4.5 GB from run-002), same tokenizer (structok-64k / standard-64k). Only swap architecture to RoPE + GQA + SwiGLU + RMSNorm. If delimiter heads still emerge and ablation results hold, architecture independence is proven. This is the cheapest possible response to the reviewer objection because you change one variable.

| Model | Architecture | Tokenizer | Vocab | Params | Context |
|-------|-------------|-----------|-------|--------|---------|
| A0 | Llama 410M | structok | 64K | 410M | 2048 |
| B0 | Llama 410M | standard | 64K | 410M | 2048 |

Llama 410M config:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden size | 1024 | Same capacity as run-002 GPT-NeoX |
| Layers | 24 | Same depth |
| Query heads | 16 | Same head count |
| KV heads | 4 | GQA with 4:1 ratio |
| Head dim | 64 | 1024 / 16 |
| Intermediate size | 2816 | ~2.75x hidden, SwiGLU |
| Activation | SwiGLU | |
| Normalization | RMSNorm | Pre-norm, no bias |
| Position encoding | RoPE | Base frequency 500,000 |
| Context length | 2048 | Same as run-002 for direct comparison |
| Total params | ~410M | |

**Why 410M first:** Eliminates the scale variable. If the mechanism replicates at 410M on Llama, the only change was architecture. If it doesn't, you learn that before spending $200+ on 1.3B.

**What you get:** One line in the paper: "Results replicated on Llama-style architecture (RoPE, GQA, SwiGLU, RMSNorm)." Preemptively closes the most common reviewer objection.

**Eval:** Core ablation (necessity, sufficiency, layer-wise, cross-format transfer). Use excess-score head identification (threshold 0.10). Compare head count, ablation deltas, and transfer results to run-002.

### Stage 2 + 3: Scale + Context (~$200-500)

Train Llama-style at 1.3B with 4K context. Addresses scale, context window, and potentially comprehension gap in one training run.

| Model | Architecture | Tokenizer | Vocab | Params | Context |
|-------|-------------|-----------|-------|--------|---------|
| A1 | Llama 1.3B | structok | 64K | 1.3B | 4096 |
| B1 | Llama 1.3B | standard | 64K | 1.3B | 4096 |

This eliminates the JSON truncation confound (4K context fits ~200 records) and is large enough to potentially answer comprehension questions. If 1.3B works, 7B is nice to have but not necessary for the paper.

**Corpus decision:** The run-002 corpus (4.5 GB, ~1.1B tokens) is undersized for 1.3B. Scale to ~50 GB (~12B tokens) while preserving category ratios, or accept that the model will overfit and note it as a limitation. A 410M sanity check on the new corpus ($20) controls for the corpus change confound.

### Stage 4: Vocabulary Size (~$200-500 additional)

Train at 32K vocabulary (Llama's native size) alongside 64K. Tests whether merge barriers work with production-standard vocabulary sizes.

| Model | Architecture | Tokenizer | Vocab | Params | Context |
|-------|-------------|-----------|-------|--------|---------|
| A2 | Llama 1.3B | structok | 32K | 1.3B | 4096 |
| B2 | Llama 1.3B | standard | 32K | 1.3B | 4096 |
| A3 | Llama 1.3B | structok | 128K | 1.3B | 4096 |
| B3 | Llama 1.3B | standard | 128K | 1.3B | 4096 |

**What you can skip:** 7B training ($500-2,000) is nice to have but 1.3B with Llama architecture, 4K context, and comprehension eval results would be sufficient for a strong paper. The 7B version is for making the artifact commercially interesting, not for proving the mechanism.

### Minimum viable run-003

Llama-style 1.3B, 4K context, 64K vocab. One training run (Stage 1 + 2/3 combined), roughly $250-600. Answers architecture + scale + context objections simultaneously. Stage 1 alone ($50-100) answers architecture.

### Cost summary by stage

| Stage | What it proves | Models | Cost | Cumulative |
|-------|---------------|--------|------|------------|
| 1: Architecture | Mechanism is architecture-independent | 2 x 410M | $50-100 | $50-100 |
| 2+3: Scale + Context | Mechanism scales, eliminates truncation confound | 2 x 1.3B | $200-500 | $250-600 |
| 4: Vocabulary | Works at 32K/128K, not just 64K | 4 x 1.3B | $200-500 | $450-1,100 |
| Optional: 7B | Production-tier model, commercial viability | 2 x 7B | $500-2,000 | $950-3,100 |

## Architecture Configs

### Llama 1.3B

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden size | 2048 | Standard for 1.3B class |
| Layers | 24 | |
| Query heads | 32 | |
| KV heads | 8 | GQA with 4:1 ratio |
| Head dim | 64 | 2048 / 32 |
| Intermediate size | 5632 | ~2.75x hidden, rounded for SwiGLU |
| Activation | SwiGLU | Gate + up projection, then down |
| Normalization | RMSNorm | Pre-norm, no bias |
| Position encoding | RoPE | Base frequency 500,000 |
| Context length (train) | 4096 | 2x run-002, eliminates truncation confound |
| Vocab size | 32K / 64K / 128K | Three variants |
| Total params (~64K vocab) | ~1.3B | |

GQA implications for head analysis: with 32 query heads and 8 KV heads per layer, there are 768 total query heads (32 x 24 layers). The delimiter head identification should operate on query heads since those determine where attention is directed. Each group of 4 query heads shares one KV head, so ablation can target individual query heads or entire KV groups. Run-002 ablated by zeroing output projections; the same method applies here but the shared KV structure means ablating one query head leaves 3 siblings still using the same KV projection. This is a methodological difference from run-002 that needs careful handling.

### Llama 7B (stretch goal)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden size | 4096 | Standard for 7B class |
| Layers | 32 | |
| Query heads | 32 | |
| KV heads | 8 | GQA with 4:1 ratio |
| Head dim | 128 | 4096 / 32 |
| Intermediate size | 11008 | ~2.69x hidden, standard Llama |
| Activation | SwiGLU | |
| Normalization | RMSNorm | |
| Position encoding | RoPE | Base frequency 500,000 |
| Context length (train) | 8192 | Large enough for 200+ record JSON |
| Vocab size | 64K | One variant only (cost constraint) |
| Total params | ~7B | |

At 7B, the model is directly comparable to Llama 3.1 8B, Mistral 7B, and Phi-3: the same tier as the production models probed in run-002. If structok-7B develops concentrated delimiter specialization where those production models have diffuse attention, that connects the controlled experiment to the production probing observation.

## Tokenizer Training

### Merge barrier tokenizers (structok)

Train new tokenizers on the corpus as needed:
- `structok-32k.json` (32,768 vocab, same 16 barrier characters)
- `structok-64k.json` (reuse from run-002 for Stage 1; retrain on new corpus for Stages 2+)
- `structok-128k.json` (131,072 vocab, same 16 barrier characters)

Barrier characters are unchanged: `|@<>"',:;\t\n{}[]()`. The pre-tokenizer prevents any merge that crosses a barrier character boundary.

### Standard tokenizers (baseline)

Train matching standard BPE tokenizers:
- `standard-32k.json`
- `standard-64k.json` (reuse from run-002 for Stage 1; retrain for Stages 2+)
- `standard-128k.json`

ByteLevel pre-tokenizer, no merge barriers. Same training corpus and vocab size as the corresponding structok tokenizer.

### Vocabulary size considerations

At 32K (Llama 3's vocab size), fewer merge steps mean fewer opportunities for delimiter corruption, but also fewer specialized tokens for structured patterns. The merge barriers may have less impact because standard BPE corrupts fewer delimiters at lower vocab sizes.

At 128K, more merge steps create more opportunities for corruption but also more specialized tokens. The barriers prevent more merges, which costs more compression. The compression penalty of merge barriers may be larger at 128K than at 64K.

We genuinely don't know which direction this will go. That's why we test all three.

## Corpus

### Stage 1: Same corpus as run-002

Reuse the ~4.5 GB rebalanced corpus (33% FineWeb, 13% code, 14% JSON, 8% GCF, 3% YAML/CSV/TOML, 3% Wikipedia). Direct comparison with run-002 results. Same tokenizers, same data, different architecture only.

### Stages 2-4: Scaled corpus (~50 GB)

Scale the corpus 10x while preserving the same ratios. Pull additional FineWeb, code, and structured data.

For the 1.3B models, 4.5 GB is too small. The model will overfit. Scale to ~50 GB (approximately 12B tokens) while keeping the same category ratios. This is still below Chinchilla optimal (26B tokens for 1.3B params) but within a reasonable range for a proof-of-concept training run.

| Source | Size | % | Notes |
|--------|------|---|-------|
| FineWeb | 16.5 GB | 33% | Subset of FineWeb-Edu or FineWeb proper |
| Code (Go, Python, TS, JS, Rust) | 6.5 GB | 13% | The Stack v2 subset |
| JSON | 7.0 GB | 14% | API responses, configs, package.json, GeoJSON |
| GCF | 4.0 GB | 8% | All 7 data shapes, generated at scale |
| YAML/CSV/TOML | 1.5 GB | 3% | Configs, data dumps, Kubernetes manifests |
| Natural language (Wikipedia) | 1.5 GB | 3% | Wikipedia articles |
| Mixed/other | 13.0 GB | 26% | Additional FineWeb for general capability |
| **Total** | **~50 GB** | | ~12B tokens |

The GCF proportion (8%) matches run-002. If anything, GCF should be slightly underrepresented to avoid inflating results.

**Corpus confound control:** Train one 410M pair on the new corpus as a sanity check (~$20). If results differ from run-002, the corpus change is identified as a variable.

## Training Plan

### Stage 1: Llama 410M (2 models)

| Parameter | Value |
|-----------|-------|
| Steps | 20,000 |
| Batch size | 16 (micro-batch 4 x 4 gradient accumulation) |
| Sequence length | 2048 |
| Tokens per step | 32,768 (16 x 2048) |
| Total tokens | ~655M |
| Learning rate | 6e-4 peak |
| LR schedule | Linear warmup 2000 steps, cosine decay to 6e-5 |
| Weight decay | 0.1 |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, eps=1e-8) |
| Gradient clipping | 1.0 |
| Mixed precision | bfloat16 |
| Checkpoints | Every 5,000 steps + final |

Same hyperparameters as run-002 for direct comparison.

### Stages 2-3: Llama 1.3B (2 models per vocab size)

| Parameter | Value |
|-----------|-------|
| Steps | 50,000 |
| Batch size | 32 (micro-batch 4 x 8 gradient accumulation) |
| Sequence length | 4096 |
| Tokens per step | 131,072 (32 x 4096) |
| Total tokens | ~6.5B |
| Learning rate | 3e-4 peak |
| LR schedule | Linear warmup 2000 steps, cosine decay to 3e-5 |
| Weight decay | 0.1 |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, eps=1e-8) |
| Gradient clipping | 1.0 |
| Mixed precision | bfloat16 |
| Checkpoints | Every 5,000 steps + final |

### 7B models (stretch goal)

| Parameter | Value |
|-----------|-------|
| Steps | 30,000 |
| Batch size | 16 (micro-batch 2 x 8 gradient accumulation) |
| Sequence length | 8192 |
| Tokens per step | 131,072 (16 x 8192) |
| Total tokens | ~3.9B |
| Learning rate | 1.5e-4 peak |
| LR schedule | Linear warmup 2000 steps, cosine decay to 1.5e-5 |
| Weight decay | 0.1 |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, eps=1e-8) |
| Gradient clipping | 1.0 |
| Mixed precision | bfloat16 |
| Checkpoints | Every 5,000 steps + final |

### Context extension (post-training, Stages 2+)

After training at 4096 (1.3B) or 8192 (7B), apply YaRN to extend context:
- 1.3B models: extend from 4K to 32K
- 7B models: extend from 8K to 128K

YaRN modifies the RoPE frequency scaling without retraining. Test extended-context PPL on 100, 200, and 500 record payloads.

## Evaluation Plan

### Tier 1: PPL comparison (same as run-002)

For each model pair, measure per-format PPL at multiple payload sizes:

| Size | JSON tokens (est.) | GCF tokens (est.) | Fits in 4K? | Fits in 8K? |
|------|-------------------|--------------------|-------------|-------------|
| 10 records | ~200 | ~120 | Yes | Yes |
| 50 records | ~1,000 | ~600 | Yes | Yes |
| 100 records | ~2,000 | ~1,200 | Yes | Yes |
| 200 records | ~4,000 | ~2,400 | Tight | Yes |
| 500 records | ~10,000 | ~6,000 | No (needs ext.) | Tight |

Formats: GCF generic, GCF graph, JSON, YAML, TOON, CSV, code, natural language.

### Tier 2: Direct comprehension eval (new for run-003)

Run the GCF comprehension eval suite (13 questions, 500 records) on the structok-trained models themselves:

1. Present the model with a structured data payload (GCF, JSON, or TOON)
2. Ask the 13 standard comprehension questions
3. Score accuracy

Requirements:
- 1.3B is marginal for instruction-following. Fallback: simpler eval (multiple choice, cloze-style).
- If 1.3B cannot answer reliably in either condition, the eval is inconclusive at that scale.

### Tier 3: Ablation replication (verifying architecture independence)

Repeat the core ablation experiments from run-002 on the Llama models:

1. **Head identification**: use excess-score method (threshold 0.10), not raw >50%. This corrects the base-rate problem discovered in run-002 where JSON's 75.7% delimiter density inflated head counts.
2. **Necessity test**: ablate delimiter heads, measure GCF PPL delta. Run-002 showed +59%.
3. **Sufficiency test**: keep only delimiter heads, measure GCF PPL. Run-002 showed -55.6% (50 excess-score heads).
4. **Sufficiency scaling**: test at 30, 50, 100, 200 rows. Run-002 showed gap narrows from -93pp to -41pp but never reverses.
5. **Layer-wise ablation**: confirm late-layer concentration. Expect layers 16-23 to carry the causal effect.
6. **Cross-format transfer**: test the 9 unseen formats. Run-002 showed 8/9 transfer (+50.6% average) with corrected excess-score identification. Transfer is effectively universal.
7. **Control**: ablate random non-delimiter heads as control, 5 seeds.

GQA-specific consideration: with GQA, ablation targets query heads. When ablating a query head, its siblings still share the same KV projection. If individual query head ablation shows weak effects, try ablating entire KV groups (all query heads sharing one KV head).

### Tier 4: Vocabulary size comparison (Stage 4 only)

Compare delimiter head counts, PPL ratios, and ablation deltas across the three vocab sizes:

| Metric | 32K | 64K | 128K |
|--------|-----|-----|------|
| Delimiter heads (structok) | ? | ? | ? |
| Delimiter heads (standard) | ? | ? | ? |
| GCF PPL ratio (standard/structok) | ? | ? | ? |
| Necessity delta (GCF) | ? | ? | ? |
| Compression ratio (bits/char) | ? | ? | ? |

## Success Criteria

The experiment succeeds if:

1. **Architecture independence confirmed** (Stage 1): Llama 410M with merge barriers develops delimiter head specialization. The qualitative finding (concentrated specialization in merge-barrier model, diffuse/absent in standard) should replicate. Ablation necessity and sufficiency hold.

2. **Scale independence confirmed** (Stage 2): the merge barrier advantage exists at 1.3B. The 3x PPL advantage should not disappear.

3. **Context window eliminates truncation confound** (Stage 3): the PPL scaling curve is monotonically increasing from 10 to 200+ records, without the reversal seen at 50+ records in run-002.

4. **Vocabulary size robustness** (Stage 4): the advantage exists at all three vocab sizes. Same direction, statistically significant.

5. **Direct comprehension** (if model is capable): structok model scores measurably higher than standard model on the 13-question comprehension eval.

## Failure Criteria

The experiment fails to support the thesis if:

1. **Architecture dependence**: Llama 410M with merge barriers does NOT develop delimiter head specialization, suggesting the mechanism depends on GPT-NeoX architecture specifics (separate KV projections, learned position embeddings).

2. **Scale collapse**: the 3x PPL advantage at 410M disappears at 1.3B, suggesting a small-model phenomenon.

3. **Vocabulary sensitivity**: the advantage holds at 64K but not at 32K (Llama's native size).

4. **Comprehension disconnect**: structok model has lower PPL but equal or worse comprehension scores. PPL advantage doesn't translate to downstream accuracy.

## Risk Factors

### GQA changes the head dynamics
Run-002's ablation targets individual heads with their own Q, K, V, and output projections. GQA shares K and V across query heads. Ablating one query head is a weaker intervention. May need to ablate by KV group or mask attention weights directly. This is methodologically tractable but needs validation in Stage 1 before scaling.

### 1.3B may not be large enough for comprehension eval
Models below 3B typically struggle with complex question-answering. The comprehension eval may require the 7B model. Mitigation: design a simplified eval (multiple choice, cloze) that works at 1.3B.

### Corpus scaling introduces a new variable (Stages 2+)
Scaling from 4.5 GB to 50 GB may affect results. Mitigation: preserve category ratios, train one 410M pair on new corpus as sanity check (~$20).

### YaRN extension may degrade quality
YaRN is well-tested on Llama-family models but structok is a novel tokenizer. Validate extended-context PPL on both models before using it for scaling comparisons.

### Training instability at 7B
Larger models are more prone to loss spikes. Mitigation: checkpoint every 5K steps, gradient clipping at 1.0, reduce LR to 1e-4 if divergence occurs.

## Timeline

| Stage | Duration | Cumulative | Cost |
|-------|----------|------------|------|
| Stage 1: Llama 410M pair | 1-2 days | Day 2 | $50-100 |
| Stage 1 eval + ablation | 1 day | Day 3 | (included) |
| Decision gate: proceed to Stage 2? | | Day 3 | |
| Corpus assembly + tokenizer training | 2-3 days | Day 6 | |
| Pre-tokenization (50 GB) | 1 day | Day 7 | |
| Stage 2+3: Llama 1.3B pair | 4-5 days | Day 12 | $200-500 |
| Stage 2+3 eval + ablation | 2-3 days | Day 15 | (included) |
| Decision gate: proceed to Stage 4? | | Day 15 | |
| Stage 4: 32K + 128K vocab pairs | 8-10 days | Day 25 | $200-500 |
| Stage 4 eval | 2-3 days | Day 28 | (included) |
| YaRN extension + extended-context eval | 1-2 days | Day 30 | |
| Analysis and writeup | 2-3 days | Day 33 | |

Stage 1 alone: 3 days, $50-100.
Stages 1-3: 15 days, $250-600.
Full run-003: ~33 days, $450-1,100.
With 7B stretch goal: add 7-10 days, $500-2,000.

## Deliverables

### Stage 1
- 2 trained models (Llama 410M) with checkpoints archived to R2
- Ablation results confirming architecture independence
- One-line paper addition: "Results replicated on Llama-style architecture"

### Stages 2-3
- 2 trained models (Llama 1.3B, 4K context) with checkpoints
- PPL comparison table across payload sizes (no truncation confound)
- Ablation replication at 1.3B scale
- Direct comprehension scores (if model is capable)
- Extended-context scaling curve via YaRN

### Stage 4
- 4 additional models (32K and 128K vocab) with checkpoints
- Vocabulary size comparison table
- Evidence for/against "works at production vocab sizes"

### Optional: 7B
- 2 models at production-tier scale
- Comprehension eval (likely to succeed at this scale)
- Commercial viability assessment

## Methodological Updates from Run-002

The following methodological improvements from run-002 should be applied to all run-003 experiments:

1. **Excess-score head identification** (threshold 0.10): raw >50% threshold inflates head counts when delimiter-dense formats (JSON at 75.7%) are used for probing. Excess score = raw attention minus base rate. Run-002 found 50 heads with this method vs 76 with raw.

2. **Universal cross-format transfer**: run-002 originally reported "6 of 9 unseen formats transfer." With corrected identification, 8 of 9 transfer (+50.6% average). The selectivity was an artifact of head identification instability. Run-003 should expect universal transfer and report any non-transfer as a genuine finding.

3. **Sufficiency scaling interpretation**: 13% of heads (50/384) outperform the full model on structured data. Gap narrows with payload size (93pp at 30 rows to 41pp at 200 rows) because the holistic improvement matters more than specialization at scale. Run-003 at 4K context should test whether this gap continues to narrow or stabilizes.

4. **Adversarial robustness**: delimiter heads contribute to structural corruption detection (~56% reduction under ablation) but are not the sole mechanism. Test this on Llama architecture to see if the partial effect replicates.
