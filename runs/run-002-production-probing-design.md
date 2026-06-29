# Production Model Probing: Experiment Design

## Question

Do production models develop delimiter-specialized attention heads? If so, does the head count correlate with comprehension accuracy from the GCF eval suite?

## Hypothesis

Models with more delimiter-specialized heads will score higher on structured data comprehension. Models with fewer will score lower. If the correlation holds, attention head specialization is a predictive mechanism for structured data comprehension across production models.

## Models

Selected based on (1) available comprehension eval data, (2) fits on RTX 3090 24GB, (3) available on HF without gating:

| Model | HF repo | Parameters | VRAM (fp16) | Comprehension score |
|-------|---------|------------|-------------|-------------------|
| Llama 3.1 8B | NousResearch/Meta-Llama-3.1-8B | 8B | ~16GB | 65.4% GCF (eval) |
| Mistral 7B v0.3 | mistralai/Mistral-7B-v0.3 | 7B | ~14GB | 64.6% GCF (eval) |
| Qwen 2.5 7B | Qwen/Qwen2.5-7B | 7B | ~14GB | (no eval data yet) |
| Phi-2 | microsoft/phi-2 | 2.7B | ~6GB | (no eval data yet) |

Llama and Mistral have comprehension eval data from the flatten experiment (generic profile, 500 orders). If probing reveals a correlation, we can then run comprehension evals on Qwen and Phi-2 to validate the prediction.

## Method

For each model:

1. Load model with `output_attentions=True` capability
2. Feed the same structured data test texts used in ablation experiments (GCF generic 50 rows, GCF graph 20 symbols, JSON 50 records)
3. For each attention head, compute delimiter attention score (same method as ablation: fraction of attention going to delimiter token positions, averaged across query positions and test texts)
4. Count delimiter-majority heads (>50% threshold)
5. Record top 20 heads by score
6. Compare head counts to comprehension eval scores

### VRAM management

- Use fp16 precision
- Cap input sequences at 512 tokens for head identification (sufficient for scoring, reduces attention map memory)
- Load one model at a time, unload before loading next
- If a model still OOMs at 512 tokens, fall back to 256

### Delimiter classification

Use the same BARRIER_CHARS set as all other ablation experiments. The production models use different tokenizers, so the delimiter positions will differ, but the characters are the same.

### Test data

Same texts as ablation v2/v3/v4:
- GCF generic: 50-row order table
- GCF graph: 20-symbol code graph
- JSON: 50-record order array

Using trained-format texts only for head identification (consistent with the methodological lesson from v4).

## Output

### Console log
Full progress output with per-model head counts, top heads, timing.

### Structured JSON
```json
{
  "metadata": { ... },
  "models": [
    {
      "name": "Llama-3.1-8B",
      "hf_repo": "NousResearch/Meta-Llama-3.1-8B",
      "parameters": 8000000000,
      "n_layers": 32,
      "n_heads": 32,
      "total_heads": 1024,
      "delimiter_heads_50pct": N,
      "delimiter_heads_60pct": N,
      "delimiter_heads_40pct": N,
      "top_20_heads": [...],
      "avg_delimiter_score": X,
      "comprehension_score_gcf": 65.4,
      "comprehension_score_json": 58.3
    },
    ...
  ],
  "correlation": {
    "head_count_vs_gcf_accuracy": R,
    "head_pct_vs_gcf_accuracy": R
  }
}
```

### Storage
- Local: `runs/run-002-production-probing-results.json`, `runs/run-002-production-probing-log.txt`
- R2: `logs/run-002-ablation/production-probing-results.json`, `logs/run-002-ablation/production-probing-log.txt`

## Controls

- **Our models as reference points**: include Model A (merge barriers, 70 heads) and Model B (standard BPE, 3 heads) in the same analysis. These are the endpoints: a model designed for delimiter specialization and one that isn't.
- **Multiple thresholds**: report head counts at 40%, 50%, 60% to show the full distribution, not just a single cutoff.

## Comprehension scores (from eval suite)

| Model | GCF avg | JSON avg | Source |
|-------|---------|----------|--------|
| Model A (structok 410M) | N/A (PPL only) | N/A | run-002 |
| Model B (standard 410M) | N/A (PPL only) | N/A | run-002 |
| Llama 3.1 8B | 65.4% | 58.3% | flatten experiment |
| Mistral 7B v0.3 | 64.6% | 63.6% | flatten experiment, Mistral Small proxy |
| Llama 3.3 70B | 84.6% | 61.5% | flatten experiment |
| Gemini 2.5 Flash | 95.0% | 74.0% | comprehension eval |
| GPT-5.5 | 100% | 100% | comprehension eval |

Note: Llama 3.3 70B and Gemini 2.5 Flash don't fit on 24GB. We can only probe models that fit. The correlation will be computed from the models we can probe plus our controlled models.

## Hardware

- GPU: NVIDIA GeForce RTX 3090 (24GB)
- Instance: Vast.ai, Spain (same instance as ablation experiments)
- Estimated runtime: ~10 min per model (load + 3 forward passes with attention), ~40 min total

## Risk

- **OOM**: 8B models at fp16 with attention outputs may exceed 24GB. Mitigated by capping sequence length at 512 and using gradient checkpointing if available.
- **Architecture differences**: different models use different attention implementations (GQA, MQA, MHA). The head counting method works on MHA (each head has independent Q/K/V) but may need adaptation for GQA (grouped queries). Llama 3.1 uses GQA with 8 KV heads and 32 query heads. We count query heads since those determine attention patterns.
- **Tokenizer differences**: each model has its own tokenizer, so "delimiter positions" will differ. This is expected and correct: we're measuring how each model's own tokenizer handles delimiters.

## Reproduction

```bash
python3 eval_production_probing.py \
  --models "NousResearch/Meta-Llama-3.1-8B,mistralai/Mistral-7B-v0.3,microsoft/phi-2" \
  --output /root/logs/run-002-production-probing-results.json \
  2>&1 | tee /root/logs/run-002-production-probing-log.txt
```
