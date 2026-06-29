# Experiment Index

All training runs and evaluations for the structok project.

## Run-001: Initial Training

Single model (structok tokenizer), proof of concept.

| File | Description |
|------|-------------|
| `run-001-structok-64k-410m.md` | Experiment design and results |
| `run-001-eval-results.json` | Structured eval data |
| `run-001-eval-log.txt` | Console output |

## Run-002: Controlled Experiment

Two identical GPT-NeoX 410M models, same corpus, same hyperparameters. Only variable: tokenizer (merge barriers vs standard BPE). 20,000 steps each.

### Training

| File | Description |
|------|-------------|
| `run-002-experiment-design.md` | Experimental design and methodology |
| `run-002-results.md` | Complete results writeup with all findings |
| `run-002-standard-training-log.json` | Model B per-step training metrics |

### Core eval (PPL comparison)

| File | Description |
|------|-------------|
| `run-002-eval-results.json` | Structured PPL data (GCF, JSON, code, NL) |
| `run-002-eval-log.txt` | Console output |

### Extended eval (more formats, scaling)

| File | Description |
|------|-------------|
| `run-002-eval-extended.json` | Extended format coverage (YAML, CSV, TOML, code languages) |
| `run-002-eval-extended-log.txt` | Console output |

### Deep eval (generation, delimiter accuracy, adversarial)

| File | Description |
|------|-------------|
| `run-002-deep-eval.json` | Generation quality, delimiter accuracy, scaling curves |
| `run-002-deep-eval-log.txt` | Console output |

### Attention analysis

| File | Description |
|------|-------------|
| `run-002-attention-results.json` | Entropy, grammar attention share, collapse analysis |
| `run-002-attention-log.txt` | Console output |

### Mechanistic analysis

| File | Description |
|------|-------------|
| `run-002-mechanistic-results.json` | Per-token loss, head specialization, embedding space, cross-format transfer |
| `run-002-mechanistic-log.txt` | Console output |

### Head ablation (causal analysis)

Tests whether delimiter-specialized heads are causally responsible for structured data comprehension.

| File | Description |
|------|-------------|
| `run-002-ablation.md` | Full methodology, results, and interpretation |
| `run-002-ablation-results.json` | Model A ablation data (5 control seeds) |
| `run-002-ablation-log.txt` | Model A console output |
| `run-002-ablation-modelb-results.json` | Model B ablation data (5 control seeds) |
| `run-002-ablation-modelb-log.txt` | Model B console output |
| `run-002-ablation-full-results.json` | Complete 8-phase experiment (ablation + reverse + layer-wise + attention) |
| `run-002-ablation-full-log.txt` | Complete 8-phase console output |
| `run-002-ablation-v3-results.json` | Follow-up: cross-format transfer + head ranking + threshold |
| `run-002-ablation-v3-log.txt` | Follow-up console output |
| `run-002-ablation-v4-results.json` | Extended cross-format transfer (9 unseen formats, raw threshold, 76 heads) |
| `run-002-ablation-v4-log.txt` | Extended transfer console output |
| `run-002-ablation-v4-excess-results.json` | Extended transfer rerun (excess-score identification, 40 heads, 8/9 transfer) |
| `run-002-ablation-v4-excess-log.txt` | Extended transfer rerun console output |
| `run-002-transplant-results.json` | Transplant v1: A's delimiter heads into B |
| `run-002-transplant-log.txt` | Transplant v1 console output |
| `run-002-transplant-v2-results.json` | Transplant v2: 5 controls (random, reverse, cross-position, subsets) |
| `run-002-transplant-v2-log.txt` | Transplant v2 console output |
| `run-002-bootstrap-results.json` | Bootstrap confidence intervals (5 seeds) |
| `run-002-bootstrap-log.txt` | Bootstrap console output |
| `run-002-scaling-ablation-results.json` | Scaling: causal effect vs payload size (10-200 rows) |
| `run-002-scaling-ablation-log.txt` | Scaling console output |
| `run-002-production-probing-design.md` | Probing experiment design |
| `run-002-production-probing-results.json` | Probing v1 (inconclusive) |
| `run-002-production-probing-log.txt` | Probing v1 console output |
| `run-002-production-probing-v2-results.json` | Probing v2: Model A, Model B, Phi-2, Mistral |
| `run-002-production-probing-v2-log.txt` | Probing v2 console output |
| `run-002-llama-probing-results.json` | Llama 3.1 8B probing |
| `run-002-llama-probing-log.txt` | Llama probing console output |
| `run-002-probing-gemma-qwen-results.json` | Gemma 2B + Qwen 7B probing |
| `run-002-probing-gemma-qwen-log.txt` | Gemma/Qwen console output |
| `run-002-structural-pattern-test-design.md` | Structural pattern test design |
| `run-002-structural-pattern-results.json` | Structural pattern test (5 formats, character vs pattern) |
| `run-002-structural-pattern-log.txt` | Structural pattern console output |
| `run-002-ablation-connections-results.json` | Per-token loss + entropy under ablation (null: not head-controlled) |
| `run-002-ablation-connections-log.txt` | Connections console output |
| `run-002-emergence-design.md` | Emergence timing experiment design |
| `run-002-emergence-results.json` | Emergence timing (steps 3500-5000, run 2) |
| `run-002-emergence-log.txt` | Emergence timing console output |
| `run-002-generation-ablation-results.json` | Generation under ablation (inconclusive at 410M) |
| `run-002-generation-ablation-log.txt` | Generation ablation console output |
| `run-002-remaining-ablation-results.json` | Combined experiments: embedding space (#19), adversarial robustness (#21), sufficiency scaling (#22) |
| `run-002-remaining-ablation-log.txt` | Combined experiments console output |

**Key findings:**

*Causal hierarchy (root cause to effects):*
- **Layer 1: Tokenizer** (root cause). Clean delimiters vs corrupted. The only variable in the controlled experiment.
- **Layer 2: Whole model** (first-order). 2.4x delimiter prediction advantage, lower entropy, 3x PPL. Distributed across all parameters, not head-controlled.
- **Layer 3: Specialized heads** (second-order). 70 causal heads, necessary and sufficient, concentrated in late layers.
- **Layer 4: Cross-format transfer** (third-order). 6/9 unseen formats, selectivity partially understood.

*Strong evidence (ablation, within-model):*
1. Delimiter heads ARE causal: removing them hurts GCF (+59%) while random removal helps (-36%)
2. Delimiter heads are sufficient: 70 heads alone beat all 384 on structured data (-44% GCF, -88% YAML)
3. Late layers (16-23) are where reasoning happens: +63% GCF degradation
4. Model B's 3 delimiter heads are non-functional
5. Cross-format transfer: 8 of 9 unseen formats degrade (+50.6% avg) with corrected excess-score identification. Original "6 of 9" was an artifact of head identification instability.
6. Top 5 heads account for 45% of total degradation
7. Bootstrap: +16.7pp delimiter-random gap, 2.0% std, all 5 seeds consistent
8. Format-adversarial mechanism: JSON improves when heads removed (-37%). L17H1 sends 99% of content attention to delimiters on JSON (0.9% to content). Heads are blind to JSON content.

*Weaker/nuanced evidence:*
9. Transplant: A's heads help B (-81% GCF) but random A heads also help (-70%). Holistic model improvement, not portable modules.
10. Scaling: gap does NOT widen with scale on 410M model (capacity limitation at 2048 context). Gap reverses at large payloads.
11. Production probing: qualitatively different attention profiles (concentrated vs diffuse) but NOT a predictive metric due to tokenizer confound. Exploratory only.
12. Emergence: heads emerge immediately (~107 at step 1000), narrow to 60-70 by step 5000. No phase transition. Concentration increases 37% to 54%. Consistent across two independent random seeds.
13. Delimiter density does NOT predict cross-format transfer (r=0.026).
14. Merge rate does NOT predict cross-format transfer. SQL uses `(` at 21.6% and transfers; TOON at 20.0% does not.
15. Merge word count does NOT predict transfer. SQL uses `(` at 2,353 words and transfers; TOON at 1,238 does not.
16. Structural pattern test: pipe becomes adversarial in wrapping layout (-54%), tab transfers in all layouts (+32% to +123%). Transfer driven by learned character-specific priors conflicting with context, not structural or tokenizer properties. Original TOON non-transfer may have been a test-data artifact.
17. Per-token loss + entropy under ablation: delimiter heads do NOT directly control per-token loss (2.4x) or attention entropy. These are holistic model properties, not head-specific (null results). Consistent with transplant finding.
18. Generation under ablation: inconclusive at 410M scale.
19. Embedding space under ablation: delimiter cohesion +1.7%, ratio -5.5%. Null result, consistent with #17/#18.
20. Adversarial robustness: ablation reduces structural corruption detection by 56% (3/4 types). Wrong-delimiter detection retained. Heads contribute to but don't solely control error detection.
21. Sufficiency holds at scale: 50 delimiter heads beat 50 random at 30-200 rows. Gap narrows (93pp->41pp) but never reverses.
22. Head identification base-rate bug: raw >50% threshold inflated count from 50 to 168 when JSON (75.7% delimiter positions) was used for probing. Fixed with excess scores (raw - base_rate, threshold 0.10).

## Run-003 Stage 1: Llama 410M Architecture Independence

Two identical Llama 410M models (RoPE, GQA 4:1, SwiGLU, RMSNorm), same corpus as run-002, same tokenizers. Only variable: tokenizer (merge barriers vs standard BPE). 40,000 steps each.

### Training

| File | Description |
|------|-------------|
| `run-003-experiment-design.md` | Full experiment design with phased rollout, architecture configs, cost estimates |
| `run-003-llama-structok-training-log.json` | Model A0 per-step training metrics (JSONL) |
| `run-003-llama-structok-training-log-phase1.txt` | Model A0 console output (steps 0-20K) |
| `run-003-llama-structok-training-log-phase2.txt` | Model A0 console output (steps 20K-40K resume) |

### Cross-format transfer (threshold sweep)

| File | Description |
|------|-------------|
| `run-003-ablation-v4-t010-results.json` | Transfer at excess threshold 0.10 (85 heads, 7/9) |
| `run-003-ablation-v4-t010-log.txt` | Console output |
| `run-003-ablation-v4-t015-results.json` | Transfer at excess threshold 0.15 (56 heads, 8/9) |
| `run-003-ablation-v4-t015-log.txt` | Console output |
| `run-003-ablation-v4-t020-results.json` | Transfer at excess threshold 0.20 (31 heads, 7/9) |
| `run-003-ablation-v4-t020-log.txt` | Console output |

### Combined ablation (layer-wise, sufficiency, ranking, attention, emergence)

| File | Description |
|------|-------------|
| `run-003-llama-ablation-results.json` | 5 experiments: layer-wise, sufficiency scaling, head ranking, attention patterns, emergence |
| `run-003-llama-ablation-log.txt` | Console output |
| `run-003-emergence-results.json` | Emergence timing across 6 checkpoints (steps 15K-40K) |
| `run-003-emergence-log.txt` | Console output |

### Remaining ablation (embedding, adversarial, sufficiency with both models)

| File | Description |
|------|-------------|
| `run-003-remaining-ablation-results.json` | Embedding space (#19), adversarial robustness (#21), sufficiency (#22) |
| `run-003-remaining-ablation-log.txt` | Console output |

### A vs B controlled comparison

| File | Description |
|------|-------------|
| `run-003-ablation-v2-results.json` | Full v2 ablation: baselines, progressive ablation, 5-seed control, reverse, layer-wise, attention |
| `run-003-ablation-v2-log.txt` | Console output |

### B0 ablation + KV-group ablation

| File | Description |
|------|-------------|
| `run-003-b0-kvgroup-results.json` | B0 head identification (35 functional heads) + KV-group ablation on A0 |
| `run-003-b0-kvgroup-log.txt` | Console output |

### Per-token loss + bootstrap

| File | Description |
|------|-------------|
| `run-003-connections-results.json` | Per-token loss and attention entropy under ablation |
| `run-003-connections-log.txt` | Console output |
| `run-003-bootstrap-results.json` | Bootstrap confidence intervals (5 seeds) |
| `run-003-bootstrap-log.txt` | Console output |

**Key findings (architecture independence):**

*What replicates across GPT-NeoX and Llama:*
1. Delimiter heads emerge: 56-85 on Llama (NeoX: 50), threshold dependent
2. Cross-format transfer: 7-8/9 unseen formats on Llama (NeoX: 8/9)
3. JSON attention saturation: L6H0 at 99.2% delimiter attention (NeoX L17H1: 99.1%)
4. Head ranking concentration: top 5 = 36% (NeoX: 45%)
5. Head count narrowing during training: 71->49 (NeoX: 107->61)
6. Embedding space null result: whole-model property on both
7. Per-token loss null result: ablation doesn't spike loss to Model B levels on either
8. NL unaffected by ablation on both

*What differs (GQA effects, not mechanism failure):*
9. A/B baseline ratio: 10x on Llama (NeoX: 46x). GQA moderates advantage.
10. Layer distribution: early/middle layers causal on Llama (NeoX: late). GQA pushes structural processing earlier.
11. Trained-format ablation directions differ: GCF helped by ablation on Llama (NeoX: hurt). Shared KV projections weaken per-query-head ablation.
12. B0 has 35 FUNCTIONAL delimiter heads (NeoX B: 3 non-functional). GQA gives standard tokenizer partial structural capability.
13. KV-group ablation: delimiter-vs-random gap is +48pp GCF, +164pp JSON. Causal signal clear as gap.
14. Concentration lower: 13-15% on Llama (NeoX: 37-54%). GQA distributes specialization.
15. Entropy change under ablation larger: +5.7% on Llama (NeoX: +1.0%)

## Checkpoints and Tokenizers

| Location | Path | Size |
|----------|------|------|
| Hugging Face | `blackwell-systems/structok-checkpoints/checkpoint-a.pt` | 5.0 GB | Run-002 NeoX structok |
| Hugging Face | `blackwell-systems/structok-checkpoints/checkpoint-b.pt` | 5.0 GB | Run-002 NeoX standard |
| Hugging Face | `blackwell-systems/structok-checkpoints/run-003-llama-a.pt` | 4.6 GB | Run-003 Llama structok |
| Hugging Face | `blackwell-systems/structok-checkpoints/run-003-llama-b.pt` | 4.6 GB | Run-003 Llama standard |
| Hugging Face | `blackwell-systems/structok-checkpoints/structok-64k.json` | 4.2 MB | |
| Hugging Face | `blackwell-systems/structok-checkpoints/standard-64k.json` | 4.5 MB | |
| R2 | `archive/run-002-structok-64k-410m/checkpoint.pt` | 5.0 GB | |
| R2 | `archive/run-002-standard-64k-410m/checkpoint.pt` | 5.0 GB | |
| R2 | `archive/run-001-structok-64k-410m/step-14500/checkpoint.pt` | 5.0 GB | |
| R2 | `checkpoints/run-003-llama-structok/step-{15K-40K}/` | 4.6 GB each | 6 checkpoints |
| R2 | `checkpoints/run-003-llama-standard/step-{5K-40K}/` | 4.6 GB each | 8 checkpoints |
| R2 | `tokenizers/structok-64k.json` | 4.2 MB | |
| R2 | `tokenizers/standard-64k.json` | 4.5 MB | |

## R2 Storage Layout (`structok-training` bucket)

```
archive/
  run-001-structok-64k-410m/
    checkpoint.pt (step 14500)     5.0 GB
    config.json                    713 B
    structok-64k.json              4.2 MB
    tokens.bin                     6.7 GB
    run-manifest.md                4 KB
  run-002-structok-64k-410m/
    checkpoint.pt (step 20000)     5.0 GB
    training_log.json              49 KB
  run-002-standard-64k-410m/
    checkpoint.pt (step 20000)     5.0 GB
    training_log.json              49 KB

checkpoints/                       (working copies, same as archive)
  run-002-structok/step-20000/
  run-002-standard/step-20000/

corpus/                            (training data)
  fineweb-v2.txt                   2.0 GB
  fineweb.txt                      4.7 GB
  json_data.txt                    0.8 GB
  gcf_examples-v2.txt              0.5 GB
  gcf_examples.txt                 0.1 GB
  code_typescript.txt              0.3 GB
  code_go.txt                      0.3 GB
  code_python.txt                  0.1 GB
  code_javascript.txt              33 MB
  code_rust.txt                    33 MB
  natural_language-v2.txt          0.2 GB
  natural_language.txt             0.1 GB
  yaml_data.txt                    17 MB
  yml_data.txt                     17 MB
  csv_data.txt                     17 MB

eval/test_data/                    (eval fixtures at 5 sizes)
  test-{5,10,20,50,100}-records.{gcf,json}

logs/
  run-002-structok/training_log.json    49 KB
  run-002-standard/training_log.json    49 KB
  run-002-ablation/
    ablation-log.txt                    6 KB
    ablation-results.json               14 KB
    ablation-modelb-log.txt             6 KB
    ablation-modelb-results.json        10 KB
    ablation-full-log.txt               9 KB
    ablation-full-results.json          19 KB
    ablation-v3-log.txt                 5 KB
    ablation-v3-results.json            14 KB
    ablation-v4-transfer-log.txt        4 KB
    ablation-v4-transfer-results.json   3 KB
    bootstrap-log.txt
    bootstrap-results.json
    scaling-ablation-log.txt
    scaling-ablation-results.json
    transplant-log.txt
    transplant-results.json
    transplant-v2-log.txt
    transplant-v2-results.json
    llama-probing-log.txt
    llama-probing-results.json
    probing-gemma-qwen-log.txt
    probing-gemma-qwen-results.json
    production-probing-log.txt
    production-probing-results.json
    production-probing-v2-log.txt
    production-probing-v2-results.json
    emergence-results.json
    emergence-log.txt
    generation-ablation-log.txt
    generation-ablation-results.json
    structural-pattern-log.txt
    structural-pattern-results.json
    ablation-connections-log.txt
    ablation-connections-results.json

logs/
  run-003-llama-structok/training_log.json
  run-003-llama-standard/training_log.json
  run-003-ablation/
    run-003-ablation-v4-t010-*.{json,txt}
    run-003-ablation-v4-t015-*.{json,txt}
    run-003-ablation-v4-t020-*.{json,txt}
    run-003-llama-ablation-*.{json,txt}
    run-003-emergence-*.{json,txt}
    run-003-remaining-ablation-*.{json,txt}
    run-003-ablation-v2-*.{json,txt}
    run-003-b0-kvgroup-*.{json,txt}
    run-003-connections-*.{json,txt}
    run-003-bootstrap-*.{json,txt}

tokenizers/
  structok-64k.json                4.2 MB
  standard-64k.json                4.5 MB

tokens/                            (tokenized corpus)
  structok-64k-v2.bin              4.7 GB
  standard-64k-v2.bin              4.7 GB
```

## Eval Scripts

| Script | Description |
|--------|-------------|
| `eval_model.py` | Core PPL comparison with `--compare-checkpoint` and `--extended` |
| `eval_deep.py` | Generation quality, delimiter accuracy, scaling, adversarial |
| `eval_attention.py` | Attention entropy, grammar share, collapse |
| `eval_mechanistic.py` | Per-token loss, head specialization, embedding, cross-format transfer |
| `eval_ablation.py` | Head ablation v1 (superseded by v2) |
| `eval_ablation_v2.py` | Head ablation v2: per-format PPL, reverse ablation, layer-wise, attention patterns |
| `eval_ablation_v3.py` | Head ablation v3: cross-format transfer, single-head ranking, threshold sensitivity |
| `eval_ablation_v4.py` | Head ablation v4: extended cross-format transfer (9 unseen formats, raw threshold) |
| `eval_ablation_v4_excess.py` | Head ablation v4 rerun: excess-score identification, confirms 8/9 universal transfer |
| `eval_transplant.py` | Head transplant v1: graft Model A's delimiter heads into Model B |
| `eval_transplant_v2.py` | Head transplant v2: comprehensive controls (random, reverse, cross-position, subsets, unseen formats) |
| `eval_bootstrap.py` | Bootstrap confidence intervals: 5 seeds, delimiter vs random gap statistics |
| `eval_scaling_ablation.py` | Scaling ablation: does the causal effect strengthen with payload size (10-200 rows)? |
| `eval_production_probing.py` | Production probing v1 (inconclusive, threshold counting doesn't transfer across model sizes) |
| `eval_production_probing_v2.py` | Production probing v2: concentration ratio, top-K, distribution shape (GCF-only) |
| `eval_emergence.py` | Emergence timing: probe checkpoints for when delimiter heads appear during training |
| `eval_structural_pattern.py` | Structural pattern test: is transfer driven by character or layout pattern? |
| `eval_ablation_connections.py` | Connect ablation to per-token loss (2.4x) and attention entropy (30% to 8.6%) |
| `eval_generation_ablation.py` | Generation quality under head ablation (inconclusive at 410M scale) |
| `eval_remaining_ablation.py` | Combined: embedding space under ablation (#19), adversarial robustness (#21), sufficiency scaling at 100/200 rows (#22). Uses excess-score head identification. |
| `generate_charts.py` | Chart generation from run-002 data |
| `charts/generate_charts.py` | Ablation chart generation (6 PNGs) |
| `charts/density_vs_delta.py` | Delimiter density vs transfer scatter plot (r=0.026) |
| `charts/attention_heatmap.py` | Attention flow heatmap: GCF vs JSON content attention |
| `charts/generate_remaining_charts.py` | Sufficiency scaling, adversarial robustness, embedding cohesion charts |
| `eval_transfer_analysis.py` | Transfer selectivity hypothesis testing (boundary clarity, positional, spacing) |
| `eval_llama_ablation.py` | Run-003: layer-wise, sufficiency, head ranking, attention patterns, emergence on Llama |
| `eval_llama_b0_and_kvgroup.py` | Run-003: B0 head identification + KV-group ablation on A0 |
