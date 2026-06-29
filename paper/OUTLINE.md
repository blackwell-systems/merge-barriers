# Merge Barriers in BPE Tokenization: How Tokenizer Design Causally Determines Attention Head Specialization

## Paper Structure

### Abstract (~200 words)
BPE tokenizers corrupt structural delimiters by merging them with adjacent content. We introduce merge barriers (16 characters forbidden from merging) and prove through controlled experiments on two architectures (GPT-NeoX 410M and Llama 410M) that this single tokenizer change causally determines attention head specialization. The merge-barrier model develops 50-66 delimiter-specialized heads that are necessary for structured data comprehension (+59% degradation when removed on NeoX), sufficient (13% of heads outperform the full model), and transfer universally to 8 of 9 unseen formats. We establish a four-layer causal hierarchy: tokenizer (root cause) > whole-model improvement > specialized heads > cross-format transfer. Architecture independence confirmed: the mechanism replicates on Llama (RoPE, GQA, SwiGLU) with GQA moderating the effect magnitude. JSON attention saturation replicates exactly (99.1% on NeoX, 99.2% on Llama). This is the first controlled experiment connecting tokenizer design to attention head organization.

### 1. Introduction (~1 page)
- The problem: structured data in LLM context windows
- JSON's tokenizer problem (30% quote merge rate across 43 tokenizers)
- Merge barriers: a one-line tokenizer fix
- What this paper proves: the causal chain from tokenizer to comprehension
- **Chart: hero chart (100% standard, 91% adversarial)**

### 2. Background (~1 page)
- BPE tokenization and the merge process
- Attention head specialization (prior work: Voita 2019, Olsson 2022)
- The gap: nobody has studied what training conditions cause specialization
- Merge barriers: preventing merges across 16 delimiter characters

### 3. The 43-Tokenizer Study (~1.5 pages)
- Methodology: scanning all 94 ASCII characters across 43 production tokenizers
- Delimiter merge rates: pipe 0.47%, quote 8.17%, tab 32.91%
- Field name corruption: "name": merges 30% of the time
- JSON overhead: 17,001 tokens at 1000 rows vs GCF's 11
- **Chart: delimiter merge rates (3 bars)**
- **Chart: structural equivalence heatmap (43 tokenizers x 3 formats)**
- Table: all 16 barrier character merge rates

### 4. Experimental Setup (~1.5 pages)

#### 4.1 Controlled Experiment Design
- Two identical models, same corpus, same hyperparameters, only tokenizer differs
- Run-002: GPT-NeoX 410M, 20K steps, 2048 context
- Run-003: Llama 410M (RoPE, GQA 4:1, SwiGLU, RMSNorm), 40K steps, 2048 context

#### 4.2 Corpus
- 4.5 GB rebalanced: 33% FineWeb, 13% code, 14% JSON, 8% GCF, 3% YAML/CSV/TOML

#### 4.3 Head Identification
- Excess-score method (raw attention minus base rate, threshold 0.10-0.15)
- Corrects for base-rate problem (JSON at 75.7% delimiter positions)
- Table: head counts by threshold and architecture

#### 4.4 Ablation Method
- Zero output projection weights for selected heads
- GQA consideration: per-query-head vs KV-group ablation

### 5. Results: The Causal Hierarchy (~4 pages)

#### 5.1 Layer 1: Tokenizer (Root Cause)
- Baselines: NeoX structok 46x better on GCF, Llama 10x
- Per-format PPL comparison table (both architectures)
- NL unaffected on both (0.7x, confirming zero NL cost)

#### 5.2 Layer 2: Whole-Model Improvement (First-Order)
- Per-token loss: 2.4x delimiter prediction advantage (NeoX), 2.1x (Llama)
- Ablation doesn't spike loss to Model B levels (null result, both architectures)
- Attention entropy: whole-model property, not head-controlled
- **Chart: (table only, no chart needed)**

#### 5.3 Layer 3: Specialized Heads (Second-Order)
- Necessity: +59% GCF degradation on NeoX when heads removed
- Sufficiency: 13% of heads outperform full model (NeoX). GQA weakens this on Llama.
- Late-layer concentration on NeoX (+63%). Early/middle on Llama (GQA effect).
- **Chart: sufficiency scaling (NeoX, 4 sizes)**
- **Chart: attention heatmap GCF vs JSON (showing saturation)**
- Table: top 5 heads attention flow comparison

#### 5.4 Layer 4: Cross-Format Transfer (Third-Order)
- 8 of 9 unseen formats transfer on NeoX (corrected identification)
- 7-8 of 9 on Llama (threshold dependent)
- Transfer selectivity resolved as artifactual (7 hypotheses tested)
- **Chart: cross-format transfer table (both architectures side by side)**

### 6. Architecture Independence (~2 pages)
- Run-003 methodology: same corpus, same tokenizers, Llama architecture
- What replicates: heads emerge, transfer works, JSON saturates, head count narrows
- What differs: GQA effects (layer distribution, ablation magnitude, B0 functional heads)
- KV-group ablation: delimiter-vs-random gap is the correct metric under GQA
- B0 has 35 functional heads: GQA enables partial specialization even without merge barriers
- **Table: NeoX vs Llama side-by-side comparison (15 findings)**

### 7. The Format-Adversarial Mechanism (~1 page)
- JSON improves when delimiter heads removed (NeoX: -37%)
- Heads trust delimiters, which hurts on corrupted formats
- L17H1/L6H0 saturation: 99% of content attention to delimiters on JSON
- Pipe becomes adversarial in wrapping layouts (-54%)
- **Chart: attention heatmap (already in Section 5.3)**

### 8. Discussion (~1 page)
- The causal hierarchy as a framework for understanding head specialization
- What the paper should claim vs what it shouldn't
- Implications for tokenizer design in production models
- The GQA finding: shared KV projections enable partial specialization

### 9. Limitations (~0.5 pages)
- 410M scale only (1.3B and 7B are future work)
- 2048 context (JSON truncation confound at 50+ records)
- PPL-to-comprehension gap (not directly measured on structok models)
- Head identification threshold sensitivity
- GQA ablation methodology differences

### 10. Related Work (~0.5 pages)
- Observational head analysis (Voita 2019, Clark 2019)
- Induction heads (Olsson 2022, closest analog)
- Tokenizer quality studies (no prior work connecting to head specialization)
- Our contribution: causal, not descriptive; bridges tokenizer and interp communities

### 11. Conclusion (~0.5 pages)
- Tokenizer design causally determines attention head organization
- Architecture-independent mechanism (proven on GPT-NeoX and Llama)
- Practical implication: protect structural delimiters during BPE training

---

## Chart Assignments

### Main paper (8-10 figures)
1. **Hero chart** (Section 1): 100% standard, 91% adversarial comprehension
2. **Delimiter merge rates** (Section 3): pipe 0.47% vs quote 8.17% vs tab 32.91%
3. **Structural equivalence heatmap** (Section 3): 43 tokenizers x 3 formats
4. **Sufficiency scaling** (Section 5.3): 13% of heads beat full model, gap narrows
5. **Attention heatmap GCF vs JSON** (Section 5.3): L17H1 saturation visual
6. **Adversarial robustness** (Section 7): detection under ablation
7. **Cross-format transfer comparison** (Section 5.4): NeoX vs Llama side by side (new chart needed)

### Supplementary
- Accuracy by model (per-model bars)
- Error magnitude scatter
- Token cost vs accuracy
- Advantage by tier
- Generation validity
- TOON heatmap
- Distance label problem
- Failure types (pie + stacked)
- Comprehension variance box plot
- Generic accuracy by model
- Scale test
- Token efficiency 15 datasets
- Vocab merge entries
- Field merge rates
- Overhead scaling
- Three-format savings (43 tokenizers)
- Savings stability bands
- ASCII adversarial surface
- Embedding cohesion ablation
- All structok training charts (delimiter heads, per-token loss, grammar attention, etc.)
- Emergence timeline (both architectures)
- Production probing
- Transplant controls
- Structural pattern test

---

## Data tables for paper

### Table 1: Baselines (both architectures)
| Format | NeoX A | NeoX B | Ratio | Llama A | Llama B | Ratio |

### Table 2: Ablation necessity (NeoX, primary)
| Format | Baseline | Ablated | Delta | Control | Delta |

### Table 3: Cross-format transfer (both architectures)
| Format | NeoX (50 heads) | Llama 0.15 (56 heads) | Transfer? |

### Table 4: Architecture comparison (15 findings)
| Finding | NeoX | Llama | Same? |

### Table 5: Head identification threshold sensitivity
| Threshold | NeoX heads | Llama heads | GCF delta |

### Table 6: Barrier character merge rates (16 chars)
| Character | Merge rate | Merge words |
