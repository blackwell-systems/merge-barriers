# Delimiter Head Emergence Timing: Experiment Design

## Question

When during training do delimiter-specialized attention heads emerge? Is the specialization gradual or a sudden phase transition?

## Method

Train a GPT-NeoX 410M model with merge barriers (same config as run-002 Model A) for 5,000 steps, saving a checkpoint every 500 steps. At each of the 10 checkpoints, probe for delimiter heads using the ablation v2 method (multi-text averaged, GCF-only, >50% excess threshold). Plot the head count and concentration ratio over training steps.

### Training config

Identical to run-002 Model A:
- Architecture: GPT-NeoX 410M (24 layers, 16 heads/layer, 384 total)
- Tokenizer: structok-64k.json (merge barriers)
- Corpus: structok-64k-v2.bin (pre-tokenized, 4.7GB)
- Learning rate: 3e-4 (flat)
- Batch size: 32 effective (adjust gradient accumulation for available GPUs)
- Precision: fp16
- Steps: 5,000 (25% of run-002's 20,000)
- Checkpoint: every 500 steps (10 checkpoints)

### Probing at each checkpoint

At each of the 10 checkpoints:
1. Load the checkpoint
2. Feed GCF test data (same 2 texts as ablation v2)
3. Count delimiter-majority heads (>50% threshold, multi-text averaged)
4. Compute concentration ratio and top-10 excess
5. Record per-layer head distribution

### Expected outcomes

**Gradual emergence:** head count increases linearly from 0 to ~70 over 5K steps. Each training step adds a small amount of specialization. This would suggest delimiter specialization is a continuous optimization process.

**Phase transition:** head count stays near 0 for the first N steps, then jumps to ~70 over a few hundred steps. This would suggest a critical point where the model suddenly "discovers" that delimiter tokens are useful for prediction.

**Early plateau:** heads emerge quickly (by step 1000) and stabilize. This would suggest the tokenizer's delimiter isolation is immediately beneficial and the model exploits it early.

### Hardware

Best option: 1x A100 40GB (~45 min training + ~20 min probing = ~1 hour total, ~$0.75)
Fallback: 1x RTX 3090 24GB (~2-3 hours training + ~20 min probing, ~$0.30-0.40)

### Data requirements

- `tokens/structok-64k-v2.bin` from R2 (4.7GB)
- `structok-64k.json` tokenizer (already on instance and HF)
- `train_model.py` training script
- `eval_ablation_v2.py` or custom probing script for checkpoint analysis

### Run command

```bash
# Download corpus
# (presigned URL or HF download)

# Train with checkpoints every 500 steps
python train_model.py \
  --size 410m \
  --tokenizer structok-64k.json \
  --data /root/data/ \
  --steps 5000 \
  --checkpoint-every 500 \
  --r2-prefix run-002-emergence

# Probe each checkpoint
python eval_emergence.py \
  --checkpoint-dir /root/checkpoints/ \
  --tokenizer structok-64k.json \
  --output emergence-results.json
```

### Output

- Per-checkpoint: delimiter head count, concentration ratio, top-10 excess, per-layer distribution
- Timeline plot: heads vs training step
- Phase transition detection: is there a step where head count jumps by >20 in 500 steps?

### Storage

- Local: `runs/run-002-emergence-results.json`, `runs/run-002-emergence-log.txt`
- R2: `logs/run-002-ablation/emergence-results.json`, checkpoints at `checkpoints/run-002-emergence/step-{N}/`
- Intermediate checkpoints can be deleted after probing (only the results matter, not the weights)
