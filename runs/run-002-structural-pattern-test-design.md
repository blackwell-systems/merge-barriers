# Structural Pattern Transfer Test: Design

## Question

Is cross-format transfer determined by the delimiter character or by the structural pattern? Specifically: do delimiter heads generalize to any format with flat-field-separator layout, regardless of which character is used?

## Hypothesis

The delimiter heads learned "field boundary in a flat layout" from GCF. They generalize to formats that match this structural pattern (flat field separation), not formats that use a different paradigm (tag wrapping, whitespace separation). The specific delimiter character doesn't matter; the structural pattern does.

## Test Design

Hold the delimiter character constant (tab), vary the structural pattern. If pattern matters more than character, the GCF-like tab format should transfer while TOON-style tab should not.

### Format A: Tab-as-field-separator (GCF-like layout)

```
## orders [5]{orderId,customer,status,total}
ORD-00001	Alice Chen	pending	29.97
ORD-00002	Bob Smith	processing	42.47
ORD-00003	Carla Rodriguez	shipped	54.97
ORD-00004	David Park	delivered	67.47
ORD-00005	Eva Johansson	cancelled	79.97
```

This is literally GCF with tabs instead of pipes. Same header with inline schema, same flat rows, same structure. Only the delimiter changed.

### Format B: TOON-style tab (header + rows)

```
orderId	customer	status	total
ORD-00001	Alice Chen	pending	29.97
ORD-00002	Bob Smith	processing	42.47
ORD-00003	Carla Rodriguez	shipped	54.97
ORD-00004	David Park	delivered	67.47
ORD-00005	Eva Johansson	cancelled	79.97
```

Standard TSV: header row declares columns, data rows follow. No inline schema, no section marker.

### Format C: Tab-as-wrapper (XML-like but with tabs)

```
	orderId	ORD-00001	
	customer	Alice Chen	
	status	pending	
	total	29.97	

	orderId	ORD-00002	
	customer	Bob Smith	
	status	processing	
	total	42.47	
```

Tab "wraps" each key-value pair, one per line. Structurally closer to XML than to GCF.

### Format D: Pipe-as-wrapper (XML-like but with pipes)

```
|orderId|ORD-00001|
|customer|Alice Chen|
|status|pending|
|total|29.97|

|orderId|ORD-00002|
|customer|Bob Smith|
|status|processing|
|total|42.47|
```

Pipe used in a wrapping pattern, not a flat-field-separator pattern. If the character matters, this should transfer (pipe always transfers). If the pattern matters, this should NOT transfer (wrapping pattern, not flat separation).

### Control: Standard GCF (pipe, flat)

```
## orders [5]{orderId,customer,status,total}
ORD-00001|Alice Chen|pending|29.97
ORD-00002|Bob Smith|processing|42.47
```

Known to be affected by delimiter head ablation.

## Predictions

| Format | Character | Pattern | Prediction if character matters | Prediction if pattern matters |
|--------|-----------|---------|-------------------------------|------------------------------|
| A (tab-as-GCF) | tab | flat separator | NO transfer (tab) | YES transfer (flat) |
| B (TOON-style) | tab | header+rows | NO transfer (tab) | NO transfer (different pattern) |
| C (tab-wrapper) | tab | wrapping | NO transfer (tab) | NO transfer (wrapping) |
| D (pipe-wrapper) | pipe | wrapping | YES transfer (pipe) | NO transfer (wrapping) |
| GCF (control) | pipe | flat separator | YES (known) | YES (known) |

**The decisive test is Format A vs Format B.** Same character (tab), different patterns. If A transfers and B doesn't, pattern wins. If neither transfers, character wins. If both transfer, something else is going on.

**Format D is the confirmation test.** If pipe-as-wrapper doesn't transfer, the heads aren't attending to pipe specifically; they're attending to the flat-field-separator pattern that pipe usually represents.

## Method

1. Generate 30 rows of each format
2. Load Model A (merge barriers)
3. Measure baseline PPL on each format
4. Ablate delimiter heads (same 70-76 heads as previous experiments)
5. Measure ablated PPL on each format
6. Compare deltas: which formats degrade (transfer) vs improve (no transfer)?

## Hardware

RTX 3090 instance (still alive). No training needed, just inference with ablation.

## Expected runtime

~5 minutes. One model load, one deep copy + ablation, 5 format measurements each.
