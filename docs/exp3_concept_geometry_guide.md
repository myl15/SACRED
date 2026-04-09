# Understanding the Concept Direction Geometry Figures (Exp3)

These four figure types are produced by the **Concept Direction Geometry** section added
to Experiment 3.  They use the PCA reading vectors extracted from the contrastive-pair
difference matrices saved by Experiment 1.  Unlike the causal intervention experiments
(Exp2, Exp4), these figures are purely **representational** — they ask where and how
strongly a concept is geometrically encoded in NLLB's residual stream, not whether
subtracting a vector changes model output.

All figures are saved to `results/figures/` and come in two variants: one per domain
(`sacred`, `kinship`). A machine-readable summary is also saved to
`results/json/exp3_concept_geometry_summary.json` for paper tables/appendix.

---

## 1. `concept_direction_alignment_{domain}.png`

### What it shows
A line chart with one curve per language pair (e.g., `arb ↔ zho`, `eng ↔ spa`).
The x-axis is encoder layer (0–23); the y-axis is the **absolute cosine similarity**
between the PCA concept directions extracted independently from each language in the
pair at that layer.

The PCA concept direction for a language is computed by running PCA on the matrix of
`(positive_activation − negative_activation)` differences across all contrastive pairs
for that language, then taking the first principal component and averaging across
concepts within the domain.  The absolute value is taken because PCA is sign-ambiguous
and sign correction is a heuristic — a value of 1.0 means the two directions are
parallel (possibly pointing in opposite senses), and a value of 0.0 means they are
orthogonal.

### How to read it
- **High values (≥ 0.7) at a layer** mean both languages have extracted concept
  directions that point nearly the same way.  This is the geometric prerequisite for
  cross-lingual causal transfer: if the directions are aligned, then subtracting
  language A's vector will also partially subtract language B's concept encoding.
- **The layer where the curves peak** is where cross-lingual concept geometry is
  strongest.  If this peak coincides with the intervention layers (10–15), that
  corroborates the choice of those layers for Exp2/Exp4.
- **Curves rising through the middle layers then falling** would indicate the model
  builds a language-neutral concept representation mid-network before diverging into
  language-specific generation representations in later layers — consistent with the
  interlingua hypothesis.
- **Curves that stay flat and low** (≤ 0.3) indicate that the two languages never
  share a concept direction, which would explain why cross-lingual causal transfer
  fails for those pairs in Exp4.
- **Comparing sacred vs. kinship panels** will tell you whether one domain has more
  geometrically unified concept representations than the other.

### What good vs. bad looks like
| Pattern | Interpretation |
|---------|---------------|
| All pairs peak near layer 12, values ≥ 0.7 | Strong shared concept subspace; cross-lingual transfer expected |
| English pairs (eng ↔ *) higher than non-English pairs | English-centric encoding; model routes through English representation |
| Flat curves ≈ 0.3–0.4 across all layers | Concept does not have a stable shared direction; mean vectors may be noise |
| Kinship lower than sacred throughout | Kinship concept is encoded more language-specifically than sacred |

---

## 2. `projection_consistency_{domain}.png`

### What it shows
A line chart with **one bold curve per language** (4 lines total: eng, arb, zho, spa)
plus a **shaded ±1 standard deviation band** representing variation across concepts
within that language.  The x-axis is encoder layer; the y-axis is the **fraction of
contrastive pairs whose difference vector projects positively onto the PCA concept
direction** for that language and layer, averaged over all concepts in the domain.

For a given layer, each contrastive pair contributes a difference vector
`(pos_activation − neg_activation)`.  The PCA direction is estimated from those same
differences.  If the direction genuinely captures the concept, every pair's difference
should point in the same half-space — yielding a consistency score near 1.0.  If some
pairs point the "wrong" way, the concept signal is noisy or multi-directional.

### How to read it
- **Score near 1.0** at a layer: the PCA direction is a reliable, consistent probe for
  that concept in that language at that layer.  Nearly all pairs agree.
- **Score near 0.5** at a layer: the PCA direction is no better than random for that
  slice.  The concept may not be linearly encoded, or the pairs may be too noisy.
- **Rising trajectory** through middle layers: the concept representation becomes more
  coherent and linearly separable as depth increases, then potentially degrades in
  late layers as the model shifts to generation.
- **Drops at layer 0**: expected — the embedding layer is largely token-identity, not
  semantic concept structure.
- **Differences across concepts within a domain**: some kinship roles (e.g., `mother`,
  `father`) may be more consistently encoded than others (e.g., `cousin`, `in-law`),
  which would show up as individual curves deviating from the cluster.

### Important caveat: circularity
The projection consistency metric has a known circularity problem: the PCA direction
is estimated from the same pairs whose projections are then measured.  PCA by
construction maximises variance, so most pairs will naturally project positively even
if the direction has no generalisable concept signal.  Treat high scores here as a
necessary — but not sufficient — condition for a good probe.  The **linear probe
accuracy** figure (Section 5) is the circularity-free replacement.

### Relationship to the PCA causal failure
The kinship-PCA vectors in Exp2 were causally inert.  This figure explains why: if
the consistency scores for kinship concepts are low (≈ 0.5–0.6) in the intervention
layers, then the PCA direction is not reliably capturing the concept direction — it is
instead latching onto sentence-structure variance that happens to dominate PC1.  High
consistency (≥ 0.85) is expected due to circularity; what matters is whether the
**linear probe accuracy** (Section 5) also stays high on held-out pairs.

### What good vs. bad looks like
| Pattern | Interpretation |
|---------|---------------|
| All curves ≥ 0.85 at layers 10–15 | PCA is a reliable concept probe in the intervention window |
| Kinship curves lower than sacred | Kinship concept is noisier / less linearly separable per-language |
| Some concepts cluster high, others cluster low | Domain is heterogeneous; pooling concepts may wash out signal |
| Score drops sharply after layer 18 | Late layers shift away from concept encoding toward language-specific generation |

---

## 3. `pc1_explained_variance_{domain}.png`

### What it shows
A line chart with one curve per language × concept combination.  The x-axis is encoder
layer; the y-axis is the **fraction of variance in the contrastive-pair difference
matrix explained by the first principal component** (PC1 explained variance ratio).

Concretely: for each language × concept × layer, we have a matrix of shape
`[n_pairs × 1024]` whose rows are `(pos − neg)` difference vectors.  PCA decomposes
this matrix.  PC1's explained variance ratio is `eigenvalue_1 / sum(all eigenvalues)`.
A high ratio means one dominant direction captures most of the pair-level variance —
i.e., the concept is encoded cleanly in a single linear direction.  A low ratio means
the variance is spread across many directions, indicating a more complex or noisy
representation.

### How to read it
- **High ratio (≥ 0.4)**: the concept has a strong, dominant linear direction at this
  layer.  The PCA reading vector is a reliable estimate of that direction, and a linear
  probe would be powerful.
- **Low ratio (≤ 0.15)**: variance is spread across many components; there is no single
  dominant concept direction.  This does NOT mean the concept is unrepresented — it may
  be encoded across multiple dimensions — but it does mean a single PCA vector is a poor
  summary.
- **Peak layer**: the layer where PC1 explained variance is highest is where the concept
  is most "linearly concentrated."  Comparing this to the intervention layers (10–15) is
  informative: ideally they align.
- **Comparing domains**: if sacred consistently has higher PC1 variance than kinship
  across layers, sacred is more cleanly linearly encoded, which is consistent with the
  stronger causal intervention effects observed for sacred in Exp2.

### Relationship to the other figures
This figure is complementary to the projection consistency figure.  High PC1 variance +
high projection consistency together indicate a clean, reliable linear concept direction.
High PC1 variance but low projection consistency would be unusual and would suggest the
dominant variance direction is not actually the concept direction (e.g., it might be
sentence length).  Low PC1 variance but high consistency is unlikely by construction.

### What good vs. bad looks like
| Pattern | Interpretation |
|---------|---------------|
| Peak at layers 10–15, ratio ≥ 0.3 | Strong linear concept encoding at intervention layers |
| Sacred ratio > kinship ratio | Sacred is more cleanly linearly encoded |
| Very flat trajectory (≈ 0.1 everywhere) | Concept not linearly concentrated in any layer |
| Sharp peak at a specific layer | Possible "bottleneck" layer where concept is most compressed |

---

## 4. `cross_lingual_projection_transfer_{domain}_layer{k}.png`

### What it shows
An annotated **NxN heatmap** (N = number of languages, currently 4: eng, arb, zho, spa)
evaluated at a single encoder layer (default: layer 12, the middle of the intervention
window).

**Row i** = language whose contrastive-pair differences are being projected.
**Column j** = language whose PCA concept direction is used as the probe.
**Cell [i, j]** = fraction of language i's contrastive-pair difference vectors that
project positively onto language j's PCA concept direction.

The diagonal [i, i] is the **self-consistency** of each language's PCA direction — how
well language i's own direction separates its own pairs.  Off-diagonal cells are
**cross-lingual generalization**: does language j's direction separate language i's
pairs?

### How to read it
- **Diagonal values**: these are the same as the projection consistency figure at
  layer 12.  Values ≥ 0.85 indicate reliable self-probing.
- **High off-diagonal values (≥ 0.7)**: language j's concept direction works well on
  language i's data.  This is a necessary (though not sufficient) condition for
  causal cross-lingual transfer.  If direction alignment is high but causal transfer
  in Exp4 is low, the bottleneck is not the shared direction but something else (e.g.,
  the encoder's sensitivity to the direction at that layer).
- **Compare to Exp4 transfer matrix**: the causal transfer matrix and this heatmap
  should broadly agree.  Language pairs with high geometric generalization (this figure)
  but low causal transfer (Exp4) suggest the concept direction is represented similarly
  but is not the causally active component — a finding worth reporting.
- **English row/column vs. others**: if the English row (projecting other languages'
  pairs onto the English direction) is uniformly high, that supports the English-pivot
  hypothesis at the representational level: all languages' concept pairs align with the
  English concept direction.
- **Asymmetry**: the matrix need not be symmetric.  `[arb, zho]` (Arabic pairs onto
  Chinese direction) can differ from `[zho, arb]` (Chinese pairs onto Arabic direction).
  Large asymmetries indicate that concept encoding is not equally universal.

### Reading the matrix as a whole
| Pattern | Interpretation |
|---------|---------------|
| All off-diagonal values ≥ 0.7 | Nearly universal concept subspace; strong geometric basis for cross-lingual transfer |
| English column highest | All languages' pairs align with English direction — English-pivot representation |
| Block structure (Romance high, Arabic-Chinese low) | Language-family-specific encoding rather than universal |
| Off-diagonal << diagonal everywhere | Each language encodes concept in a private direction; no shared subspace |
| Off-diagonal ≈ diagonal everywhere | Concept direction is essentially the same across all languages |

### Relationship to Exp4 causal transfer matrix
This figure is the **representational prerequisite check** for Exp4.  The logic is:

```
High geometric generalization [this figure]
    ↓ necessary condition for
High causal transfer [Exp4]
    ↓ both together imply
Shared, causally active concept direction
```

If Exp4 shows high transfer but this figure shows low geometric generalization, the
causal mechanism is not a shared linear direction — it might be a shared nonlinear
feature or a translation-path artifact.  If this figure shows high generalization but
Exp4 shows low transfer, the direction is shared at the representation level but is not
causally manipulable via simple subtraction (e.g., it is not linearly separable in the
causal sense).

---

## 5. `linear_probe_accuracy_{domain}.png`

### What it shows
A line chart with one bold curve per language (4 lines) plus a shaded ±1 standard
deviation band across concepts, identical in layout to the projection consistency chart.
The y-axis is **5-fold cross-validated probe accuracy** (0.5 = chance, 1.0 = perfect).

The probe works as follows: for each language × concept × layer, the n contrastive-pair
difference vectors are split into 5 folds. Each fold withholds 20% of pairs as a test
set. A probe direction is estimated as the **mean of the training diff rows only** — the
held-out pairs never influence the direction. The fraction of held-out pairs that project
positively onto this direction is the fold accuracy. The reported score is the mean
across all 5 folds.

### Why this is more trustworthy than projection consistency
Projection consistency is circular: the PCA direction is fit to the same data it is
evaluated on, guaranteeing high scores by construction. This chart breaks that circularity
by using a train/test split. A score significantly above 0.5 here is a **genuine finding**
— it means the concept direction estimated from some pairs reliably predicts the direction
of unseen pairs at that layer.

### How to read it
- **Score near 1.0 on held-out data**: the concept direction is stable and replicable
  across pairs — the layer robustly encodes the concept linearly.
- **Score near 0.5**: the direction learned from training pairs does not generalize to
  held-out pairs. The concept may not be linearly separable at this layer, or the
  sample size is too small for the direction to stabilise (n=15–20 pairs per concept
  is small; expect some noise).
- **Layer where the score peaks**: the most reliable layer for linear concept encoding.
  If this matches the intervention window (layers 10–15), it corroborates the layer
  selection for Exp2 and Exp4.
- **Gap between this chart and projection consistency**: a large gap (high consistency
  but low CV accuracy) is evidence of circularity inflating the consistency scores.
  A small gap (both high) indicates the PCA direction is genuinely stable.
- **Comparing languages**: if English has higher CV accuracy than Arabic or Chinese, the
  concept is more cleanly linearly encoded in English encoder representations — consistent
  with English being the implicit pivot language.

### What good vs. bad looks like

| Pattern | Interpretation |
|---------|---------------|
| CV accuracy ≥ 0.80 at layers 10–15 | Concept reliably linearly encoded in intervention window |
| CV accuracy ≈ 0.55–0.65 throughout | Weak linear signal; small sample size or multi-directional encoding |
| CV accuracy drops after layer 18 | Late layers shift away from concept representation |
| English higher than non-English | English-centric concept encoding |
| Large domain gap (sacred > kinship or vice versa) | One domain has more stable linear representation |

---

## 6. `cross_lingual_probe_transfer_{domain}_layer{k}.png`

### What it shows
An annotated **NxN heatmap** at a single encoder layer (default: layer 12), where:

- **Row i** = language whose diff rows were used to **train** the probe direction
- **Column j** = language whose diff rows were used to **test** the probe direction
- **Cell [i, j]** = fraction of language j's contrastive-pair differences that project
  positively onto the probe direction trained on language i's data

The **diagonal [i, i]** is the k-fold CV self-accuracy from the linear probe accuracy
chart — each language tested against its own held-out pairs.

The **off-diagonal [i, j]** is genuine cross-lingual generalization: the probe is trained
entirely on language i's data and evaluated on language j's data, which it has never seen.
There is no circularity in the off-diagonal entries.

### Why this figure matters most
This is the **representation-space analogue of the Exp4 causal transfer matrix**. Exp4
asks: "Can language i's concept vector causally suppress the concept in language j's
translations?" This figure asks: "Does language i's concept direction even separate
language j's concept-positive from concept-negative activations?" The former is a
causal question; this is a geometric prerequisite.

Placing this heatmap alongside the Exp4 causal matrix enables a direct comparison:

```
High probe transfer [this figure] + High causal transfer [Exp4]
    → Shared linear direction that is also causally active (strongest result)

High probe transfer + Low causal transfer
    → Shared direction but not the causally manipulable one
      (possibly a non-causal correlate, or alpha needs tuning)

Low probe transfer + High causal transfer
    → Causal transfer does not require a shared linear direction
      (unexpected; check concept detection validity)

Low probe transfer + Low causal transfer
    → No shared direction, no causal transfer; languages encode concept differently
```

### How to read it

- **High off-diagonal values (≥ 0.70)**: language i's concept direction generalizes
  to language j — strong evidence of a shared concept subspace.
- **Values near 0.50**: language i's direction is no better than random on language j's
  pairs. The two languages do not share a concept direction at this layer.
- **English row elevated**: if the English probe direction works well for all other
  languages (all entries in the English row are high), English's concept direction is
  a "universal" probe — consistent with the English-pivot hypothesis.
- **English column elevated**: if all languages' probes work well on English test data,
  English representations are easy to separate along any language's concept direction —
  a different form of English centrality.
- **Asymmetry [i,j] ≠ [j,i]**: language i's probe may generalize to language j but not
  vice versa. This indicates directional asymmetry in concept encoding — one language's
  representation is "easier" to probe with a foreign-language direction.
- **Language-family clustering**: if Romance languages (spa) have higher mutual transfer
  than either has with Arabic or Chinese, concept encoding follows typological proximity.

### What good vs. bad looks like

| Pattern | Interpretation |
|---------|---------------|
| All off-diagonal ≥ 0.70 | Near-universal concept subspace; strongest support for cross-lingual hypothesis |
| English row highest | English concept direction is most generalisable — pivot-consistent |
| Diagonal ≈ off-diagonal | Concept direction is essentially language-independent |
| Off-diagonal ≈ 0.50 everywhere | No shared concept geometry; cross-lingual transfer in Exp4 is not explained by shared direction |
| Block structure (e.g., spa↔eng high, arb↔zho high, cross-block low) | Language-family-specific encoding |

---

## Summary: reading all six figures together

| # | Figure | Question answered | Circularity-free? | Scale |
|---|--------|-------------------|--------------------|-------|
| 1 | Direction alignment | Do languages share the same concept direction? | Yes | Per language-pair, per layer |
| 2 | Projection consistency | Do all pairs agree with the PCA direction? | **No** — use Fig 5 instead | Per language, per layer |
| 3 | PC1 explained variance | Is the concept signal concentrated in one direction? | Yes | Per language×concept, per layer |
| 4 | Cross-lingual projection transfer | Does language j's PCA direction separate language i's pairs? | Off-diagonal only | NxN at one layer |
| 5 | Linear probe accuracy | Does the probe direction generalise to held-out pairs? | **Yes** (k-fold CV) | Per language, per layer |
| 6 | Cross-lingual probe transfer | Does language i's probe direction generalise to language j's data? | **Yes** (off-diagonal fully held-out) | NxN at one layer |

Figures 2 and 4 are useful diagnostic aids. Figures 5 and 6 are the rigorous primary
results. Figure 2 should broadly agree with Figure 5 in shape — if it shows much higher
scores, that gap quantifies the circularity inflation. Figures 4 and 6 should agree in
off-diagonal structure; divergence would mean the PCA and mean-differencing directions
are capturing different aspects of the geometry.

The ideal result pattern for supporting the paper's central claims:

1. **Direction alignment (Fig 1)** peaks at layers 10–15, all pairs ≥ 0.7 → shared subspace in the intervention window
2. **Linear probe accuracy (Fig 5)** ≥ 0.75 at those layers → direction is stable across held-out pairs
3. **PC1 explained variance (Fig 3)** highest at those layers → concept signal is most linearly concentrated there
4. **Cross-lingual probe transfer (Fig 6)** off-diagonal ≥ 0.7 at layer 12 → shared direction generalises across languages
5. **Exp4 causal matrix** agrees with Fig 6 off-diagonal structure → the shared direction is also causally active

Deviations are themselves informative findings: they reveal which domains, language
pairs, or layers fail to show universal concept encoding, and explain the asymmetries
and failures observed in the causal experiments.
