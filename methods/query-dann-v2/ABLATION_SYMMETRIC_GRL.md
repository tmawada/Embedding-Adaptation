# Symmetric GRL ablation: does making the adversary balanced fix the harm?

## Question

In `../query_dann/ABLATION_GAMMA0.md`, the adversarial term was mildly harmful: with
`gamma=0.1` the discriminator was never fooled (dev accuracy 0.5 → ~0.95) and a
`gamma=0` control scored +0.0047 dev / +0.0005 test nDCG@10 better.

Hypothesis: the asymmetry is the cause — the domain target was the cached constant
`z_form`, so GRL reached the adapter only from the informal branch (at `gamma/2 = 0.05`),
while D was trained on both domains at full strength. This variant passes formal queries
through the adapter too, so both branches carry gradient back to A.

## Setup

Identical to `../query_dann/ABLATION_GAMMA0.md`: BGE-M3, `--adapter zeroinit`,
`--bottleneck 128`, `--discriminator sanitized`, `mu=0.5`, `eta=10`, `tau=0.05`,
1,350 steps, `lr=3e-5`, `batch=64`, cosine schedule with 10% warmup, `grad_clip=1`,
`weight_decay=0.01`, checkpoint by best dev informal nDCG@10. Seeds 42 / 84 / 123 /
256 / 512, `gamma ∈ {0.1, 0}`. Only change: `dom_tgt = D(GRL(A(z_form)), alpha)`.

Sanity check: the `gamma=0` runs are numerically **identical** to the original
`gamma=0` runs (no gradient flows through the domain branch there), which confirms the
change is isolated to the adversarial path.

## Discriminator behaviour

With symmetric GRL, D accuracy no longer climbs monotonically to ~0.95. It oscillates
(e.g. seed 42: 0.50 → 0.86 at step 300 → 0.53 at step 600 → 0.81 at step 1350), ending
around **0.81–0.85**. This is a genuinely competitive minimax game, though D still is
not fooled to 0.5 — so the result must **not** be described as domain invariance.

## Result — the sign flips

Best **dev** informal nDCG@10:

| seed | v2 γ=0.1 | v2 γ=0 | diff |
|---:|---:|---:|---:|
| 42  | 0.5735 | 0.5693 | +0.0042 |
| 84  | 0.5695 | 0.5658 | +0.0037 |
| 123 | 0.5650 | 0.5606 | +0.0044 |
| 256 | 0.5737 | 0.5679 | +0.0058 |
| 512 | 0.5684 | 0.5613 | +0.0071 |
| **mean** | **0.5700** | **0.5650** | **+0.0051** |

Held-out **test** informal nDCG@10:

| seed | v2 γ=0.1 | v2 γ=0 | diff |
|---:|---:|---:|---:|
| 42  | 0.5584 | 0.5520 | +0.0064 |
| 84  | 0.5597 | 0.5545 | +0.0052 |
| 123 | 0.5555 | 0.5467 | +0.0088 |
| 256 | 0.5611 | 0.5504 | +0.0107 |
| 512 | 0.5514 | 0.5488 | +0.0026 |
| **mean** | **0.5572** | **0.5505** | **+0.0067** |

Paired bootstrap over the 5 seeds: 95% CI of the test difference `[+0.0044, +0.0092]`,
positive on **5/5** seeds.

Against the *original* asymmetric `gamma=0.1` checkpoints, v2 gains **+0.0072** test
nDCG@10 (mean 0.5572 vs 0.5500; 95% CI `[+0.0022, +0.0123]`, 4/5 seeds positive).

## Interpretation

The asymmetry was indeed the main cause of the harm: restoring a two-sided gradient flow
flips the sign of the adversarial contribution. The benefit is best described as
**balanced adversarial regularisation**, not domain invariance — D still ends at
~0.82–0.85, and even dips to ~0.34–0.60 mid-training, so the game is genuinely
competitive but never converges to D = 0.5.

A caution on mechanism: simple geometric probes do **not** explain the gain. On dev,
v2 γ=0.1 has *higher* mean pairwise cosine among adapted queries than γ=0
(0.0523 vs 0.0257) and an essentially unchanged own-twin alignment margin
(0.668 vs 0.675, against 0.668 for the original γ=0.1). So the improvement is not
from preserving query-specific variance or from tighter twin alignment. The remaining
plausible account is that a weaker, oscillating discriminator supplies a differently
shaped (better-balanced) gradient to the adapter; establishing that would need a
gradient-level analysis, which we have not done. We therefore report the empirical
effect and avoid a causal claim beyond the asymmetry diagnosis.

## Query-level paired bootstrap (553 test queries, 5-seed-averaged per-query nDCG@10)

| comparison | diff | 95% CI | p | queries favouring first |
|---|---:|---:|---:|---:|
| v2 γ=0.1 vs v1 γ=0.1 | +0.0072 | [+0.0043, +0.0102] | 0.0001 | 229/553 |
| v2 γ=0.1 vs v2 γ=0 | +0.0067 | [+0.0035, +0.0099] | 0.0001 | 215/553 |
| v1 γ=0.1 vs v1 γ=0 | −0.0005 | [−0.0034, +0.0025] | 0.7291 | 157/553 |
| v2 γ=0 vs v1 γ=0 | 0.0000 | [0.0000, 0.0000] | 1.0 | 0/553 |

The last row is a sanity check: with `gamma=0` the two variants are identical. The first
two rows confirm that the symmetric variant's gain is statistically significant at the
query level, while the original variant's GRL contribution is indistinguishable from
zero. Raw scores: `results_raw.json`, bootstrap: `query_level_bootstrap.json`.

Caveats: five seeds, one encoder/dataset, dev-selected checkpoints.
