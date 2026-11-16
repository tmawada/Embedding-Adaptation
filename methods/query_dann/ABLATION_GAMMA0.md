# Query-DANN: does the adversarial (GRL) term help?

## Question

The proposed loss is
`L = L_InfoNCE + γ · L_domain + μ · L_preserve`.
At the harmonised default `γ = 0.1`, is the adversarial branch responsible for the
retrieval gain, or is the gain entirely from InfoNCE + preservation?

## Setup

All runs use the same protocol as the harmonised BGE-M3 experiments:
`--adapter zeroinit --bottleneck 128 --discriminator sanitized --mu 0.5 --eta 10`
`--total_steps 1350 --lr 3e-5 --batch_size 64 --warmup_ratio 0.1 --scheduler cosine`
`--grad_clip 1.0 --weight_decay 0.01 --eval_every 50 --mixed_precision bf16`,
differing only in `γ ∈ {0.1, 0}` and seed. Checkpoints selected by best dev informal
nDCG@10. Five seeds: 42, 84, 123, 256, 512.

## Result 1 — the discriminator is never fooled (γ = 0.1)

From `outputs/harmonized_seed256/log.jsonl`:

| step | loss_domain | GRL α | train D-acc | dev nDCG (informal) |
|---:|---:|---:|---:|---:|
| 10  | 0.694 (≈ ln 2) | 0.03 | 0.50 | 0.539 |
| 300 | 0.441 | 0.80 | 0.90 | 0.556 |
| 1350| 0.157 | 1.00 | 0.95 | 0.556 |

The domain loss *decreases* and discriminator accuracy *increases*: D wins, so the
adapted representations are not domain-invariant. Two structural reasons:

1. `γ = 0.1` makes the adversarial branch only a few percent of the total gradient.
2. The target branch feeds the cached constant `z_form` to the discriminator, so GRL
   has no gradient path to the adapter there. D is trained on both domains; the
   adapter is pushed only on the informal side, at 0.1× weight.

## Result 2 — γ = 0 matches or beats γ = 0.1

Best **dev** informal nDCG@10:

| seed | γ = 0.1 | γ = 0 | diff |
|---:|---:|---:|---:|
| 42  | 0.5614 | 0.5693 | −0.0079 |
| 84  | 0.5602 | 0.5658 | −0.0057 |
| 123 | 0.5606 | 0.5606 | +0.0001 |
| 256 | 0.5588 | 0.5679 | −0.0090 |
| 512 | 0.5604 | 0.5613 | −0.0009 |
| **mean** | **0.5603** | **0.5650** | **−0.0047** |

Paired bootstrap over the five seeds: 95% CI of the mean difference
`[−0.0079, −0.0015]`.

Held-out **test** informal nDCG@10 (same checkpoints):

| seed | γ = 0.1 | γ = 0 | diff |
|---:|---:|---:|---:|
| 42  | 0.5488 | 0.5520 | −0.0033 |
| 84  | 0.5504 | 0.5545 | −0.0041 |
| 123 | 0.5543 | 0.5467 | +0.0076 |
| 256 | 0.5446 | 0.5504 | −0.0058 |
| 512 | 0.5518 | 0.5488 | +0.0030 |
| **mean** | **0.5500** | **0.5505** | **−0.0005** |

95% CI `[−0.0044, +0.0040]` — indistinguishable from zero.

## Conclusion

With only five seeds this is not a definitive significance result, but the direction is
consistent on dev and neutral on test: **the GRL term provides no measurable benefit at
`γ = 0.1` and is mildly harmful on dev.** The retrieval gain comes from InfoNCE plus
formal preservation. Therefore:

- `γ = 0.1` is kept as the reported default for continuity with the published runs;
- the adversarial term is described as a *weak regulariser*, not as the mechanism of
  adaptation and not as evidence of domain-invariant representations;
- the method is framed as **paired query-style adaptation**, not unsupervised domain
  adaptation (the unpaired UDA setting is the separate Pipeline 5).

**Follow-up:** making the adversary symmetric (formal queries also pass through the
adapter) equalises gradient flow and flips the sign — `γ=0.1` then improves test nDCG@10
by +0.0067 over `γ=0`. See `../query-dann-v2/ABLATION_SYMMETRIC_GRL.md`.

Raw scores: `outputs/gamma0_ablation/results.json`.
