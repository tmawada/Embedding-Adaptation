# Query-DANN v2 — symmetric adversarial branch

This variant tests one specific fix for the finding in `../query_dann/ABLATION_GAMMA0.md`:
at `gamma=0.1` the original discriminator is never fooled (dev accuracy 0.5 → ~0.95) and
a `gamma=0` control performs equally well or better.

## What changed

The only functional difference from `../query_dann/train.py` is in `train_step`:

| | original | v2 (here) |
|---|---|---|
| domain target (label 1) | cached raw `z_form` | `A(z_form)` — formal queries pass through the adapter |
| GRL gradient to adapter | informal branch only | both branches (symmetric game) |
| documents (`formal+doc`) | cached, unchanged | cached, unchanged |
| `L_preserve` | `A(z_form)` vs `z_form` | unchanged |
| `L_InfoNCE` | on `A(z_inf)` | unchanged |

Rationale: in the original, `z_form` is a constant with no gradient path to the adapter,
so the adversary is trained on both domains while the adapter is pushed only on the
informal side (at `gamma/2 = 0.05` weight). Making both branches trainable restores the
canonical DANN minimax symmetry. The head architecture is identical, so checkpoints load
with `methods.eval.load_head` and can be inspected with `methods/query_dann/inference.py`.

The dev discriminator probe also scores `A(z_form)` rather than raw `z_form`, matching
what the discriminator is actually trained on in this variant.

## Running

```bash
cd /home/yd/adapt-dann-retrieve/bge_dann
python3 methods/query-dann-v2/train.py \
  --run_name harmonized_seed42 --seed 42 \
  --adapter zeroinit --bottleneck 128 --discriminator sanitized \
  --gamma 0.1 --mu 0.5 --eta 10.0 --tau 0.05 --domain_targets formal \
  --total_steps 1350 --lr 3e-5 --lr_min 1e-6 --scheduler cosine --warmup_ratio 0.1 \
  --grad_clip 1.0 --weight_decay 0.01 --eval_every 50 \
  --hn_range 10 100 --adapter_dropout 0.1 --disc_dropout 0.2 \
  --mixed_precision bf16
```

Checkpoints and logs go to `methods/query-dann-v2/outputs/<run_name>/`, consistent with
the rest of the project.

## Ablation result

Seeds 42 / 84 / 123 / 256 / 512, `gamma ∈ {0.1, 0}`, same harmonized protocol as
`../query_dann/ABLATION_GAMMA0.md`. The sign flips relative to the original asymmetric
variant: symmetric `gamma=0.1` beats `gamma=0` by **+0.0051 dev** and **+0.0067 test**
informal nDCG@10 (5/5 seeds positive; test 95% CI `[+0.0044, +0.0092]`), and beats the
original `gamma=0.1` by **+0.0072 test**. D accuracy ends at ~0.82–0.85 rather than
~0.95, so this is balanced adversarial regularisation, **not** domain invariance.
Full report: `ABLATION_SYMMETRIC_GRL.md`, raw scores in `results_raw.json`.

Terminology note: this is still **paired query-style adaptation**, not UDA — see the
terminology note in `../../README.md`.
