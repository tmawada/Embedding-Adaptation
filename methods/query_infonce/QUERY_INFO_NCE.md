# InfoNCE query adaptation for BGE-M3 — design record

**Status: implemented and evaluated — positive result on informal, partial cost on formal.** Code is
`train_infonce.py` (authoritative for the method); `prepare_infonce_cache.py` builds the query cache and
`evaluate_infonce.py` measures it. Results in the final section.

This folder is **not** TSDAE-SAPT and does not modify it. `../tsdae_sapt/` is left byte-identical to its
published negative result — code, doc, and `outputs/bge-m3-sapt-informal-id/` (the baseline encoder) are
never written by anything here.

## Why this folder exists

`../tsdae_sapt` fine-tuned the encoder with pure reconstruction on informal reviews. Its objective was
**never connected to retrieval** — nothing in the loss knew which passage a query should find. The result
was style merging by degradation, not adaptation:

| condition | formal nDCG@10 | informal nDCG@10 | drop |
|---|---|---|---|
| base BGE-M3 | 0.6166 | 0.5373 | −0.0793 |
| TSDAE-SAPT `query_only` | 0.5340 | 0.3604 | −0.1736 |

The formal side fell 0.083 **despite formal text never appearing in the training data**. `../tsdae_sapt_queries/`
reproduced the failure on the task's own queries (0.5266 vs 0.5373), so the corpus was not the problem —
the `[CLS]`-only reconstruction bottleneck was.

This folder keeps the encoder-finetuning shape but replaces the **primary** loss with one that is
anchored to retrieval, and demotes reconstruction to a regularizer.

## Method (as implemented)

Two encoder forwards per step, deliberately on **different** inputs:

```
branch 1  CLEAN informal query      ──► BGE-M3 encoder (TRAINABLE, CLS pool, L2 norm, fp32)
                                           │
                                           ├──► InfoNCE vs frozen passage vectors      w = 1.0
                                           │
branch 2  CORRUPTED (60% words gone) ──► same encoder ──► CLS ──► tied decoder ──► CE vs the CLEAN query
                                                                             w = w_tsdae (0.1)
```

- **Branch 1 is clean on purpose.** `prepare_infonce_cache.py` encodes the *uncorrupted*
  `generated_informal_query`, so feeding a 40%-of-words string into InfoNCE would train and evaluate on
  different distributions. Retrieval is scored on the text the user actually typed.
- **Branch 2 is TSDAE**, mechanism byte-identical to `../tsdae_sapt/train_sapt.py` (same `delete_words`,
  same tied decoder, 389 tied / 240 new tensors, same cross-entropy against the uncorrupted original).
- **Passages are frozen.** Candidates are mmap-gathered rows of `../cache/corpus_emb.npy` (base BGE-M3).
  The document space never moves, so there is no index rebuild and no `--mode symmetric`.
- **No query adapter, no GRL, no discriminator, no LLM.** The only trainable thing is the encoder
  (plus the reconstruction branch's new cross-attention tensors when that branch is on).

Why passages instead of an explicit formal target: documents are formal-register Wikipedia text, so
pulling informal queries toward the frozen document space *is* the informal → formal pressure, obtained
from retrieval rather than from a rewrite step. That is the substitution this method bets on.

## Data (`data/train_pairs.csv` + `../cache`)

MIRACL-id query pairs. The TSDAE reconstruction branch is label-free, but the InfoNCE term uses labelled positive passages and mined hard negatives, so the method as a whole is **paired and label-using** (not UDA):

- `train_pairs.csv` → `informal_query`, `formal_query` per `query_id`
- `build_triples()` joins `miracl_pairs.json` for the positive passage
- hard negatives from `../cache/hard_negs.json`, ranks `[10, 100)` (mined by base BGE-M3 on *formal*
  queries, so they are a proxy for informal — the same choice `dataset.TrainQueryDataset` makes)
- **2,891 train / 386 dev / 578 test**, disjoint by query (`dataset.make_splits`, seed 42)

InfoNCE construction mirrors `dataset.TrainQueryDataset.collate`: `[B positives ; B hard negatives]` with
a `false_neg` mask zeroing other queries' labelled positives.

## Hyperparameters (as implemented)

| | value | note |
|---|---|---|
| objective | `infonce_tsdae` | alt: `infonce` (reconstruction off) |
| `w_tsdae` | 0.1 | |
| `del_ratio` | 0.6 | TSDAE corruption |
| `tau` | 0.05 | matches `query_dann/train.py` |
| batch | **32** | 63 candidates/query; see below |
| epochs | **4** | 361 steps total |
| lr | 3e-5 | constant (default scheduler) |
| weight decay | 0.0 | biases/LayerNorm excluded from decay regardless |
| max_length | 128 | eval truncates at 64, matching `../cache` |
| grad clip | 1.0 | |
| seed | 42 | |

**Batch size is not copied from `../tsdae_sapt` (8).** InfoNCE quality scales with candidate count —
B=8 gives 15 negatives, B=32 gives 63. Measured on the RTX 3090: **B=8 → 13.5 GB, B=32 → 15.8 GB,
B=64 → 21.9 GB of 25.3 GB**. B=64 fits but leaves only 3.4 GB of headroom; B=32 keeps 9.5 GB.

Eval-time truncation stays at `query_max_length=64` (`methods/encoders.py`
`ENCODERS["bge_m3"]`), *not* the 128 used in training, so the base and InfoNCE conditions differ only
in weights — no truncation confound in the comparison.

## Deviations from `../tsdae_sapt/train_sapt.py`

`../tsdae_sapt/train_sapt.py` has four bugs introduced by commit `d1dfdab` ("fix: L2 Norm Output"), which
silently changed three unrelated hyperparameters alongside the L2 fix. That folder is left frozen, so
this file ships correct from the start:

| | `tsdae_sapt` after `d1dfdab` | here | why |
|---|---|---|---|
| `weight_decay` | 0.01 | **0.0** | `sapt_config.baseline.json` records 0.0 |
| `max_length` | 64 | **128** | baseline records 128 |
| scheduler | OneCycleLR | **constant** | `outputs/retrain_train.log` prints `lr 3e-05 constant` |
| `avg` at config write | `NameError` if `--max_steps 0` | **initialized `None`** | |
| AdamW decay | all params | **excluded for `ndim < 2`** | HF convention: never decay bias/LayerNorm |
| config contents | no schedule recorded | **records objective, `w_tsdae`, `tau`, scheduler, torch/transformers, query counts** | |

The baseline encoder also has no backup copy, and `train_sapt.py` writes to it unconditionally. Here the
default `--output_dir` is `methods/query_infonce/outputs/bge-m3-infonce-id`, so that path cannot be clobbered.

## Design decisions already taken

Recorded so they are not silently re-litigated:

- **Formal-side InfoNCE declined.** Only informal queries are encoded and trained. Nothing anchors
  formal queries, so formal drift is possible again — this is the single biggest risk in the method.
- **Preserve term declined.** No `1 − cos(E(formal), z_base_formal)` term. The signal that this was the
  wrong call is now the **formal column** of `evaluate_infonce.py`: a drop the way TSDAE's had
  (0.6166 -> 0.5340) means formal queries drifted. The fix is this term (`query_dann/train.py:229`) and
  requires no new architecture. (`evaluate_infonce.py` is retrieval-only now — for the old
  `drift_vs_base.queries_formal` diagnostic, run `../tsdae_sapt/evaluate_sapt.py`.)
- **Second view for InfoNCE declined.** Corrupted text does not enter branch 1; only branch 2 sees it.
- **`--mode symmetric` removed.** Re-encoding passages would contradict the frozen-document-space
  constraint and cost ~1 GB this machine does not have (8.2 GB free at 97% utilisation).

## Reproducing `../tsdae_sapt`'s baseline

Not run from here — that folder's own `run.sh` does it, but note it will **not** reproduce
`sapt_config.baseline.json` as shipped, because of the `d1dfdab` drift documented above. The recorded
baseline artifacts are `tsdae_sapt/sapt_config.baseline.json`, `tsdae_sapt/outputs/zero_shot_{test,dev}.baseline.json`
and `tsdae_sapt/cache/*.baseline/`.

## Results

MIRACL-id, 578 test queries / 500,000 passages, nDCG@10. The `base` row is identical to
`../tsdae_sapt`'s table, confirming the two folders run the **same protocol** — those rows are directly
comparable. (`comparison/runs/harmonized/` is a *different* protocol; Query-DANN's +3.95 and LLM's
+6.44 must not be compared against the numbers below.)

| condition | queries | formal | informal | drop (pts) | cos(inf, formal) |
|---|---|---|---|---|---|
| base BGE-M3 | base | 0.6166 | 0.5373 | −7.93 | 0.8976 |
| TSDAE-SAPT `query_only` | SAPT | 0.5340 | 0.3604 | −17.36 | 0.8085 |
| TSDAE-SAPT `symmetric` | SAPT | 0.4289 | 0.2468 | −18.20 | 0.8085 |
| **InfoNCE `query_only`** | **this folder** | **0.5911** | **0.5650** | **−2.62** | **0.9357** |

Paired bootstrap vs base, 1000 samples, two-sided:

| comparison | metric | diff | CI95 | p |
|---|---|---|---|---|
| informal | nDCG@10 | **+0.0276** | [+0.0093, +0.0459] | **0.002 \*** |
| informal | MRR@10 | +0.0325 | [+0.0062, +0.0586] | 0.012 \* |
| informal | Recall@100 | +0.0001 | [−0.0167, +0.0164] | 0.992 |
| formal | nDCG@10 | −0.0255 | [−0.0399, −0.0101] | 0.001 \* |
| **gap** | nDCG@10 | **+0.0531** | [+0.0367, +0.0685] | **0.001 \*** |

Dev reproduces test almost exactly (386 queries): base 0.6144 / 0.5328 (gap −8.16) → InfoNCE
0.5928 / 0.5779 (gap −1.49), cos 0.9015 → 0.9438.

**Pass/fail against the criteria set before the run:**

- ✅ **Informal beats base** — 0.5650 vs 0.5373, p = 0.002. The stated goal was to beat 0.5373.
- ✅ **67% of the gap removed** — −7.93 → −2.62 pts, p = 0.001.
- ✅ **No TSDAE-style collapse** — formal fell −0.0255 (p = 0.001) against TSDAE-SAPT's −0.0826, i.e.
  **3.2× less damage**, and informal rose instead of falling −0.177.
- ⚠️ **Formal is not free** — the loss is real and significant, the consequence of declining the
  preserve term (see "Design decisions already taken"). Recall@100 on informal is flat (+0.0001,
  p = 0.992): the gain is in *ranking* the top-100, not in reaching it, since R@100 was already 0.9003.

Embedding drift vs base: formal queries **0.7659**, informal **0.7124**, passages **1.0000** — the last
confirms the symlink held and the document space never moved.

**The `cos(inf, formal twin)` column moved the opposite way to TSDAE-SAPT**, and this is the cleanest
signal in the table: ours 0.8976 → **0.9357** (the two representations of the same query converged),
TSDAE-SAPT 0.8976 → 0.8085 (they diverged). Note that `../tsdae_sapt/TSDAE_SAPT_IMPLEMENTATION.md:92`
describes *its* fall to 0.808 as "the style gap did collapse" — under the reading that cosine measures
similarity, a fall means the representations came *apart*, so that sentence looks inverted. Their
headline conclusion (TSDAE failed) is unaffected either way.

### Reading the result

The mechanism the method bet on — anchor the query side to a frozen document space instead of
reconstructing text — is what worked. InfoNCE gives the encoder a retrieval objective, so it can no
longer improve reconstruction by eroding retrieval semantics; formal still degrades (it gets no
gradient), but far less, because the weights that would damage it are being pulled the other way by
the 2,891 informal queries' gradient.

What would push this further, in rough order of expected value:

1. **Preserve term** — `1 − cos(E(formal), z_base_formal)` from `query_dann/train.py:222`, against the
   already-cached `../cache/formal_emb.npy`. Directly targets the remaining −0.0255. No new
   architecture; this is the explicitly declined item.
2. **Formal-side InfoNCE** — run branch 1 on formal queries too, same frozen positives. Anchors formal
   for free and pulls both styles to identical targets. ~2× branch-1 cost.
3. **Drop `--w_tsdae` to 0** (`--objective infonce`) — isolates how much of the formal loss the
   reconstruction term is responsible for. One training run, ~90 s.
4. **Add this row to `comparison/report.py`** — the harmonized protocol is the one that produces the
   Query-DANN / LLM ranking; until this model appears there, no claim can be made against them.

### Baseline reference (harmonized protocol — different protocol, do not compare with the table above)

From `comparison/runs/harmonized/report_harmonized_test.md`:

| model | formal nDCG@10 | informal nDCG@10 | narrowing |
|---|---|---|---|
| BGE-M3 base | 0.6166 | 0.5373 | +0.00 |
| TSDAE-SAPT (reviews) | 0.6143 | 0.5351 | +0.01 |
| Query-DANN v2 | 0.6136 | 0.5738 | +3.95 |
| LLM normalisation | 0.6166 | 0.6017 | +6.44 |
