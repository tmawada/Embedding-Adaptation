"""Retrieval-anchored query adaptation for BGE-M3: InfoNCE on informal queries + a TSDAE regularizer.

Motivation: ../tsdae_sapt trained the encoder with pure reconstruction and merged the two query styles
by degrading BOTH (informal 0.5373 -> 0.3604, formal 0.6166 -> 0.5340) because nothing in that objective
was connected to retrieval. This replaces the primary loss with InfoNCE, which anchors the query side to
the frozen passage space; documents are formal-register, so pulling informal queries toward them is the
informal -> formal pressure, obtained from retrieval instead of from an LLM rewrite.

Two encoder forwards per step, on different inputs (this is the point, not an accident):

  branch 1  CLEAN informal query   -> encoder -> CLS -> L2 norm -> InfoNCE vs frozen passages   w = 1.0
  branch 2  CORRUPTED (60% deleted) -> encoder -> CLS -> tied decoder -> CE vs the clean query  w = w_tsdae

  branch 1 matches evaluation: prepare_infonce_cache.py encodes the UNCORRUPTED query, so feeding a
  40%-of-words string into InfoNCE would train/eval on different distributions.
  branch 2 is TSDAE, byte-identical in mechanism to ../tsdae_sapt/train_sapt.py, demoted to a regularizer.

The passage side is NEVER re-encoded: candidates are rows of ../cache/corpus_emb.npy (base BGE-M3).
No index rebuild, no adapter head, no GRL, no discriminator, no LLM.

  python3 models/query_infonce/train_infonce.py                          # default: objective infonce_tsdae, w_tsdae 0.1
  python3 models/query_infonce/train_infonce.py --objective infonce      # ablation: reconstruction branch off
  python3 models/query_infonce/train_infonce.py --max_steps 20 --no_save # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import build_triples  # noqa: E402
from methods.encoders import ENCODERS, pool  # noqa: E402
from methods.eval import TextDevSelector  # noqa: E402
from methods.model import warmup_cosine  # noqa: E402


def delete_words(text: str, del_ratio: float, rng: random.Random) -> str:
    """TSDAE deletion noise: drop each word with probability del_ratio, always keeping at least one word."""
    words = text.split()
    kept = [w for w in words if rng.random() > del_ratio]
    return " ".join(kept) if kept else rng.choice(words)


def tie_decoder_to_encoder(encoder: torch.nn.Module, decoder: XLMRobertaForCausalLM):
    """Share every decoder parameter whose name exists in the encoder; cross-attention etc. stay decoder-only."""
    enc_params = dict(encoder.named_parameters())
    base = decoder.roberta
    tied, new = [], []
    for name, _ in list(base.named_parameters()):
        if name in enc_params:
            module_path, attr = name.rsplit(".", 1)
            setattr(base.get_submodule(module_path), attr, enc_params[name])
            tied.append(name)
        else:
            new.append(name)
    # Output projection shares the (now encoder-owned) word embedding matrix, as in the pretrained LM.
    decoder.lm_head.decoder.weight = base.embeddings.word_embeddings.weight
    return tied, new


class InformalRetrievalDataset(Dataset):
    """Informal query TEXT plus frozen passage candidates for InfoNCE.

    dataset.TrainQueryDataset serves cached *query* embeddings; that cannot drive a trainable encoder,
    so this yields the raw informal string instead. Passage candidates stay in the frozen cache
    (mmap-gathered, ~2 rows/batch -> never materialises the 1 GB corpus on the GPU).

    Negatives and the false-negative mask follow dataset.TrainQueryDataset exactly, so the InfoNCE
    construction is comparable with query_dann's.
    """

    def __init__(self, triples, train_qids, doc2row, corpus_path, hn_range=(10, 100), seed: int = 42):
        self.corpus = np.load(corpus_path, mmap_mode="r")
        self.n_docs = len(doc2row)
        train = set(train_qids)
        self.text, self.positives, self.negatives = {}, defaultdict(list), {}
        for t in triples:
            qid = t["query_id"]
            if qid not in train or t["positive_doc_id"] not in doc2row:
                continue
            self.text[qid] = t["informal_query"]
            self.positives[qid].append(doc2row[t["positive_doc_id"]])
            # hard_negs.json already excludes this query's labelled positives; [10:100] skips the top ranks.
            self.negatives[qid] = [doc2row[d] for d in t["negative_doc_ids"][hn_range[0]:hn_range[1]]
                                   if d in doc2row]
        self.qids = sorted(self.positives, key=int)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, i):
        qid = self.qids[i]
        negs = self.negatives[qid]
        return {
            "text": self.text[qid],
            "pos_row": self.rng.choice(self.positives[qid]),
            "neg_row": self.rng.choice(negs) if negs else self.rng.randrange(self.n_docs),
            "pos_set": set(self.positives[qid]),
        }

    def collate(self, items):
        rows = [it["pos_row"] for it in items] + [it["neg_row"] for it in items]
        z_cand = torch.from_numpy(np.asarray(self.corpus[rows])).float()  # fp16 -> fp32, [2B, 1024]
        B = len(items)
        false_neg = torch.zeros(B, 2 * B, dtype=torch.bool)
        for i, it in enumerate(items):
            pos = it["pos_set"]
            for j, r in enumerate(rows):
                if j != i and r in pos:  # another query's positive, but labelled relevant for i
                    false_neg[i, j] = True
        return {"texts": [it["text"] for it in items], "z_cand": z_cand, "false_neg": false_neg}

    def loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)


def parse_args():
    ap = argparse.ArgumentParser()
    # --- objective ---
    ap.add_argument("--objective", default="infonce_tsdae", choices=["infonce_tsdae", "infonce"],
                    help="infonce_tsdae = both branches; infonce = reconstruction branch off (ablation)")
    ap.add_argument("--w_tsdae", type=float, default=0.1, help="Weight of the TSDAE reconstruction term")
    ap.add_argument("--del_ratio", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=0.05, help="InfoNCE temperature (query_dann default)")
    ap.add_argument("--hn_range", type=int, nargs=2, default=[10, 100],
                    help="Sample hard negatives from these ranks of the base-model ranking")
    # --- data ---
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"),
                    help="Frozen base-BGE-M3 cache; corpus_emb.npy / hard_negs.json / splits.json are read-only")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--encoder", default="bge_m3", choices=list(ENCODERS),
                    help="Encoder mode (methods/encoders.py): query pooling / prefix / dev max length; pass the "
                         "matching --model and --cache_dir (that encoder's frozen passage index)")
    # --- optimisation (loss-scale defaults reproduce tsdae_sapt's recorded baseline settings) ---
    # batch_size/epochs are NOT copied from tsdae_sapt: InfoNCE quality scales with candidate count
    # (B=8 -> 15 negatives, B=32 -> 63), and 361 steps matches that run's 614 in magnitude.
    # Measured on the RTX 3090: B=8 13.5 GB, B=32 15.8 GB, B=64 21.9 GB of 25.3.
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=None, help="Stop early (smoke tests)")
    ap.add_argument("--total_steps", type=int, default=None,
                    help="Fixed optimiser-step budget, overriding --epochs (default: epochs x steps/epoch)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_min", type=float, default=1e-6, help="Floor of --scheduler cosine")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="constant", choices=["constant", "onecycle", "cosine"],
                    help="cosine = warmup then cosine to --lr_min (harmonized protocol)")
    ap.add_argument("--warmup_ratio", type=float, default=0.1, help="Used by --scheduler onecycle / cosine")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="Dev evaluation interval in optimiser steps; saves the best dev informal nDCG@10 encoder "
                         "instead of the last one (0 = off, save the final encoder)")
    # --- io ---
    ap.add_argument("--output_dir", default=os.path.join(HERE, "outputs", "bge-m3-infonce-id"),
                    help="New path: ../tsdae_sapt/outputs holds the baseline encoder and is never written here")
    ap.add_argument("--min_free_gb", type=float, default=3.0)
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    need_decoder = args.objective == "infonce_tsdae" and args.w_tsdae > 0
    spec = ENCODERS[args.encoder]

    # ---- data -------------------------------------------------------------
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    with open(os.path.join(args.cache_dir, "hard_negs.json")) as f:
        hard_negs = json.load(f)
    with open(os.path.join(args.cache_dir, "corpus_ids.json")) as f:
        doc_ids = json.load(f)
    doc2row = {d: i for i, d in enumerate(doc_ids)}
    triples = build_triples(args.data_dir, hard_negs)
    ds = InformalRetrievalDataset(triples, splits["train"], doc2row,
                                  os.path.join(args.cache_dir, "corpus_emb.npy"),
                                  tuple(args.hn_range), args.seed)
    if len(ds) == 0:
        raise SystemExit(f"no training queries: {args.data_dir} / {args.cache_dir} mismatch?")

    # ---- model ------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    tied = new = None
    decoder = None
    if need_decoder:
        dec_config = AutoConfig.from_pretrained(args.model, is_decoder=True, add_cross_attention=True)
        dec_config._attn_implementation = "eager"
        decoder = XLMRobertaForCausalLM(dec_config).to(device)
        tied, new = tie_decoder_to_encoder(encoder, decoder)
    parts = list(encoder.parameters()) + (list(decoder.parameters()) if decoder else [])
    params = list({id(p): p for p in parts}.values())  # tied tensors are shared: dedupe by identity
    n_enc = sum(p.numel() for p in encoder.parameters())
    print(f"objective={args.objective} | train queries={len(ds)} | batches/epoch={len(ds) // args.batch_size} | "
          f"encoder {n_enc / 1e6:.0f}M (trainable) | "
          + (f"decoder: {len(tied)} tied, {len(new)} new" if need_decoder else "decoder: off"))

    loader = ds.loader(args.batch_size)
    # HF convention: weight decay on 2-D+ weights only, never on bias / LayerNorm (ndim < 2).
    decay, no_decay = ([p for p in params if p.ndim >= 2], [p for p in params if p.ndim < 2])
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    total = args.total_steps or (len(loader) * args.epochs if args.max_steps is None
                                 else min(args.max_steps, len(loader) * args.epochs))
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total,
                                                        pct_start=args.warmup_ratio, anneal_strategy="linear")
    elif args.scheduler == "cosine":
        scheduler = warmup_cosine(optimizer, total, args.warmup_ratio, args.lr_min)
    else:
        scheduler = None
    print(f"steps: {total} | batch {args.batch_size} | lr {args.lr} {args.scheduler} | "
          f"w_tsdae {args.w_tsdae if need_decoder else 0} | tau {args.tau}")

    def tokenize(texts):
        return tokenizer(texts, padding=True, truncation=True, max_length=args.max_length,
                         return_tensors="pt").to(device)

    def save(extra=None):
        free_gb = shutil.disk_usage(HERE).free / 1e9
        if free_gb < args.min_free_gb:
            raise SystemExit(f"only {free_gb:.1f} GB free, need {args.min_free_gb} GB for the encoder")
        print(f"saving encoder to {args.output_dir} (free disk {free_gb:.1f} GB)")
        os.makedirs(args.output_dir, exist_ok=True)
        encoder.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        import transformers
        with open(os.path.join(args.output_dir, "infonce_config.json"), "w") as f:
            json.dump({**vars(args), "n_train_queries": len(ds), "n_epochs_seen": epoch, "steps": step,
                       "final_loss": avg, "decoder_used": need_decoder,
                       "decoder_tied_tensors": len(tied) if tied else 0,
                       "decoder_new_tensors": len(new) if new else 0,
                       "torch": torch.__version__, "transformers": transformers.__version__, **(extra or {})}, f, indent=2)

    # --eval_every: keep the best dev informal nDCG@10 encoder instead of the last one (harmonized protocol)
    # The passage index stays in RAM: batch 64 alone needs ~21.9 GB of the 24 GB RTX 3090.
    selector = TextDevSelector(args.cache_dir, args.data_dir, device, store_device="cpu", pooling=spec.pooling,
                               prefix=spec.query_prefix, max_length=spec.query_max_length) if args.eval_every else None
    best, selected = -1.0, {}

    os.makedirs(args.output_dir, exist_ok=True)
    log = open(os.path.join(args.output_dir, "log.jsonl"), "w")
    noise_rng = random.Random(args.seed)
    encoder.train()
    if decoder is not None:
        decoder.train()
    step, epoch, t0, avg, recent_nce, recent_t = 0, 0, time.time(), None, [], []
    while step < total:
        for batch in loader:
            B = len(batch["texts"])
            labels = torch.arange(B, device=device)
            false_neg = batch["false_neg"].to(device)
            z_cand = batch["z_cand"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # branch 1: CLEAN query -> the frozen document space (pooled as that index was encoded)
                q = tokenize([spec.query_prefix + t for t in batch["texts"]])
                pooled_clean = pool(encoder(**q).last_hidden_state, q["attention_mask"], spec.pooling)
                # branch 2: CORRUPTED query -> reconstruct the clean one (TSDAE)
                if need_decoder:
                    noisy = [delete_words(t, args.del_ratio, noise_rng) for t in batch["texts"]]
                    n = tokenize(noisy)
                    pooled_noisy = pool(encoder(**n).last_hidden_state, n["attention_mask"], spec.pooling)
                    tgt = tokenize(batch["texts"])
                    cls_d = F.normalize(pooled_noisy.float(), dim=-1).to(torch.bfloat16)
                    logits = decoder(input_ids=tgt["input_ids"][:, :-1], attention_mask=tgt["attention_mask"][:, :-1],
                                     encoder_hidden_states=cls_d[:, None, :],
                                     encoder_attention_mask=torch.ones(B, 1, device=device, dtype=torch.long),
                                     use_cache=False).logits
            # L2 norm + InfoNCE in fp32, outside autocast: masking -inf into bf16 logits is lossy.
            # Eval (encoders.py:110) normalises the same way, so train and eval see the same vectors.
            z_q = F.normalize(pooled_clean.float(), dim=-1)
            sims = (z_q @ z_cand.T) / args.tau
            sims = sims.masked_fill(false_neg, float("-inf"))
            loss_nce = F.cross_entropy(sims, labels)
            loss = loss_nce
            loss_t = None
            if need_decoder:
                labels_t = tgt["input_ids"][:, 1:]
                loss_t = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels_t.reshape(-1),
                                         ignore_index=tokenizer.pad_token_id)
                loss = loss_nce + args.w_tsdae * loss_t

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            recent_nce.append(loss_nce.item())
            if loss_t is not None:
                recent_t.append(loss_t.item())
            if step % args.log_every == 0 or step == total:
                avg = sum(recent_nce) / len(recent_nce)
                rec = {"step": step, "epoch": epoch, "loss": float(loss.item()), "loss_nce": avg,
                       "lr": optimizer.param_groups[0]["lr"]}
                if recent_t:
                    rec["loss_tsdae"] = sum(recent_t) / len(recent_t)
                    recent_t = []
                recent_nce = []
                log.write(json.dumps(rec) + "\n")
                log.flush()
                extra = f" | TSDAE {rec['loss_tsdae']:.4f}" if "loss_tsdae" in rec else ""
                print(f"step {step:5d}/{total} | epoch {epoch} | InfoNCE {avg:.4f}{extra} | {time.time() - t0:.0f}s")
            if selector and (step % args.eval_every == 0 or step == total):
                info = {"step": step, **selector.run(encoder, tokenizer)}
                improved = info["dev_informal"]["nDCG@10"] > best
                log.write(json.dumps(info) + "\n")
                log.flush()
                print(f"step {step:5d} | dev informal nDCG@10 {info['dev_informal']['nDCG@10']:.4f} | "
                      f"dev formal {info['dev_formal']['nDCG@10']:.4f}{' *' if improved else ''}")
                if improved:
                    best, selected = info["dev_informal"]["nDCG@10"], info
                    if not args.no_save:
                        save({"selection": "best dev informal nDCG@10", **info})
            if step >= total:
                break
        epoch += 1
    log.close()

    torch.cuda.synchronize()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} / {total_gb:.1f} GB")
    if args.no_save:
        print("--no_save: model not written")
        return
    if args.eval_every:  # the best dev checkpoint was already written during training
        print(f"best dev informal nDCG@10 {best:.4f} at step {selected['step']} -> {args.output_dir}")
    else:
        save()
    print("done")


if __name__ == "__main__":
    main()
