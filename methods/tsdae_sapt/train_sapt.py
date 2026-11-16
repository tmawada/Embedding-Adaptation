"""Sentence-Adaptive Pre-Training (SAPT) of BGE-M3 with TSDAE on informal Indonesian reviews.

    informal review -> delete ~60% of its words -> BGE-M3 encoder -> single [CLS] vector
    -> XLM-RoBERTa decoder cross-attends to that ONE vector -> reconstruct the ORIGINAL review
    loss = token cross-entropy of the reconstruction.

WHICH MODEL LAYERS ARE MODIFIED:

    TRAINED  : the FULL BGE-M3 encoder (embeddings + 24 transformer layers)
    TIED     : decoder self-attention / FFN / embeddings share the encoder's parameters
               BY NAME (389 tensors); only its cross-attention + LM-head transform are new
    DISCARDED: the whole decoder after training; only the encoder is saved (~2.3 GB)
    FROZEN   : nothing — the encoder weights change, so the passage index must be re-encoded

    python3 models/tsdae_sapt/train_sapt.py                          # spec defaults: 1 epoch, bs 8, lr 3e-5
    python3 models/tsdae_sapt/train_sapt.py --max_steps 20 --no_save # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from methods.eval import TextDevSelector  # noqa: E402
from methods.model import warmup_cosine  # noqa: E402


# --------------------------------------------------------------------------- #
# Data: informal reviews (labels dropped) + TSDAE word-deletion noise
# --------------------------------------------------------------------------- #
def load_reviews(path: str, column: str, min_words: int):
    """Review texts only (labels dropped), de-duplicated, without reviews too short to survive 60% deletion."""
    texts = pd.read_csv(path)[column].dropna().astype(str).str.strip()
    texts = texts[texts != ""].drop_duplicates()
    kept = texts[texts.str.split().str.len() >= min_words].tolist()
    return kept, len(texts)


def delete_words(text: str, del_ratio: float, rng: random.Random) -> str:
    """Drop each word with probability del_ratio, always keeping at least one word."""
    words = text.split()
    kept = [w for w in words if rng.random() > del_ratio]
    return " ".join(kept) if kept else rng.choice(words)


class DenoisingDataset(Dataset):
    """(noisy, original) pairs; the noise is re-sampled every time an item is drawn (every epoch)."""

    def __init__(self, texts, del_ratio: float, seed: int):
        self.texts, self.del_ratio, self.rng = texts, del_ratio, random.Random(seed)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return delete_words(self.texts[i], self.del_ratio, self.rng), self.texts[i]


# --------------------------------------------------------------------------- #
# Model wiring: BGE-M3 encoder (TRAINED) + XLM-R decoder (weights tied)
# --------------------------------------------------------------------------- #
def tie_decoder_to_encoder(encoder, decoder):
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
    decoder.lm_head.decoder.weight = base.embeddings.word_embeddings.weight
    return tied, new


def build_model(args, device):
    """Encoder (trainable) + decoder (tied to encoder); returns everything needed to train and save."""
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    dec_config = AutoConfig.from_pretrained(args.model, is_decoder=True, add_cross_attention=True)
    dec_config._attn_implementation = "eager"
    decoder = XLMRobertaForCausalLM(dec_config).to(device)
    tied, new = tie_decoder_to_encoder(encoder, decoder)
    params = list({id(p): p for p in list(encoder.parameters()) + list(decoder.parameters())}.values())
    return tokenizer, encoder, decoder, params, tied, new


# --------------------------------------------------------------------------- #
# One training step: corrupt -> [CLS] bottleneck -> reconstruct -> loss
# --------------------------------------------------------------------------- #
def reconstruction_loss(encoder, decoder, tokenizer, batch, max_length, device):
    noisy, original = batch
    src = tokenizer(noisy, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    tgt = tokenizer(original, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw_cls = encoder(**src).last_hidden_state[:, 0]
        cls = F.normalize(raw_cls.float(), dim=-1).to(torch.bfloat16)  # bottleneck: ONE L2-normalised vector
        logits = decoder(input_ids=tgt["input_ids"][:, :-1], attention_mask=tgt["attention_mask"][:, :-1],
                         encoder_hidden_states=cls[:, None, :],
                         encoder_attention_mask=torch.ones(cls.size(0), 1, device=device, dtype=torch.long),
                         use_cache=False).logits
    labels = tgt["input_ids"][:, 1:]
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=tokenizer.pad_token_id)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(args, texts, device):
    tokenizer, encoder, decoder, params, tied, new = build_model(args, device)
    n_enc = sum(p.numel() for p in encoder.parameters())
    print(f"encoder {n_enc / 1e6:.0f}M params (trainable) | decoder: {len(tied)} tensors tied, "
          f"{len(new)} new | total {sum(p.numel() for p in params) / 1e6:.0f}M")

    loader = DataLoader(DenoisingDataset(texts, args.del_ratio, args.seed), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, collate_fn=lambda b: b)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    per_epoch = len(loader) // args.grad_accum  # optimiser steps per epoch
    total = args.total_steps or (per_epoch * args.epochs if args.max_steps is None
                                 else min(args.max_steps, per_epoch * args.epochs))
    if args.scheduler == "cosine":
        scheduler = warmup_cosine(optimizer, total, args.warmup_ratio, args.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total,
                                                        pct_start=args.warmup_ratio, anneal_strategy="linear")
    print(f"steps: {total} ({per_epoch}/epoch, bs {args.batch_size} x {args.grad_accum} accum, lr {args.lr}, "
          f"{args.scheduler})")
    # --eval_every: keep the best dev informal nDCG@10 encoder instead of the last one (harmonized protocol)
    selector = TextDevSelector(args.cache_dir, args.data_dir, device) if args.eval_every else None
    best, selected, last = -1.0, {}, {}

    os.makedirs(args.output_dir, exist_ok=True)
    log = open(os.path.join(args.output_dir, "log.jsonl"), "w")
    encoder.train()
    decoder.train()
    step, micro, epoch, t0, recent, avg = 0, 0, 0, time.time(), [], None
    optimizer.zero_grad(set_to_none=True)
    while step < total:
        for raw_batch in loader:
            noisy, original = zip(*raw_batch)
            loss = reconstruction_loss(encoder, decoder, tokenizer, (list(noisy), list(original)),
                                       args.max_length, device)
            (loss / args.grad_accum).backward()
            recent.append(loss.item())
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total:
                avg = sum(recent) / len(recent)
                recent = []
                log.write(json.dumps({"step": step, "epoch": epoch, "loss": avg}) + "\n")
                log.flush()
                print(f"step {step:5d}/{total} | loss {avg:.4f} | {time.time() - t0:.0f}s")
            if selector and (step % args.eval_every == 0 or step == total):
                info = last = {"step": step, **selector.run(encoder, tokenizer)}
                improved = info["dev_informal"]["nDCG@10"] > best
                log.write(json.dumps(info) + "\n")
                log.flush()
                print(f"step {step:5d} | dev informal nDCG@10 {info['dev_informal']['nDCG@10']:.4f} | "
                      f"dev formal {info['dev_formal']['nDCG@10']:.4f}{' *' if improved else ''}")
                if improved:
                    best, selected = info["dev_informal"]["nDCG@10"], info
                    if not args.no_save:
                        save_encoder(args, encoder, tokenizer, tied, new, len(texts), step, avg,
                                     {"selection": "best dev informal nDCG@10", **info})
            if step >= total:
                break
        epoch += 1
    log.close()
    return encoder, tokenizer, tied, new, step, avg, selected, last


def save_encoder(args, encoder, tokenizer, tied, new, n_texts, step, avg, extra=None, out_dir=None):
    out_dir = out_dir or args.output_dir
    print(f"saving encoder to {out_dir} (free disk {shutil.disk_usage(ROOT).free / 1e9:.1f} GB)")
    os.makedirs(out_dir, exist_ok=True)
    encoder.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "sapt_config.json"), "w") as f:
        json.dump({**vars(args), "reviews_used": n_texts, "steps": step, "final_loss": avg,
                   "decoder_tied_tensors": len(tied), "decoder_new_tensors": len(new), **(extra or {})}, f, indent=2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "prdect-id.csv"))
    ap.add_argument("--text_column", default="Customer Review")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--output_dir", default=os.path.join(HERE, "outputs", "bge-m3-sapt-informal-id"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None, help="Stop early (smoke tests)")
    ap.add_argument("--total_steps", type=int, default=None,
                    help="Fixed optimiser-step budget, cycling the reviews as often as needed (overrides --epochs)")
    ap.add_argument("--batch_size", type=int, default=8, help="Micro-batch; effective batch = this x --grad_accum")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_min", type=float, default=1e-6, help="Floor of --scheduler cosine")
    ap.add_argument("--scheduler", default="onecycle", choices=["onecycle", "cosine"],
                    help="onecycle = linear warmup/anneal (original); cosine = warmup then cosine to --lr_min")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--del_ratio", type=float, default=0.6)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="Dev evaluation interval in optimiser steps; saves the best dev informal nDCG@10 encoder "
                         "instead of the last one (0 = off, save the final encoder)")
    ap.add_argument("--final_dir", default=None,
                    help="Also save the encoder after the last step here (default: only --output_dir)")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"), help="Dev split + frozen passage index")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    texts, n_unique = load_reviews(args.data, args.text_column, args.min_words)
    print(f"reviews: {n_unique} unique, {len(texts)} kept (>= {args.min_words} words); labels discarded")

    encoder, tokenizer, tied, new, step, avg, selected, last = train(args, texts, device)

    if args.no_save:
        print("--no_save: model not written")
    elif args.eval_every:  # the best dev checkpoint was already written during training
        print(f"best dev informal nDCG@10 {selected['dev_informal']['nDCG@10']:.4f} at step {selected['step']} "
              f"-> {args.output_dir}")
    else:
        save_encoder(args, encoder, tokenizer, tied, new, len(texts), step, avg)
    if args.final_dir and not args.no_save:  # what training itself produced, whatever the dev selection kept
        save_encoder(args, encoder, tokenizer, tied, new, len(texts), step, avg,
                     {"selection": "none (final step)", **last}, out_dir=args.final_dir)
    print("done")


if __name__ == "__main__":
    main()
