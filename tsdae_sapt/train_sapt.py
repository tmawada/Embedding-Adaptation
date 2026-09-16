"""Sentence-Adaptive Pre-Training (SAPT) of BGE-M3 with TSDAE on informal Indonesian reviews (prdect-id).

TSDAE (Wang et al., 2021), implemented directly with transformers (sentence-transformers needs Python >= 3.10):
  noisy review (60% of words deleted) -> BGE-M3 encoder -> [CLS] vector (the only information bottleneck)
  -> Transformer decoder that cross-attends to that single vector -> reconstruct the ORIGINAL review
  loss = token cross-entropy between the decoder output and the uncorrupted review.
The decoder shares all embedding / self-attention / feed-forward weights with the encoder (tie_encoder_decoder);
only its cross-attention layers and LM-head transform are new. The ENCODER IS FULLY FINE-TUNED (a frozen encoder
could not change the embeddings). Sentiment labels are discarded: the objective is unsupervised.

  python3 tsdae_sapt/train_sapt.py                          # spec defaults: 1 epoch, bs 8, lr 3e-5 constant
  python3 tsdae_sapt/train_sapt.py --max_steps 20 --no_save # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_reviews(path: str, column: str, min_words: int):
    """Review texts only (labels dropped), de-duplicated, without reviews too short to survive 60% deletion."""
    texts = pd.read_csv(path)[column].dropna().astype(str).str.strip()
    texts = texts[texts != ""].drop_duplicates()
    kept = texts[texts.str.split().str.len() >= min_words].tolist()
    return kept, len(texts)


def delete_words(text: str, del_ratio: float, rng: random.Random) -> str:
    """TSDAE deletion noise: drop each word with probability del_ratio, always keeping at least one word."""
    words = text.split()
    kept = [w for w in words if rng.random() > del_ratio]
    return " ".join(kept) if kept else rng.choice(words)


class DenoisingDataset(Dataset):
    """Returns (noisy, original) pairs; the noise is re-sampled every time an item is drawn (i.e. every epoch)."""

    def __init__(self, texts, del_ratio: float, seed: int):
        self.texts, self.del_ratio, self.rng = texts, del_ratio, random.Random(seed)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return delete_words(self.texts[i], self.del_ratio, self.rng), self.texts[i]


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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "prdect-id.csv"))
    ap.add_argument("--text_column", default="Customer Review")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--output_dir", default=os.path.join(HERE, "output", "bge-m3-sapt-informal-id"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None, help="Stop early (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--del_ratio", type=float, default=0.6)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--log_every", type=int, default=10)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    dec_config = AutoConfig.from_pretrained(args.model, is_decoder=True, add_cross_attention=True)
    dec_config._attn_implementation = "eager"
    decoder = XLMRobertaForCausalLM(dec_config).to(device)
    tied, new = tie_decoder_to_encoder(encoder, decoder)
    params = list({id(p): p for p in list(encoder.parameters()) + list(decoder.parameters())}.values())
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_total = sum(p.numel() for p in params)
    print(f"encoder {n_enc / 1e6:.0f}M params (trainable) | decoder: {len(tied)} tensors tied to encoder, "
          f"{len(new)} new tensors | total trainable {n_total / 1e6:.0f}M")

    loader = DataLoader(DenoisingDataset(texts, args.del_ratio, args.seed), batch_size=args.batch_size,
                        shuffle=True, drop_last=False, collate_fn=lambda b: b)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = len(loader) * args.epochs if args.max_steps is None else min(args.max_steps, len(loader) * args.epochs)
    print(f"steps: {total} ({len(loader)}/epoch, batch {args.batch_size}, lr {args.lr} constant)")

    log_dir = os.path.join(HERE, "runs", "train")
    os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, "log.jsonl"), "w")
    encoder.train()
    decoder.train()
    step, t0, recent = 0, time.time(), []
    for epoch in range(args.epochs):
        for batch in loader:
            if step >= total:
                break
            noisy, original = zip(*batch)
            src = tokenizer(list(noisy), padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            tgt = tokenizer(list(original), padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cls = encoder(**src).last_hidden_state[:, 0]  # BGE-M3 dense pooling, before normalisation
                logits = decoder(input_ids=tgt["input_ids"][:, :-1], attention_mask=tgt["attention_mask"][:, :-1],
                                 encoder_hidden_states=cls[:, None, :],
                                 encoder_attention_mask=torch.ones(cls.size(0), 1, device=device, dtype=torch.long),
                                 use_cache=False).logits
            labels = tgt["input_ids"][:, 1:]
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                                   ignore_index=tokenizer.pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            step += 1
            recent.append(loss.item())
            if step % args.log_every == 0 or step == total:
                avg = sum(recent) / len(recent)
                recent = []
                log.write(json.dumps({"step": step, "epoch": epoch, "loss": avg}) + "\n")
                log.flush()
                print(f"step {step:5d}/{total} | epoch {epoch} | reconstruction loss {avg:.4f} | {time.time() - t0:.0f}s")

    if args.no_save:
        print("--no_save: model not written")
        return
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"saving encoder to {args.output_dir} (free disk {free_gb:.1f} GB)")
    os.makedirs(args.output_dir, exist_ok=True)
    encoder.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "sapt_config.json"), "w") as f:
        json.dump({**vars(args), "reviews_used": len(texts), "steps": step, "final_loss": avg,
                   "decoder_tied_tensors": len(tied), "decoder_new_tensors": len(new)}, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
