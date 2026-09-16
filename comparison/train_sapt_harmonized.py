"""SAPT (TSDAE) under the SHARED training protocol used by every trained model in the comparison.

Reuses the TSDAE machinery of ../tsdae_sapt/train_sapt.py unchanged (data loading, deletion noise,
decoder tying) and only replaces the training protocol so it matches the adapters:

  optimiser steps 1350 | effective batch 64 (micro-batch x grad accumulation) | AdamW lr 3e-5 -> 1e-6
  cosine with 10% warmup | weight decay 0.01 | grad clip 1.0 | bf16 | max_length 128 | seed 42
  checkpoint selection: best dev nDCG@10 on informal queries, evaluated every 50 optimiser steps

Full fine-tuning cannot hold an effective batch of 64 in 24 GB, so it is reached by accumulation
(micro-batch 8 x 8) - the optimiser sees the same 64 examples per step as the adapters do.

  python3 comparison/train_sapt_harmonized.py --source reviews   # prdect-id informal reviews
  python3 comparison/train_sapt_harmonized.py --source queries    # the task's own informal train queries
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tsdae_sapt"))

from dataset import EmbeddingStore, load_train_pairs  # noqa: E402
from eval import Evaluator  # noqa: E402
import train_sapt as sapt  # noqa: E402  (../tsdae_sapt/train_sapt.py, unmodified)


@torch.no_grad()
def encode_cls(encoder, tokenizer, texts, device, max_length: int, batch_size: int = 64) -> torch.Tensor:
    """BGE-M3 dense pooling: [CLS] then L2 norm - the representation the frozen passage index uses."""
    was_training = encoder.training
    encoder.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i: i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = encoder(**enc).last_hidden_state[:, 0]
        out.append(F.normalize(cls.float(), dim=-1))
    encoder.train(was_training)
    return torch.cat(out)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="reviews", choices=["reviews", "queries"],
                    help="Unlabelled SAPT text: prdect-id informal reviews, or the task's own informal train queries")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "prdect-id.csv"))
    ap.add_argument("--text_column", default="Customer Review")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--output_dir", default=os.path.join(HERE, "runs", "harmonized", "sapt_reviews", "model"))
    # shared protocol
    ap.add_argument("--total_steps", type=int, default=1350)
    ap.add_argument("--batch_size", type=int, default=8, help="Micro-batch; effective batch = this x grad_accum")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    # SAPT-specific
    ap.add_argument("--del_ratio", type=float, default=0.6)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.dirname(os.path.abspath(args.output_dir))
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "train_log.jsonl"), "a")

    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    tp = load_train_pairs(args.data_dir).set_index("query_id")
    if args.source == "reviews":
        texts, n_unique = sapt.load_reviews(args.data, args.text_column, args.min_words)
        print(f"SAPT text: {len(texts)} informal reviews kept of {n_unique} unique (labels discarded)")
    else:
        texts = [tp.at[q, "generated_informal_query"] for q in splits["train"]]
        print(f"SAPT text: {len(texts)} informal TRAIN queries (no labels, no dev/test text)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    dec_config = AutoConfig.from_pretrained(args.model, is_decoder=True, add_cross_attention=True)
    dec_config._attn_implementation = "eager"
    decoder = XLMRobertaForCausalLM(dec_config).to(device)
    tied, new = sapt.tie_decoder_to_encoder(encoder, decoder)
    params = list({id(p): p for p in list(encoder.parameters()) + list(decoder.parameters())}.values())
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.0f}M params | decoder: {len(tied)} tied, {len(new)} new")

    loader = DataLoader(sapt.DenoisingDataset(texts, args.del_ratio, args.seed), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, collate_fn=lambda b: b)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_ratio * args.total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup if s < warmup else
        max(args.lr_min / args.lr, 0.5 * (1 + torch.cos(torch.tensor(
            3.141592653589793 * (s - warmup) / max(1, args.total_steps - warmup))).item()))))
    effective = args.batch_size * args.grad_accum
    print(f"protocol: {args.total_steps} optimiser steps, effective batch {effective} "
          f"({args.batch_size} x {args.grad_accum} accum), lr {args.lr}->{args.lr_min} cosine, "
          f"wd {args.weight_decay}, max_length {args.max_length}, eval every {args.eval_every}")

    store = EmbeddingStore(args.cache_dir, device=device)
    dev = Evaluator(store, splits["dev"], args.data_dir)
    dev_informal = [tp.at[q, "generated_informal_query"] for q in dev.qids]
    dev_formal = [tp.at[q, "formal_query"] for q in dev.qids]
    base_inf, base_form = dev.run(dev.informal()), dev.run(dev.formal())
    print(f"dev base informal nDCG@10 {base_inf['nDCG@10']:.4f} | base formal nDCG@10 {base_form['nDCG@10']:.4f}")

    best, step, micro, t0, recent = -1.0, 0, 0, time.time(), []
    encoder.train()
    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    done = False
    while not done:
        for batch in loader:
            noisy, original = zip(*batch)
            src = tokenizer(list(noisy), padding=True, truncation=True, max_length=args.max_length,
                            return_tensors="pt").to(device)
            tgt = tokenizer(list(original), padding=True, truncation=True, max_length=args.max_length,
                            return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cls = encoder(**src).last_hidden_state[:, 0]
                logits = decoder(input_ids=tgt["input_ids"][:, :-1], attention_mask=tgt["attention_mask"][:, :-1],
                                 encoder_hidden_states=cls[:, None, :],
                                 encoder_attention_mask=torch.ones(cls.size(0), 1, device=device, dtype=torch.long),
                                 use_cache=False).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt["input_ids"][:, 1:].reshape(-1),
                                   ignore_index=tokenizer.pad_token_id)
            (loss / args.grad_accum).backward()
            recent.append(loss.item())
            micro += 1
            if micro % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            sched.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.eval_every == 0 or step == args.total_steps:
                m_inf = dev.run(encode_cls(encoder, tokenizer, dev_informal, device, args.max_length))
                m_form = dev.run(encode_cls(encoder, tokenizer, dev_formal, device, args.max_length))
                avg = sum(recent) / len(recent)
                recent = []
                improved = m_inf["nDCG@10"] > best
                log.write(json.dumps({"step": step, "loss": avg, "lr": optimizer.param_groups[0]["lr"],
                                      "dev_informal": m_inf, "dev_formal": m_form}) + "\n")
                log.flush()
                print(f"step {step:5d}/{args.total_steps} | loss {avg:.4f} | dev informal nDCG@10 {m_inf['nDCG@10']:.4f} "
                      f"| dev formal {m_form['nDCG@10']:.4f}{' *' if improved else ''} ({time.time() - t0:.0f}s)")
                if improved:
                    best = m_inf["nDCG@10"]
                    encoder.save_pretrained(args.output_dir, safe_serialization=True)
                    tokenizer.save_pretrained(args.output_dir)
                    with open(os.path.join(args.output_dir, "sapt_config.json"), "w") as f:
                        json.dump({**vars(args), "effective_batch": effective, "texts": len(texts),
                                   "selection": "best dev informal nDCG@10", "step": step,
                                   "dev_informal": m_inf, "dev_formal": m_form, "reconstruction_loss": avg}, f, indent=2)
            if step >= args.total_steps:
                done = True
                break

    print(f"done: best dev informal nDCG@10 {best:.4f} (base {base_inf['nDCG@10']:.4f}) -> {args.output_dir}")


if __name__ == "__main__":
    main()
