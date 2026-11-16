"""TSDAE sentence-adaptive pretraining (SAPT) - ablation variant: trained on the TASK's own queries of the BGE-M3 QUERY tower on unlabelled informal queries.

Stage 1 of the ADAPT-DANN stack (sentence-level objective, unlike DAPT's word-level masked LM).
This variant adapts on the task's own 2,891 informal queries; tsdae_sapt/train_sapt.py adapts on
informal Indonesian reviews (prdect-id) instead:
  - unsupervised: only the informal queries of the TRAIN split are used (no labels, no dev/test text)
  - TSDAE (Wang et al. 2021): delete ~60% of the words, encode the damaged query to ONE vector (CLS),
    and make a decoder reconstruct the original query from that vector only
  - the document tower stays frozen, so the cached 500k passage embeddings remain valid; the query tower
    becomes asymmetric and is re-aligned afterwards by the adapter stage (DANN / distillation)

Checkpoint selection is the same protocol as every other method: best dev nDCG@10 on informal queries.

  python3 models/tsdae_sapt_queries/tsdae_train.py --run_name tsdae_informal --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import InputExample, SentenceTransformer, models
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import EmbeddingStore, load_train_pairs  # noqa: E402
from methods.encoders import ENCODERS  # noqa: E402
from methods.eval import Evaluator  # noqa: E402
from methods.model import warmup_cosine  # noqa: E402


def delete_words(text: str, del_ratio: float, rng: np.random.Generator) -> str:
    """TSDAE word-deletion noise on whitespace tokens (the library default needs nltk + punkt data)."""
    words = text.split()
    if len(words) <= 1:
        return text
    keep = rng.random(len(words)) > del_ratio
    if not keep.any():
        keep[rng.integers(len(words))] = True
    return " ".join(w for w, k in zip(words, keep) if k)


class NoisyQueryDataset(Dataset):
    """(damaged query, original query) pairs - the TSDAE training signal.

    Equivalent to sentence_transformers.datasets.DenoisingAutoEncoderDataset, without its nltk dependency.
    """

    def __init__(self, sentences, del_ratio: float, seed: int = 42):
        self.sentences = list(sentences)
        self.del_ratio = del_ratio
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        original = self.sentences[i]
        return InputExample(texts=[delete_words(original, self.del_ratio, self.rng), original])


def build_model(model_id: str, max_seq_length: int, pooling: str = "cls") -> SentenceTransformer:
    """BGE-M3 backbone with CLS pooling and L2 normalization: matching dense retrieval embedding space
    (--encoder me5: mean pooling, as its frozen passage index was encoded)."""
    word = models.Transformer(model_id, max_seq_length=max_seq_length)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=pooling)
    norm = models.Normalize()
    return SentenceTransformer(modules=[word, pool, norm])


@torch.no_grad()
def encode_queries(model: SentenceTransformer, texts, device, batch_size: int = 128) -> torch.Tensor:
    was_training = model.training
    model.eval()
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    model.train(was_training)
    return F.normalize(torch.from_numpy(emb).to(device).float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="tsdae_informal")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--encoder", default="bge_m3", choices=list(ENCODERS),
                    help="Encoder mode (methods/encoders.py): pooling and query prefix for dev selection; pass the "
                         "matching --model and --cache_dir")
    ap.add_argument("--output_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"))
    ap.add_argument("--include_formal", action="store_true",
                    help="Also use the train-split FORMAL queries as unlabelled SAPT text (default: informal only)")
    ap.add_argument("--del_ratio", type=float, default=0.6, help="TSDAE word-deletion noise")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--total_steps", type=int, default=None,
                    help="Fixed optimiser-step budget, overriding --epochs (default: epochs x steps/epoch)")
    ap.add_argument("--batch_size", type=int, default=16, help="Micro-batch; effective batch = this x --grad_accum")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_min", type=float, default=1e-6, help="Floor of --scheduler cosine")
    ap.add_argument("--scheduler", default="onecycle", choices=["onecycle", "cosine"],
                    help="onecycle = linear warmup/anneal (original); cosine = warmup then cosine to --lr_min")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="Dev evaluation / checkpointing interval in optimiser steps (0 = once per epoch)")
    ap.add_argument("--save_best", default="true", choices=["true", "false"],
                    help="false = do not write the dev-selected encoder to <run>/model (still logged)")
    ap.add_argument("--final_dir", default=None,
                    help="Also save the encoder after the last step here (default: only the dev-selected one)")
    ap.add_argument("--max_seq_length", type=int, default=64)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.output_dir, args.run_name)
    best_dir = os.path.join(run_dir, "model")
    os.makedirs(run_dir, exist_ok=True)
    spec = ENCODERS[args.encoder]
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    tp = load_train_pairs(args.data_dir).set_index("query_id")
    sentences = [tp.at[q, "generated_informal_query"] for q in splits["train"]]
    if args.include_formal:
        sentences += [tp.at[q, "formal_query"] for q in splits["train"]]

    model = build_model(args.model, args.max_seq_length, spec.pooling).to(device)
    loss_fn = DenoisingAutoEncoderLoss(model, decoder_name_or_path=args.model, tie_encoder_decoder=True).to(device)
    ds = NoisyQueryDataset(sentences, args.del_ratio, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=model.smart_batching_collate)

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = args.total_steps or steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = warmup_cosine(optimizer, total_steps, args.warmup_ratio, args.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                                                        pct_start=args.warmup_ratio, anneal_strategy="linear")
    trainable = sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)
    print(f"[{args.run_name}] SAPT sentences={len(sentences)} (informal only={not args.include_formal}) "
          f"steps/epoch={steps_per_epoch} total={total_steps} trainable params={trainable / 1e6:.1f}M "
          f"del_ratio={args.del_ratio}")

    # Frozen document tower + dev queries: the selection signal (dev nDCG@10 on informal queries).
    store = EmbeddingStore(args.cache_dir, device=device)
    dev = Evaluator(store, splits["dev"], args.data_dir)
    # Retrieval encoding uses the encoder's query prefix; the TSDAE objective itself trains on the raw queries.
    dev_informal = [spec.query_prefix + tp.at[q, "generated_informal_query"] for q in dev.qids]
    dev_formal = [spec.query_prefix + tp.at[q, "formal_query"] for q in dev.qids]
    base_inf, base_form = dev.run(dev.informal()), dev.run(dev.formal())
    print(f"dev base informal nDCG@10 {base_inf['nDCG@10']:.4f} | base formal nDCG@10 {base_form['nDCG@10']:.4f}")

    def evaluate(epoch: int, step: int) -> dict:
        m_inf = dev.run(encode_queries(model, dev_informal, device))
        m_form = dev.run(encode_queries(model, dev_formal, device))
        return {"epoch": epoch, "step": step, "dev_informal": m_inf, "dev_formal": m_form}

    best, step, micro, t0 = -1.0, 0, 0, time.time()

    def select(epoch: int, loss_value: float) -> dict:
        nonlocal best
        info = evaluate(epoch, step)
        info["reconstruction_loss"] = loss_value
        log.write(json.dumps(info) + "\n")
        log.flush()
        improved = info["dev_informal"]["nDCG@10"] > best
        print(f"epoch {epoch:3d} step {step:5d} | TSDAE loss {loss_value:.3f} | "
              f"dev informal nDCG@10 {info['dev_informal']['nDCG@10']:.4f} | dev formal {info['dev_formal']['nDCG@10']:.4f}"
              f"{' *' if improved else ''} ({time.time() - t0:.0f}s)")
        if improved:
            best = info["dev_informal"]["nDCG@10"]
        if improved and args.save_best == "true":
            os.makedirs(best_dir, exist_ok=True)
            model[0].auto_model.save_pretrained(best_dir)  # plain HF dir: loadable by model.BGEM3Encoder
            model[0].tokenizer.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "tsdae_info.json"), "w") as f:
                json.dump({"run": args.run_name, "selection": "dev informal nDCG@10", **info}, f, indent=2)
        return info

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" and device == "cuda"
                else torch.autocast("cuda", enabled=False))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while step < total_steps:
        epoch += 1
        running = 0.0
        for features, labels in loader:
            features = [{k: v.to(device) for k, v in f.items()} for f in features]
            with autocast:
                loss = loss_fn(features, labels)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running += loss.item()
            if step % 50 == 0:
                log.write(json.dumps({"step": step, "loss": running / 50, "lr": optimizer.param_groups[0]["lr"]}) + "\n")
                log.flush()
                running = 0.0
            if args.eval_every and (step % args.eval_every == 0 or step == total_steps):
                last = select(epoch, loss.item())
            if step >= total_steps:
                break
        if not args.eval_every:
            last = select(epoch, loss.item())

    print(f"[{args.run_name}] best dev informal nDCG@10 {best:.4f} (base {base_inf['nDCG@10']:.4f})"
          + (f" -> {best_dir}" if args.save_best == "true" else " (not saved: --save_best false)"))
    if args.final_dir:  # what training itself produced, whatever the dev selection kept
        model[0].auto_model.save_pretrained(args.final_dir)
        model[0].tokenizer.save_pretrained(args.final_dir)
        with open(os.path.join(args.final_dir, "tsdae_info.json"), "w") as f:
            json.dump({"run": args.run_name, "selection": "none (final step)", **last}, f, indent=2)
        print(f"[{args.run_name}] final-step encoder (step {step}) -> {args.final_dir}")


if __name__ == "__main__":
    main()
