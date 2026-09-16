"""UDA Query-DANN: unsupervised domain adaptation from labelled formal queries to unlabelled informal queries.

The ADAPT-DANN / DANN setting carried over to dense retrieval:
  source domain     = formal queries WITH relevance labels          -> InfoNCE task loss
  target domain     = informal queries WITHOUT labels, WITHOUT pairs -> seen only by the domain discriminator
  feature extractor = frozen BGE-M3 + one shared query adapter A_psi, applied to both domains
  L = L_InfoNCE(source) + gamma * L_domain(source vs target) + mu * L_preserve(source)
gamma = 0 is the classic source-only UDA baseline. mu = 0.5 by default (as in Query-DANN v2): without it the
contrastive source loss distorts the pretrained space within a few dozen steps. mu = 0 is the pure DANN objective.

--target_split disjoint (default): training queries are split in half; the source uses only the formal queries
    (+ labels) of one half and the target only the informal queries of the other, so no question is in both domains.
--target_split overlap: source = all training formal queries, target = all training informal queries, sampled
    independently (unpaired, but the same questions exist in both domains).

Checkpoints (strict UDA may not use target labels for model selection):
  best_source/         best dev nDCG@10 of formal queries through the adapter   (strict UDA selection)
  best_target_oracle/  best dev nDCG@10 of informal queries (uses target labels; oracle, report as such)
  last/                final step
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
from accelerate import Accelerator
from accelerate.utils import set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import EmbeddingStore, TrainQueryDataset, build_triples  # noqa: E402
from eval import Evaluator  # noqa: E402
from model import ADAPTERS, DISCRIMINATORS, QueryDANN, grl_alpha  # noqa: E402


class Logger:
    """TensorBoard (if installed) + a JSONL file that is always written."""

    def __init__(self, run_dir: str):
        self.jsonl = open(os.path.join(run_dir, "log.jsonl"), "a")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
        except ImportError:
            self.tb = None

    def log(self, step: int, **scalars):
        self.jsonl.write(json.dumps({"step": step, **scalars}) + "\n")
        self.jsonl.flush()
        if self.tb:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)


def split_domains(train_qids, mode: str, seed: int):
    """Returns (source_qids, target_qids)."""
    if mode == "overlap":
        return list(train_qids), list(train_qids)
    qids = sorted(train_qids, key=int)
    random.Random(seed).shuffle(qids)
    half = len(qids) // 2
    return sorted(qids[:half], key=int), sorted(qids[half:], key=int)


def target_batches(rows, batch_size: int, seed: int):
    """Endless, independently shuffled batches of target (informal) query rows."""
    rng = random.Random(seed)
    while True:
        perm = list(rows)
        rng.shuffle(perm)
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            yield perm[i : i + batch_size]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="uda")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--output_dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--target_split", default="disjoint", choices=["disjoint", "overlap"])
    ap.add_argument("--total_steps", type=int, default=1350, help="Same optimisation budget as Query-DANN v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1.0, help="Domain loss weight (0 = source-only baseline)")
    ap.add_argument("--mu", type=float, default=0.5,
                    help="Preservation loss on source (formal) queries, as in Query-DANN v2; uses no target information. "
                         "0 = pure DANN objective")
    ap.add_argument("--eta", type=float, default=10.0)
    ap.add_argument("--hn_range", type=int, nargs=2, default=[10, 100])
    ap.add_argument("--adapter", default="zeroinit", choices=list(ADAPTERS))
    ap.add_argument("--discriminator", default="sanitized", choices=list(DISCRIMINATORS))
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--adapter_dropout", type=float, default=0.1)
    ap.add_argument("--disc_dropout", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger = Logger(run_dir)

    store = EmbeddingStore(args.cache_dir, device=device)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    with open(os.path.join(args.cache_dir, "hard_negs.json")) as f:
        hard_negs = json.load(f)

    source_qids, target_qids = split_domains(splits["train"], args.target_split, args.seed)
    with open(os.path.join(run_dir, "domains.json"), "w") as f:
        json.dump({"source_formal_labelled": source_qids, "target_informal_unlabelled": target_qids}, f)
    # Hard negatives come from the base model's formal-query ranking with source labels removed: source-only information.
    source_ds = TrainQueryDataset(build_triples(args.data_dir, hard_negs), source_qids, store,
                                  hn_range=tuple(args.hn_range), seed=args.seed)
    loader = source_ds.loader(args.batch_size)
    target_iter = target_batches([store.qid2row[q] for q in target_qids], args.batch_size, args.seed + 1)
    dev = Evaluator(store, splits["dev"], args.data_dir)

    head = QueryDANN(args.bottleneck, args.adapter_dropout, args.disc_dropout, args.adapter, args.discriminator).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.lr_min)
    head, optimizer = accelerator.prepare(head, optimizer)
    raw = accelerator.unwrap_model(head)

    accelerator.print(f"[{args.run_name}] split={args.target_split} source(formal+labels)={len(source_ds)} "
                      f"target(informal, unlabelled)={len(target_qids)} steps={args.total_steps} gamma={args.gamma} mu={args.mu}")
    base_inf, base_form = dev.run(dev.informal()), dev.run(dev.formal())
    accelerator.print(f"dev base informal nDCG@10 {base_inf['nDCG@10']:.4f} | base formal nDCG@10 {base_form['nDCG@10']:.4f}")

    best = {"source": -1.0, "target_oracle": -1.0}
    bce = F.binary_cross_entropy_with_logits
    step, t0 = 0, time.time()

    def save(name: str, info: dict):
        d = os.path.join(run_dir, name)
        os.makedirs(d, exist_ok=True)
        accelerator.save(raw.adapter.state_dict(), os.path.join(d, "adapter.pt"))
        accelerator.save(raw.discriminator.state_dict(), os.path.join(d, "discriminator.pt"))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"method": "uda_query_dann", "adapter": args.adapter, "discriminator": args.discriminator,
                       "bottleneck": args.bottleneck, "adapter_dropout": args.adapter_dropout,
                       "disc_dropout": args.disc_dropout, "step": step, **info}, f, indent=2)

    @torch.no_grad()
    def evaluate(alpha: float):
        z_inf, z_form = dev.adapted(raw, "informal"), dev.adapted(raw, "formal")
        m_tgt, m_src = dev.run(z_inf), dev.run(z_form)
        raw.eval()
        disc_acc = 0.5 * ((raw.discriminator(z_inf) < 0).float().mean() + (raw.discriminator(z_form) > 0).float().mean()).item()
        head.train()
        cos_twin = (z_inf * dev.formal()).sum(-1).mean().item()
        info = {"dev_informal": m_tgt, "dev_formal_through_adapter": m_src, "dev_disc_accuracy": disc_acc,
                "dev_cos_adapted_informal_vs_base_formal": cos_twin}
        logger.log(step, **{f"dev/informal_{k}": v for k, v in m_tgt.items()},
                   **{f"dev/formal_{k}": v for k, v in m_src.items()}, **{"dev/disc_accuracy": disc_acc, "dev/cos_twin": cos_twin})
        marks = ""
        for key, metrics in (("source", m_src), ("target_oracle", m_tgt)):
            if metrics["nDCG@10"] > best[key]:
                best[key] = metrics["nDCG@10"]
                save(f"best_{key}", info)
                marks += " S" if key == "source" else " T"
        accelerator.print(f"step {step:5d} a {alpha:.3f} | dev informal nDCG@10 {m_tgt['nDCG@10']:.4f} | "
                          f"formal {m_src['nDCG@10']:.4f} | D acc {disc_acc:.3f} | cos(A(inf),form) {cos_twin:.4f}{marks} "
                          f"({time.time() - t0:.0f}s)")
        return info

    info = {}
    head.train()
    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps:
                break
            B = batch["z_form"].size(0)
            z_target_in = store.informal[torch.tensor(next(target_iter), device=device)].float()
            alpha = grl_alpha(step / args.total_steps, args.eta)
            with accelerator.autocast():
                z_s = raw.adapt(batch["z_form"])  # source: formal queries through the shared adapter
                z_t = raw.adapt(z_target_in)  # target: unpaired, unlabelled informal queries
                dom_s = raw.domain_logits(z_s, alpha)
                dom_t = raw.domain_logits(z_t, alpha)
            with torch.autocast(device.type, enabled=False):
                sims = (z_s.float() @ batch["z_cand"].T) / args.tau
                sims = sims.masked_fill(batch["false_neg"], float("-inf"))
                labels = torch.arange(B, device=device)
                loss_nce = F.cross_entropy(sims, labels)
                dom_s, dom_t = dom_s.float(), dom_t.float()
                # y=1 source (formal), y=0 target (informal), class-balanced.
                loss_dom = 0.5 * bce(dom_s, torch.ones_like(dom_s)) + 0.5 * bce(dom_t, torch.zeros_like(dom_t))
                loss_pres = (1 - (z_s.float() * batch["z_form"]).sum(-1)).mean()
                loss = loss_nce + args.gamma * loss_dom + args.mu * loss_pres

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 10 == 0:
                logger.log(step, **{"loss/total": loss.item(), "loss/infonce_source": loss_nce.item(),
                                    "loss/domain": loss_dom.item(), "loss/preserve": loss_pres.item(),
                                    "grl/alpha": alpha, "lr": optimizer.param_groups[0]["lr"],
                                    "disc/accuracy": 0.5 * ((dom_s > 0).float().mean() + (dom_t < 0).float().mean()).item()})
            if step % args.eval_every == 0 or step == args.total_steps:
                info = evaluate(alpha)

    save("last", info)
    accelerator.print(f"[{args.run_name}] best dev formal (strict selection) {best['source']:.4f} | "
                      f"best dev informal (oracle) {best['target_oracle']:.4f}")


if __name__ == "__main__":
    main()
