"""Stacked training: Query-DANN v2 + score distillation in ONE adapter (optionally on a TSDAE-SAPT tower).

L_total = L_InfoNCE + gamma * L_domain + mu * L_preserve + kd_weight * L_KD
  L_KD = KL( softmax(z_formal @ C_k / kd_tau) || softmax(A(z_informal) @ C_k / kd_tau) ), C_k = the teacher's
         top-kd_topk passages over the full corpus (teacher = the tower's own formal query embedding)
Train on a TSDAE tower by pointing --cache_dir at its cache. --kd_weight 0 reduces exactly to query_dann/train.py.
  L_preserve = mean(1 - cos(A(z_form), z_form))   keeps formal queries unchanged by the adapter
GRL alpha_p = 2/(1+exp(-eta*p)) - 1. Dev is evaluated every --eval_every steps and the
adapter + discriminator are checkpointed whenever dev nDCG@10 improves.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import CACHE_DIR, DATA_DIR, EmbeddingStore, TrainQueryDataset, build_triples
from eval import Evaluator
from model import ADAPTERS, DISCRIMINATORS, QueryDANN, grl_alpha


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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="query_dann")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--output_dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.1, help="Domain loss weight (0 = no adversarial training)")
    ap.add_argument("--mu", type=float, default=0.5, help="Formal preservation loss weight")
    ap.add_argument("--kd_weight", type=float, default=1.0, help="Score-distillation weight lambda (0 = plain Query-DANN)")
    ap.add_argument("--kd_topk", type=int, default=100, help="Teacher candidate passages for score distillation")
    ap.add_argument("--kd_tau", type=float, default=0.05, help="Softmax temperature for teacher and student scores")
    ap.add_argument("--eta", type=float, default=10.0)
    ap.add_argument("--domain_targets", default="formal", choices=["formal", "formal+doc"],
                    help="What the discriminator treats as label 1: formal queries, or formal queries and positive docs")
    ap.add_argument("--hn_range", type=int, nargs=2, default=[10, 100],
                    help="Sample hard negatives from these ranks of the base-model ranking")
    ap.add_argument("--adapter", default="zeroinit", choices=list(ADAPTERS))
    ap.add_argument("--use_adapter", default="true", choices=["true", "false"],
                    help="false = ablation with no query adapter at all (identity). The encoder is frozen, so "
                         "nothing on the query side is trainable and retrieval equals the base model")
    ap.add_argument("--discriminator", default="sanitized", choices=list(DISCRIMINATORS))
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--adapter_dropout", type=float, default=0.1)
    ap.add_argument("--disc_dropout", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=50, help="Dev evaluation / checkpointing interval in steps")
    ap.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def dev_discriminator_accuracy(raw: QueryDANN, dev: Evaluator) -> float:
    """Balanced accuracy of D on dev: adapted informal -> 0, formal -> 1 (0.5 = domains aligned)."""
    was_training = raw.training
    raw.eval()
    acc = 0.5 * ((raw.discriminator(dev.adapted(raw)) < 0).float().mean()
                 + (raw.discriminator(dev.formal()) > 0).float().mean()).item()
    raw.train(was_training)
    return acc


def main():
    args = parse_args()
    if args.use_adapter == "false":
        args.adapter = "identity"
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    run_dir = os.path.join(args.output_dir, args.run_name)
    best_dir = os.path.join(run_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger = Logger(run_dir)

    store = EmbeddingStore(args.cache_dir, device=device)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    with open(os.path.join(args.cache_dir, "hard_negs.json")) as f:
        hard_negs = json.load(f)
    train_ds = TrainQueryDataset(build_triples(args.data_dir, hard_negs), splits["train"], store,
                                 hn_range=tuple(args.hn_range), seed=args.seed)
    loader = train_ds.loader(args.batch_size)
    dev = Evaluator(store, splits["dev"], args.data_dir)

    head = QueryDANN(args.bottleneck, args.adapter_dropout, args.disc_dropout, args.adapter, args.discriminator).to(device)
    # Only A_psi and D_phi are optimised; the encoder is not even in this process.
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr_min)
                 if args.scheduler == "cosine" else None)
    head, optimizer = accelerator.prepare(head, optimizer)
    raw = accelerator.unwrap_model(head)

    n_adapter = sum(p.numel() for p in raw.adapter.parameters())
    if n_adapter == 0:
        accelerator.print("--use_adapter false: no trainable query-side parameters; only the discriminator "
                          "trains and retrieval stays at the base model's level")
    accelerator.print(f"train queries={len(train_ds)} steps/epoch={len(loader)} total={total_steps} "
                      f"dev queries={len(dev.qids)} adapter params={n_adapter:,}")
    base_inf, base_form = dev.run(dev.informal()), dev.run(dev.formal())
    init = dev.run(dev.adapted(raw))
    accelerator.print(f"dev base informal {base_inf}\ndev base formal   {base_form}\ndev adapter@init  {init}")
    logger.log(0, **{f"dev/base_informal_{k}": v for k, v in base_inf.items()},
               **{f"dev/base_formal_{k}": v for k, v in base_form.items()},
               **{f"dev/adapted_{k}": v for k, v in init.items()})

    best, step, t0 = -1.0, 0, time.time()
    bce = F.binary_cross_entropy_with_logits

    def evaluate_and_checkpoint(epoch: int, alpha: float):
        nonlocal best
        metrics = dev.run(dev.adapted(raw))
        form_metrics = dev.run(dev.adapted(raw, "formal"))
        disc_acc = dev_discriminator_accuracy(raw, dev)
        head.train()
        logger.log(step, epoch=epoch, **{f"dev/adapted_{k}": v for k, v in metrics.items()},
                   **{f"dev/adapted_formal_{k}": v for k, v in form_metrics.items()}, **{"dev/disc_accuracy": disc_acc})
        improved = metrics["nDCG@10"] > best
        accelerator.print(
            f"ep {epoch:2d} step {step:5d} a {alpha:.3f} lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"dev inf nDCG@10 {metrics['nDCG@10']:.4f} MRR@10 {metrics['MRR@10']:.4f} R@100 {metrics['Recall@100']:.4f} | "
            f"formal nDCG@10 {form_metrics['nDCG@10']:.4f} | D acc {disc_acc:.3f} {'*' if improved else ''} ({time.time() - t0:.0f}s)")
        if improved:
            best = metrics["nDCG@10"]
            accelerator.save(raw.adapter.state_dict(), os.path.join(best_dir, "adapter.pt"))
            accelerator.save(raw.discriminator.state_dict(), os.path.join(best_dir, "discriminator.pt"))
            with open(os.path.join(best_dir, "config.json"), "w") as f:
                json.dump({"adapter": args.adapter, "discriminator": args.discriminator, "bottleneck": args.bottleneck,
                           "adapter_dropout": args.adapter_dropout, "disc_dropout": args.disc_dropout,
                           "epoch": epoch, "step": step, "dev": metrics, "dev_formal_through_adapter": form_metrics,
                           "dev_disc_accuracy": disc_acc,
                           "method": "stacked_dann_kd", "gamma": args.gamma, "mu": args.mu,
                           "kd_weight": args.kd_weight, "kd_topk": args.kd_topk, "kd_tau": args.kd_tau}, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        head.train()
        for batch in loader:
            B = batch["z_inf"].size(0)
            alpha = grl_alpha(step / total_steps, args.eta)
            if args.kd_weight > 0:
                # Teacher: the formal twin ranked over the full corpus (no gradient).
                with torch.no_grad(), torch.autocast(device.type, enabled=False):
                    t_scores, t_idx = (batch["z_form"].half() @ store.corpus.T).float().topk(args.kd_topk, dim=1)
                    kd_cand = store.corpus[t_idx].float()
            with accelerator.autocast():
                z_q = raw.adapt(batch["z_inf"])  # adapted informal queries, L2-normalised
                z_pres = raw.adapt(batch["z_form"]) if args.mu > 0 else None
                dom_inf = raw.domain_logits(z_q, alpha)
                targets = [batch["z_form"]] + ([batch["z_cand"][:B]] if args.domain_targets == "formal+doc" else [])
                dom_tgt = raw.domain_logits(torch.cat(targets), alpha)
            with torch.autocast(device.type, enabled=False):
                # InfoNCE: candidates = in-batch positives + mined hard negatives; mask other labelled positives.
                sims = (z_q.float() @ batch["z_cand"].T) / args.tau
                sims = sims.masked_fill(batch["false_neg"], float("-inf"))
                labels = torch.arange(B, device=device)
                loss_nce = F.cross_entropy(sims, labels)
                # Class-balanced BCE: y=0 informal (adapted), y=1 formal target domain.
                dom_inf, dom_tgt = dom_inf.float(), dom_tgt.float()
                loss_dom = 0.5 * bce(dom_inf, torch.zeros_like(dom_inf)) + 0.5 * bce(dom_tgt, torch.ones_like(dom_tgt))
                loss_pres = ((1 - (z_pres.float() * batch["z_form"]).sum(-1)).mean()
                             if z_pres is not None else torch.zeros((), device=device))
                loss_kd = torch.zeros((), device=device)
                if args.kd_weight > 0:
                    s_scores = torch.einsum("bd,bkd->bk", z_q.float(), kd_cand)
                    loss_kd = F.kl_div(F.log_softmax(s_scores / args.kd_tau, dim=-1),
                                       F.softmax(t_scores / args.kd_tau, dim=-1), reduction="batchmean")
                loss = loss_nce + args.gamma * loss_dom + args.mu * loss_pres + args.kd_weight * loss_kd

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1

            if step % 10 == 0:
                disc_acc = 0.5 * ((dom_inf < 0).float().mean() + (dom_tgt > 0).float().mean())
                logger.log(step, **{
                    "loss/total": loss.item(), "loss/infonce": loss_nce.item(), "loss/domain": loss_dom.item(),
                    "loss/preserve": loss_pres.item(), "loss/kd": loss_kd.item(), "grl/alpha": alpha, "lr": optimizer.param_groups[0]["lr"],
                    "disc/accuracy": disc_acc.item(), "train/acc@1": (sims.argmax(1) == labels).float().mean().item(),
                })
            if step % args.eval_every == 0 or step == total_steps:
                evaluate_and_checkpoint(epoch, alpha)

    accelerator.print(f"best dev nDCG@10 {best:.4f} -> {best_dir}")


if __name__ == "__main__":
    main()
