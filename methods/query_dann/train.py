"""Train the Query-DANN adapter + discriminator on cached frozen BGE-M3 embeddings

L_total = L_InfoNCE + gamma * L_domain + mu * L_preserve
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
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from preprocessing.dataset import CACHE_DIR, DATA_DIR, EmbeddingStore, TrainQueryDataset, build_triples
from methods.eval import Evaluator
from methods.model import ADAPTERS, DISCRIMINATORS, QueryDANN, grl_alpha, warmup_cosine


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
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "methods", "cache"))
    ap.add_argument("--output_dir", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--total_steps", type=int, default=None,
                    help="Fixed optimiser-step budget, overriding --epochs (default: epochs x steps/epoch)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup fraction before the cosine (0 = none)")
    ap.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm (0 = no clipping)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.1, help="Domain loss weight (0 = no adversarial training)")
    ap.add_argument("--mu", type=float, default=0.5, help="Formal preservation loss weight")
    ap.add_argument("--eta", type=float, default=10.0)
    ap.add_argument("--domain_targets", default="formal", choices=["formal", "formal+doc"],
                    help="What the discriminator treats as label 1: formal queries, or formal queries and positive docs")
    ap.add_argument("--hn_range", type=int, nargs=2, default=[10, 100],
                    help="Sample hard negatives from these ranks of the base-model ranking")
    ap.add_argument("--adapter", default="zeroinit", choices=list(ADAPTERS),
                    help="Adapter architecture. Use 'zeroinit' for the bottleneck adapter and 'linear' for the "
                         "no-bottleneck ablation; 'identity' is the no-adapter lower bound.")
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


def setup_experiment(args):
    """Setup experiment: seeding, accelerator, directories, and logger."""
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    run_dir = os.path.join(args.output_dir, args.run_name)
    best_dir = os.path.join(run_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger = Logger(run_dir)
    return accelerator, device, run_dir, best_dir, logger


def load_data_and_model(args, device, accelerator):
    """Load datasets, create dataloaders, and initialize model components."""
    store = EmbeddingStore(args.cache_dir, device=device)
    with open(os.path.join(args.cache_dir, "splits.json")) as f:
        splits = json.load(f)
    with open(os.path.join(args.cache_dir, "hard_negs.json")) as f:
        hard_negs = json.load(f)
    train_ds = TrainQueryDataset(build_triples(args.data_dir, hard_negs), splits["train"], store,
                                 hn_range=tuple(args.hn_range), seed=args.seed)
    loader = train_ds.loader(args.batch_size)
    dev = Evaluator(store, splits["dev"], args.data_dir)

    # The adapter width follows the encoder (1024 for BGE-M3 / mE5, 768 for multilingual-mpnet).
    head = QueryDANN(args.bottleneck, args.adapter_dropout, args.disc_dropout, args.adapter, args.discriminator,
                     dim=store.formal.shape[1]).to(device)
    # Only A_psi and D_phi are optimised; the encoder is not even in this process.
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.total_steps or args.epochs * len(loader)
    if args.scheduler == "none":
        scheduler = None
    elif args.warmup_ratio > 0:
        scheduler = warmup_cosine(optimizer, total_steps, args.warmup_ratio, args.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr_min)
    head, optimizer = accelerator.prepare(head, optimizer)
    raw = accelerator.unwrap_model(head)

    n_adapter = sum(p.numel() for p in raw.adapter.parameters())
    if n_adapter == 0:
        accelerator.print("--adapter identity: no trainable query-side parameters; only the discriminator "
                          "trains and retrieval stays at the base model's level")
    accelerator.print(f"train queries={len(train_ds)} steps/epoch={len(loader)} total={total_steps} "
                      f"dev queries={len(dev.qids)} adapter params={n_adapter:,}")

    return store, splits, hard_negs, train_ds, loader, dev, head, optimizer, scheduler, raw, n_adapter, total_steps


def run_initial_evaluation(raw, dev, logger, args):
    """Run initial evaluation and log results."""
    base_inf, base_form = dev.run(dev.informal()), dev.run(dev.formal())
    init = dev.run(dev.adapted(raw))

    # Print initial evaluation in table format
    print(f"\n{'='*70}")
    print(f"Initial evaluation (step 0)")
    print(f"{'='*70}")
    print(f"{'Model':<12} | {'Formal':^26} | {'Informal':^26}")
    print(f"{'':<12} | {'R@100':>8} {'MRR@10':>8} {'nDCG@10':>8} | {'R@100':>8} {'MRR@10':>8} {'nDCG@10':>8}")
    print(f"{'-'*70}")
    print(f"{'Base':<12} | {base_form['Recall@100']:>8.4f} {base_form['MRR@10']:>8.4f} {base_form['nDCG@10']:>8.4f} | "
          f"{base_inf['Recall@100']:>8.4f} {base_inf['MRR@10']:>8.4f} {base_inf['nDCG@10']:>8.4f}")
    print(f"{'DANN@init':<12} | {'-':>8} {'-':>8} {'-':>8} | "
          f"{init['Recall@100']:>8.4f} {init['MRR@10']:>8.4f} {init['nDCG@10']:>8.4f}")
    print(f"{'='*70}\n")

    logger.log(0, **{f"dev/base_informal_{k}": v for k, v in base_inf.items()},
               **{f"dev/base_formal_{k}": v for k, v in base_form.items()},
               **{f"dev/adapted_{k}": v for k, v in init.items()})

    return base_inf, base_form, init


def train_step(batch, raw, optimizer, scheduler, accelerator, logger, args, step, total_steps, device):
    """Execute a single training step."""
    B = batch["z_inf"].size(0)
    alpha = grl_alpha(step / total_steps, args.eta)
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
        loss_dom = 0.5 * F.binary_cross_entropy_with_logits(dom_inf, torch.zeros_like(dom_inf)) + 0.5 * F.binary_cross_entropy_with_logits(dom_tgt, torch.ones_like(dom_tgt))
        loss_pres = ((1 - (z_pres.float() * batch["z_form"]).sum(-1)).mean()
                     if z_pres is not None else torch.zeros((), device=device))
        loss = loss_nce + args.gamma * loss_dom + args.mu * loss_pres

    optimizer.zero_grad(set_to_none=True)
    accelerator.backward(loss)
    if args.grad_clip > 0:
        accelerator.clip_grad_norm_(raw.parameters(), args.grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    step += 1

    disc_acc = 0.5 * ((dom_inf < 0).float().mean() + (dom_tgt > 0).float().mean())
    if step % 10 == 0:
        logger.log(step, **{
            "loss/total": loss.item(), "loss/infonce": loss_nce.item(), "loss/domain": loss_dom.item(),
            "loss/preserve": loss_pres.item(), "grl/alpha": alpha, "lr": optimizer.param_groups[0]["lr"],
            "disc/accuracy": disc_acc.item(), "train/acc@1": (sims.argmax(1) == labels).float().mean().item(),
        })

    return loss, loss_nce, loss_dom, loss_pres, alpha, disc_acc, step


def evaluate_and_checkpoint(raw, dev, accelerator, logger, optimizer, step, epoch, alpha, best, best_dir, args, base_inf, base_form, total_steps):
    """Evaluate model and checkpoint if improved."""
    metrics = dev.run(dev.adapted(raw))
    form_metrics = dev.run(dev.adapted(raw, "formal"))
    disc_acc = dev_discriminator_accuracy(raw, dev)
    raw.train()  # restore train mode: evaluation helpers temporarily switched the head to eval()
    logger.log(step, epoch=epoch, **{f"dev/adapted_{k}": v for k, v in metrics.items()},
               **{f"dev/adapted_formal_{k}": v for k, v in form_metrics.items()}, **{"dev/disc_accuracy": disc_acc})
    improved = metrics["nDCG@10"] > best

    # Final comparison table, printed once when the step budget is spent.
    if step == total_steps:
        print(f"\n{'='*70}")
        print(f"ep {epoch:2d} step {step:5d}  alpha {alpha:.3f}  lr {optimizer.param_groups[0]['lr']:.2e}  D_acc {disc_acc:.3f}  {'*' if improved else ''}")
        print(f"{'='*70}")
        print(f"{'Model':<12} | {'Formal':^26} | {'Informal':^26}")
        print(f"{'':<12} | {'R@100':>8} {'MRR@10':>8} {'nDCG@10':>8} | {'R@100':>8} {'MRR@10':>8} {'nDCG@10':>8}")
        print(f"{'-'*70}")
        print(f"{'Base':<12} | {base_form['Recall@100']:>8.4f} {base_form['MRR@10']:>8.4f} {base_form['nDCG@10']:>8.4f} | "
              f"{base_inf['Recall@100']:>8.4f} {base_inf['MRR@10']:>8.4f} {base_inf['nDCG@10']:>8.4f}")
        print(f"{'DANN':<12} | {form_metrics['Recall@100']:>8.4f} {form_metrics['MRR@10']:>8.4f} {form_metrics['nDCG@10']:>8.4f} | "
              f"{metrics['Recall@100']:>8.4f} {metrics['MRR@10']:>8.4f} {metrics['nDCG@10']:>8.4f}")
        print(f"{'Delta':<12} | {form_metrics['Recall@100'] - base_form['Recall@100']:>+8.4f} "
              f"{form_metrics['MRR@10'] - base_form['MRR@10']:>+8.4f} {form_metrics['nDCG@10'] - base_form['nDCG@10']:>+8.4f} | "
              f"{metrics['Recall@100'] - base_inf['Recall@100']:>+8.4f} {metrics['MRR@10'] - base_inf['MRR@10']:>+8.4f} "
              f"{metrics['nDCG@10'] - base_inf['nDCG@10']:>+8.4f}")
        print(f"{'='*70}\n")
    elif step % 100 == 0:  # Print minimal progress every 100 steps
        accelerator.print(
            f"ep {epoch:2d} step {step:5d}  alpha {alpha:.3f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
            f"D_acc {disc_acc:.3f}"
        )

    if improved:
        best = metrics["nDCG@10"]
        accelerator.save(raw.adapter.state_dict(), os.path.join(best_dir, "adapter.pt"))
        accelerator.save(raw.discriminator.state_dict(), os.path.join(best_dir, "discriminator.pt"))
        with open(os.path.join(best_dir, "config.json"), "w") as f:
            json.dump({"adapter": args.adapter, "discriminator": args.discriminator, "bottleneck": args.bottleneck,
                       "dim": raw.dim,
                       "adapter_dropout": args.adapter_dropout, "disc_dropout": args.disc_dropout,
                       "epoch": epoch, "step": step, "dev": metrics, "dev_formal_through_adapter": form_metrics,
                       "dev_disc_accuracy": disc_acc}, f, indent=2)

    return improved, best


def main():
    args = parse_args()

    # Setup experiment
    accelerator, device, run_dir, best_dir, logger = setup_experiment(args)

    # Load data and initialize model
    store, splits, hard_negs, train_ds, loader, dev, head, optimizer, scheduler, raw, n_adapter, total_steps = load_data_and_model(args, device, accelerator)

    # Run initial evaluation
    base_inf, base_form, init = run_initial_evaluation(raw, dev, logger, args)

    # Training variables
    best, step, t0 = -1.0, 0, time.time()

    # Training loop: epochs until the step budget is spent (exactly --epochs when --total_steps is unset)
    epoch = 0
    while step < total_steps:
        epoch += 1
        head.train()
        for batch in loader:
            loss, loss_nce, loss_dom, loss_pres, alpha, disc_acc, step = train_step(
                batch, raw, optimizer, scheduler, accelerator, logger, args, step, total_steps, device
            )

            if step % args.eval_every == 0 or step == total_steps:
                _, best = evaluate_and_checkpoint(
                    raw, dev, accelerator, logger, optimizer, step, epoch, alpha, best, best_dir, args,
                    base_inf, base_form, total_steps
                )
            if step >= total_steps:
                break

    accelerator.print(f"best dev nDCG@10 {best:.4f} -> {best_dir}")


if __name__ == "__main__":
    main()
