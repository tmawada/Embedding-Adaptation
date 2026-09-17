"""Query distillation: the paired / supervised alternative to Query-DANN.

Uses the informal<->formal query pairs directly; the frozen base model on the formal query is the teacher.
  --distill embed : L_kd = mean(1 - cos(A(z_informal), z_formal))                     (embedding-level)
  --distill score : L_kd = KL( softmax(z_formal @ C_k / T) || softmax(A(z_informal) @ C_k / T) ),
                    C_k = the teacher's top-k passages over the full corpus             (score-level, listwise)
Both are label-free: they need the pairs, not relevance judgements. Optional terms:
  + w_infonce * InfoNCE(A(z_informal), labelled positives + hard negatives)   (uses relevance labels)
  + mu * mean(1 - cos(A(z_formal), z_formal))                                  (formal preservation)
Adapter, optimiser, schedule, step budget, dev protocol and checkpoint selection (dev informal nDCG@10)
match Query-DANN v2 so the two are directly comparable.
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

from dataset import EmbeddingStore, TrainQueryDataset, build_triples  # noqa: E402
from eval import Evaluator  # noqa: E402
from model import ADAPTERS, EMB_DIM  # noqa: E402


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


class AdapterHead(torch.nn.Module):
    """Adapter only (no discriminator), exposing the same .adapt() interface Evaluator expects."""

    def __init__(self, adapter: str, bottleneck: int, dropout: float):
        super().__init__()
        self.adapter = ADAPTERS[adapter](EMB_DIM, bottleneck, dropout)

    def adapt(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.adapter(z).float(), dim=-1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="distill")
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--output_dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--distill", default="score", choices=["embed", "score"])
    ap.add_argument("--kd_topk", type=int, default=100, help="Teacher candidate passages for score distillation")
    ap.add_argument("--kd_tau", type=float, default=0.05, help="Softmax temperature for teacher and student scores")
    ap.add_argument("--w_infonce", type=float, default=0.0, help="Weight of the supervised InfoNCE term (0 = label-free)")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.05, help="InfoNCE temperature")
    ap.add_argument("--total_steps", type=int, default=1350)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--hn_range", type=int, nargs=2, default=[10, 100])
    ap.add_argument("--adapter", default="zeroinit", choices=list(ADAPTERS))
    ap.add_argument("--use_adapter", default="true", choices=["true", "false"],
                    help="false = ablation with no query adapter at all (identity): nothing is trainable, "
                         "so the run only measures the base model")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--adapter_dropout", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.use_adapter == "false":
        args.adapter = "identity"
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
    train_ds = TrainQueryDataset(build_triples(args.data_dir, hard_negs), splits["train"], store,
                                 hn_range=tuple(args.hn_range), seed=args.seed)
    loader = train_ds.loader(args.batch_size)
    dev = Evaluator(store, splits["dev"], args.data_dir)

    head = AdapterHead(args.adapter, args.bottleneck, args.adapter_dropout).to(device)
    trainable = [p for p in head.parameters() if p.requires_grad]
    if not trainable:  # --use_adapter false: evaluate the base model once and stop
        accelerator.print("--use_adapter false: no trainable parameters, evaluating the base model only")
        raw = head
        m_inf, m_form = dev.run(dev.adapted(raw)), dev.run(dev.adapted(raw, "formal"))
        accelerator.print(f"dev informal nDCG@10 {m_inf['nDCG@10']:.4f} | dev formal {m_form['nDCG@10']:.4f}")
        d = os.path.join(run_dir, "best")
        os.makedirs(d, exist_ok=True)
        accelerator.save(raw.adapter.state_dict(), os.path.join(d, "adapter.pt"))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"method": "no_adapter_ablation", "adapter": args.adapter, "bottleneck": args.bottleneck,
                       "adapter_dropout": args.adapter_dropout, "step": 0,
                       "dev_informal": m_inf, "dev_formal_through_adapter": m_form,
                       "dev": m_inf}, f, indent=2)
        return
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.lr_min)
    head, optimizer = accelerator.prepare(head, optimizer)
    raw = accelerator.unwrap_model(head)

    accelerator.print(f"[{args.run_name}] distill={args.distill} w_infonce={args.w_infonce} mu={args.mu} "
                      f"train pairs={len(train_ds)} steps={args.total_steps}")
    base_inf, base_form = dev.run(dev.informal()), dev.run(dev.formal())
    accelerator.print(f"dev base informal nDCG@10 {base_inf['nDCG@10']:.4f} | base formal nDCG@10 {base_form['nDCG@10']:.4f}")

    best, step, t0 = -1.0, 0, time.time()

    def save(name: str, info: dict):
        d = os.path.join(run_dir, name)
        os.makedirs(d, exist_ok=True)
        accelerator.save(raw.adapter.state_dict(), os.path.join(d, "adapter.pt"))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"method": "query_distillation", "distill": args.distill, "w_infonce": args.w_infonce,
                       "adapter": args.adapter, "bottleneck": args.bottleneck,
                       "adapter_dropout": args.adapter_dropout, "step": step, **info}, f, indent=2)

    @torch.no_grad()
    def evaluate():
        nonlocal best
        z_inf = dev.adapted(raw, "informal")
        m_inf, m_form = dev.run(z_inf), dev.run(dev.adapted(raw, "formal"))
        cos_twin = (z_inf * dev.formal()).sum(-1).mean().item()
        info = {"dev_informal": m_inf, "dev_formal_through_adapter": m_form, "dev_cos_adapted_informal_vs_base_formal": cos_twin}
        logger.log(step, **{f"dev/informal_{k}": v for k, v in m_inf.items()},
                   **{f"dev/formal_{k}": v for k, v in m_form.items()}, **{"dev/cos_twin": cos_twin})
        improved = m_inf["nDCG@10"] > best
        if improved:
            best = m_inf["nDCG@10"]
            save("best", info)
        accelerator.print(f"step {step:5d} lr {optimizer.param_groups[0]['lr']:.2e} | dev informal nDCG@10 {m_inf['nDCG@10']:.4f} "
                          f"| formal {m_form['nDCG@10']:.4f} | cos(A(inf),form) {cos_twin:.4f}{' *' if improved else ''} "
                          f"({time.time() - t0:.0f}s)")
        return info

    info = {}
    head.train()
    while step < args.total_steps:
        for batch in loader:
            if step >= args.total_steps:
                break
            B = batch["z_inf"].size(0)
            z_form = batch["z_form"]
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                if args.distill == "score":
                    # Teacher: base model on the formal twin, ranked over the full corpus.
                    t_scores, t_idx = (z_form.half() @ store.corpus.T).float().topk(args.kd_topk, dim=1)
                    cand = store.corpus[t_idx].float()  # (B, k, d)
            with accelerator.autocast():
                z_q = raw.adapt(batch["z_inf"])
                z_pres = raw.adapt(z_form) if args.mu > 0 else None
            with torch.autocast(device.type, enabled=False):
                z_q = z_q.float()
                if args.distill == "embed":
                    loss_kd = (1 - (z_q * z_form).sum(-1)).mean()
                else:
                    s_scores = torch.einsum("bd,bkd->bk", z_q, cand)
                    loss_kd = F.kl_div(F.log_softmax(s_scores / args.kd_tau, dim=-1),
                                       F.softmax(t_scores / args.kd_tau, dim=-1), reduction="batchmean")
                loss_nce = torch.zeros((), device=device)
                if args.w_infonce > 0:
                    sims = (z_q @ batch["z_cand"].T) / args.tau
                    sims = sims.masked_fill(batch["false_neg"], float("-inf"))
                    loss_nce = F.cross_entropy(sims, torch.arange(B, device=device))
                loss_pres = (1 - (z_pres.float() * z_form).sum(-1)).mean() if z_pres is not None else torch.zeros((), device=device)
                loss = loss_kd + args.w_infonce * loss_nce + args.mu * loss_pres

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 10 == 0:
                logger.log(step, **{"loss/total": loss.item(), "loss/distill": loss_kd.item(), "loss/infonce": loss_nce.item(),
                                    "loss/preserve": loss_pres.item(), "lr": optimizer.param_groups[0]["lr"]})
            if step % args.eval_every == 0 or step == args.total_steps:
                info = evaluate()

    save("last", info)
    accelerator.print(f"[{args.run_name}] best dev informal nDCG@10 {best:.4f} -> {os.path.join(run_dir, 'best')}")


if __name__ == "__main__":
    main()
