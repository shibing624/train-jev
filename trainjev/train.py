"""JEV-recipe System 1 trainer: LoRA + 24-slot decision head on a Qwen3.5 backbone, DDP over N GPUs.

Recipe (autotrust/JEV-27B model card, "Training details"):
  * LoRA r16 / alpha 32 / dropout 0.05 on all 10 projection types of both layer kinds
  * 24-slot fp32 head initialised from the backbone's lm_head rows of the verbalizer tokens
  * loss = KL(teacher || model) + 0.5 * RPS on score rows
  * 30% choice-option permutation augmentation
  * 128 rows / step, kind-stratified (each kind >= 1/6 of the batch), one epoch
  * AdamW betas (0.9, 0.98), lr 1e-4 (LoRA) / 2e-4 (head), 3% warm-up, cosine to 0.02x, grad clip 1.0
  * jev-distill ``yuri_v1`` placeholder rows down-weighted to x0.05
  * validation KL on a fixed 4k-row subset every 500 steps; ``best/`` keeps the lowest

Usage::

    torchrun --nproc_per_node 3 -m trainjev.train --base Qwen/Qwen3.5-4B \
        --data data/jev-distill-corpus-v3 --out runs/jev-4b --no_gc --micro_tokens 4096

``--mix`` adds extra jsonl files in the same row schema (``kind/state/question/options/target[/weight]``),
each optionally ``path:repeat``.
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist

from trainjev.data import encode, is_placeholder, load_rows, micro_batches
from trainjev.losses import kl_rps
from trainjev.metrics import metrics
from trainjev.modeling import LORA_TARGETS, VERBALIZER_IDS, DecisionHead, check_verbalizers, last_hidden


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B", help="HF id or local path of the Qwen3.5 backbone")
    ap.add_argument("--data", default="data/jev-distill-corpus-v3", help="dir with train.jsonl / validation.jsonl")
    ap.add_argument("--mix", default="", help="comma-separated extra training jsonl files, each optionally path:N")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = one epoch at --batch rows/step")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke tests); schedule still uses --steps")
    ap.add_argument("--batch", type=int, default=128, help="global rows per optimiser step")
    ap.add_argument("--micro_tokens", type=int, default=12288, help="padded tokens per micro-batch per GPU")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=2e-4)
    ap.add_argument("--rps_w", type=float, default=0.5)
    ap.add_argument("--perm", type=float, default=0.3, help="probability of shuffling a choice row's options")
    ap.add_argument("--placeholder_w", type=float, default=0.05, help="loss weight of placeholder rows (0 drops them)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--val_n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_gc", action="store_true", help="disable gradient checkpointing (faster, more memory)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = "cuda"
    log = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(a.base)
    check_verbalizers(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    # ------------------------------------------------------------------ data
    def prep(rows):
        out = []
        for r in rows:
            w = r.get("weight", 1.0) * (a.placeholder_w if is_placeholder(r) else 1.0)
            if w <= 0:
                continue
            if r["kind"] == "choice" and len(r["options"]) > 16:
                continue
            out.append(dict(r, weight=w))
        return out

    train = prep(load_rows(f"{a.data}/train.jsonl"))
    log(f"jev-distill train rows kept: {len(train)}")
    for spec in [p for p in a.mix.split(",") if p]:
        p, rep = spec.rsplit(":", 1) if spec.rsplit(":", 1)[-1].isdigit() else (spec, "1")
        extra = prep(load_rows(p)) * int(rep)
        log(f"mix: +{len(extra)} rows from {p} (x{rep})")
        train += extra
    val = load_rows(f"{a.data}/validation.jsonl", a.val_n, seed=1)
    steps = a.steps or len(train) // a.batch
    log(f"train rows {len(train)}  steps {steps}  world {world}  batch {a.batch}")

    by_kind = {}
    for i, r in enumerate(train):
        by_kind.setdefault(r["kind"], []).append(i)

    def step_rows(step):
        """Kind-stratified (>= 1/6 of the batch per kind) sample for one global step, identical on every rank."""
        rng = random.Random(a.seed * 1_000_003 + step)
        floor = a.batch // 6
        idx = [i for k in by_kind for i in rng.sample(by_kind[k], floor)]
        idx += rng.sample(range(len(train)), a.batch - len(idx))
        return [train[i] for i in idx]

    def augment(r, rng):
        if r["kind"] != "choice" or rng.random() >= a.perm:
            return r
        perm = list(range(len(r["options"])))
        rng.shuffle(perm)
        return dict(r, options=[r["options"][j] for j in perm], target=[r["target"][j] for j in perm])

    # ------------------------------------------------------------------ model
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    init_rows = base.get_output_embeddings().weight[VERBALIZER_IDS].detach().float().clone()
    body = base.model  # Qwen3.5 text model (embeddings + layers + final norm); lm_head is not used
    body.config.use_cache = False
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
    body = get_peft_model(body, cfg)
    if not a.no_gc:
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        body.enable_input_require_grads()
    body.to(dev)
    head = DecisionHead(init_rows).to(dev)
    lora_params = [p for p in body.parameters() if p.requires_grad]
    log(f"trainable: LoRA {sum(p.numel() for p in lora_params) / 1e6:.1f}M  "
        f"head {sum(p.numel() for p in head.parameters()) / 1e3:.0f}k")

    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}],
                            betas=(0.9, 0.98), weight_decay=0.0, fused=True)
    base_lrs = [a.lr, a.head_lr]
    warm = max(1, int(0.03 * steps))

    def lr_mult(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def all_reduce_grads():
        ps = lora_params + list(head.parameters())
        for p in ps:  # same flat layout on every rank, even if a parameter got no gradient
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat = torch.cat([p.grad.flatten() for p in ps])
        dist.all_reduce(flat)
        flat /= world
        o = 0
        for p in ps:
            n = p.numel()
            p.grad.copy_(flat[o:o + n].view_as(p))
            o += n

    @torch.no_grad()
    def evaluate(rows):
        body.eval()
        probs = [None] * len(rows)
        mine = list(range(rank, len(rows), world))
        items = [(len(encode(tok, rows[i], a.max_len)), i) for i in mine]
        res = {}
        for mb in micro_batches(items, a.micro_tokens * 2):
            h = last_hidden(body, [encode(tok, rows[i], a.max_len) for i in mb], pad, dev)
            for i, hh in zip(mb, h):
                res[i] = torch.softmax(head.logits(hh, rows[i]["kind"], len(rows[i]["options"])), -1).tolist()
        gathered = [None] * world
        dist.all_gather_object(gathered, res)
        for g in gathered:
            for i, p in g.items():
                probs[i] = p
        body.train()
        return metrics(rows, probs)

    def save(tag):
        if rank:
            return
        d = os.path.join(a.out, tag)
        os.makedirs(d, exist_ok=True)
        body.save_pretrained(d)
        torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(),
                    "verbalizer_ids": VERBALIZER_IDS, "base": a.base, "args": vars(a)}, os.path.join(d, "head.pt"))

    # ------------------------------------------------------------------ train
    hist = []
    if rank == 0:
        with open(os.path.join(a.out, "args.json"), "w") as f:
            json.dump(vars(a), f, indent=1)
    best = (1e9, None)
    t0 = time.time()
    tok_seen = 0
    body.train()
    last = a.max_steps or steps
    for s in range(last):
        rng = random.Random(a.seed * 7919 + s * 131 + rank)
        rows = [augment(r, rng) for r in step_rows(s)[rank::world]]
        enc = [(len(e), (e, r)) for r in rows for e in [encode(tok, r, a.max_len)]]
        for i, g in enumerate(opt.param_groups):
            g["lr"] = base_lrs[i] * lr_mult(s)
        tot_loss, tot_w = 0.0, 0.0
        # ranks may run different numbers of micro-batches: gradients are reduced manually after all of them
        for mb in micro_batches(enc, a.micro_tokens):
            h = last_hidden(body, [e for e, _ in mb], pad, dev)
            loss = 0.0
            for (e, r), hh in zip(mb, h):
                z = head.logits(hh, r["kind"], len(r["options"]))
                t = torch.tensor(r["target"], device=dev, dtype=torch.float32)
                lv = kl_rps(z, t / t.sum(), r["kind"], a.rps_w) * r["weight"]
                loss = loss + lv
                tot_loss += lv.item()
                tot_w += r["weight"]
                tok_seen += len(e)
            (loss / a.batch * world).backward()  # mean over the global batch after the all-reduce mean
        all_reduce_grads()
        gn = torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or s == last - 1:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9
            log(f"step {s + 1}/{steps} loss {tot_loss / max(tot_w, 1e-9):.4f} gnorm {gn.item():.3f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {el / (s + 1):.2f}s/step  ~{tok_seen * world / el:.0f} tok/s  "
                f"eta {(steps - s - 1) * el / (s + 1) / 3600:.2f}h  mem {mem:.1f}GB")
        if (s + 1) % a.eval_every == 0 or s == last - 1:
            m = evaluate(val)
            hist.append({"step": s + 1, "val": m, "elapsed_h": (time.time() - t0) / 3600})
            log(f"[eval] step {s + 1} val KL {m['all']['kl']:.4f} | " + " | ".join(
                f"{k} kl {v['kl']:.3f} top1 {v.get('top1') or 0:.3f}" for k, v in m.items() if k != "all"))
            if rank == 0:
                with open(os.path.join(a.out, "history.json"), "w") as f:
                    json.dump(hist, f, indent=1)
            save(f"step{s + 1:05d}")
            if m["all"]["kl"] < best[0]:
                best = (m["all"]["kl"], s + 1)
                save("best")
    log(f"done; best val KL {best[0]:.4f} at step {best[1]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
