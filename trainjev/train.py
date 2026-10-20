# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: JEV System 1 训练：Qwen3.5 上的 LoRA + 24 槽位决策头，多卡 DDP。

JEV-recipe System 1 trainer: LoRA + 24-slot decision head on a Qwen3.5 backbone, DDP over N GPUs.

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

``--full_ft`` trains every weight of the text backbone instead of a LoRA (FSDP2: fp32 master weights, gradients and
AdamW states sharded across GPUs, bf16 compute; ``--cpu_offload`` moves the shards to host memory for 27B on 4 GPUs).
Checkpoints then hold the text-model weights (``model.safetensors``, bf16) plus ``head.pt``.

``--save_state_every N`` writes a resumable training state (trainable weights, optimiser, RNG, step, best, history) to
``<out>/state`` every N steps; ``--resume`` continues from it with the same data order (see ``trainjev.resume``).
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from trainjev.attention import add_attention_argument, configure_prompt_attention, save_attention_metadata
from trainjev.data import encode_many, is_placeholder, load_rows, micro_batches
from trainjev.losses import kl_rps_batch, pack_targets
from trainjev.metrics import metrics
from trainjev.modeling import (
    LORA_TARGETS,
    RANGES,
    VERBALIZER_IDS,
    DecisionHead,
    check_verbalizers,
    last_hidden,
    load_trained,
)
from trainjev.resume import load_state, require_state, save_state


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B", help="HF id or local path of the Qwen3.5 backbone")
    ap.add_argument("--data", default="data/jev-distill-corpus-v3", help="dir with train.jsonl / validation.jsonl")
    ap.add_argument("--mix", default="", help="comma-separated extra training jsonl files, each optionally path:N")
    ap.add_argument("--mix_fraction", type=float, default=None,
                    help="exact fraction of rows replaced by same-kind --mix rows; default retains concatenation")
    ap.add_argument("--kind_counts", default="", help="fixed global kind counts, e.g. choice:64,noul:32,score:32")
    ap.add_argument("--init_adapter", default="", help="continue a LoRA + head checkpoint with a fresh optimizer")
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
    ap.add_argument("--brier_w", type=float, default=0.0, help="optional per-row Brier loss coefficient (LoRA only)")
    ap.add_argument("--perm", type=float, default=0.3, help="probability of shuffling a choice row's options")
    ap.add_argument("--placeholder_w", type=float, default=0.05, help="loss weight of placeholder rows (0 drops them)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--val_n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_gc", action="store_true", help="disable gradient checkpointing (faster, more memory)")
    ap.add_argument("--full_ft", action="store_true", help="full-parameter training of the text backbone (FSDP2)")
    ap.add_argument("--cpu_offload", action="store_true", help="--full_ft: keep sharded params/grads/optimizer on CPU")
    ap.add_argument("--save_steps", action="store_true", help="--full_ft: also keep stepNNNNN/ checkpoints")
    ap.add_argument("--save_state_every", type=int, default=0, help="write <out>/state every N steps (0 = never)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/state if it exists")
    add_attention_argument(ap)
    return ap.parse_args(argv)


def setup_data(a, log):
    """-> ``(train, val, steps, step_rows, augment)``, shared by the LoRA and the full-parameter trainers."""

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

    train = [dict(r, _mixed=False) for r in prep(load_rows(f"{a.data}/train.jsonl"))]
    log(f"jev-distill train rows kept: {len(train)}")
    base_n = len(train)
    mix_fraction = getattr(a, "mix_fraction", None)
    if mix_fraction is not None and not 0 <= mix_fraction <= 1:
        raise ValueError("--mix_fraction must lie in [0, 1]")
    for spec in [p for p in a.mix.split(",") if p]:
        p, rep = spec.rsplit(":", 1) if spec.rsplit(":", 1)[-1].isdigit() else (spec, "1")
        extra = prep(load_rows(p)) * int(rep)
        extra = [dict(r, _mixed=True) for r in extra]
        log(f"mix: +{len(extra)} rows from {p} (x{rep})")
        train += extra
    val = load_rows(f"{a.data}/validation.jsonl", a.val_n, seed=1)
    steps = a.steps or (base_n if mix_fraction is not None else len(train)) // a.batch

    by_kind = {}
    sample_n = base_n if mix_fraction is not None else len(train)
    for i, r in enumerate(train[:sample_n]):
        by_kind.setdefault(r["kind"], []).append(i)
    mix_by_kind = {}
    if mix_fraction is not None:
        for i in range(base_n, len(train)):
            mix_by_kind.setdefault(train[i]["kind"], []).append(i)
        if mix_fraction > 0 and not mix_by_kind:
            raise ValueError("A nonzero --mix_fraction requires nonempty --mix data")
    fixed_counts = {}
    for spec in getattr(a, "kind_counts", "").split(","):
        if spec:
            kind, count = spec.split(":")
            if kind in fixed_counts or kind not in by_kind or int(count) < 0:
                raise ValueError("Invalid or duplicated --kind_counts entry")
            fixed_counts[kind] = int(count)
    if fixed_counts and sum(fixed_counts.values()) != a.batch:
        raise ValueError("--kind_counts must sum to --batch")

    def step_rows(step):
        """Kind-stratified (>= 1/6 of the batch per kind) sample for one global step, identical on every rank."""
        rng = random.Random(a.seed * 1_000_003 + step)
        if fixed_counts:
            idx = [i for k, count in fixed_counts.items() for i in rng.sample(by_kind[k], count)]
        else:
            floor = a.batch // 6
            idx = [i for k in by_kind for i in rng.sample(by_kind[k], floor)]
            idx += rng.sample(range(sample_n), a.batch - len(idx))
        rows = [train[i] for i in idx]
        if mix_fraction:
            mrng = random.Random(a.seed * 1_000_003 + step + 97_531)
            eligible = [i for i, r in enumerate(rows) if r["kind"] in mix_by_kind]
            count = round(a.batch * mix_fraction)
            if count > len(eligible):
                raise ValueError("Requested mix fraction exceeds same-kind replacement capacity")
            for i in mrng.sample(eligible, count):
                replacement = train[mrng.choice(mix_by_kind[rows[i]["kind"]])]
                rows[i] = dict(replacement, weight=rows[i]["weight"])
        return rows

    def augment(r, rng):
        if r["kind"] != "choice" or rng.random() >= a.perm:
            return r
        perm = list(range(len(r["options"])))
        rng.shuffle(perm)
        return dict(r, options=[r["options"][j] for j in perm], target=[r["target"][j] for j in perm])

    return train, val, steps, step_rows, augment


def lr_schedule(steps):
    """Linear warm-up over 3% of the steps, then one cosine down to 0.02x."""
    warm = max(1, int(0.03 * steps))

    def lr_mult(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return lr_mult


def step_batches(a, tok, step_rows, augment, rank, world, start, stop):
    """Yield ``(step, rows, token ids)`` of this rank for ``start .. stop - 1``. Sampling, augmentation and
    tokenisation of step ``s + 1`` run in a background thread while step ``s`` trains; both are seeded by the step,
    so the result does not depend on timing."""

    def build(s):
        rng = random.Random(a.seed * 7919 + s * 131 + rank)
        rows = [augment(r, rng) for r in step_rows(s)[rank::world]]
        return rows, encode_many(tok, rows, a.max_len)

    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(build, start) if start < stop else None
        for s in range(start, stop):
            rows, ids = fut.result()
            if s + 1 < stop:
                fut = ex.submit(build, s + 1)
            yield s, rows, ids


def micro_loss(head, h, rows, rps_w, brier_w=0.0):
    """Weighted sum of the per-row KL (+ RPS) over a micro-batch, as a tensor (no host sync)."""
    lo, n, tgt, sc, w = (x.to(h.device, non_blocking=True) for x in pack_targets(rows))
    return (kl_rps_batch(head.slot_logits(h), lo, n, tgt, sc, rps_w, brier_w) * w).sum()


@torch.no_grad()
def head_probs(head, h, rows):
    """Per-row softmax over each row's own slots, with one device-to-host copy per micro-batch."""
    z = head.slot_logits(h).cpu()
    out = []
    for zz, r in zip(z, rows):
        lo, _ = RANGES[r["kind"]]
        out.append(torch.softmax(zz[lo: lo + len(r["options"])], -1).tolist())
    return out


def main(argv=None):
    a = parse_args(argv)
    if a.brier_w < 0:
        raise ValueError("--brier_w must be nonnegative")
    if a.resume:
        require_state(os.path.join(a.out, "state"))
    if a.full_ft:
        if a.init_adapter or a.brier_w:
            raise ValueError("--init_adapter and --brier_w currently support LoRA training only")
        from trainjev.train_full import main_full

        return main_full(a)
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
    train, val, steps, step_rows, augment = setup_data(a, log)
    log(f"train rows {len(train)}  steps {steps}  world {world}  batch {a.batch}")
    val_ids = encode_many(tok, val, a.max_len)

    # ------------------------------------------------------------------ model
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    init_rows = base.get_output_embeddings().weight[VERBALIZER_IDS].detach().float().clone()
    body = base.model  # Qwen3.5 text model (embeddings + layers + final norm); lm_head is not used
    body.config.use_cache = False
    configure_prompt_attention(body, a.experimental_prompt_bidirectional)
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
    head = DecisionHead(init_rows)
    if a.init_adapter:
        body = load_trained(body, head, a.init_adapter, trainable=True,
                            experimental_prompt_bidirectional=a.experimental_prompt_bidirectional)
        if body.peft_config["default"].r != a.lora_r:
            raise ValueError("--lora_r does not match --init_adapter")
        log(f"initialized LoRA and decision head from {a.init_adapter}; optimizer reset")
    else:
        body = get_peft_model(body, cfg)
    if not a.no_gc:
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        body.enable_input_require_grads()
    body.to(dev)
    head.to(dev)
    lora_params = [p for p in body.parameters() if p.requires_grad]
    log(f"trainable: LoRA {sum(p.numel() for p in lora_params) / 1e6:.1f}M  "
        f"head {sum(p.numel() for p in head.parameters()) / 1e3:.0f}k")

    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}],
                            betas=(0.9, 0.98), weight_decay=0.0, fused=True)
    base_lrs = [a.lr, a.head_lr]
    lr_mult = lr_schedule(steps)

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
        items = [(len(val_ids[i]), i) for i in mine]
        res = {}
        for mb in micro_batches(items, a.micro_tokens * 2):
            h = last_hidden(body, [val_ids[i] for i in mb], pad, dev)
            res.update(zip(mb, head_probs(head, h, [rows[i] for i in mb])))
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
        attention_config = save_attention_metadata(d, body)
        torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(),
                    "verbalizer_ids": VERBALIZER_IDS, "base": a.base, "args": vars(a),
                    "attention_config": attention_config}, os.path.join(d, "head.pt"))

    # ------------------------------------------------------------------ train
    hist = []
    if rank == 0:
        with open(os.path.join(a.out, "args.json"), "w") as f:
            json.dump(vars(a), f, indent=1)
    best = (1e9, None)
    start = 0
    state_dir = os.path.join(a.out, "state")
    if a.resume:
        meta = load_state(state_dir, body, head, opt)
        if meta is None:
            raise FileNotFoundError("Training state disappeared during loading; refusing a fresh run with --resume")
        if meta:
            start, best, hist = meta["step"], tuple(meta["best"]), meta["hist"]
            log(f"resumed from {state_dir} at step {start}")
    t0 = time.time()
    tok_seen = 0
    draw_counts = Counter()
    draw_weights = Counter()
    body.train()
    last = a.max_steps or steps
    for s, rows, ids in step_batches(a, tok, step_rows, augment, rank, world, start, last):
        for r in rows:
            key = "mixed" if r.get("_mixed") else "base"
            draw_counts[key] += 1
            draw_counts[f"source:{r.get('domain') or r.get('source', 'unknown')}"] += 1
            draw_weights[key] += r["weight"]
        enc = [(len(e), (e, r)) for e, r in zip(ids, rows)]
        for i, g in enumerate(opt.param_groups):
            g["lr"] = base_lrs[i] * lr_mult(s)
        tot_loss, tot_w = torch.zeros((), device=dev), 0.0
        # ranks may run different numbers of micro-batches: gradients are reduced manually after all of them
        for mb in micro_batches(enc, a.micro_tokens):
            h = last_hidden(body, [e for e, _ in mb], pad, dev)
            loss = micro_loss(head, h, [r for _, r in mb], a.rps_w, a.brier_w)
            tot_loss += loss.detach()
            tot_w += sum(r["weight"] for _, r in mb)
            tok_seen += sum(len(e) for e, _ in mb)
            (loss / a.batch * world).backward()  # mean over the global batch after the all-reduce mean
        all_reduce_grads()
        gn = torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or s == last - 1:
            el, done = time.time() - t0, s + 1 - start
            mem = torch.cuda.max_memory_allocated() / 1e9
            log(f"step {s + 1}/{steps} loss {tot_loss.item() / max(tot_w, 1e-9):.4f} gnorm {gn.item():.3f} "
                f"lr {opt.param_groups[0]['lr']:.2e} {el / done:.2f}s/step  ~{tok_seen * world / el:.0f} tok/s  "
                f"eta {(steps - s - 1) * el / done / 3600:.2f}h  mem {mem:.1f}GB")
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
        if a.save_state_every and (s + 1) % a.save_state_every == 0:
            save_state(state_dir, body, head, opt, {"step": s + 1, "best": best, "hist": hist})
    local_stats = {"counts": dict(draw_counts), "weights": dict(draw_weights), "tokens": tok_seen}
    gathered_stats = [None] * world
    dist.all_gather_object(gathered_stats, local_stats)
    if rank == 0:
        counts, weights = Counter(), Counter()
        for st in gathered_stats:
            counts.update(st["counts"])
            weights.update(st["weights"])
        total = counts["base"] + counts["mixed"]
        stats = {"step_start": start, "step_stop": last, "counts": dict(counts), "weights": dict(weights),
                 "actual_mix_fraction": counts["mixed"] / max(1, total),
                 "tokens": sum(st["tokens"] for st in gathered_stats), "seconds": time.time() - t0}
        with open(os.path.join(a.out, f"sampling_{start:05d}_{last:05d}.json"), "w") as f:
            json.dump(stats, f, indent=2)
        log("[sampling]", json.dumps(stats))
    log(f"done; best val KL {best[0]:.4f} at step {best[1]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
