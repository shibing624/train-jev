# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 全参训练：FSDP2 切分 fp32 主权重、梯度和 AdamW 状态，bf16 计算，其余配方与 LoRA 版一致。

Full-parameter variant of ``trainjev.train`` (selected with ``--full_ft``).

Everything except the trainable parameters is shared with the LoRA trainer: the same data, kind-stratified sampling,
option permutation, KL + RPS loss, lr schedule, validation and checkpoint selection. Differences:

  * every weight of the Qwen3.5 text model is trained (embeddings, all layers, final norm); ``lm_head`` is not used
  * the model is built on the meta device on every rank and sharded first; only rank 0 reads the checkpoint and the
    weights are broadcast into the shards one tensor at a time, so host memory holds one copy of the weights (on
    rank 0) instead of one fp32 copy per rank
  * FSDP2 (``fully_shard`` per decoder layer + root): fp32 master weights, gradients and AdamW states sharded across
    ranks; parameters all-gathered in bf16 for compute, gradients reduce-scattered in fp32
  * the 24-slot head stays a replicated fp32 module whose gradients are all-reduced by hand, exactly as with LoRA
  * ranks pad their micro-batch count to the global maximum with zero-weight dummy passes, because FSDP collectives
    run once per forward / backward on every rank
  * gradient clipping at global norm 1.0 over (sharded backbone grads, replicated head grads)
  * checkpoints: ``model.safetensors`` (text-model weights, bf16, keys as in ``Qwen3_5TextModel``) + ``head.pt``
"""
import json
import os
import time

import torch
import torch.distributed as dist

from trainjev.attention import configure_prompt_attention, save_attention_metadata
from trainjev.data import encode_many, micro_batches
from trainjev.metrics import metrics
from trainjev.modeling import VERBALIZER_IDS, DecisionHead, check_verbalizers, last_hidden
from trainjev.resume import load_state, save_state


def init_rotary(body):
    """Recompute the non-persistent rotary buffers (``inv_freq``), which a meta-device model leaves uninitialised and
    which are not in the checkpoint."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    for m in body.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_type"):
            fn = m.compute_default_rope_parameters if m.rope_type == "default" else ROPE_INIT_FUNCTIONS[m.rope_type]
            inv, scaling = fn(m.config, m.inv_freq.device)
            m.inv_freq.copy_(inv)
            m.original_inv_freq.copy_(inv)
            m.attention_scaling = scaling


def load_sharded(base, rank, dev, shard, cpu_offload=False, *, experimental_prompt_bidirectional=False):
    """Build the Qwen3.5 text model on the meta device, shard it with ``shard(body)``, then fill the shards from rank
    0's copy of the checkpoint. -> ``(body, init_rows)`` with ``init_rows`` the fp32 ``lm_head`` rows of the
    verbalizers (from rank 0, broadcast)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM

    with torch.device("meta"):
        body = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(base), dtype=torch.float32).model
    configure_prompt_attention(body, experimental_prompt_bidirectional)
    shard(body)
    if cpu_offload:  # parameter shards live on the host; buffers (rotary) still go to the GPU, as fully_shard does
        body.to_empty(device="cpu")
        for m in body.modules():
            for k, b in m._buffers.items():
                if b is not None:
                    m._buffers[k] = b.to(dev)
    else:
        body.to_empty(device=dev)
    with torch.no_grad():
        init_rotary(body)
    full, rows = {}, torch.empty(len(VERBALIZER_IDS), body.config.hidden_size, device=dev)
    if rank == 0:
        m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        rows.copy_(m.get_output_embeddings().weight[VERBALIZER_IDS].detach())
        full = m.model.state_dict()
        del m
    set_model_state_dict(body, full, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    del full
    dist.broadcast(rows, src=0)
    return body, rows.float().clone()


def main_full(a):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
    from transformers import AutoTokenizer

    from trainjev.train import head_probs, lr_schedule, micro_loss, setup_data, step_batches

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

    train, val, steps, step_rows, augment = setup_data(a, log)
    log(f"train rows {len(train)}  steps {steps}  world {world}  batch {a.batch}  [full-parameter, FSDP2]")
    val_ids = encode_many(tok, val, a.max_len)

    # ------------------------------------------------------------------ model
    t_load = time.time()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    kw = {"mp_policy": mp}
    if a.cpu_offload:
        kw["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

    def shard(body):
        for layer in body.layers:
            fully_shard(layer, **kw)
        fully_shard(body, **kw)

    body, init_rows = load_sharded(a.base, rank, dev, shard, a.cpu_offload,
                                  experimental_prompt_bidirectional=a.experimental_prompt_bidirectional)
    body.config.use_cache = False
    if not a.no_gc:
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head = DecisionHead(init_rows).to(dev)
    body_params = [p for p in body.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    log(f"trainable: backbone {sum(p.numel() for p in body_params) / 1e9:.3f}B (full)  "
        f"head {sum(p.numel() for p in head_params) / 1e3:.0f}k  load+shard {time.time() - t_load:.0f}s  "
        f"mem {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    opt_kw = {"betas": (0.9, 0.98), "weight_decay": 0.0}
    groups = [{"params": body_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}]
    opt = torch.optim.AdamW(groups, **opt_kw, **({"foreach": True} if a.cpu_offload else {"fused": True}))
    base_lrs = [a.lr, a.head_lr]
    lr_mult = lr_schedule(steps)

    def global_max(n):
        t = torch.tensor([n], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def reduce_head_grads():
        for p in head_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        flat = torch.cat([p.grad.flatten() for p in head_params])
        dist.all_reduce(flat)
        flat /= world
        o = 0
        for p in head_params:
            n = p.numel()
            p.grad.copy_(flat[o:o + n].view_as(p))
            o += n

    def clip_grads(max_norm=1.0):
        sq = torch.zeros((), device=dev)
        for p in body_params:
            if p.grad is not None:
                g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                sq += g.float().pow(2).sum().to(dev)
        dist.all_reduce(sq)  # sum of the disjoint shards
        sq += sum(p.grad.float().pow(2).sum() for p in head_params)  # replicated: count once
        total = sq.sqrt()
        coef = float(torch.clamp(max_norm / (total + 1e-6), max=1.0))
        for p in body_params + head_params:
            if p.grad is not None:
                p.grad.mul_(coef)
        return total

    @torch.no_grad()
    def evaluate(rows):
        body.eval()
        mine = list(range(rank, len(rows), world))
        items = [(len(val_ids[i]), i) for i in mine]
        mbs = micro_batches(items, a.micro_tokens * 2)
        res = {}
        for j in range(global_max(len(mbs))):
            if j < len(mbs):
                mb = mbs[j]
                h = last_hidden(body, [val_ids[i] for i in mb], pad, dev)
                res.update(zip(mb, head_probs(head, h, [rows[i] for i in mb])))
            else:
                last_hidden(body, [[pad]], pad, dev)  # keep FSDP collectives in step
        gathered = [None] * world
        dist.all_gather_object(gathered, res)
        probs = [None] * len(rows)
        for g in gathered:
            for i, p in g.items():
                probs[i] = p
        body.train()
        return metrics(rows, probs)

    def save(tag):
        sd = get_model_state_dict(body, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            from safetensors.torch import save_file

            d = os.path.join(a.out, tag)
            os.makedirs(d, exist_ok=True)
            t = time.time()
            save_file({k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()},
                      os.path.join(d, "model.safetensors"), metadata={"format": "pt", "trainjev": "full-v1"})
            attention_config = save_attention_metadata(d, body)
            torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(),
                        "verbalizer_ids": VERBALIZER_IDS, "base": a.base, "full": True, "args": vars(a),
                        "attention_config": attention_config},
                       os.path.join(d, "head.pt"))
            log(f"saved {d} ({time.time() - t:.0f}s)")
        del sd
        dist.barrier()

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
    body.train()
    last = a.max_steps or steps
    for s, rows, ids in step_batches(a, tok, step_rows, augment, rank, world, start, last):
        enc = [(len(e), (e, r)) for e, r in zip(ids, rows)]
        for i, g in enumerate(opt.param_groups):
            g["lr"] = base_lrs[i] * lr_mult(s)
        tot_loss, tot_w = torch.zeros((), device=dev), 0.0
        mbs = micro_batches(enc, a.micro_tokens)
        for j in range(global_max(len(mbs))):
            if j < len(mbs):
                mb = mbs[j]
                h = last_hidden(body, [e for e, _ in mb], pad, dev)
                loss = micro_loss(head, h, [r for _, r in mb], a.rps_w)
                tot_loss += loss.detach()
                tot_w += sum(r["weight"] for _, r in mb)
                tok_seen += sum(len(e) for e, _ in mb)
            else:  # zero-weight pass so every rank runs the same FSDP all-gathers / reduce-scatters
                h = last_hidden(body, [[pad]], pad, dev)
                loss = head.logits(h[0], "noul", 2).sum() * 0.0
            (loss / a.batch * world).backward()  # FSDP averages over ranks -> mean over the global batch
        reduce_head_grads()
        gn = clip_grads(1.0)
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
            if a.save_steps:
                save(f"step{s + 1:05d}")
            if m["all"]["kl"] < best[0]:
                best = (m["all"]["kl"], s + 1)
                save("best")
        if a.save_state_every and (s + 1) % a.save_state_every == 0:
            save_state(state_dir, body, head, opt, {"step": s + 1, "best": best, "hist": hist})
    if not a.save_steps:
        save("last")
    log(f"done; best val KL {best[0]:.4f} at step {best[1]}")
    dist.destroy_process_group()
