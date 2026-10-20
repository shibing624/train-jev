"""Experimental complete-prompt bidirectionality for the transformers 5.17.0 Qwen3.5 text decoder.

Independent implementation using the stock decoder's per-layer mask mapping, not a copied decoder loop or kernel.
Concept reference: StartLux (Apache-2.0), startlux_decide/backends/bidirectional.py at
83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01. No code from that 5.8-only helper is reused.
The stock layer calls, rotary embeddings, output capture and gradient checkpointing remain in transformers.
"""
import json
from pathlib import Path
from types import MethodType

import torch

FLAG = "experimental_prompt_bidirectional"
CAUSAL = "causal"
BIDIRECTIONAL = "prompt_bidirectional_full_attention_v1"
SUPPORTED_TRANSFORMERS = "5.17.0"


def add_attention_argument(parser):
    parser.add_argument(f"--{FLAG}", action="store_true",
                        help="EXPERIMENTAL: full-attention layers read each complete text prompt bidirectionally; "
                             "linear layers stay causal; no vision, caching or chunked prompts")


def attention_metadata(body):
    config = getattr(body, "config", None)
    enabled = getattr(config, FLAG, False)
    return {"format_version": 1, "mode": BIDIRECTIONAL if enabled else CAUSAL,
            "transformers": SUPPORTED_TRANSFORMERS if enabled else None,
            "full_attention_layers": [i for i, t in enumerate(getattr(config, "layer_types", []))
                                      if t == "full_attention"],
            "linear_attention": "causal", "prompt_scope": "complete_independent_right_padded_rows",
            "kv_cache": False if enabled else None, "vision_supported": False if enabled else None}


def _metadata_mode(metadata):
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        raise ValueError("Unsupported train-jev attention metadata")
    mode = metadata.get("mode")
    if mode not in (CAUSAL, BIDIRECTIONAL):
        raise ValueError(f"Unsupported checkpoint attention mode: {mode!r}")
    if mode == BIDIRECTIONAL and metadata.get("transformers") != SUPPORTED_TRANSFORMERS:
        raise ValueError("Bidirectional checkpoint requires a different transformers implementation")
    return mode


def validate_attention_metadata(metadata, enabled, *, strict=False, vision=False):
    mode = CAUSAL if metadata is None else _metadata_mode(metadata)
    if vision and (enabled or mode == BIDIRECTIONAL):
        raise ValueError("Experimental prompt-bidirectional + vision/MRoPE is not supported")
    if mode == BIDIRECTIONAL and not enabled:
        raise ValueError(f"Bidirectional checkpoint requires --{FLAG}; refusing causal loading")
    if strict and (mode == BIDIRECTIONAL) != enabled:
        raise ValueError("Resume attention mode differs from the saved training state")


def checkpoint_attention(checkpoint, head_state=None):
    """Check both the sidecar and the head's redundant marker; legacy unmarked checkpoints are causal."""
    metadata = None
    sidecar = Path(checkpoint) / "attention_config.json"
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        _metadata_mode(metadata)
    if head_state is None:
        head_path = Path(checkpoint) / "head.pt"
        if head_path.exists():
            head_state = torch.load(head_path, map_location="cpu", weights_only=True)
    if head_state is not None:
        head_metadata = head_state.get("attention_config")
        if head_metadata is not None:
            _metadata_mode(head_metadata)
            if metadata is not None and head_metadata != metadata:
                raise ValueError("Checkpoint head and attention sidecar disagree")
            metadata = head_metadata
        declared = head_state.get("args", {}).get(FLAG)
        if declared is not None:
            if type(declared) is not bool:
                raise ValueError("Invalid checkpoint bidirectional flag")
            if metadata is not None and (_metadata_mode(metadata) == BIDIRECTIONAL) != declared:
                raise ValueError("Checkpoint args and attention metadata disagree")
            if declared and metadata is None:
                raise ValueError("Bidirectional checkpoint is missing its implementation metadata")
    return metadata


def check_checkpoint_attention(checkpoint, enabled=False, *, vision=False):
    validate_attention_metadata(checkpoint_attention(checkpoint) if checkpoint else None, enabled, vision=vision)


def save_attention_metadata(path, body):
    metadata = attention_metadata(body)
    (Path(path) / "attention_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _prompt_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, use_cache=None, **kwargs):
    from transformers.masking_utils import create_recurrent_attention_mask

    if not getattr(self.config, FLAG, False) or self.config._attn_implementation != "sdpa":
        raise ValueError("Experimental prompt attention config/backend changed after enabling")
    if past_key_values is not None or use_cache:
        raise ValueError("Experimental bidirectional prompts require use_cache=False and no past_key_values")
    if inputs_embeds is not None or input_ids is None:
        raise ValueError("Experimental bidirectional prompts currently require text input_ids, not image embeddings")
    if set(kwargs) - {"return_dict", "output_attentions", "output_hidden_states"}:
        raise ValueError(f"Unsupported experimental prompt arguments: {sorted(kwargs)}")
    if input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("Each independent prompt row must contain at least one token")
    rows, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device).expand(rows, -1)
    if position_ids is not None and (position_ids.shape != positions.shape or not torch.equal(position_ids, positions)):
        raise ValueError("Experimental bidirectional prompts require ordinary unchunked text positions; no MRoPE")
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("Experimental prompt attention requires a 2D right-padding mask, not packed/custom masks")
        mask = attention_mask.to(input_ids.device)
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("Prompt padding mask must contain only 0 and 1")
        valid = mask.bool()
        lengths = valid.sum(-1)
        if not bool((lengths > 0).all()) or not torch.equal(valid, positions < lengths[:, None]):
            raise ValueError("Experimental prompt rows must be nonempty and right-padded")
        attention_mask = mask
    embeddings = self.embed_tokens(input_ids)
    # SDPA broadcasts [B, 1, 1, L] over heads/queries. Every valid query sees ALL valid keys in its own row.
    # A non-None bool mask disables SDPA's implicit causal term, even in an unpadded batch.
    masks = {"full_attention": valid[:, None, None, :],
             "linear_attention": create_recurrent_attention_mask(
                 config=self.config, inputs_embeds=embeddings, attention_mask=attention_mask, past_key_values=None)}
    return self._trainjev_causal_forward(inputs_embeds=embeddings, attention_mask=masks,
                                        position_ids=position_ids, past_key_values=None, use_cache=False, **kwargs)


def configure_prompt_attention(body, enabled=False):
    """Opt in per model, never globally. False leaves the existing forward/config untouched."""
    config = getattr(body, "config", None)
    if not enabled:
        if getattr(config, FLAG, False):
            raise ValueError(f"Model config requires --{FLAG}; refusing causal loading")
        return body
    import transformers

    decoder = body.get_base_model() if hasattr(body, "get_base_model") else body
    config = decoder.config
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(f"Experimental prompt attention supports transformers {SUPPORTED_TRANSFORMERS}, "
                           f"found {transformers.__version__}")
    if config.model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise ValueError("Experimental prompt attention supports only Qwen3.5 text decoders")
    types = config.layer_types
    if len(types) != config.num_hidden_layers or len(decoder.layers) != len(types) or "full_attention" not in types:
        raise ValueError("Experimental prompt attention requires config-declared full-attention layers")
    for layer, kind in zip(decoder.layers, types):
        if (kind not in {"full_attention", "linear_attention"} or layer.block_type != kind
                or (kind == "full_attention") != hasattr(layer, "self_attn")):
            raise ValueError("Decoder layer does not match its configured attention type")
    # Use the installed SDPA interface; never patch the linear mixer, layer loop or a global attention registry.
    decoder.set_attn_implementation("sdpa")
    if not hasattr(decoder, "_trainjev_causal_forward"):
        decoder._trainjev_causal_forward = decoder.forward
        decoder.forward = MethodType(_prompt_forward, decoder)
    setattr(config, FLAG, True)
    config.use_cache = False
    return body
