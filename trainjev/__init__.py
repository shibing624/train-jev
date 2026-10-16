# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: 在 Qwen3.5 上训练 JEV 风格的 System 1 决策模型（LoRA + 24 槽位决策头）。
"""

__version__ = "0.2.0"

from trainjev.data import LABELS, encode, is_placeholder, load_rows, micro_batches, prompt
from trainjev.losses import kl_rps
from trainjev.metrics import ece, metrics
from trainjev.modeling import (
    LORA_TARGETS,
    RANGES,
    VERBALIZER_IDS,
    VERBALIZERS,
    DecisionHead,
    check_verbalizers,
    last_hidden,
    resolve_checkpoint,
)

__all__ = [
    "__version__",
    "LABELS", "LORA_TARGETS", "RANGES", "VERBALIZERS", "VERBALIZER_IDS",
    "DecisionHead", "check_verbalizers", "last_hidden", "resolve_checkpoint",
    "prompt", "encode", "load_rows", "is_placeholder", "micro_batches",
    "kl_rps", "metrics", "ece",
]
