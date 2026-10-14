"""train-jev: train JEV-style System 1 typed-decision models (LoRA + 24-slot decision head) on Qwen3.5."""

__version__ = "0.1.0"

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
)

__all__ = [
    "__version__",
    "LABELS", "LORA_TARGETS", "RANGES", "VERBALIZERS", "VERBALIZER_IDS",
    "DecisionHead", "check_verbalizers", "last_hidden",
    "prompt", "encode", "load_rows", "is_placeholder", "micro_batches",
    "kl_rps", "metrics", "ece",
]
