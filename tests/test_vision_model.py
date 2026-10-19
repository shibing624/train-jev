"""Exercise the real Qwen visual tower on a tiny random model; no downloads or training."""
import base64
import io
import re
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

import trainjev.modeling as modeling
import trainjev.serve as serve
from trainjev.assets import VISION_TOKEN, prepare_assets


def tiny_model():
    text = Qwen3_5TextConfig(
        vocab_size=4096, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        layer_types=["full_attention"], num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0,
                         "mrope_section": [3, 3, 2]},
    )
    vision = Qwen3_5VisionConfig(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=4, in_channels=3,
        patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=32,
        num_position_embeddings=64,
    )
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict(), image_token_id=120,
                         vision_start_token_id=121, vision_end_token_id=122)
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(config, dtype=torch.float32).eval()


class Tokenizer:
    def __init__(self):
        self.ids = dict(zip(modeling.VERBALIZERS, modeling.VERBALIZER_IDS))
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for index, label in enumerate(alphabet):
            self.ids[label] = 32 + index
        for index, label in enumerate(a + b for a in alphabet for b in alphabet):
            self.ids[label] = 400 + index

    def encode(self, text, add_special_tokens=False):
        if text in self.ids:
            return [self.ids[text]]
        label = re.search(r"\n([A-Z]{1,2})\)", text)
        return [10, self.ids[label[1]], 11] if label else [10, 11]

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[10, 11])


class Processor:
    """Use real image patching and tiny-vocabulary token layout; avoid a large tokenizer download."""
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.image_processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2)

    def __call__(self, *, text, images, return_tensors, truncation):
        assert return_tensors == "pt" and truncation is False
        assert text[0].count(VISION_TOKEN) == len(images)
        pixels = self.image_processor(images=images, return_tensors="pt")
        token_ids = [1]
        for grid in pixels["image_grid_thw"]:
            count = int(grid.prod()) // 4
            token_ids += [121] + [120] * count + [122]
        token_ids.append(2)
        ids = torch.tensor([token_ids])
        self.last_inputs = BatchFeature(dict(pixels, input_ids=ids, attention_mask=torch.ones_like(ids),
                                            mm_token_type_ids=ids.eq(120).long()))
        return self.last_inputs


def test_vision_loader_merges_adapter_and_shares_the_real_text_body(tmp_path, monkeypatch):
    from peft import LoraConfig, get_peft_model

    original = tiny_model()
    base = tmp_path / "base"
    original.save_pretrained(base)
    processor = Processor()
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *a, **k: processor)

    # Construct and save an adapter, but never take an optimizer/training step.
    text = get_peft_model(original.model.language_model,
                         LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], lora_dropout=0.0))
    with torch.no_grad():
        for name, parameter in text.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.02)
    adapter = tmp_path / "adapter"
    text.save_pretrained(adapter)
    weights = torch.randn(24, 32)
    bias = torch.arange(24).float() / 100
    torch.save({"weight": weights, "bias": bias, "verbalizer_ids": modeling.VERBALIZER_IDS}, adapter / "head.pt")

    tokenizer, body, head, loaded_processor, vl_body = modeling.load_vision_model(
        str(base), str(adapter), dtype=torch.float32, device="cpu")
    assert tokenizer is processor.tokenizer and loaded_processor is processor
    assert body is vl_body.language_model
    assert not any("lora_" in name for name, _ in body.named_parameters())
    assert torch.equal(head.weight, weights) and torch.equal(head.bias, bias)
    assert head.wide.shape == (240, 32)
    assert not vl_body.training


@pytest.mark.parametrize("size", [(280, 280), (1, 1), (16, 16), (2048, 64), (4096, 1)])
def test_pixels_reach_real_qwen_tower_and_calibrated_head(monkeypatch, size):
    model = tiny_model()
    processor = Processor()
    processor.tokenizer.choice_labels, _ = modeling.choice_labels(processor.tokenizer)
    head = modeling.DecisionHead(model.get_output_embeddings().weight[modeling.VERBALIZER_IDS])
    monkeypatch.setattr(serve, "S", {
        "args": serve.parse_args(["--vision", "--device", "cpu", "--max_len", "1024"]),
        "tok": processor.tokenizer, "body": model.model.language_model, "head": head,
        "processor": processor, "vision": model.model, "temps": {"noul": 2.0},
    })
    results = []
    for color in ["red", "blue"]:
        output = io.BytesIO()
        Image.new("RGB", size, color).save(output, format="PNG")
        normalized, assets = prepare_assets(
            {"state": {"door": "photo-ref"}, "assets": {"photo-ref": {
                "format": "png", "data_base64": base64.b64encode(output.getvalue()).decode()}},
             "questions": {"q": {"type": "noul", "instructions": "Is door red?"}}}, {})
        try:
            state = serve._state_text(normalized["state"])
            probabilities, count = serve._hf_image("noul", state, "Is door red?", ["false", "true"], assets)
            inputs = processor.last_inputs
            with torch.inference_mode():
                h = model.model(**inputs).last_hidden_state[0, -1]
                expected = torch.softmax(head.logits(h, "noul", 2) / 2.0, -1).tolist()
            assert probabilities == pytest.approx(expected)
            assert sum(probabilities) == pytest.approx(1.0)
            assert count == inputs["input_ids"].shape[-1]
            results.append(probabilities)
        finally:
            for asset in assets.markers.values():
                asset.image.close()
    assert abs(results[0][1] - results[1][1]) > 1e-6


def test_cuda_device_index_keeps_bfloat16_setup(monkeypatch):
    recorded = []
    processor = Processor()
    monkeypatch.setattr(serve, "S", {})
    head = type("Head", (), {"attention_mode": "causal", "lm_head": None})()
    monkeypatch.setattr(serve, "load_vision_model", lambda *args, **kwargs: (
        recorded.append(kwargs) or processor.tokenizer, object(), head, processor, object()))
    serve.setup(serve.parse_args(["--vision", "--base_only", "--device", "cuda:0"]))
    assert recorded[0]["dtype"] == torch.bfloat16
    assert recorded[0]["device"] == "cuda:0"
