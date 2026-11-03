# NOESIS-Σ Unit Tests: Vision Tower (ViT + Aligner), Image Processor & Multimodal Backbone
# Path: tests/test_vision_and_multimodal.py

from __future__ import annotations
import base64
from pathlib import Path

import pytest
import torch

from vision import ViT, Aligner, get_vision_cos_sin, apply_rotary
from image_processor import (
    TEXT,
    IMAGE_START,
    IMAGE,
    IMAGE_NEW_LINE,
    IMAGE_END,
    ImageInput,
    num_image_tokens,
    llm_grid,
    solve_resize_ratio,
    safe_resize,
    load_image_bytes,
    plan_image_grid,
    image_token_types,
)
from model import ModelArgs, NoesisSigmaTransformer


def test_image_processor_geometry_and_loaders(tmp_path: Path) -> None:
    """Verify grid sizing, aspect-ratio resize solver, token layout, and byte loaders."""
    # Grid & token count formulas
    n_h, n_w = llm_grid(best_height=84, best_width=84, patch_size=14, downsample_ratio=3)
    assert (n_h, n_w) == (2, 2)
    assert num_image_tokens(n_h, n_w) == 2 * (2 + 1) + 2 == 8

    # Aspect ratio solver: normal, very tall, very wide
    h_norm, w_norm = solve_resize_ratio(height=1000, width=1000, patch_size=14, downsample_ratio=3, max_n_token=50)
    assert h_norm > 0 and w_norm > 0
    h_tall, w_tall = solve_resize_ratio(height=5000, width=50, patch_size=14, downsample_ratio=3, max_n_token=20)
    assert w_tall == 14 * 3
    h_wide, w_wide = solve_resize_ratio(height=50, width=5000, patch_size=14, downsample_ratio=3, max_n_token=20)
    assert h_wide == 14 * 3

    # safe_resize shrinking when exceeding max_n_token
    sh, sw, bh, bw = safe_resize(840, 840, 840, 840, patch_size=14, downsample_ratio=3, max_n_token=30)
    assert num_image_tokens(sh, sw) <= 30

    # plan_image_grid with small image (triggers vision_min_pixels upscaling)
    args = ModelArgs(vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=64, vision_min_pixels=14700)
    gh, gw, _, _ = plan_image_grid(width=20, height=20, args=args)
    assert num_image_tokens(gh, gw) <= 64

    # image_token_types structure
    types = image_token_types(2, 3)
    assert types[0].item() == IMAGE_START
    assert types[-1].item() == IMAGE_END
    assert (types == IMAGE).sum().item() == 6
    assert (types == IMAGE_NEW_LINE).sum().item() == 2

    # load_image_bytes from raw bytes, base64, Anthropic source, data URL, and file path
    raw = b"\x89PNG\r\n\x1a\nfakedata"
    b64 = base64.b64encode(raw).decode("ascii")
    assert load_image_bytes({"data": raw}) == raw
    assert load_image_bytes({"data": b64}) == raw
    assert load_image_bytes({"source": {"data": b64}}) == raw
    assert load_image_bytes({"url": f"data:image/png;base64,{b64}"}) == raw

    img_file = tmp_path / "sample.bin"
    img_file.write_bytes(raw)
    assert load_image_bytes({"url": str(img_file)}) == raw

    with pytest.raises(ValueError):
        load_image_bytes({"url": "data:image/png,notbase64"})
    with pytest.raises(ValueError):
        load_image_bytes({"invalid": 123})


def test_vit_aligner_and_multimodal_transformer_integration(device: torch.device) -> None:
    """Verify ViT + Aligner 2D RoPE forward pass and multimodal span merging inside NoesisSigmaTransformer."""
    torch.manual_seed(20)
    args = ModelArgs(
        max_batch_size=2,
        max_seq_len=64,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=512,
        dim=64,
        moe_inter_dim=128,
        n_layers=2,
        n_mtp_layers=0,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=32,
        window_size=32,
        compress_ratios=(0, 0),
        kv_source_layers=(),
        index_source_layers=(),
        candidate_source_layer=-1,
        hc_mult=2,
        engram_layer_ids=(),
        vision_n_layers=2,
        vision_dim=32,
        vision_inter_dim=64,
        vision_n_heads=4,
        vision_patch_size=4,
        vision_downsample_ratio=2,
        image_token_id=500,
        osc_enabled=True,
        osc_rank=16,
        osc_icnn_hidden=32,
        osc_steps=2,
    )
    assert args.vision_enabled

    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    model = NoesisSigmaTransformer(args).to(device)
    assert model.vision is not None

    # Create synthetic 4x4 ViT patch grid -> 2x2 LLM token grid after 2x2 downsample
    n_vit_h, n_vit_w = 4, 4
    n_llm_h, n_llm_w = 2, 2
    p = args.vision_patch_size
    patches = torch.randn(n_vit_h * n_vit_w, 3, p, p, device=device, dtype=torch.bfloat16)
    types = image_token_types(n_llm_h, n_llm_w).to(device)  # length = 2*(2+1)+2 = 8 tokens
    img_input = ImageInput(start=2, patches=patches, n_vit_h=n_vit_h, n_vit_w=n_vit_w, types=types)

    seq_len = 14
    input_ids = torch.randint(0, 400, (1, seq_len), device=device)
    input_ids[0, 2 : 2 + types.numel()] = args.image_token_id

    token_types = torch.full((1, seq_len), TEXT, dtype=torch.long, device=device)
    token_types[0, 2 : 2 + types.numel()] = types

    # Differentiable multimodal training pass
    logits, osc_metrics, hidden = model.forward_train(input_ids, images=[[img_input]], token_types=token_types)
    assert logits.shape == (1, seq_len, args.vocab_size)
    loss = logits.float().square().mean()
    loss.backward()
    assert model.vision.patch_embed.proj.weight.grad is not None
    assert model.image_start.grad is not None
