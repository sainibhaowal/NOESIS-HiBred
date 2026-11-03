# NOESIS-Σ Comprehensive CLI, Sharding, Quantized Layers & Edge-Case Coverage Suite
# Path: tests/test_cli_pipelines_and_edge_cases.py

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import verify_noesis_sigma
import engram
import image_processor
import convert
import generate
from model import (
    ModelArgs,
    NoesisSigmaTransformer,
    Linear,
    RowParallelLinear,
    Compressor,
    Gate,
)


def test_verify_noesis_sigma_script_runs_end_to_end() -> None:
    """Execute core/verify_noesis_sigma.py main() directly inside pytest."""
    verify_noesis_sigma.main()


def test_engram_build_compressed_token_map_all_branches() -> None:
    """Test build_compressed_token_map with int, custom list, and backend_tokenizer normalizer."""
    m1, n1 = engram.build_compressed_token_map(16)
    assert n1 == 16 and len(m1) == 16

    m2, n2 = engram.build_compressed_token_map(["a", "b", "c"])
    assert n2 == 3 and m2 == [0, 1, 2]

    class DummyBackend:
        tokens = [" The", "the", "THE", " ", "\ufffdbyte", ""]

        def decode(self, ids, skip_special_tokens=False):
            return self.tokens[ids[0]]

        def id_to_token(self, tid):
            return f"raw_{tid}"

    class DummyTokenizer:
        backend_tokenizer = DummyBackend()

        def __len__(self):
            return len(DummyBackend.tokens)

    lookup, unique_count = engram.build_compressed_token_map(DummyTokenizer())
    assert len(lookup) == 6
    assert lookup[0] == lookup[1] == lookup[2]
    assert unique_count == 4


def test_image_processor_full_branches(tmp_path: Path) -> None:
    """Test source.url, http:// urlopen, load_image, and prepare_vl_inputs branches."""
    raw_file = tmp_path / "img.bin"
    raw_file.write_bytes(b"hello-image")
    assert image_processor.load_image_bytes({"source": {"url": str(raw_file)}}) == b"hello-image"

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"http-bytes"

    with patch.object(image_processor, "urlopen", return_value=DummyResp()):
        assert image_processor.load_image_bytes({"url": "https://example.com/a.png"}) == b"http-bytes"

    class FakePILImage:
        def __init__(self, w: int, h: int):
            self.width = w
            self.height = h

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode: str):
            return self

        def resize(self, size: tuple[int, int]):
            return FakePILImage(size[0], size[1])

        def __array__(self, dtype=None):
            return np.full((self.height, self.width, 3), 128, dtype=dtype or np.float32)

    class FakeImageMod:
        @staticmethod
        def open(_stream):
            return FakePILImage(32, 8)

    class FakeImageOpsMod:
        @staticmethod
        def pad(img, size, color=(127, 127, 127)):
            return FakePILImage(size[0], size[1])

    args = ModelArgs(
        vision_n_layers=1,
        vision_patch_size=4,
        vision_downsample_ratio=2,
        vision_max_n_token=64,
        vision_min_pixels=16,
        vision_max_wh_ratio=2.0,
        image_token_id=500,
    )

    with patch.object(image_processor, "Image", FakeImageMod), patch.object(
        image_processor, "ImageOps", FakeImageOpsMod
    ):
        patches, n_vh, n_vw, n_lh, n_lw = image_processor.load_image({"data": b"raw"}, args)
        assert patches.shape == (n_vh * n_vw, 3, 4, 4)

        args_no_ratio = ModelArgs(
            vision_n_layers=1,
            vision_patch_size=4,
            vision_downsample_ratio=2,
            vision_max_n_token=64,
            vision_min_pixels=16,
            vision_max_wh_ratio=None,
            image_token_id=500,
        )
        patches2, _, _, _, _ = image_processor.load_image({"data": b"raw"}, args_no_ratio)
        assert patches2.ndim == 4

        class FakeTok:
            unk_token_id = 0

            def convert_tokens_to_ids(self, tok):
                return 500

            def encode(self, prompt):
                return [10, 500, 20] if "<image>" in prompt or "<｜image｜>" in prompt else [10, 20]

        tok = FakeTok()
        tokens, t_types, img_inputs = image_processor.prepare_vl_inputs(
            "hello <image> world", [{"data": b"raw"}], tok, args
        )
        assert img_inputs is not None and len(img_inputs) == 1
        assert len(tokens) == len(t_types)

        with pytest.raises(ValueError, match="Found 1 image tokens"):
            image_processor.prepare_vl_inputs("hello <image> world", [], tok, args)

        args_no_vis = ModelArgs(vision_n_layers=0, image_token_id=500)
        with pytest.raises(ValueError, match="no vision tower"):
            image_processor.prepare_vl_inputs("hello <image> world", [{"data": b"raw"}], tok, args_no_vis)


def test_model_quantized_layers_compressor_ratio1_and_window_wraparound(device: torch.device) -> None:
    """Test FP8/FP4 Linear, RowParallelLinear bias, Compressor ratio=1 & remainder, Engram in Transformer, and wraparound."""
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    # 1. get_moe_config for backbone vs DSpark MTP layers
    args_moe = ModelArgs(
        n_layers=2,
        n_routed_experts=4,
        n_activated_experts=2,
        dspark_n_routed_experts=8,
        dspark_n_activated_experts=4,
    )
    assert args_moe.get_moe_config(0) == (4, 2)
    assert args_moe.get_moe_config(2) == (8, 4)

    # 2. FP8 and FP4 Linear + bias + RowParallelLinear bias
    lin_fp8 = Linear(64, 32, bias=False, dtype=torch.float8_e4m3fn).to(device)
    x = torch.randn(2, 4, 64, device=device, dtype=torch.bfloat16)
    y_fp8 = lin_fp8(x)
    assert y_fp8.shape == (2, 4, 32)

    lin_fp4 = Linear(64, 32, bias=False, dtype=torch.float4_e2m1fn_x2).to(device)
    y_fp4 = lin_fp4(x)
    assert y_fp4.shape == (2, 4, 32)

    lin_bias = Linear(64, 32, bias=True, dtype=torch.bfloat16).to(device)
    assert lin_bias.bias is not None

    rpl = RowParallelLinear(64, 32, bias=True, dtype=torch.bfloat16).to(device)
    y_rpl = rpl(x)
    assert y_rpl.shape == (2, 4, 32)

    # 3. Compressor with compress_ratio == 1 and compress_ratio == 4 with trailing remainder & decode
    args_comp = ModelArgs(
        max_batch_size=2,
        max_seq_len=64,
        dim=64,
        head_dim=32,
        compress_ratios=(1, 4),
    )
    comp1 = Compressor(args_comp, layer_id=0).to(device)
    latents1, first1 = comp1(x, start_pos=0)
    assert latents1.shape == (2, 4, 32) and first1 == 0

    comp4 = Compressor(args_comp, layer_id=1).to(device)
    # 6 tokens with ratio 4 -> 4 compressed into 1 latent, 2 remainder tokens stored in state
    x6 = torch.randn(2, 6, 64, device=device, dtype=torch.bfloat16)
    lat, first = comp4(x6, start_pos=0)
    assert lat is not None and lat.shape == (2, 1, 32) and first == 0
    # Decode steps: pos=6 (slot 2 -> None), pos=7 (slot 3 -> completes group of 4 -> returns latent)
    assert comp4(x[:, :1], start_pos=6)[0] is None
    lat_dec, first_dec = comp4(x[:, :1], start_pos=7)
    assert lat_dec is not None and lat_dec.shape == (2, 1, 32) and first_dec == 1

    # 4. Gate with vision_enabled and image_mask
    args_vl_gate = ModelArgs(dim=64, n_routed_experts=4, n_activated_experts=2, vision_n_layers=1)
    gate_vl = Gate(0, args_vl_gate).to(device)
    img_mask = torch.tensor([True, False, True, False], device=device)
    w_vl, i_vl = gate_vl(torch.randn(4, 64, device=device), image_mask=img_mask)
    assert w_vl.shape == (4, 2) and i_vl.shape == (4, 2)

    # 5. Full Transformer with Engram, 2-stage candidate Indexer (uses_candidates), DSpark & ring-buffer wraparound
    args_wrap = ModelArgs(
        max_batch_size=2,
        max_seq_len=64,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=128,
        dim=64,
        moe_inter_dim=128,
        n_layers=3,
        n_mtp_layers=1,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=32,
        window_size=4,  # tiny window so seqlen=10 wraps around
        compress_ratios=(4, 4, 0, 0),
        kv_source_layers=(0,),
        index_source_layers=(0, 1),  # layer 0 is candidate_source, layer 1 uses_candidates
        candidate_source_layer=0,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=4,
        hc_mult=1,
        engram_layer_ids=(0,),
        engram_num_embeddings=(400,),
        engram_vocab_size=50,
        engram_compressed_vocab_size=128,
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_id=0,
        vision_n_layers=0,
        dspark_block_size=2,
        dspark_target_layer_ids=(0,),
        dspark_markov_rank=16,
    )
    model_wrap = NoesisSigmaTransformer(args_wrap).to(device)
    with torch.inference_mode():
        # Short sequence seqlen=2 < compress_ratio=4 and <= window_size=4 -> hits compress_len == 0 and DSpark seqlen <= win
        short_toks = torch.randint(0, 128, (2, 2), device=device)
        out_s, _, main_s = model_wrap(short_toks, start_pos=0)
        model_wrap.forward_spec(out_s, main_s, start_pos=0)

        # Long sequence seqlen=10 > window_size=4 -> hits ring-buffer wraparound in Attention & DSparkAttention
        toks = torch.randint(0, 128, (2, 10), device=device)
        out_ids, _, main_hid = model_wrap(toks, start_pos=0)
        model_wrap.forward_spec(out_ids, main_hid, start_pos=0)
        out_ids2, _, main_hid2 = model_wrap(toks[:, :1], start_pos=10)
        res = model_wrap.forward_spec(out_ids2, main_hid2, start_pos=10)
        assert res is not None


def test_convert_checkpoint_sharding_pipeline(tmp_path: Path) -> None:
    """Test convert.main() sharding logic across mp=2 ranks with mocked safetensors I/O."""
    ckpt_dir = tmp_path / "hf_ckpt"
    save_dir = tmp_path / "out_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "dummy.safetensors").write_bytes(b"")
    (ckpt_dir / "tokenizer.json").write_text("{}")

    fake_tensors = {
        "model.embed.weight": torch.randn(16, 8, dtype=torch.bfloat16, device="cpu"),
        "model.layers.0.self_attn.wq_b.weight": torch.randn(16, 8, dtype=torch.bfloat16, device="cpu"),
        "model.layers.0.self_attn.wo_a.weight": torch.randn(64, 32, device="cpu").to(torch.float8_e4m3fn),
        "model.layers.0.self_attn.wo_a.weight_scale_inv": torch.ones(2, 1, dtype=torch.float32, device="cpu"),
        "model.layers.0.self_attn.attn_sink": torch.zeros(4, dtype=torch.float32, device="cpu"),
        "model.layers.0.mlp.experts.0.w1.weight": torch.randint(-10, 10, (32, 16), dtype=torch.int8, device="cpu"),
        "model.layers.0.mlp.experts.0.w1.weight_scale_inv": torch.ones(32, 1, dtype=torch.float32, device="cpu"),
        "model.layers.0.mlp.experts.1.w1.weight": torch.randint(-10, 10, (32, 16), dtype=torch.int8, device="cpu"),
        "model.layers.0.mlp.experts.1.w1.weight_scale_inv": torch.ones(32, 1, dtype=torch.float32, device="cpu"),
        "model.layers.0.engram.embed.weight": torch.randn(5, 8, dtype=torch.bfloat16, device="cpu"),
        "model.layers.0.engram.embed.scale": torch.ones(5, 1, dtype=torch.float32, device="cpu"),
        "mtp.0.embed.weight": torch.randn(16, 8, dtype=torch.bfloat16, device="cpu"),
    }

    # Write model.safetensors.index.json as well to cover expected_names verification branch
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "dummy.safetensors" for k in fake_tensors}})
    )

    class FakeSafeOpen:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return list(fake_tensors.keys())

        def get_tensor(self, k):
            return fake_tensors[k].clone()

    saved_shards = {}

    def fake_save_file(sd, path):
        saved_shards[path] = sd

    with patch.object(convert, "safe_open", FakeSafeOpen), patch.object(
        convert, "save_file", fake_save_file
    ):
        convert.main(str(ckpt_dir), str(save_dir), mp=2, expert_dtype="fp8")
        assert len(saved_shards) == 2
        assert (save_dir / "tokenizer.json").exists()

        # Also test without index.json and with expert_dtype=None (fp4 view)
        (ckpt_dir / "model.safetensors.index.json").unlink()
        convert.main(str(ckpt_dir), str(save_dir), mp=2, expert_dtype=None)


def test_generate_module_vl_eos_and_cli_modes(device: torch.device, config_10m_dict: dict, tmp_path: Path) -> None:
    """Test generate() with eos_id & images, prepare_case(), and generate.main() interactive & batch modes."""
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    args = ModelArgs(**config_10m_dict)
    model = NoesisSigmaTransformer(args).to(device)

    # Test prepare_case context error & normal path
    with pytest.raises(ValueError, match="context"):
        generate.prepare_case({"context": "old", "messages": []}, "chat", None, args)

    class FakeAutoTok:
        eos_token_id = 1
        unk_token_id = 0

        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def convert_tokens_to_ids(self, _tok):
            return args.image_token_id

        def encode(self, _text):
            return [10, 11, 12]

        def decode(self, _ids):
            return "decoded response<｜end▁of▁sentence｜>"

        def batch_decode(self, id_lists):
            return ["decoded response<｜end▁of▁sentence｜>" for _ in id_lists]

    p_str, p_toks, p_types, p_imgs = generate.prepare_case(
        {"messages": [{"role": "user", "content": "Hello"}]}, "chat", FakeAutoTok(), args
    )
    assert p_imgs is None and len(p_toks) == 3

    # Test generate() with eos_id hit in completion_tokens and with images
    call_count = 0

    def fake_forward(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        # First step emit token 55, second step emit eos_id=99
        tok_id = 55 if call_count == 1 else 99
        return (torch.tensor([tok_id], device=device), None, None)

    fake_img = image_processor.ImageInput(
        start=0,
        patches=torch.zeros(1, 3, 4, 4, device=device),
        n_vit_h=1,
        n_vit_w=1,
        types=torch.tensor([0, 1, 2, 3], device=device),
    )
    with patch.object(model, "forward", side_effect=fake_forward):
        comp = generate.generate(
            model,
            [[1, 2, 3, 4]],
            max_new_tokens=4,
            eos_id=99,
            prompt_token_types=[[0, 1, 2, 3]],
            images=[[fake_img]],
        )
        assert comp == [[55]]

    # Test generate.main() in interactive and batch modes (text, json, and VL)
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps(config_10m_dict))
    txt_file = tmp_path / "prompts.txt"
    txt_file.write_text("Hello NOESIS\n\nSecond prompt")
    json_file = tmp_path / "prompts.json"
    json_file.write_text(json.dumps([{"messages": [{"role": "user", "content": "Hi"}]}]))

    with patch.object(generate, "AutoTokenizer", FakeAutoTok), patch.object(
        generate, "load_model", lambda m, p: None
    ), patch.object(generate, "Transformer", return_value=model), patch.object(
        generate, "generate", return_value=[[10, 11]]
    ):
        # 1. Interactive mode (/clear then message then /exit)
        inputs = iter(["/clear", "What is NOESIS-Sigma?", "/exit"])
        with patch("builtins.input", lambda _prompt="": next(inputs)):
            generate.main(str(tmp_path), str(cfg_file), interactive=True, max_new_tokens=2)

        # 1b. Multi-rank worker (WORLD_SIZE=2, RANK=1) receiving broadcast "/exit"
        with patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "0"}), patch(
            "torch.distributed.init_process_group"
        ), patch("torch.distributed.destroy_process_group"), patch(
            "torch.distributed.broadcast_object_list", side_effect=lambda objs, _src: objs.__setitem__(0, "/exit")
        ):
            generate.main(str(tmp_path), str(cfg_file), interactive=True, max_new_tokens=2)

        # 2. Batch TXT mode
        generate.main(str(tmp_path), str(cfg_file), input_file=str(txt_file), interactive=False, max_new_tokens=2)

        # 3. Batch JSON mode (and VL branch when prepare_case returns image_inputs)
        generate.main(str(tmp_path), str(cfg_file), input_file=str(json_file), interactive=False, max_new_tokens=2)
        with patch.object(
            generate, "prepare_case", return_value=("p", [1, 2, 3, 4], [0, 1, 2, 3], [fake_img])
        ):
            generate.main(
                str(tmp_path), str(cfg_file), input_file=str(json_file), interactive=False, max_new_tokens=2
            )


def test_config_1_5b_specification() -> None:
    """Verifies that config_1.5b.json correctly specifies the 1.5B production model."""
    cfg_path = Path(__file__).resolve().parent.parent / "core" / "config_1.5b.json"
    assert cfg_path.exists()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["dim"] == 1536
    assert cfg["n_layers"] == 18
    assert cfg["n_heads"] == 12
    assert cfg["n_routed_experts"] == 8
    assert cfg["vocab_size"] == 32000
    assert cfg["osc_enabled"] is True
    args = ModelArgs(**cfg)
    assert args.dim == 1536


def test_model_arbitrary_parameter_scales_and_distribution() -> None:
    """Verifies that NOESIS-Σ instantiates and runs across arbitrary parameter configurations,
    including the active 1.5B config, 671B blueprint calculations, and split pipeline distribution.
    """
    # 1. Verify 1.5B active config loads cleanly into ModelArgs
    cfg_1_5b_path = Path(__file__).resolve().parent.parent / "core" / "config_1.5b.json"
    assert cfg_1_5b_path.exists(), "config_1.5b.json must exist"
    with open(cfg_1_5b_path, "r", encoding="utf-8") as f:
        cfg_1_5b = json.load(f)
    args_1_5b = ModelArgs(**cfg_1_5b)
    assert args_1_5b.dim == 1536
    assert args_1_5b.n_layers == 18
    assert args_1_5b.n_routed_experts == 8

    # 2. Verify 671B frontier blueprint config loads cleanly into ModelArgs
    cfg_671b_path = Path(__file__).resolve().parent.parent / "core" / "config_671b.json"
    assert cfg_671b_path.exists(), "config_671b.json must exist"
    with open(cfg_671b_path, "r", encoding="utf-8") as f:
        cfg_671b = json.load(f)
    args_671b = ModelArgs(**cfg_671b)
    assert args_671b.dim == 7168
    assert args_671b.n_layers == 61
    assert args_671b.n_routed_experts == 256

    # 3. Test multi-device pipeline distribution on a lightweight test model
    small_args = ModelArgs(
        dim=64,
        n_layers=4,
        n_heads=2,
        n_routed_experts=2,
        n_activated_experts=1,
        vocab_size=100,
        max_seq_len=32,
        osc_enabled=True,
    )
    test_model = NoesisSigmaTransformer(small_args)
    test_model.distribute_to_gpus(devices=("cpu", "cpu"))
    inp = torch.randint(0, 100, (1, 8))
    out, logits, _ = test_model(inp)
    assert out.shape == (1,)
    train_logits, osc_m, _ = test_model.forward_train(inp)
    assert train_logits.shape == (1, 8, 100)
