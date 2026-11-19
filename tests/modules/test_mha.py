# Copyright (c) 2026, Tri Dao, Albert Gu.

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


# Load the Python modules directly so these CPU tests do not need the package's
# unrelated CUDA / Triton imports in mamba_ssm.__init__.
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[2]
mha = _load_module("mamba_test_mha_cpu", _ROOT / "mamba_ssm/modules/mha.py")
InferenceParams = _load_module(
    "mamba_test_generation_cpu", _ROOT / "mamba_ssm/utils/generation.py"
).InferenceParams


@pytest.mark.parametrize("num_heads_kv", [1, 2, 4])
@pytest.mark.parametrize("chunks", [(3, 1, 1, 1), (2, 2, 2), (6,)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch_size_offset", [0, 2])
def test_mha_cached_attention(num_heads_kv, chunks, causal, batch_size_offset, monkeypatch):
    monkeypatch.setattr(mha, "flash_attn_with_kvcache", None)
    torch.manual_seed(0)
    model = mha.MHA(
        embed_dim=16, num_heads=4, num_heads_kv=num_heads_kv,
        causal=causal, softmax_scale=0.3, layer_idx=0, dtype=torch.float64,
    )
    x = torch.randn(2, sum(chunks), 16, dtype=torch.float64)
    inference_params = InferenceParams(
        max_seqlen=x.shape[1], max_batch_size=x.shape[0] + batch_size_offset + 1,
        batch_size_offset=batch_size_offset,
    )
    cache = model.allocate_inference_cache(inference_params.max_batch_size, x.shape[1])
    cache[0].fill_(float("nan"))
    inference_params.key_value_memory_dict[0] = cache

    start = 0
    with torch.inference_mode():
        for chunk in chunks:
            end = start + chunk
            expected = model(x[:, :end])[:, start:end]
            actual = model(x[:, start:end], inference_params=inference_params)
            torch.testing.assert_close(actual, expected)
            inference_params.seqlen_offset = end
            start = end
    assert cache[0][:batch_size_offset].isnan().all()
    assert cache[0][batch_size_offset + x.shape[0]:].isnan().all()
