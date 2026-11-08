import json

import numpy as np
import pytest
import torch

from wspdelta import ChannelScale, LowRank, analyze, bf16, fit_channel_scale, fit_low_rank
from wspdelta.export import peft_adapter
from wspdelta.runtime import MultiVersion, materialized


def random_bits(rng, shape, scale=0.05):
    return bf16.from_f32((rng.standard_normal(shape) * scale).astype(np.float32))


def test_bf16_rounding_matches_torch():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.standard_normal(200_000).astype(np.float32) * 10.0 ** rng.integers(-30, 30, 200_000),
                             np.array([0.0, -0.0, 1e-40, -1e-40, 3.0e38, 1.00390625, 1.01171875], dtype=np.float32)])
    values = values[np.isfinite(values)].astype(np.float32)
    expected = torch.from_numpy(values).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(bf16.from_f32(values), expected)


def test_every_finite_bf16_pattern_round_trips():
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    finite = np.isfinite(bf16.to_f32(bits))
    assert np.array_equal(bf16.from_f32(bf16.to_f32(bits[finite])), bits[finite])
    assert np.array_equal(bf16.from_f64(bf16.to_f64(bits[finite])), bits[finite])


def test_rounding_interval_contains_only_same_bits():
    rng = np.random.default_rng(2)
    bits = random_bits(rng, 50_000)
    bits[:10] = 0
    bits[10:20] = 0x8000
    low, high = bf16.rounding_interval(bits)
    # Interior points only: low + 1.0 * (high - low) can turn the bound -0.0 into +0.0.
    for t in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.999):
        assert np.array_equal(bf16.from_f64(low + t * (high - low)), bits)


def editor_scale(base_bits, columns, factor, assign_zero=False):
    """What the logged editor did: fp32 tensor *= Python float, cast to bf16 at evaluation."""
    w = bf16.to_f32(base_bits).copy()
    if assign_zero:
        w[:, columns] = 0
    else:
        w[:, columns] *= np.float32(factor)
    return bf16.from_f32(w)


def test_channel_scale_recovered_exactly_with_readable_scales():
    rng = np.random.default_rng(3)
    base = random_bits(rng, (96, 40))
    edited = base
    edits = {(3, 7): 0.5, (10,): 1.5, (12,): 1.1, (14,): -0.5, (21,): 0.0}
    for cols, c in edits.items():
        edited = editor_scale(edited, list(cols), c)
    edited = editor_scale(edited, [20], 0, assign_zero=True)
    best, _, details = analyze(base, edited)
    assert isinstance(best, ChannelScale)
    assert np.array_equal(best.materialize(base), edited)
    found = dict(zip(best.columns.tolist(), best.scales.tolist()))
    assert found[3] == found[7] == 0.5 and found[10] == 1.5 and found[14] == -0.5
    assert found[12] == np.float32(1.1) and found[20] == 0.0 and found[21] == 0.0
    assert best.assigned_zero.tolist() == [col == 20 for col in best.columns.tolist()]


def editor_projection(base_bits, u, alpha, columns=None):
    """W - alpha * outer(u, (u @ W) masked to columns), in float32 like the logged editor."""
    w = bf16.to_f32(base_bits)
    writer = (u.astype(np.float32) @ w).astype(np.float32)
    if columns is not None:
        masked = np.zeros_like(writer)
        masked[columns] = writer[columns]
        writer = masked
    return bf16.from_f32(w - np.float32(alpha) * np.outer(u.astype(np.float32), writer).astype(np.float32))


@pytest.mark.parametrize("masked", [False, True])
def test_rank_one_projection_is_rebuilt_bit_exactly(masked):
    rng = np.random.default_rng(4)
    base = random_bits(rng, (128, 300))
    u = rng.standard_normal(128)
    u /= np.linalg.norm(u)
    columns = np.sort(rng.choice(300, 24, replace=False)) if masked else None
    edited = editor_projection(base, u, 4.0, columns)
    best, options, details = analyze(base, edited)
    assert isinstance(best, LowRank) and best.rank == 1
    assert np.array_equal(best.materialize(base), edited)
    stats = details["low_rank_1"]
    assert stats["refined_mismatches"] <= stats["plain_svd_mismatches"]
    assert stats["relative_residual"] < 0.01
    if masked:
        assert set(best.support.tolist()) <= set(columns.tolist())


def test_mixed_version_batch_matches_separate_materialized_runs():
    torch.random.default_generator.manual_seed(0)
    rng = np.random.default_rng(5)
    model = torch.nn.Sequential(torch.nn.Linear(24, 32, bias=False), torch.nn.Linear(32, 16, bias=False)).double()
    base_bits = {"0.weight": random_bits(rng, (32, 24), 0.2), "1.weight": random_bits(rng, (16, 32), 0.2)}
    with torch.no_grad():
        for name, bits in base_bits.items():
            dict(model.named_parameters())[name].copy_(torch.from_numpy(bf16.to_f64(bits)))
    u = rng.standard_normal(32)
    u /= np.linalg.norm(u)
    edited = {"scale": {"1.weight": editor_scale(base_bits["1.weight"], [2, 5, 9], 0.0)},
              "project": {"0.weight": editor_projection(base_bits["0.weight"], u, 2.0)}}
    versions = {}
    for version, tensors in edited.items():
        versions[version] = {name: analyze(base_bits[name], bits)[0] for name, bits in tensors.items()}
    x = torch.randn(4, 3, 24, dtype=torch.float64)
    names = ["base", "scale", "project", "scale"]
    serving = MultiVersion(model, versions, base_bits, factor_dtype=torch.float64)
    with serving.active(names):
        mixed = model(x)
    for row, version in enumerate(names):
        reference_bits = {} if version == "base" else {
            n: versions[version][n].materialize(base_bits[n]) for n in versions[version]}
        with torch.no_grad():
            for name, bits in reference_bits.items():
                dict(model.named_parameters())[name].copy_(torch.from_numpy(bf16.to_f64(bits)))
            single = model(x[row:row + 1])
            for name, bits in base_bits.items():
                dict(model.named_parameters())[name].copy_(torch.from_numpy(bf16.to_f64(bits)))
        # The low-rank row uses unrounded factors, so it matches only to BF16 weight rounding.
        tolerance = 1e-12 if version in ("base", "scale") else 2e-2
        assert torch.allclose(mixed[row:row + 1], single, rtol=0, atol=tolerance * single.abs().max().item())


def test_materialized_context_restores_weights():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False)).to(torch.bfloat16)
    before = model[0].weight.detach().clone()
    bits = np.zeros((3, 4), dtype=np.uint16)
    with materialized(model, {"0.weight": bits}):
        assert torch.count_nonzero(model[0].weight) == 0
    assert torch.equal(model[0].weight, before)


def test_peft_export_reproduces_delta(tmp_path):
    from safetensors.torch import load_file

    rng = np.random.default_rng(6)
    base = {"model.layers.0.mlp.down_proj.weight": random_bits(rng, (20, 30))}
    edited = editor_scale(base["model.layers.0.mlp.down_proj.weight"], [1, 4], 0.0)
    delta = analyze(base["model.layers.0.mlp.down_proj.weight"], edited)[0]
    info = peft_adapter({"model.layers.0.mlp.down_proj.weight": delta}, base, tmp_path,
                        base_model="toy", revision="0")
    weights = load_file(str(tmp_path / "adapter_model.safetensors"))
    a = weights["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"].double().numpy()
    b = weights["base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"].double().numpy()
    expected = bf16.to_f64(edited) - bf16.to_f64(base["model.layers.0.mlp.down_proj.weight"])
    assert np.allclose(b @ a, expected, atol=0)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["r"] == config["lora_alpha"] == info["rank"] == 2 and config["target_modules"] == ["down_proj"]


def test_serialize_round_trip(tmp_path):
    from wspdelta.serialize import load, save

    rng = np.random.default_rng(7)
    base = random_bits(rng, (64, 50))
    u = rng.standard_normal(64)
    u /= np.linalg.norm(u)
    edits = {"a.weight": editor_scale(base, [3, 9], 0.5), "b.weight": editor_projection(base, u, 2.0, [1, 2, 30])}
    deltas = {name: analyze(base, bits)[0] for name, bits in edits.items()}
    save(tmp_path / "c.npz", deltas)
    loaded = load(tmp_path / "c.npz")
    for name, bits in edits.items():
        assert type(loaded[name]) is type(deltas[name])
        assert np.array_equal(loaded[name].materialize(base), bits)


def test_batched_versions_match_loop_runtime():
    from wspdelta.runtime import BatchedVersions

    torch.random.default_generator.manual_seed(1)
    rng = np.random.default_rng(8)
    model = torch.nn.Sequential(torch.nn.Linear(24, 32, bias=False), torch.nn.Linear(32, 16, bias=False))
    base_bits = {"0.weight": random_bits(rng, (32, 24), 0.2), "1.weight": random_bits(rng, (16, 32), 0.2)}
    with torch.no_grad():
        for name, bits in base_bits.items():
            dict(model.named_parameters())[name].copy_(torch.from_numpy(bf16.to_f32(bits)))
    u = rng.standard_normal(32)
    u /= np.linalg.norm(u)
    versions = {"scale": {"1.weight": analyze(base_bits["1.weight"], editor_scale(base_bits["1.weight"], [2, 5], 0.5))[0]},
                "project": {"0.weight": analyze(base_bits["0.weight"], editor_projection(base_bits["0.weight"], u, 2.0))[0]}}
    x = torch.randn(5, 3, 24)
    names = ["project", "base", "scale", "scale", "project"]
    with MultiVersion(model, versions, base_bits).active(names):
        loop = model(x)
    with BatchedVersions(model, versions, base_bits).active(names):
        batched = model(x)
    assert torch.allclose(loop, batched, atol=1e-5, rtol=1e-5)
