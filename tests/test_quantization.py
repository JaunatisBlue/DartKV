import pytest
import torch

from dart import dequantize, quantize, quantize_axis, quantize_key_mixed, select_key_channels


@pytest.mark.parametrize("bits,tolerance", [(2, 0.55), (4, 0.12), (8, 0.01)])
def test_round_trip_supports_non_divisible_dimension(bits, tolerance):
    torch.manual_seed(1)
    source = torch.randn(2, 3, 7, dtype=torch.float32)
    packed = quantize(source, bits=bits, group_size=4, metadata_dtype=torch.float32)
    restored = dequantize(packed)
    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    assert torch.max(torch.abs(source - restored)).item() <= tolerance
    assert packed.values.dtype == torch.uint8
    assert packed.nbytes < packed.dense_nbytes


def test_constant_group_is_stable():
    source = torch.full((1, 5), 3.25, dtype=torch.float32)
    restored = quantize(source, bits=2, group_size=8, metadata_dtype=torch.float32).dequantize()
    torch.testing.assert_close(restored, source)


def test_invalid_quantization_inputs():
    with pytest.raises(ValueError, match="bits"):
        quantize(torch.ones(2, 3), bits=3)
    with pytest.raises(ValueError, match="group_size"):
        quantize(torch.ones(2, 3), group_size=0)
    with pytest.raises(TypeError, match="floating"):
        quantize(torch.ones(2, 3, dtype=torch.int32))
    with pytest.raises(ValueError, match="finite"):
        quantize(torch.tensor([[float("nan")]]))


def test_quantize_axis_restores_original_layout():
    torch.manual_seed(3)
    source = torch.randn(2, 3, 9, 5)
    packed = quantize_axis(source, axis=-2, bits=2, group_size=4, metadata_dtype=torch.float32)
    restored = packed.dequantize()
    assert restored.shape == source.shape
    assert packed.scale.shape == (2, 3, 5, 3)
    assert torch.max((source - restored).abs()).item() < 0.9


def test_mixed_key_stores_only_promoted_high_bits():
    torch.manual_seed(4)
    source = torch.randn(1, 2, 17, 16)
    mask, indices = select_key_channels(source, 0.25, strategy="magnitude")
    assert mask.shape == (1, 2, 16)
    assert indices.shape == (1, 2, 4)
    mixed = quantize_key_mixed(source, group_size=8, promote_ratio=0.25, metadata_dtype=torch.float32)
    restored = mixed.dequantize()
    assert restored.shape == source.shape
    assert mixed.high_values.shape[2] == 4
    assert mixed.promote_mask.sum().item() == 8
    assert mixed.nbytes < mixed.dense_nbytes
    assert torch.max((source - restored).abs()).item() < 0.9
