import pytest
import torch

from dart import dequantize, quantize


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
