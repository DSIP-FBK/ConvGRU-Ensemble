import torch

from convgru_ensemble.model import EncoderDecoder


def test_forward_single_output_shape():
    model = EncoderDecoder(channels=1, num_blocks=2)
    x = torch.randn(2, 4, 1, 16, 16)
    out = model(x, steps=3, noisy_decoder=False, ensemble_size=1)
    assert out.shape == (2, 3, 1, 16, 16)


def test_forward_ensemble_output_shape():
    model = EncoderDecoder(channels=1, num_blocks=2)
    x = torch.randn(2, 4, 1, 16, 16)
    out = model(x, steps=3, noisy_decoder=False, ensemble_size=5)
    assert out.shape == (2, 3, 5, 16, 16)


def test_forward_different_num_blocks():
    model = EncoderDecoder(channels=1, num_blocks=3)
    x = torch.randn(1, 4, 1, 32, 32)
    out = model(x, steps=2, ensemble_size=1)
    assert out.shape == (1, 2, 1, 32, 32)


def test_noisy_decoder_produces_different_outputs():
    model = EncoderDecoder(channels=1, num_blocks=2)
    model.eval()
    x = torch.randn(1, 4, 1, 16, 16)
    out1 = model(x, steps=2, noisy_decoder=True, ensemble_size=1)
    out2 = model(x, steps=2, noisy_decoder=True, ensemble_size=1)
    # Noisy decoder should produce different outputs (with very high probability)
    assert not torch.allclose(out1, out2)
