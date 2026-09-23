import torch

from gaim.metrics import (
    average_edge_strength_metric,
    fgresq_pair_from_logits,
    gaussian_scale_gradient_entropy,
    gaussian_smooth,
    gradient_entropy_metric,
    normalized_gradient_squared_metric,
    to_iqa_input,
)


def test_to_iqa_input_shared_scale():
    img = torch.tensor([[0.0, 2.0], [4.0, 8.0]])
    rgb = to_iqa_input(img, scale=8.0)
    assert rgb.shape == (1, 3, 2, 2)
    assert torch.allclose(rgb[0, 0], rgb[0, 1])
    assert float(rgb.max()) == 1.0
    assert float(rgb.min()) == 0.0


def test_fgresq_pair_p_candidate_is_first_image_class():
    quality0 = torch.tensor([0.6])
    quality1 = torch.tensor([0.2])
    logits = torch.tensor([[0.0, 5.0, 0.0]])
    out = fgresq_pair_from_logits(quality0, quality1, logits)
    assert int(out['rank']) == 1
    assert float(out['p_candidate_better']) > 0.9
    assert float(out['p_other_better']) < 0.1

    swapped = fgresq_pair_from_logits(quality1, quality0, torch.tensor([[5.0, 0.0, 0.0]]))
    assert int(swapped['rank']) == 0
    assert abs(float(out['p_candidate_better']) - float(swapped['p_other_better'])) < 1e-6


def test_fgresq_pair_identical_prefers_similar_class():
    logits = torch.tensor([[0.0, 0.0, 4.0]])
    out = fgresq_pair_from_logits(torch.ones(1), torch.ones(1), logits)
    assert int(out['rank']) == 2
    assert float(out['p_similar']) > 0.8


def test_normalized_gradient_squared_formula():
    img = torch.zeros(4, 4)
    img[1, 1] = 1.0
    g = torch.gradient(img.abs())
    mag = (g[0].square() + g[1].square() + 1e-12).sqrt()
    expected = mag.square().sum() / (mag.sum().square() + 1e-12)
    assert torch.allclose(normalized_gradient_squared_metric(img), expected)
    assert torch.allclose(
        normalized_gradient_squared_metric(3 * img),
        normalized_gradient_squared_metric(img),
        rtol=1e-5,
    )


def test_gaussian_sigma_zero_matches_gradient_entropy():
    img = torch.randn(8, 8)
    ge = gradient_entropy_metric(img)
    stacked = gaussian_scale_gradient_entropy(img, sigmas=(0.0, 0.5))
    assert torch.allclose(stacked[0], ge)
    assert gaussian_smooth(img, 0.0).shape == img.shape


def test_aes_is_mean_sobel_magnitude():
    img = torch.linspace(0, 1, 6).repeat(6, 1)
    score = average_edge_strength_metric(img)
    assert torch.isfinite(score)
    assert score.ndim == 0
    assert average_edge_strength_metric(img).shape == average_edge_strength_metric(img[None])[0].shape
