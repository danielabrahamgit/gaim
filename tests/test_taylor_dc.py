"""Small CPU checks for the forward Taylor data-consistency experiment."""
import torch

from gaim.optim import phase_bound
from gaim.taylor_dc import quadratic_model, quadratic_step, least_squares_image, taylor_dc


def test_quadratic_matches_complex_residual_and_is_convex():
    torch.manual_seed(4)
    responses = torch.randn(2, 3, 9, dtype=torch.complex128)
    g = torch.randn(4, 9, dtype=torch.float64)
    r = torch.randn(3, 9, dtype=torch.complex128)
    d = torch.randn(2, 4, dtype=torch.float64)
    H, q = quadratic_model(responses, g, r, chunk=4)
    change = -2j * torch.pi * torch.einsum('bp,pt,bct->ct', d.to(responses.dtype), g.to(responses.dtype), responses)
    actual = (r + change).abs().square().sum()
    predicted = r.abs().square().sum() + 2 * q.dot(d.flatten()) + d.flatten().dot(H @ d.flatten())
    torch.testing.assert_close(actual, predicted)
    assert torch.linalg.eigvalsh(H).min() > -1e-10
    phi = torch.randn(2, 7, dtype=torch.float64)
    delta, gain, _ = quadratic_step(H, q, phi, g, radius=0.02)
    assert phase_bound(delta, phi, g) <= 0.02
    assert gain > 0


def test_cg_matches_weighted_and_unweighted_least_squares():
    torch.manual_seed(5)
    matrix = torch.randn(15, 5, dtype=torch.complex128)
    y = torch.randn(15, dtype=torch.complex128)
    for weight in [torch.ones(15), torch.linspace(0.1, 2, 15)]:
        W = weight.sqrt()[:, None] * matrix

        class Operator:
            forward = lambda self, x: W @ x
            adjoint = lambda self, k: W.H @ k

        data = weight.sqrt() * y
        x, info = least_squares_image(Operator(), data, tolerance=1e-10)
        expected = torch.linalg.lstsq(W, data).solution
        torch.testing.assert_close(x, expected)
        assert info['normal_residual'] < 1e-10


def test_exact_forward_derivative_and_alternating_fit():
    torch.manual_seed(6)
    matrix = torch.randn(30, 5, dtype=torch.complex128)
    phi = torch.randn(1, 5, dtype=torch.float64)
    g = torch.randn(1, 30, dtype=torch.float64)

    class Operator:
        def __init__(self, alphas):
            self.matrix = matrix * torch.exp(-2j * torch.pi * (alphas.T @ phi))

        def forward(self, x):
            return (self.matrix @ x)[None]

        def adjoint(self, k):
            return self.matrix.H @ k.flatten()

    x_true = torch.randn(5, dtype=torch.complex128)
    y = Operator(0.03 * g).forward(x_true)
    A = Operator(torch.zeros_like(g))
    derivative = -2j * torch.pi * g[0] * A.forward(phi[0] * x_true)
    difference = (Operator(1e-6 * g).forward(x_true) - Operator(-1e-6 * g).forward(x_true)) / 2e-6
    torch.testing.assert_close(derivative, difference)
    result = taylor_dc(phi, g, y, Operator, outer_steps=15, radius=0.05)
    initial = (A.forward(result['initial_image']) - y).norm()
    final = (Operator(result['alphas']).forward(result['image']) - y).norm()
    assert final < 0.02 * initial
    for item in result['history']:
        assert item['phase_bound'] <= item['radius'] * (1 + 1e-6)
        if item['accepted']:
            assert item['reconstructed_error'] <= item['fixed_image_error'] + 1e-10
