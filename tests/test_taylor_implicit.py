"""CPU checks of implicit derivatives against independently solved dense systems."""
import pytest
import torch

from gaim.optim import (_implicit_responses, implicit_image_recon,
                        taylor_implicit_trust)


class DenseEncoding:
    def __init__(self, matrix, weights):
        self.matrix, self.weights = matrix, weights

    def forward(self, x):
        return self.matrix @ x

    def adjoint(self, y):
        return self.matrix.mH @ (self.weights * y)


def problem(samples):
    torch.manual_seed(34)
    n = 8
    phi = 2 * torch.randn(2, n, dtype=torch.float64)
    bases = 3 * torch.randn(2, samples, dtype=torch.float64)
    matrix = torch.randn(samples, n, dtype=torch.complex128) / n**0.5
    weights = torch.linspace(0.2, 1.4, samples, dtype=torch.float64)
    weights[0] = 0  # A zero density weight is valid.
    y = torch.randn(samples, dtype=torch.complex128)

    def build(alphas):
        phase = alphas.T @ phi
        return DenseEncoding(matrix * torch.exp(-2j * torch.pi * phase), weights)

    def exact(F, lam):
        A = build(F @ bases).matrix
        H = A.mH @ (weights[:, None] * A) + lam * torch.eye(n, dtype=A.dtype)
        return torch.linalg.solve(H, A.mH @ (weights * y))

    return phi, bases, y, build, exact


@pytest.mark.parametrize('samples', [5, 13])
def test_weighted_implicit_derivative(samples):
    phi, bases, y, build, exact = problem(samples)
    F = torch.randn(2, 2, dtype=torch.float64) * 0.01
    settings = dict(regularization=0.15, cg_max_iter=100, cg_tolerance=1e-11)
    operator = build(F @ bases)
    x = implicit_image_recon(operator, y, **settings)
    torch.testing.assert_close(x, exact(F, 0.15), atol=1e-10, rtol=1e-10)
    J = _implicit_responses(operator, x, y, phi, bases, **settings)
    # All coefficients exercise signs, residual term, and nonuniform weighting.
    for b in range(2):
        for p in range(2):
            direction = torch.zeros_like(F)
            direction[b, p] = 1
            eps = 1e-6
            finite_difference = (exact(F + eps * direction, 0.15)
                                 - exact(F - eps * direction, 0.15)) / (2 * eps)
            torch.testing.assert_close(J[b, p], finite_difference, atol=2e-5, rtol=2e-6)


def test_trust_initialization_and_acceptance():
    phi, bases, y, build, exact = problem(5)
    F = torch.tensor([[0.01, -0.02], [0.03, 0.01]], dtype=torch.float64)
    target = exact(F + 0.001, 0.15)
    metric = lambda image: -(image - target).abs().square().sum()
    result = taylor_implicit_trust(
        phi, bases, y, build, metric=metric, F_init=F,
        outer_steps=3, inner_steps=10, radius=0.005, max_radius=0.02,
        regularization=0.15, cg_max_iter=100, cg_tolerance=1e-11, verbose=False)
    torch.testing.assert_close(result['initial_image'], exact(F, 0.15))
    torch.testing.assert_close(result['alphas'], result['F'] @ bases)
    torch.testing.assert_close(result['image'], exact(result['F'], 0.15))
    assert result['final_score'] > result['initial_score']
    assert any(step['accepted'] for step in result['history'])
    for step in result['history']:
        assert step['phase_bound'] <= step['radius'] * (1 + 1e-6)
        if step['accepted']:
            assert step['actual_gain'] > 0


def test_unconverged_solve_is_reported():
    phi, bases, y, build, _ = problem(5)
    with pytest.raises(RuntimeError, match='did not converge'):
        implicit_image_recon(build(torch.zeros_like(bases)), y,
                             cg_max_iter=1, cg_tolerance=1e-12)


def test_zero_data_and_initial_guess():
    _, bases, y, build, exact = problem(5)
    A = build(torch.zeros_like(bases))
    zero = implicit_image_recon(A, torch.zeros_like(y), initial=torch.ones(8, dtype=y.dtype))
    assert torch.count_nonzero(zero) == 0
    x = implicit_image_recon(A, y, regularization=0.15, cg_tolerance=1e-11,
                            initial=torch.ones(8, dtype=y.dtype))
    torch.testing.assert_close(x, exact(torch.zeros(2, 2, dtype=torch.float64), 0.15))
