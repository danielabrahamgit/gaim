import torch

from gaim.optim import local_step, taylor_trust
from gaim.taylor_hybrid import WeightedForward, hybrid_residual, taylor_hybrid


def test_weighted_forward_gradient_including_zero_dcf():
    torch.manual_seed(1)
    matrix = torch.randn(9, 4, dtype=torch.complex128)

    class Operator:
        dcf = torch.linspace(0, 2, 9, dtype=torch.float64)
        def forward(self, x): return matrix @ x
        def adjoint(self, k): return matrix.H @ (self.dcf * k)

    A = Operator()
    x = torch.randn(4, dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: WeightedForward.apply(v, A, A.dcf.sqrt()), (x,))


def test_residual_linearizes_both_image_and_encoding():
    torch.manual_seed(2)
    matrix = torch.randn(13, 4, dtype=torch.complex128)
    phi = torch.randn(2, 4, dtype=torch.float64)
    g = torch.randn(3, 13, dtype=torch.float64)
    delta = torch.randn(2, 3, dtype=torch.float64)
    x, dx = torch.randn(2, 4, dtype=torch.complex128)

    class Operator:
        dcf = torch.linspace(0.1, 2, 13, dtype=torch.float64)
        def forward(self, v): return (matrix @ v)[None]
        def adjoint(self, k): return matrix.H @ (self.dcf * k).flatten()

    A = Operator()
    w = A.dcf.sqrt()
    responses = torch.stack([w * A.forward(ph * x) for ph in phi])
    data = torch.randn(1, 13, dtype=torch.complex128)
    errors = []
    for epsilon in [1e-4, 5e-5]:
        actual_matrix = matrix * torch.exp(-2j * torch.pi * epsilon * ((delta @ g).T @ phi))
        actual = w * (actual_matrix @ (x + epsilon * dx))[None] - data
        model = hybrid_residual(epsilon * delta, x + epsilon * dx, A, w, responses, g, data)
        errors.append((actual - model).norm())
    torch.testing.assert_close(errors[0] / errors[1], torch.tensor(4., dtype=torch.float64), rtol=0.01, atol=0)


def test_local_step_optimizes_coefficient_penalty():
    delta, info = local_step(torch.ones(2, 2), torch.zeros(1, 2, 2),
                             torch.ones(1, 2, 2), torch.ones(1, 3),
                             lambda x: x.sum() * 0, 0.05, 25,
                             coefficient_penalty=lambda f, x: (f - 0.02).square().sum())
    torch.testing.assert_close(delta, torch.full_like(delta, 0.02), atol=1e-5, rtol=1e-5)
    assert info['predicted_gain'] > 0


def test_zero_weight_matches_existing_sharpness_solver():
    torch.manual_seed(3)
    matrix = torch.randn(20, 6, dtype=torch.complex128)
    phi = torch.randn(1, 2, 3, dtype=torch.float64)
    g = torch.randn(1, 20, dtype=torch.float64)

    class Operator:
        dcf = torch.linspace(0.5, 2, 20, dtype=torch.float64)
        def __init__(self, alphas):
            self.matrix = matrix * torch.exp(-2j * torch.pi * (alphas.T @ phi.flatten(1)))
        def forward(self, x): return (self.matrix @ x.flatten())[None]
        def adjoint(self, k): return (self.matrix.H @ (self.dcf * k).flatten()).reshape(2, 3)

    def recon(A, y):
        return torch.linalg.lstsq(A.dcf.sqrt()[:, None] * A.matrix,
                                  (A.dcf.sqrt() * y).flatten()).solution.reshape(2, 3)

    y = Operator(0.02 * g).forward(torch.randn(2, 3, dtype=torch.complex128))
    settings = dict(outer_steps=2, inner_steps=5, radius=0.05, max_radius=0.1)
    old = taylor_trust(phi, g, y, Operator, recon, verbose=False, **settings)
    new = taylor_hybrid(phi, g, y, Operator, recon, dc_weight=0, **settings)
    torch.testing.assert_close(new['alphas'], old['alphas'], atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(new['image'], old['image'], atol=1e-9, rtol=1e-7)
