import torch

from gaim.optim import (
    lstsq_max_phase_wrap,
    taylor_sph_lbfgs,
    taylor_sph_optim,
    _alpha_wrap_max,
    _net_phase_max,
    _sampled_phase_max,
    _taylor_sph_loss,
    _taylor_sph_setup,
)


def test_lstsq_max_phase_wrap_caps_phase():
    torch.manual_seed(0)
    B, P, n_r, n_t, M = 3, 4, 8, 9, 40
    phis = torch.randn(B, n_r)
    g = torch.randn(P, n_t)
    F_true = torch.randn(B, P)
    A = torch.randn(M, B * P)
    b = A @ F_true.reshape(-1) + 0.05 * torch.randn(M)
    unconstrained = torch.linalg.lstsq(A, b).solution.reshape(B, P)
    wrap = 0.25 * min(
        _alpha_wrap_max(unconstrained, phis, g).item(),
        _net_phase_max(unconstrained, phis, g).item(),
    )
    F = lstsq_max_phase_wrap(A, b, phis, g, max_phase_wrap=wrap)
    assert _alpha_wrap_max(F, phis, g) <= wrap * (1 + 1e-4)
    assert _net_phase_max(F, phis, g) <= wrap * (1 + 1e-4)
    unc_res = (A @ unconstrained.reshape(-1) - b).norm()
    con_res = (A @ F.reshape(-1) - b).norm()
    assert con_res >= unc_res * (1 - 1e-4)


def test_lstsq_max_phase_wrap_kills_cancelling_alphas():
    torch.manual_seed(2)
    n_r, n_t, P = 12, 10, 3
    phi0 = torch.linspace(-0.5, 0.5, n_r)
    phis = torch.stack([phi0, phi0.clone()])
    g = torch.randn(P, n_t)
    # Kernel of net phase: opposite rows, huge coefficients.
    F_true = torch.tensor([[40.0, -25.0, 15.0], [-40.0, 25.0, -15.0]])
    A = torch.eye(phis.shape[0] * P)
    b = F_true.reshape(-1)
    wrap = 0.25
    assert _net_phase_max(F_true, phis, g) < wrap
    assert _alpha_wrap_max(F_true, phis, g) > 10 * wrap
    F = lstsq_max_phase_wrap(A, b, phis, g, max_phase_wrap=wrap)
    assert _alpha_wrap_max(F, phis, g) <= wrap * (1 + 1e-4)
    assert _net_phase_max(F, phis, g) <= wrap * (1 + 1e-4)


def test_lstsq_max_phase_wrap_unconstrained_when_feasible():
    torch.manual_seed(1)
    A = torch.eye(6)
    b = torch.zeros(6)
    phis = torch.ones(2, 5)
    g = torch.ones(3, 4)
    F = lstsq_max_phase_wrap(A, b, phis, g, max_phase_wrap=1.0)
    assert torch.allclose(F, torch.zeros(2, 3), atol=1e-5)


def _toy_bases(seed=0):
    torch.manual_seed(seed)
    return torch.randn(2, 20), torch.randn(3, 16)


def test_phase_penalty_zero_inside_cap():
    phis, g = _toy_bases()
    F = torch.zeros(2, 3)
    loss, metric = _taylor_sph_loss(
        F, loss_fn=lambda x: x.square().sum() + 1.5,
        g_flt=g, phis_s=phis, g_s=g,
        alpha_reg=0.0, penalty_weight=4.0, constrain=True, wrap=0.25)
    assert torch.allclose(loss, metric)
    assert loss.item() == 1.5


def test_lbfgs_armijo_skips_backward_on_rejected_trials():
    phis, g = _toy_bases(1)
    F_true = torch.tensor([[0.2, -0.1, 0.05], [0.0, 0.15, -0.08]])
    n_bwd = {'n': 0}

    class Count(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad):
            n_bwd['n'] += 1
            return grad

    def loss_fn(F):
        return Count.apply((F - F_true).square().sum())

    n_iter = 8
    taylor_sph_lbfgs(
        torch.zeros_like(F_true), phis, g, loss_fn,
        max_phase_wrap=None, n_iter=n_iter, lr=1.0,
        n_space=8, n_time=8, n_check_space=8, n_check_time=8)
    assert n_bwd['n'] <= n_iter + 1


def test_lbfgs_recovers_quadratic():
    phis, g = _toy_bases(2)
    F_true = torch.tensor([[0.3, -0.2, 0.1], [0.05, 0.4, -0.15]])

    def loss_fn(F):
        return (F - F_true).square().sum()

    F = taylor_sph_lbfgs(
        torch.zeros_like(F_true), phis, g, loss_fn,
        max_phase_wrap=None, n_iter=40, lr=1.0,
        n_space=8, n_time=8, n_check_space=8, n_check_time=8)
    assert (F - F_true).square().sum() < 1e-4


def test_lbfgs_keeps_phase_under_cap():
    phis = torch.ones(2, 8)
    g = torch.ones(3, 6)
    wrap = 0.2
    setup = _taylor_sph_setup(phis, g, wrap, 8, 6, 8, 6)

    def loss_fn(F):
        return (F - 5).square().sum()

    F = taylor_sph_lbfgs(
        torch.zeros(2, 3), phis, g, loss_fn,
        max_phase_wrap=wrap, n_iter=20, lr=1.0,
        penalty_weight=0.0, setup=setup)
    _, phis_s, g_s, *_ = setup
    phase = _sampled_phase_max(F, phis_s, g_s)
    assert phase <= wrap * (1 + 1e-4)
    assert F.abs().sum() > 0


def test_adam_and_lbfgs_share_objective_and_setup():
    phis, g = _toy_bases(3)
    F_true = torch.tensor([[0.12, -0.08, 0.04], [0.02, 0.09, -0.05]])
    setup = _taylor_sph_setup(phis, g, None, 8, 8, 8, 8)

    def loss_fn(F):
        return (F - F_true).square().sum()

    F0 = torch.zeros_like(F_true)
    loss0 = loss_fn(F0).item()
    F_ad = taylor_sph_optim(
        F0, phis, g, loss_fn, max_phase_wrap=None, n_iter=25, lr=0.2, setup=setup)
    F_lb = taylor_sph_lbfgs(
        F0, phis, g, loss_fn, max_phase_wrap=None, n_iter=25, lr=1.0, setup=setup)
    assert loss_fn(F_ad).item() < loss0
    assert loss_fn(F_lb).item() < loss0
