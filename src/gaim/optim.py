"""
Contains many tools for optimization
"""
import torch

from einops import einsum
from tqdm import tqdm

def _phase_subsample(phis_flt, g_flt, n_space, n_time):
    """Fixed spatial/temporal samples. Prefer support voxels so the mask is not wasted."""
    n_space = min(n_space, phis_flt.shape[-1])
    n_time = min(n_time, g_flt.shape[-1])
    on = phis_flt.abs().sum(0) > 0
    pool = torch.where(on)[0] if on.any() else torch.arange(phis_flt.shape[-1], device=phis_flt.device)
    n_space = min(n_space, pool.numel())
    idxr = pool[torch.randperm(pool.numel(), device=phis_flt.device)[:n_space]]
    idxt = torch.randperm(g_flt.shape[-1], device=g_flt.device)[:n_time]
    return phis_flt[:, idxr], g_flt[:, idxt]


def _coeff_scale(phis_flt, g_flt, wrap):
    """Per-entry scale so raw[b,p] ~ 1 is about ``wrap`` from basis b, p."""
    s_phi = phis_flt.abs().amax(dim=1).clamp_min(1e-12)
    s_g = g_flt.abs().amax(dim=1).clamp_min(1e-12)
    scale = 1.0 / (s_phi[:, None] * s_g[None, :])
    if wrap is not None and wrap < float('inf'):
        scale = scale * wrap
    return scale


def _sampled_phase_stats(F, phis_s, g_s, p=8, time_chunk=256,
                         want_max=False, want_pnorm=False):
    """Chunked |phi^T F g| stats. Does not form the full n_space x n_time array."""
    Fg = F @ g_s
    n = phis_s.shape[1] * Fg.shape[1]
    acc = F.new_zeros(()) if want_pnorm else None
    m = F.new_zeros(()) if want_max else None
    for t0 in range(0, Fg.shape[1], time_chunk):
        phase_abs = (phis_s.T @ Fg[:, t0:t0 + time_chunk]).abs()
        if want_pnorm:
            acc = acc + phase_abs.pow(p).sum()
        if want_max:
            m = phase_abs.max() if m is None else torch.maximum(m, phase_abs.max())
    pnorm = (acc / n + 1e-24).pow(1 / p) if want_pnorm else None
    return pnorm, m


def _sampled_phase_max(F, phis_s, g_s):
    """max |phi^T F g| on a fixed spatial/temporal subsample."""
    _, m = _sampled_phase_stats(F, phis_s, g_s, want_max=True)
    return m


def _sampled_phase_pnorm(F, phis_s, g_s, p=8):
    """Mean p-norm of |phi^T F g|; <= the sampled max."""
    pnorm, _ = _sampled_phase_stats(F, phis_s, g_s, p=p, want_pnorm=True)
    return pnorm


def _taylor_sph_setup(phis, g_bases, max_phase_wrap,
                      n_space, n_time, n_check_space, n_check_time):
    g_flt = g_bases.reshape(g_bases.shape[0], -1)
    phis_flt = phis.reshape(phis.shape[0], -1)
    constrain = max_phase_wrap is not None and max_phase_wrap < float('inf')
    wrap = max_phase_wrap if constrain else 1.0
    phis_s, g_s = _phase_subsample(phis_flt, g_flt, n_space, n_time)
    phis_c, g_c = _phase_subsample(phis_flt, g_flt, n_check_space, n_check_time)
    scale = _coeff_scale(phis_flt, g_flt, wrap).to(dtype=phis.dtype)
    return g_flt, phis_s, g_s, phis_c, g_c, scale, constrain, wrap


def _taylor_sph_loss(F, loss_fn, g_flt, phis_s, g_s,
                     alpha_reg, penalty_weight, constrain, wrap, phase_p=8):
    """Shared Adam / L-BFGS objective. Phase max is not computed here."""
    metric = loss_fn(F)
    loss = metric
    if alpha_reg:
        loss = loss + alpha_reg * (F @ g_flt).square().mean()
    if constrain:
        phase_pnorm, _ = _sampled_phase_stats(
            F, phis_s, g_s, p=phase_p, want_pnorm=True)
        over = phase_pnorm / wrap - 1
        loss = loss + penalty_weight * torch.relu(over).square()
    return loss, metric


def _report_dense_phase(F, phis_c, g_c, constrain, wrap):
    """Log max |phi^T F g| on a denser sample. Does not modify F."""
    if not constrain:
        return F
    with torch.no_grad():
        phase_chk = _sampled_phase_max(F, phis_c, g_c)
        tqdm.write(f'dense phase max {phase_chk.item():.4f} (cap {wrap:g})')
    return F


def taylor_sph_optim(F_phi_init: torch.tensor,
                     phis: torch.tensor,
                     g_bases: torch.tensor,
                     loss_fn: callable,
                     n_iter: int = 100,
                     lr: float = 1e-2,
                     alpha_reg: float = 0.0,):
    """
    Adam on the Taylor SPH metric. The wrap cap is not enforced here;
    ``taylor_sph_lbfgs`` rejects steps that exceed it.
    """
    M = 1_000
    phis_flt = phis.reshape(phis.shape[0], -1)
    g_flt = g_bases.reshape(g_bases.shape[0], -1)
    rnd_vox = torch.randperm(phis_flt.shape[1], device=phis_flt.device)[:M]
    rnd_time = torch.randperm(g_flt.shape[1], device=g_bases.device)[:M]
    F_opt = torch.nn.Parameter(F_phi_init.clone(), requires_grad=True)
    opt = torch.optim.Adam([F_opt], lr=lr)
    tbar = tqdm(range(n_iter), desc='Taylor SPH ADAM Loop')
    for n in tbar:
        opt.zero_grad()
        loss = loss_fn(F_opt)
        phase = (phis_flt[:, rnd_vox].T @ F_opt @ g_flt[:, rnd_time])
        loss += alpha_reg * phase.abs().mean()
        # alphas = einsum(F_opt, g_bases, 'B P, P ... -> B ...')
        # loss += alpha_reg * alphas.square().mean()
        loss.backward()
        opt.step()
        if n % 100 == 0:
            tbar.set_postfix(metric=f'{loss.item():.6g}', phase=f'{phase.max().item():.4f}')
    return F_opt.detach()


def taylor_patch_optim(f_init: torch.tensor,
                       g_bases: torch.tensor,
                       loss_fn: callable,
                       n_iter: int = 100,
                       lr: float = 1e-2,
                       alpha_reg: float = 1.5e-3):
    """
    Same Adam loop as taylor_sph_optim, batched over patches.

    Args
    ----
    f_init: torch.tensor
        Per-patch coefficients, shape (G, P)
    g_bases: torch.tensor
        Temporal bases, shape (P, *trj_size)
    loss_fn: callable
        Maps (G, P) to per-patch losses of shape (G,)
    """
    g_flt = g_bases.reshape((g_bases.shape[0], -1))
    f = torch.nn.Parameter(f_init.clone())
    opt = torch.optim.Adam([f], lr=lr)

    tbar = tqdm(range(n_iter), desc='Taylor patch ADAM')
    for _ in tbar:
        opt.zero_grad()
        losses = loss_fn(f)
        losses = losses + alpha_reg * (f @ g_flt).norm(dim=-1)
        loss = losses.sum()
        loss.backward()
        opt.step()
        tbar.set_postfix(loss=losses.mean().log10().item())
    return f.detach()


def _phi_inf(phis):
    return phis.reshape(phis.shape[0], -1).abs().amax(dim=1).clamp_min(1e-12)


def _alpha_wrap_max(F, phis, g_bases):
    """max_b ||phi_b||_∞ · max_t | (F g)_b(t) |."""
    alphas = F @ g_bases.reshape(g_bases.shape[0], -1).to(dtype=F.dtype)
    return (alphas.abs() * _phi_inf(phis).to(dtype=F.dtype)[:, None]).max()


def _net_phase_max(F, phis, g_bases, time_chunk=64):
    """max |phi^T F g| over support voxels, chunked in time."""
    g_flt = g_bases.reshape(g_bases.shape[0], -1).to(dtype=F.dtype)
    phis_flt = phis.reshape(phis.shape[0], -1).to(dtype=F.dtype)
    alphas = F @ g_flt
    on = phis_flt.abs().sum(0) > 0
    phi_on = phis_flt[:, on] if on.any() else phis_flt
    m = F.new_zeros(())
    for t0 in range(0, alphas.shape[1], time_chunk):
        m = torch.max(m, (phi_on.T @ alphas[:, t0:t0 + time_chunk]).abs().max())
    return m


def _lbfgs_direction(grad, s_hist, y_hist):
    """Two-loop recursion: d = -H g. Empty history is steepest descent."""
    q = grad.clone()
    alphas = []
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        rho = 1.0 / y.dot(s).clamp_min(1e-12)
        a = rho * s.dot(q)
        q = q - a * y
        alphas.append(a)
    if y_hist:
        y, s = y_hist[-1], s_hist[-1]
        q = q * (s.dot(y) / y.dot(y).clamp_min(1e-12))
    else:
        gnorm = grad.norm().clamp_min(1e-12)
        q = q / gnorm
    for s, y, a in zip(s_hist, y_hist, reversed(alphas)):
        rho = 1.0 / y.dot(s).clamp_min(1e-12)
        q = q + s * (a - rho * y.dot(q))
    return -q


def taylor_sph_lbfgs(F_phi_init: torch.tensor,
                     phis: torch.tensor,
                     g_bases: torch.tensor,
                     loss_fn: callable,
                     max_phase_wrap: float = 0.25,
                     n_iter: int = 50,
                     lr: float = 1.0,
                     alpha_reg: float = 0.0,
                     penalty_weight: float = 0.0,
                     n_space: int = 256,
                     n_time: int = 256,
                     n_check_space: int = 4096,
                     n_check_time: int = 4096,
                     history_size: int = 20,
                     setup=None):
    """
    L-BFGS on the B×P Taylor SPH matrix.

    Armijo trials evaluate the loss under ``no_grad`` and are rejected when
    ``max |phi^T F g|`` on the optimization samples exceeds ``max_phase_wrap``.
    A backward pass runs only at an accepted point. The optional p-norm
    penalty is off unless ``penalty_weight`` is set. A denser sample is
    logged at the end and does not rescale ``F``.
    """
    g_flt, phis_s, g_s, phis_c, g_c, scale, constrain, wrap = (
        setup if setup is not None else _taylor_sph_setup(
            phis, g_bases, max_phase_wrap, n_space, n_time, n_check_space, n_check_time))
    raw = torch.nn.Parameter((F_phi_init.clone() / scale).contiguous())

    def pack_loss(F):
        return _taylor_sph_loss(
            F, loss_fn, g_flt, phis_s, g_s, alpha_reg, penalty_weight, constrain, wrap)

    def eval_loss():
        with torch.no_grad():
            loss, metric = pack_loss(raw * scale)
            return loss.detach(), metric.detach()

    def eval_with_grad():
        if raw.grad is not None:
            raw.grad.zero_()
        loss, metric = pack_loss(raw * scale)
        loss.backward()
        return loss.detach(), raw.grad.detach().reshape(-1).clone(), metric.detach()

    def phase_ok():
        if not constrain:
            return True
        with torch.no_grad():
            phase = _sampled_phase_max(raw * scale, phis_s, g_s)
        return bool(phase <= wrap * (1 + 1e-5))

    def armijo(x0, direction, t_init, loss, gtd):
        t = t_init
        for _ in range(20):
            raw.data.copy_((x0 + t * direction).view_as(raw))
            trial_loss, trial_metric = eval_loss()
            if (phase_ok() and torch.isfinite(trial_loss)
                    and trial_loss.item() <= loss.item() + 1e-4 * t * gtd.item()):
                return True, t, trial_loss, trial_metric
            t *= 0.5
        raw.data.copy_(x0.view_as(raw))
        return False, t, loss, None

    def log_phase(F):
        if not constrain:
            return F.new_zeros(())
        with torch.no_grad():
            return _sampled_phase_max(F, phis_s, g_s)

    s_hist, y_hist = [], []
    loss, grad, metric = eval_with_grad()
    phase_max = log_phase(raw.detach() * scale)
    t_trial = min(float(lr), 1.0 / grad.abs().sum().clamp_min(1e-12).item())

    tbar = tqdm(range(n_iter), desc='Taylor SPH L-BFGS')
    for _ in tbar:
        direction = _lbfgs_direction(grad, s_hist, y_hist)
        gtd = grad.dot(direction)
        if gtd >= 0:
            s_hist, y_hist = [], []
            direction = -grad / grad.norm().clamp_min(1e-12)
            gtd = grad.dot(direction)

        x0 = raw.detach().reshape(-1).clone()
        accepted, t, _, _ = armijo(x0, direction, t_trial, loss, gtd)

        if not accepted:
            s_hist, y_hist = [], []
            sd = -grad / grad.norm().clamp_min(1e-12)
            gtd_sd = grad.dot(sd)
            accepted, t, _, _ = armijo(x0, sd, t_trial, loss, gtd_sd)
            if not accepted:
                tqdm.write('L-BFGS Armijo failed; keeping previous iterate')
                t_trial = max(t_trial * 0.5, 1e-8)
                tbar.set_postfix(metric=f'{metric.item():.6g}',
                                 phase=f'{phase_max.item():.4f}',
                                 status='armijo_fail')
                continue

        new_loss, new_grad, new_metric = eval_with_grad()
        new_phase = log_phase(raw.detach() * scale)

        s = raw.detach().reshape(-1) - x0
        y = new_grad - grad
        sy = s.dot(y)
        s_norm = s.norm().clamp_min(1e-12)
        y_norm = y.norm().clamp_min(1e-12)
        if sy > 1e-8 * s_norm * y_norm:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        loss, grad = new_loss, new_grad
        metric, phase_max = new_metric, new_phase
        t_trial = min(lr, max(t * 1.5, 1e-8))
        tbar.set_postfix(metric=f'{metric.item():.6g}', phase=f'{phase_max.item():.4f}')

    F = (raw.detach() * scale)
    return _report_dense_phase(F, phis_c, g_c, constrain, wrap)


def lstsq_max_phase_wrap(A: torch.Tensor,
                         b: torch.Tensor,
                         phis: torch.Tensor,
                         g_bases: torch.Tensor,
                         max_phase_wrap: float = 1.0,
                         n_admm: int = 50,
                         rho: float = 1.0) -> torch.Tensor:
    """
    min_F ||A vec(F) - b||^2 with wrap caps on alphas and net phase.

    ``A`` is ``(M, B*P)`` and ``F`` is row-major ``(B, P)``. SPH / temporal
    bases are ill-conditioned, so capping only ``|phi^T F g|`` leaves a
    kernel of huge cancelling ``F`` (and huge ``alpha = F g``) with small
    net phase. This therefore enforces

        |alpha_b(t)| * ||phi_b||_∞  <=  max_phase_wrap     for all b, t

    by ADMM, then radially scales so ``max |phi^T alpha|`` also respects
    the cap. ``max_phase_wrap=None`` or ``inf`` is plain least squares.
    """
    B = phis.shape[0]
    P = g_bases.shape[0]
    x = torch.linalg.lstsq(A, b).solution.reshape(B, P)
    constrain = max_phase_wrap is not None and max_phase_wrap < float('inf')
    if not constrain:
        return x

    g_flt = g_bases.reshape(P, -1).to(dtype=x.dtype)
    phi_scale = _phi_inf(phis).to(dtype=x.dtype)
    wrap_b = max_phase_wrap / phi_scale
    alphas = x @ g_flt
    alpha_ok = bool((alphas.abs() <= wrap_b[:, None] * (1 + 1e-5)).all())
    if alpha_ok and _net_phase_max(x, phis, g_bases) <= max_phase_wrap:
        return x

    Gt = g_flt @ g_flt.T
    gram_A = A.T @ A
    CTC = torch.kron(torch.eye(B, dtype=x.dtype, device=x.device), Gt)
    sA = gram_A.diag().mean().clamp_min(1e-12)
    sC = CTC.diag().mean().clamp_min(1e-12)
    rho_eff = rho * (sA / sC)
    H = gram_A + rho_eff * CTC
    H.diagonal().add_(1e-6 * sA)
    Atb = A.T @ b

    z = alphas.clamp(-wrap_b[:, None], wrap_b[:, None])
    u = torch.zeros_like(z)
    for _ in range(n_admm):
        rhs = Atb + rho_eff * ((z - u) @ g_flt.T).reshape(-1)
        x = torch.linalg.solve(H, rhs).reshape(B, P)
        alphas = x @ g_flt
        z = (alphas + u).clamp(-wrap_b[:, None], wrap_b[:, None])
        u = u + alphas - z

    over = (alphas.abs() / wrap_b[:, None]).amax(dim=1).clamp_min(1.0)
    x = x / over[:, None]
    phase_max = _net_phase_max(x, phis, g_bases)
    if phase_max > max_phase_wrap:
        x = x * (max_phase_wrap / phase_max.clamp_min(1e-12))
    return x