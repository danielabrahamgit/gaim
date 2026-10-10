"""Implicit-adjoint optimization of image quality plus weighted data consistency.

Uses one adjoint image solve per gradient, rather than B*P derivative solves.
The image objective, including ridge regularization, is identical to
implicit_image_recon. Operator adjoint must include the fixed diagonal weight.
"""
import torch
from .optim import _implicit_cg, _lbfgs_direction, implicit_image_recon, phase_bound


@torch.no_grad()
def objective_value(A, x, y, metric, metric_scale=1., dc_weight=0., dc_scale=1.):
    weight = A.dcf
    energy = (weight * y.abs().square()).sum()
    residual = y - A.forward(x)
    error = (weight * residual.abs().square()).sum() / energy
    sharpness = metric(x)
    return sharpness.double() / metric_scale - dc_weight * error.double() / dc_scale, sharpness, error


@torch.no_grad()
def implicit_objective_gradient(A, x, y, phis, bases, metric, *,
                                regularization=1e-3, cg_max_iter=500,
                                cg_tolerance=1e-5, metric_scale=1.,
                                dc_weight=0., dc_scale=1.):
    """Real coefficient gradient of S(x*)/s - beta*||sqrt(W)(Ax*-y)||²/(E*d).

    v = H^-1 dS_total/dx uses PyTorch's real-loss complex gradient convention.
    Re <dA v, W r> - Re <A v, W dA x> plus the explicit DC derivative
    contracts with temporal bases. Derivatives assume exp(-2j*pi*phi^T F g).
    """
    weight = A.dcf
    energy = (weight * y.abs().square()).sum()
    residual = y - A.forward(x)
    with torch.enable_grad():
        variable = x.detach().requires_grad_(True)
        score = metric(variable) / metric_scale
        gx, = torch.autograd.grad(score, variable)
    beta = dc_weight / dc_scale / energy
    gx = gx + 2 * beta * A.adjoint(residual)
    normal = lambda z: A.adjoint(A.forward(z)) + regularization * z
    v = _implicit_cg(normal, gx, max_iter=cg_max_iter, tolerance=cg_tolerance)
    wr = weight * residual
    wx = weight * (-A.forward(v) + 2 * beta * residual)
    result = bases.new_empty(len(phis), len(bases))
    for b, phi in enumerate(phis):
        # Coil axes precede the sampling axes. Sum all leading axes.
        terms = (-2j * torch.pi * (wr.conj() * A.forward(phi * v)
                                  + wx.conj() * A.forward(phi * x))).real
        terms = terms.reshape(-1, bases[0].numel()).sum(0)
        result[b] = bases.flatten(1) @ terms
    return result


def implicit_objective_trust(phis, bases, y, build_linop, metric, *,
                             F_init=None, metric_scale=1., dc_weight=0.,
                             dc_scale=1., regularization=1e-3,
                             cg_max_iter=500, cg_tolerance=1e-5,
                             max_steps=80, radius=.1, max_radius=1.,
                             max_retries=12, callback=None, verbose=True):
    """Phase-capped L-BFGS with exact reconstructed-objective acceptance.

    This uses the same implicit derivative as taylor_implicit_trust, evaluated
    as a vector-Jacobian product. It does not store an image Taylor model or
    claim second-order agreement. Returns all attempted steps and a stop reason.
    Only k-space/metric enter optimization; references are for external scoring.
    """
    if min(metric_scale, dc_scale) <= 0 or dc_weight < 0:
        raise ValueError('Positive objective scales and nonnegative DC weight required')
    if min(max_steps, max_retries) < 1 or not 0 < radius <= max_radius:
        raise ValueError('Positive iteration counts and valid radii required')
    ps = phis.flatten(1).abs().amax(1).clamp_min(1e-12)
    gs = bases.flatten(1).abs().amax(1).clamp_min(1e-12)
    phi = phis / ps.reshape(-1, *([1]*(phis.ndim-1)))
    g = bases / gs.reshape(-1, *([1]*(bases.ndim-1)))
    F = bases.new_zeros(len(phi), len(g)) if F_init is None else F_init * ps[:, None] * gs[None]
    def alphas(f):
        return ((f / ps[:, None]) @ g.flatten(1)).reshape(len(phi), *bases.shape[1:])
    settings = dict(regularization=regularization, cg_max_iter=cg_max_iter, cg_tolerance=cg_tolerance)
    scales = dict(metric_scale=metric_scale, dc_weight=dc_weight, dc_scale=dc_scale)
    with torch.no_grad():
        A = build_linop(alphas(F))
        x = implicit_image_recon(A, y, **settings)
        value, sharp, error = objective_value(A, x, y, metric, **scales)
    initial_image, initial_score = x.clone(), float(value)
    gradient = implicit_objective_gradient(A, x, y, phi, g, metric, **settings, **scales)
    s_hist, y_hist, history = [], [], []
    stop = 'iteration_limit'
    for iteration in range(max_steps):
        if not torch.isfinite(gradient).all():
            raise RuntimeError('Nonfinite implicit objective gradient')
        if gradient.abs().max() < 1e-7:
            stop = 'gradient_small'
            break
        # _lbfgs_direction minimizes; our objective is maximized.
        direction = _lbfgs_direction(-gradient.flatten(), s_hist, y_hist).reshape_as(F)
        if (gradient * direction).sum() <= 0:
            s_hist, y_hist = [], []
            direction = gradient / gradient.norm().clamp_min(1e-20)
        bound = float(phase_bound(direction, phi, g))
        length = min(1., radius / max(bound, 1e-20)) if s_hist else radius / max(bound, 1e-20)
        accepted = False
        for retry in range(max_retries):
            delta = length * direction
            prediction = float((gradient * delta).sum())
            if prediction <= 1e-9:
                break
            with torch.no_grad():
                trial_F = F + delta
                trial_A = build_linop(alphas(trial_F))
                trial_x = implicit_image_recon(trial_A, y, **settings)
                trial_value, trial_sharp, trial_error = objective_value(trial_A, trial_x, y, metric, **scales)
            gain = float(trial_value - value)
            ratio = gain / prediction
            accepted = bool(torch.isfinite(trial_value)) and gain > 0 and ratio >= .05
            row = dict(iteration=iteration+1, retry=retry, score=float(trial_value),
                       sharpness=float(trial_sharp), weighted_error=float(trial_error),
                       actual_gain=gain, predicted_gain=prediction, ratio=ratio,
                       phase_bound=float(phase_bound(delta, phi, g)), radius=radius,
                       gradient_norm=float(gradient.norm()), accepted=accepted)
            history.append(row)
            if verbose:
                print(f'iter={iteration+1} retry={retry} accepted={accepted} '
                      f'objective={float(trial_value):.6f} DC={float(trial_error):.6f} '
                      f'ratio={ratio:.3f}', flush=True)
            if accepted:
                break
            length *= .5
        if not accepted:
            stop = 'no_acceptable_step'
            break
        next_gradient = implicit_objective_gradient(trial_A, trial_x, y, phi, g, metric, **settings, **scales)
        s = delta.flatten()
        curvature = (gradient - next_gradient).flatten()  # gradient of loss=-score
        if s.dot(curvature) > 1e-8 * s.norm() * curvature.norm():
            s_hist = (s_hist + [s])[-10:]
            y_hist = (y_hist + [curvature])[-10:]
        F, A, x = trial_F, trial_A, trial_x
        value, sharp, error, gradient = trial_value, trial_sharp, trial_error, next_gradient
        if ratio < .25:
            radius *= .5
        elif ratio > .75 and row['phase_bound'] >= .9 * radius:
            radius = min(2 * radius, max_radius)
        if callback:
            callback(dict(F=F/ps[:, None]/gs[None], alphas=alphas(F), image=x,
                          history=history, score=float(value)))
    return dict(F=F/ps[:, None]/gs[None], alphas=alphas(F), image=x,
                initial_image=initial_image, initial_score=initial_score,
                final_score=float(value), sharpness=float(sharp), weighted_error=float(error),
                gradient_norm=float(gradient.norm()), history=history, stop_reason=stop)
