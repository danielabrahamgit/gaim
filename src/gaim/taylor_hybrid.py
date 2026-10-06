"""Sharpness minus normalized density-weighted data error, with Taylor steps."""
import torch

from .metrics import gradient_entropy_metric
from .optim import local_step


class WeightedForward(torch.autograd.Function):
    """Autograd for sqrt(D) A; the MRI library's A.adjoint already includes D.

    The custom backward supports forward operators implemented with Triton.
    Zero-weight samples contribute neither to the forward nor its adjoint.
    """
    @staticmethod
    def forward(ctx, image, operator, sqrt_weight):
        ctx.operator = operator
        ctx.save_for_backward(sqrt_weight)
        return sqrt_weight * operator.forward(image)

    @staticmethod
    def backward(ctx, gradient):
        sqrt_weight, = ctx.saved_tensors
        scaled = torch.where(sqrt_weight > 0,
                             gradient / sqrt_weight.clamp_min(torch.finfo(sqrt_weight.dtype).tiny), 0)
        return ctx.operator.adjoint(scaled), None, None


def hybrid_residual(delta, image, operator, sqrt_weight, responses, g, data):
    """First-order residual includes BOTH A_k*delta_x and delta_A*x_k.

    responses[b] = sqrt(D)*A_k(phi_b*x_k). The cross term delta_A*delta_x
    is second order and is deliberately omitted. Ground truth is not used.
    """
    B = len(responses)
    alpha_delta = delta @ g.flatten(1)
    phase_change = -2j * torch.pi * (
        responses.reshape(B, responses.shape[1], -1) * alpha_delta[:, None]
    ).sum(0).reshape_as(data)
    return WeightedForward.apply(image, operator, sqrt_weight) + phase_change - data


def taylor_hybrid(phis, bases, y, build_linop, recon, metric, dc_weight=10.0,
                  outer_steps=60, inner_steps=25, radius=0.1, max_radius=1.0,
                  max_retries=4, callback=None):
    """Maximize S(x(F))-lambda*||sqrt(D)(A(F)x(F)-y)||²/||sqrt(D)y||².

    Uses the existing early-stopped image recon and image Taylor surrogate.
    Adding sharpness makes this local objective nonconvex. L-BFGS proposes
    phase-capped updates; acceptance uses a true candidate reconstruction.
    """
    if dc_weight < 0 or min(outer_steps, inner_steps, max_retries) < 1 or not 0 < radius <= max_radius:
        raise ValueError('Nonnegative DC weight, positive counts, and valid radii required')
    ps = phis.flatten(1).abs().amax(1).clamp_min(1e-12)
    gs = bases.flatten(1).abs().amax(1).clamp_min(1e-12)
    phi = phis / ps.reshape(-1, *([1] * (phis.ndim - 1)))
    g = bases / gs.reshape(-1, *([1] * (bases.ndim - 1)))
    F = bases.new_zeros(len(phi), len(g))
    alphas = lambda f: ((f / ps[:, None]) @ g.flatten(1)).reshape(len(phi), *bases.shape[1:])
    with torch.no_grad():
        A = build_linop(alphas(F))
        sqrt_weight = A.dcf.sqrt()
        data = sqrt_weight * y
        energy = data.abs().square().sum()
        x = recon(A, y)

    def evaluate(operator, image):
        sharp = metric(image).item()
        error = ((sqrt_weight * operator.forward(image) - data).abs().square().sum() / energy).item()
        return sharp - dc_weight * error, sharp, error

    with torch.no_grad():
        score, sharp, error = evaluate(A, x)
    initial_image, initial_score = x.clone(), score
    history = []
    for outer in range(outer_steps):
        with torch.no_grad():
            image_responses = torch.stack([recon(A, y * (2j * torch.pi * gp)) for gp in g])
            responses = (torch.stack([sqrt_weight * A.forward(ph * x) for ph in phi])
                         if dc_weight else None)

        def penalty(delta, image):
            residual = hybrid_residual(delta, image, A, sqrt_weight, responses, g, data)
            return dc_weight * residual.abs().square().sum() / energy

        accepted = False
        for retry in range(max_retries):
            delta, info = local_step(x, image_responses, phi, g, metric,
                                     radius, inner_steps,
                                     coefficient_penalty=penalty if dc_weight else None)
            if info['predicted_gain'] <= 1e-8:
                break
            with torch.no_grad():
                candidate_A = build_linop(alphas(F + delta))
                candidate_x = recon(candidate_A, y)
                candidate_score, candidate_sharp, candidate_error = evaluate(candidate_A, candidate_x)
            gain = candidate_score - score
            ratio = gain / info['predicted_gain']
            accepted = gain > 0 and ratio >= 0.1
            record = dict(outer=outer+1, retry=retry, radius=radius, **info,
                          score_before=score, candidate_score=candidate_score,
                          sharpness=candidate_sharp, weighted_error=candidate_error,
                          actual_gain=gain, ratio=ratio, accepted=accepted)
            history.append(record)
            print(f'lambda={dc_weight:g} outer={outer+1} accepted={accepted} '
                  f'score={candidate_score:.6f} metric={candidate_sharp:.6f} '
                  f'DC={candidate_error:.6f} ratio={ratio:.3f}', flush=True)
            if not accepted or ratio < 0.25:
                radius *= 0.5
            elif ratio > 0.75 and info['stop_reason'] == 'phase_boundary':
                radius = min(2 * radius, max_radius)
            record['next_radius'] = radius
            if accepted:
                F, A, x = F + delta, candidate_A, candidate_x
                score, sharp, error = candidate_score, candidate_sharp, candidate_error
                if callback:
                    callback(alphas(F), x, history)
                break
        if not accepted:
            break
    return dict(F=F / ps[:, None] / gs[None, :], alphas=alphas(F), image=x,
                initial_image=initial_image, initial_score=initial_score,
                final_score=score, sharpness=sharp, weighted_error=error,
                dc_weight=dc_weight, history=history)
