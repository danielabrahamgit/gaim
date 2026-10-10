"""
Contains many tools for optimization
"""
import torch

from tqdm import tqdm
from einops import einsum
from .metrics import gradient_entropy_metric
from typing import Optional

def _lbfgs_direction(grad: torch.Tensor, 
                     s_hist: list[torch.Tensor], 
                     y_hist: list[torch.Tensor]) -> torch.Tensor:
    """
    Two-loop recursion: d = -H g. Empty history is steepest descent.
    
    Args
    ----
    grad : torch.Tensor
        The gradient of the loss function.
    s_hist : list[torch.Tensor]
        The history of the search directions.
    y_hist : list[torch.Tensor]
        The history of the search directions.
        
    Returns    
    ------- 
    q : torch.Tensor
        The search direction.
    """
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

def phase_bound(F: torch.Tensor, 
                phis: torch.Tensor, 
                bases: torch.Tensor) -> torch.Tensor:
    """
    Conservative bound on max_{r,t}|phi(r)^T F g(t)|, in cycles.

    Triangle inequality avoids allocating the huge space-by-time phase array.
    Unlike sampled checks, this covers every voxel and trajectory sample.

    Args
    F : torch.Tensor
        The encoding operator.
    phis : torch.Tensor
        The phase encoding.
    bases : torch.Tensor
        The basis functions.

    Returns
    -------
    bound : torch.Tensor
        The conservative bound on max_{r,t}|phi(r)^T F g(t)|, in cycles.
    """
    phi_max = phis.flatten(1).abs().amax(1)
    return (phi_max[:, None] * (F @ bases.flatten(1)).abs()).sum(0).amax()

def local_step(x0: torch.Tensor, 
               responses: torch.Tensor, 
               phis: torch.Tensor, 
               bases: torch.Tensor, 
               metric: callable, 
               radius: float, 
               max_steps: int,
               coefficient_penalty=None, full_jacobian=False) -> tuple[torch.Tensor, dict]:
    """
    L-BFGS ascent of the Taylor metric, stopping at the phase boundary.
    
    Args
    ----
    x0 : torch.Tensor
        The initial guess with shape (*im_size)
    responses : torch.Tensor
        The responses with shape (P, *num_samples)
    phis : torch.Tensor
        The spatial phase basis functions with shape (B, *im_size)
    bases : torch.Tensor
        The temporal basis functions with shape (P, *trj_size)
    metric : callable
        The metric function with signature metric(image: torch.Tensor) -> float.
    radius : float
        The radius of the trust region.
    max_steps : int
        The maximum number of steps.
    full_jacobian : bool
        If True, responses has shape (B, P, *im_size) and contains the full
        image Jacobian instead of separable temporal responses.
    coefficient_penalty : callable, optional
        Additional loss penalty(delta_F, Taylor_image), e.g. data consistency.
        
    Returns
    """
    # Consts
    B = len(phis)
    P = len(bases)
    
    # We start with a zero F_param
    delta_F = torch.zeros(B, P, dtype=phis.dtype, device=phis.device)

    # Loss function eval using taylor model
    def loss(F):
        if full_jacobian:
            image = x0 + (F.flatten().to(responses.dtype) @
                          responses.reshape(F.numel(), -1)).reshape_as(x0)
        else:
            # Preserve the separable model without B*P derivative images.
            change = (F.to(responses.dtype) @ responses.flatten(1)).reshape_as(phis)
            image = x0 + (phis * change).sum(0)
        value = -metric(image)
        if coefficient_penalty is not None:
            value = value + coefficient_penalty(F, image)
        return value

    # Value and gradient of the loss function
    def value_gradient(F):
        F = F.detach().requires_grad_(True)
        value = loss(F)
        gradient, = torch.autograd.grad(value, F)
        return value.detach(), gradient.detach().flatten()

    value, gradient = value_gradient(delta_F)
    initial_value = value.clone()
    s_history, y_history = [], []
    steps, reason = 0, 'iteration_limit'
    for _ in range(max_steps):
        if gradient.abs().max() < 1e-7:
            reason = 'gradient_small'
            break
        direction = _lbfgs_direction(gradient, s_history, y_history).reshape_as(delta_F)
        if gradient.dot(direction.flatten()) >= 0:
            s_history, y_history = [], []
            direction = (-gradient / gradient.norm().clamp_min(1e-12)).reshape_as(delta_F)
        # A short first step gathers curvature before attempting the boundary.
        length = (0.25 * radius / phase_bound(direction, phis, bases).clamp_min(1e-12).item()
                  if not s_history else 1.0)
        accepted = False
        with torch.no_grad():
            for _ in range(20):
                trial = delta_F + length * direction
                bound = phase_bound(trial, phis, bases).item()
                boundary = bound >= radius
                if boundary:
                    trial *= (1 - 1e-6) * radius / bound
                # Never evaluate the Taylor objective outside its trust region.
                displacement = (trial - delta_F).flatten()
                slope = gradient.dot(displacement)
                trial_value = loss(trial)
                if (slope < 0 and torch.isfinite(trial_value)
                        and trial_value <= value + 1e-4 * slope):
                    accepted = True
                    break
                length *= 0.5
        if not accepted:
            reason = 'line_search_failed'
            break
        new_value, new_gradient = value_gradient(trial)
        y = new_gradient - gradient
        # Positive curvature keeps the approximate inverse Hessian well behaved.
        if displacement.dot(y) > 1e-8 * displacement.norm() * y.norm():
            s_history = (s_history + [displacement])[-10:]
            y_history = (y_history + [y])[-10:]
        improvement = (value - new_value).item()
        delta_F, value, gradient = trial, new_value, new_gradient
        steps += 1
        if boundary or phase_bound(delta_F, phis, bases) >= 0.99 * radius:
            reason = 'phase_boundary'
            break
        if improvement <= 1e-8:
            reason = 'objective_stalled'
            break
    return delta_F, dict(predicted_gain=(initial_value - value).item(), inner_steps=steps,
                       phase_bound=phase_bound(delta_F, phis, bases).item(), stop_reason=reason)

def taylor_trust(phis: torch.Tensor, 
                 bases: torch.Tensor, 
                 ksp: torch.Tensor, 
                 build_linop: callable, 
                 recon: callable, 
                 metric: callable=gradient_entropy_metric,
                 F_init: Optional[torch.Tensor] = None,
                 outer_steps=30, inner_steps=25, radius=0.1, max_radius=1,
                 P_batch_size=1, max_retries=4, verbose=True, *,
                 _response_builder=None):
    """
    Iteratively builds a taylor expansion around the current operating point, then 
    optimizes the taylor model using local_step. Rejects inaccurate predictions and 
    retries
    
    Args 
    ----
    phis : torch.Tensor
        The spatial phase basis functions with shape (B, *im_size)
    bases : torch.Tensor
        The temporal basis functions with shape (P, *trj_size)
    ksp : torch.Tensor
        The k-space data with shape (C, *ksp_size)
    build_linop : callable
        The function to build the encoding operator from alphas = F @ bases as input
    recon : callable
        The function to reconstruct the image from the encoding operator and k-space data
    metric : callable
        The metric function with signature metric(image: torch.Tensor) -> float.
    outer_steps : int
        The number of outer steps.
    inner_steps : int
        The number of inner L-BFGS steps.
    radius : float
        The radius of the trust region in cycles.
    max_radius : float
        The maximum radius of the trust region in cycles.
    max_retries : int
        The maximum number of retries.
    verbose : bool
        Whether to print verbose output.
        
    Returns
    -------
    dict
        A dictionary containing the following keys:
        - F : torch.Tensor
            The parameters we're solving for with shape (B, P)
        - alphas : torch.Tensor
            The same as F @ bases, with shape (B, *trj_size)
        - image : torch.Tensor
            The reconstructed image with shape (*im_size)
        - initial_image : torch.Tensor
            The initial image with shape (*im_size)
        - initial_score : float
            The initial score
        - final_score : float
            The final score
        - history : list[dict]
            The history of the optimization.
            Each dictionary contains the following keys:
            - outer : int
                The outer step.
            - retry : int
                The retry number.
            - radius : float
                The radius of the trust region.
    """
    # Consts
    B = len(phis)
    P = len(bases)
    
    # Check 
    if min(outer_steps, inner_steps, max_retries) < 1 or not 0 < radius <= max_radius:
        raise ValueError('Positive iteration counts and 0 < radius <= max_radius required')
    
    # Rescale phis and bases to improve conditioning without changing phi^T F g or returned alphas.
    phi_scale = phis.flatten(1).abs().amax(1).clamp_min(1e-12)
    g_scale = bases.flatten(1).abs().amax(1).clamp_min(1e-12)
    phi = phis / phi_scale.reshape(-1, *([1] * (phis.ndim - 1)))
    g = bases / g_scale.reshape(-1, *([1] * (bases.ndim - 1)))
    
    # We start with a zero F_param
    if F_init is None:
        F_param = torch.zeros(B, P, dtype=phis.dtype, device=phis.device)
    else:
        F_param = F_init * phi_scale[:, None] * g_scale[None, :]

    # Gets alphas from F 
    def alphas_from(F):
        return ((F / phi_scale[:, None]) @ g.flatten(1)).reshape(len(phis), *bases.shape[1:])

    # Build initial forward model, image, and score
    with torch.no_grad():
        operator = build_linop(alphas_from(F_param))
        image = recon(operator, ksp)
        score = metric(image).item()
        
    # Save initial image and score to see how much we improve
    initial_image, initial_score = image.clone(), score
    
    # Iterate over outer steps
    history = []
    for outer in tqdm(range(outer_steps), desc='Outer steps', disable=not verbose):
        
        # Build image responses at the current operating point.
        if verbose:
            count = B * P if _response_builder is not None else P
            print(f'Building {count} image responses', flush=True)
        with torch.no_grad():
            if _response_builder is not None:
                responses = _response_builder(operator, image, phi, g)
            elif P_batch_size == 1:
                responses = torch.stack([recon(operator, ksp * (2j * torch.pi * gp)) for gp in g])
            else:
                responses = []
                for p1 in range(0, P, P_batch_size):
                    p2 = min(p1 + P_batch_size, P)
                    tup = (slice(p1, p2),) + (None,) * (1 + ksp.ndim - g.ndim)
                    responses.append(recon(operator, ksp[None,] * (2j * torch.pi * g[tup])))
                responses = torch.cat(responses, dim=0)
        
        # Now we will see if the taylor model is accurate enough to accept the update
        accepted = False
        for retry in range(max_retries):
            
            # Optimize the taylor model with L-BFGS
            delta_F, info = local_step(image, responses, phi, g, metric, radius, inner_steps,
                                          full_jacobian=_response_builder is not None)
            
            # Convergence check
            if info['predicted_gain'] <= 1e-8:
                break
            
            # Check the real model at the proposed update
            with torch.no_grad():
                candidate = F_param + delta_F
                candidate_operator = build_linop(alphas_from(candidate))
                candidate_image = recon(candidate_operator, ksp)
                candidate_score = metric(candidate_image).item()
            actual_gain = candidate_score - score
            ratio = actual_gain / info['predicted_gain']
            accepted = actual_gain > 0 and ratio >= 0.1
            history.append(dict(outer=outer + 1, retry=retry, radius=radius, **info,
                                score_before=score, candidate_score=candidate_score,
                                actual_gain=actual_gain, ratio=ratio, accepted=accepted,
                                delta_F=(delta_F / phi_scale[:, None] / g_scale[None, :]).cpu()))
            if verbose:
                print(f'  accepted={accepted}, metric={candidate_score:.6f}, '
                      f'phase<={info["phase_bound"]:.4f} cycles, ratio={ratio:.3f}', flush=True)
            
            # Rejected because the real gain is smaller than expected, decrease the radius
            if not accepted:
                radius *= 0.5
                continue
            
            # Update the operating point
            F_param, operator, image, score = candidate, candidate_operator, candidate_image, candidate_score
            
            # If the real gain is smaller than expected, decrease the radius
            if ratio < 0.25:
                radius *= 0.5
            # If the real gain is limited by the phase boundary, increase the radius
            elif ratio > 0.75 and info['stop_reason'] == 'phase_boundary':
                radius = min(2 * radius, max_radius)
                
            # Exit the retry loop if the update was accepted
            break
        
        # Exit the outer loop if the update was not accepted
        if not accepted:
            if verbose:
                print(f'Outer {outer + 1} failed to accept update after {max_retries} retries', flush=True)
            break
    
    # Return the final parameters, alphas, image, and score
    F = F_param / phi_scale[:, None] / g_scale[None, :]
    return dict(F=F.detach(), alphas=alphas_from(F_param).detach(), image=image.detach(),
                initial_image=initial_image, initial_score=initial_score, final_score=score,
                history=history)


@torch.no_grad()
def _implicit_cg(normal, rhs, initial=None, max_iter=100, tolerance=1e-4):
    """Solve a Hermitian positive-definite system, checking the true residual."""
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    scale = rhs.norm()
    if scale == 0:
        return torch.zeros_like(rhs)
    r = rhs - normal(x)
    p = r.clone()
    rr = torch.vdot(r.flatten(), r.flatten()).real
    for _ in range(max_iter):
        if r.norm() <= tolerance * scale:
            # Recursive CG residuals can drift, especially in complex64.
            r = rhs - normal(x)
            if r.norm() <= tolerance * scale:
                return x
            p = r.clone()
            rr = torch.vdot(r.flatten(), r.flatten()).real
        Hp = normal(p)
        curvature = torch.vdot(p.flatten(), Hp.flatten()).real
        if not torch.isfinite(curvature) or curvature <= 0:
            raise RuntimeError('Implicit CG requires a positive-definite normal operator')
        step = rr / curvature
        x += step * p
        r -= step * Hp
        rr_new = torch.vdot(r.flatten(), r.flatten()).real
        p = r + (rr_new / rr) * p
        rr = rr_new
    relative_residual = ((rhs - normal(x)).norm() / scale).item()
    if not relative_residual <= tolerance:
        raise RuntimeError(f'Implicit CG did not converge: relative residual '
                           f'{relative_residual:.3g} > {tolerance:.3g}; increase '
                           'cg_max_iter or regularization, or relax cg_tolerance')
    return x


@torch.no_grad()
def implicit_image_recon(operator, ksp, *, regularization=1e-3,
                         cg_max_iter=100, cg_tolerance=1e-4, initial=None):
    """Solve ||W**(1/2)(Ax-y)||² + regularization*||x||².

    operator.forward is A; operator.adjoint MUST be A^H W for fixed real
    diagonal W (or the true adjoint for W=I). HOFFT uses its dcf as W.
    Use explicit forward/adjoint products rather than a cached normal.
    regularization is an absolute coefficient, not scaled by an eigenvalue.
    """
    if not regularization > 0 or cg_max_iter < 1 or not 0 < cg_tolerance < 1:
        raise ValueError('Positive regularization/iterations and 0 < tolerance < 1 required')
    normal = lambda x: operator.adjoint(operator.forward(x)) + regularization * x
    return _implicit_cg(normal, operator.adjoint(ksp), initial,
                        cg_max_iter, cg_tolerance)


@torch.no_grad()
def _implicit_responses(operator, image, ksp, phis, bases, *,
                        regularization, cg_max_iter, cg_tolerance):
    """J[b,p] solves H J = dA^H W(y-Ax) - A^H W dA x.

    Assumes real phis/bases and encoding exp(-2j*pi*phi^T F g).
    dA_bp = -2j*pi*M_g[p]*A*M_phi[b]. Spatial/temporal multipliers
    broadcast over image/k-space dimensions; diagonal W commutes with M_g.
    """
    normal = lambda x: operator.adjoint(operator.forward(x)) + regularization * x
    residual = ksp - operator.forward(image)
    residual_responses = torch.stack([operator.adjoint(g * residual) for g in bases])
    responses = image.new_empty(len(phis), len(bases), *image.shape)
    for b, phi in enumerate(phis):
        encoded = operator.forward(phi * image)
        for p, g in enumerate(bases):
            rhs = 2j * torch.pi * (phi * residual_responses[p]
                                  + operator.adjoint(g * encoded))
            responses[b, p] = _implicit_cg(normal, rhs, max_iter=cg_max_iter,
                                           tolerance=cg_tolerance)
    return responses


def taylor_implicit_trust(phis, bases, ksp, build_linop, recon=None,
                          metric=gradient_entropy_metric, F_init=None,
                          outer_steps=30, inner_steps=25, radius=0.1,
                          max_radius=1, P_batch_size=1, max_retries=4,
                          verbose=True, *, regularization=1e-3,
                          cg_max_iter=100, cg_tolerance=1e-4):
    """Taylor trust-region calibration using implicit image derivatives.

    Mirrors taylor_trust's arguments and result. Every base/trial image solves
    the same ridge-regularized weighted least-squares problem; optional recon
    provides only an initial guess, which is refined to cg_tolerance. Failed
    image or derivative solves raise rather than silently using a bad model.

    The operator must encode exp(-2j*pi*phi^T F g), with forward=A and
    adjoint=A^H W for fixed nonnegative diagonal W. phis/bases must be real.
    No differentiation through build_linop is needed. For approximate encoding
    implementations these are derivatives of the physical phase model.

    Caches B*P complex derivative images, unlike taylor_trust's P images.
    P_batch_size is retained for call compatibility; solves run sequentially.
    Use implicit_image_recon with the same settings for reference images.
    """
    if phis.is_complex() or bases.is_complex():
        raise ValueError('Implicit phase derivatives require real phis and bases')
    if P_batch_size < 1:
        raise ValueError('P_batch_size must be positive')
    settings = dict(regularization=regularization, cg_max_iter=cg_max_iter,
                    cg_tolerance=cg_tolerance)

    def reconstruct(operator, data):
        initial = None if recon is None else recon(operator, data)
        return implicit_image_recon(operator, data, initial=initial, **settings)

    def responses(operator, image, phi, g):
        return _implicit_responses(operator, image, ksp, phi, g, **settings)

    return taylor_trust(phis, bases, ksp, build_linop, reconstruct, metric,
                        F_init, outer_steps, inner_steps, radius, max_radius,
                        P_batch_size, max_retries, verbose,
                        _response_builder=responses)
