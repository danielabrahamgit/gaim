"""Fixed-image, forward-Taylor data-consistency calibration (real F)."""
import torch

from .optim import phase_bound


@torch.no_grad()
def least_squares_image(A, y, initial=None, max_iter=100, tolerance=1e-5):
    """CG for unweighted ||Ax-y||²; A.adjoint must be the true adjoint.

    Explicit forward/adjoint products avoid cached density-weighted normals.
    Return the normal-equation residual so truncated image solves are visible.
    """
    rhs = A.adjoint(y)
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    r = rhs - A.adjoint(A.forward(x))
    p = r.clone()
    rr = r.abs().square().sum()
    denominator = rhs.norm().clamp_min(1e-20)
    iterations = 0
    for i in range(max_iter):
        if r.norm() <= tolerance * denominator:
            break
        Ap = A.forward(p)
        curvature = Ap.abs().square().sum()
        if curvature <= 0:
            break
        step = rr / curvature
        x += step * p
        r -= step * A.adjoint(Ap)
        rr_new = r.abs().square().sum()
        p = r + (rr_new / rr) * p
        rr = rr_new
        iterations = i + 1
    # Recompute to catch drift from single-precision recursive CG residuals.
    normal_residual = (A.adjoint(A.forward(x)) - rhs).norm() / denominator
    return x, dict(cg_iterations=iterations, normal_residual=normal_residual.item())


@torch.no_grad()
def quadratic_model(responses, g, residual, chunk=4096):
    """Return H=Re(JᴴJ), q=Re(Jᴴr), with J_bp=-2πi*g_p*A(phi_b*x).

    Chunking avoids storing B*P complete multi-coil k-space arrays. Dividing
    by ||y||² is done by the caller; it does not change the minimizer.
    """
    B, C = responses.shape[:2]
    P = len(g)
    responses = responses.reshape(B, C, -1)
    g = g.flatten(1)
    residual = residual.reshape(C, -1)
    H = g.new_zeros(B * P, B * P)
    q = g.new_zeros(B * P)
    for start in range(0, g.shape[1], chunk):
        sl = slice(start, start + chunk)
        J = (-2j * torch.pi * responses[:, None, :, sl]
             * g[None, :, None, sl]).reshape(B * P, -1)
        H += (J.conj() @ J.T).real
        q += (J.conj() @ residual[:, sl].flatten()).real
    return H.double(), q.double()


@torch.no_grad()
def quadratic_step(H, q, phi, g, radius):
    """Damped quadratic minimizer; increase damping until the phase cap holds.

    The local objective is convex, but its Hessian can be singular (phase
    ambiguities / redundant bases). Column scaling and a tiny damping floor
    stabilize the solve. This is not an exact constrained phase-norm solve.
    """
    scale = H.diag().clamp_min(H.diag().max() * 1e-12).sqrt()
    scaled = H / scale[:, None] / scale[None, :]
    values, vectors = torch.linalg.eigh((scaled + scaled.T) / 2)
    values = values.clamp_min(0)
    rhs = vectors.T @ (q / scale)
    floor = max(values.max().item() * 1e-6, 1e-12)

    def solve(damping):
        return (-(vectors @ (rhs / (values + damping))) / scale).reshape(len(phi), len(g)).to(g.dtype)

    damping = floor
    delta = solve(damping)
    # Geometric search does not assume phase_bound is monotone in damping.
    while phase_bound(delta, phi, g) > radius:
        damping *= 2
        delta = solve(damping)
    d = delta.flatten().double()
    predicted_gain = -(2 * q.dot(d) + d.dot(H @ d)).item()
    return delta, predicted_gain, damping


@torch.no_grad()
def taylor_dc(phis, bases, y, build_linop, outer_steps=60, radius=0.1,
              max_radius=1.0, max_retries=6, cg_iterations=100, callback=None):
    """Alternate unweighted image LS and fixed-image phase calibration.

    Acceptance compares the nonlinear and Taylor residuals at the SAME x_k.
    Only after acceptance do we reconstruct x_{k+1}. Alphas are (B,*trj_size).
    """
    if min(outer_steps, max_retries, cg_iterations) < 1 or not 0 < radius <= max_radius:
        raise ValueError('Positive iteration counts and 0 < radius <= max_radius required')
    ps = phis.flatten(1).abs().amax(1).clamp_min(1e-12)
    gs = bases.flatten(1).abs().amax(1).clamp_min(1e-12)
    phi = phis / ps.reshape(-1, *([1] * (phis.ndim - 1)))
    g = bases / gs.reshape(-1, *([1] * (bases.ndim - 1)))
    F = bases.new_zeros(len(phi), len(g))
    alphas = lambda f: ((f / ps[:, None]) @ g.flatten(1)).reshape(len(phi), *bases.shape[1:])
    energy = y.abs().square().sum().item()
    if energy <= 0:
        raise ValueError('Data must have nonzero energy')
    A = build_linop(alphas(F))
    x, cg = least_squares_image(A, y, max_iter=cg_iterations)
    initial_image = x.clone()
    history = []
    for outer in range(outer_steps):
        r = A.forward(x) - y
        before = r.abs().square().sum().item() / energy
        responses = torch.stack([A.forward(ph * x) for ph in phi])
        H, q = quadratic_model(responses, g, r)
        H, q = H / energy, q / energy
        accepted = False
        for retry in range(max_retries):
            delta, predicted, damping = quadratic_step(H, q, phi, g, radius)
            if predicted <= 1e-12:
                break
            candidate_A = build_linop(alphas(F + delta))
            fixed_error = (candidate_A.forward(x) - y).abs().square().sum().item() / energy
            ratio = (before - fixed_error) / predicted
            accepted = before > fixed_error and ratio >= 0.1
            bound = phase_bound(delta, phi, g).item()
            record = dict(outer=outer + 1, retry=retry, radius=radius,
                          before=before, fixed_image_error=fixed_error,
                          predicted_gain=predicted, ratio=ratio, phase_bound=bound,
                          damping=damping, accepted=accepted, **cg)
            if not accepted or ratio < 0.25:
                radius *= 0.5
            elif ratio > 0.75 and bound >= 0.7 * radius:
                radius = min(2 * radius, max_radius)
            record['next_radius'] = radius
            history.append(record)
            if accepted:
                F, A = F + delta, candidate_A
                x, cg = least_squares_image(A, y, initial=x, max_iter=cg_iterations)
                record['reconstructed_error'] = (A.forward(x) - y).abs().square().sum().item() / energy
            print(f"DC outer={outer+1} retry={retry} accepted={accepted} "
                  f"error={record.get('reconstructed_error', fixed_error):.7g} "
                  f"ratio={ratio:.3f} phase={bound:.4g}", flush=True)
            if accepted:
                if callback is not None:
                    callback(outer + 1, alphas(F), x, history)
                break
        if not accepted:
            break
    return dict(F=F / ps[:, None] / gs[None, :], alphas=alphas(F), image=x,
                initial_image=initial_image, history=history, **cg)
