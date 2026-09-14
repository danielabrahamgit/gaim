"""
Least-squares reconstruction for a fixed forward model A(θ):

    x̂(θ) = argmin_x ||A(θ) x - b||_2^2
"""

import torch
from typing import Any, Callable, Optional


def _as_normal(A: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    if hasattr(A, "normal"):
        return A.normal
    return lambda x: A.adjoint(A.forward(x))


def conjugate_gradient(AHA: Callable[[torch.Tensor], torch.Tensor],
                       AHb: torch.Tensor,
                       num_iters: Optional[int] = 15,
                       lamda_l2: Optional[float] = 0.0,
                       tolerance: Optional[float] = 1e-8,
                       verbose: Optional[bool] = False) -> torch.Tensor:
    """
    Conjugate gradient for (AHA + λI) x = AHb.

    Args
    ----
    AHA : callable
        Normal operator mapping image -> image.
    AHb : torch.Tensor
        Right-hand side with the image shape.
    num_iters : int
        Maximum CG iterations.
    lamda_l2 : float
        Tikhonov weight on ||x||_2^2.
    tolerance : float
        Relative residual stopping tolerance.
    verbose : bool
        Print residual norms.

    Returns
    -------
    x : torch.Tensor
        Least-squares estimate, same shape as `AHb`.
    """
    def AHA_reg(x: torch.Tensor) -> torch.Tensor:
        y = AHA(x)
        if lamda_l2:
            y = y + lamda_l2 * x
        return y

    x = torch.zeros_like(AHb)
    r = AHb - AHA_reg(x)
    p = r.clone()
    rs_old = (r.conj() * r).real.sum()
    rhs_norm = (AHb.conj() * AHb).real.sum().clamp_min(1e-12)

    for i in range(num_iters):
        Ap = AHA_reg(p)
        denom = (p.conj() * Ap).real.sum().clamp_min(1e-12)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r.conj() * r).real.sum()
        if verbose:
            print(f"CG iter {i}: residual = {rs_new.item():.3e}")
        if (rs_new / rhs_norm).sqrt() < tolerance:
            break
        p = r + (rs_new / rs_old.clamp_min(1e-12)) * p
        rs_old = rs_new

    return x


def least_squares_recon(A: Any,
                        measurements: torch.Tensor,
                        num_iters: Optional[int] = 15,
                        lamda_l2: Optional[float] = 0.0,
                        tolerance: Optional[float] = 1e-8,
                        verbose: Optional[bool] = False) -> torch.Tensor:
    """
    Reconstruct an image from measurements under a fixed encoding A.

    This is the inner reconstruction used by GAIM:

        x̂(θ) = argmin_x ||A(θ) x - b||_2^2 + λ ||x||_2^2

    `A` should expose `.forward` / `.adjoint`, and optionally `.normal`.
    This matches `mr_recon.linops.linop`.

    Args
    ----
    A : linop-like
        Forward model at a fixed θ.
    measurements : torch.Tensor
        k-space data b, with shape matching `A.forward` output.
    num_iters : int
        CG iterations.
    lamda_l2 : float
        Tikhonov regularization.
    tolerance : float
        CG stopping tolerance.
    verbose : bool
        Print CG residuals.

    Returns
    -------
    x_hat : torch.Tensor
        Reconstructed image.
    """
    AHb = A.adjoint(measurements)
    return conjugate_gradient(
        AHA=_as_normal(A),
        AHb=AHb,
        num_iters=num_iters,
        lamda_l2=lamda_l2,
        tolerance=tolerance,
        verbose=verbose,
    )
