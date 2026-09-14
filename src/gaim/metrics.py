"""
Image-domain metrics m(x) for GAIM.

GAIM cannot use k-space data consistency to search over model-imperfection
parameters, so it needs a metric that is larger for clean images than for
artifacted reconstructions:

    m(x_clean) > m(x_corrupted)
"""

import torch
from typing import Optional


def _magnitude(img: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(img):
        return img.abs()
    return img.real if img.dtype.is_floating_point else img.float()


def _gradient_magnitude(img: torch.Tensor,
                        eps: float = 1e-12) -> torch.Tensor:
    grads = torch.gradient(img)
    g2 = grads[0].square()
    for g in grads[1:]:
        g2 = g2 + g.square()
    return (g2 + eps).sqrt()


def gradient_entropy(img: torch.Tensor,
                     eps: Optional[float] = 1e-12) -> torch.Tensor:
    """
    Shannon entropy of the spatial-gradient magnitude.

    Lower values typically indicate a more focused / less artifacted image.
    Use `negative_gradient_entropy` if you want a higher-is-better score
    that matches the GAIM maximization of J(θ).

    Args
    ----
    img : torch.Tensor
        Reconstructed image with shape (*im_size). Complex images are
        converted to magnitude.
    eps : float
        Numerical floor for logs and normalization.

    Returns
    -------
    H : torch.Tensor
        Scalar entropy.
    """
    mag = _magnitude(img)
    grad = _gradient_magnitude(mag, eps=eps)
    p = grad / grad.sum().clamp_min(eps)
    return -(p * (p + eps).log()).sum()


def negative_gradient_entropy(img: torch.Tensor,
                              eps: Optional[float] = 1e-12) -> torch.Tensor:
    """
    Higher-is-better gradient-entropy metric for GAIM.

    Args
    ----
    img : torch.Tensor
        Reconstructed image with shape (*im_size).
    eps : float
        Numerical floor forwarded to `gradient_entropy`.

    Returns
    -------
    m : torch.Tensor
        Scalar score; larger means a cleaner reconstruction.
    """
    return -gradient_entropy(img, eps=eps)
