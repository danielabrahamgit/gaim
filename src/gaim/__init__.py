"""
Generalized Auto-Focus for Imperfect Models in MRI (GAIM).

For candidate model-imperfection parameters θ, reconstruct

    x̂(θ) = argmin_x ||A(θ) x - b||_2^2

and search

    θ_GAIM = argmax_{θ ∈ Θ} m(x̂(θ))

with an image-domain metric m that scores clean images above artifacted ones.
"""

from .metrics import (
    gradient_entropy,
    negative_gradient_entropy,
)
from .recon import least_squares_recon
from .search import GaimResult, evaluate_objective, gaim

__version__ = "0.0.1"

__all__ = [
    "GaimResult",
    "evaluate_objective",
    "gaim",
    "gradient_entropy",
    "least_squares_recon",
    "negative_gradient_entropy",
    "__version__",
]
