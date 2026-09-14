"""
GAIM search:

    J(θ) = m(x̂(θ))
    θ_GAIM = argmax_{θ ∈ Θ} J(θ)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from .recon import least_squares_recon


EncodeFn = Callable[[Any], Any]
MetricFn = Callable[[torch.Tensor], torch.Tensor]
ReconFn = Callable[..., torch.Tensor]


@dataclass
class GaimResult:
    """Best parameters, reconstruction, and objective found by GAIM."""
    params: Any
    image: torch.Tensor
    objective: torch.Tensor
    history: List[Tuple[Any, torch.Tensor]] = field(default_factory=list)


def evaluate_objective(params: Any,
                       encode: EncodeFn,
                       measurements: torch.Tensor,
                       metric: MetricFn,
                       recon_fn: Optional[ReconFn] = None,
                       recon_kwargs: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate J(θ) = m(x̂(θ)) for one parameter value.

    Args
    ----
    params : any
        Candidate model-imperfection parameters θ.
    encode : callable
        Maps θ to a linop-like forward model A(θ).
    measurements : torch.Tensor
        k-space data b.
    metric : callable
        Image-domain metric m(x). Larger should mean a cleaner image.
    recon_fn : callable, optional
        Maps (A, b) to x̂. Defaults to least-squares CG.
    recon_kwargs : dict, optional
        Extra keyword arguments forwarded to `recon_fn`.

    Returns
    -------
    J : torch.Tensor
        Scalar objective m(x̂(θ)).
    x_hat : torch.Tensor
        Reconstruction at this θ.
    """
    if recon_fn is None:
        recon_fn = least_squares_recon
    if recon_kwargs is None:
        recon_kwargs = {}

    A = encode(params)
    x_hat = recon_fn(A, measurements, **recon_kwargs)
    return metric(x_hat), x_hat


def gaim(encode: EncodeFn,
         measurements: torch.Tensor,
         metric: MetricFn,
         param_candidates: Iterable[Any],
         recon_fn: Optional[ReconFn] = None,
         recon_kwargs: Optional[dict] = None,
         verbose: Optional[bool] = True) -> GaimResult:
    """
    Search over candidate parameters by maximizing the image-domain metric.

        θ_GAIM = argmax_{θ ∈ Θ} m(x̂(θ))

    Args
    ----
    encode : callable
        Maps θ to a linop-like forward model A(θ).
    measurements : torch.Tensor
        k-space data b.
    metric : callable
        Image-domain metric m(x). Must satisfy m(x_clean) > m(x_corrupted).
    param_candidates : iterable
        Discrete search set Θ. Each item is a candidate θ.
    recon_fn : callable, optional
        Maps (A, b) to x̂. Defaults to least-squares CG.
    recon_kwargs : dict, optional
        Extra keyword arguments forwarded to `recon_fn`.
    verbose : bool
        Show a progress bar over candidates.

    Returns
    -------
    result : GaimResult
        Best θ, corresponding reconstruction, objective, and search history.
    """
    candidates: Sequence[Any]
    if verbose and not isinstance(param_candidates, Sequence):
        candidates = list(param_candidates)
    else:
        candidates = param_candidates  # type: ignore[assignment]

    iterator = candidates
    if verbose:
        iterator = tqdm(candidates, desc="GAIM search")

    best: Optional[GaimResult] = None
    history: List[Tuple[Any, torch.Tensor]] = []

    for params in iterator:
        J, x_hat = evaluate_objective(
            params,
            encode=encode,
            measurements=measurements,
            metric=metric,
            recon_fn=recon_fn,
            recon_kwargs=recon_kwargs,
        )
        history.append((params, J.detach()))
        if best is None or J > best.objective:
            best = GaimResult(params=params, image=x_hat, objective=J)

    if best is None:
        raise ValueError("param_candidates was empty")

    best.history = history
    return best
