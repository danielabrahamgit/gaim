import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Sequence


def _spatial_dims(spatial_ndim: int) -> tuple:
    return tuple(range(-spatial_ndim, 0))


def _discrete_laplacian(mag: torch.Tensor,
                        spatial_ndim: int) -> torch.Tensor:
    batch_shape = mag.shape[:-spatial_ndim]
    x = mag.reshape(-1, 1, *mag.shape[-spatial_ndim:])
    xp = F.pad(x, (1, 1) * spatial_ndim, mode='replicate')
    xp = xp.reshape(*batch_shape, *xp.shape[-spatial_ndim:])
    center = [slice(None)] * mag.ndim
    for i in range(spatial_ndim):
        center[-spatial_ndim + i] = slice(1, -1)
    lap = -2.0 * spatial_ndim * xp[tuple(center)]
    for i in range(spatial_ndim):
        sl_hi = list(center)
        sl_lo = list(center)
        ax = -spatial_ndim + i
        sl_hi[ax] = slice(2, None)
        sl_lo[ax] = slice(0, -2)
        lap = lap + xp[tuple(sl_hi)] + xp[tuple(sl_lo)]
    return lap


def _haar1d(x: torch.Tensor, dim: int):
    n = x.shape[dim] - x.shape[dim] % 2
    if n < 2:
        return None, None
    sl_even = [slice(None)] * x.ndim
    sl_odd = [slice(None)] * x.ndim
    sl_even[dim] = slice(0, n, 2)
    sl_odd[dim] = slice(1, n, 2)
    even = x[tuple(sl_even)]
    odd = x[tuple(sl_odd)]
    scale = 0.5 ** 0.5
    return (even + odd) * scale, (even - odd) * scale


def _haar_details(mag: torch.Tensor, spatial_ndim: int):
    spatial_dims = _spatial_dims(spatial_ndim)
    approx = mag
    details = []
    while all(approx.shape[dim] >= 2 for dim in spatial_dims):
        bands = [approx]
        for dim in spatial_dims:
            nxt = []
            for band in bands:
                lo, hi = _haar1d(band, dim)
                if lo is None:
                    return details
                nxt.extend((lo, hi))
            bands = nxt
        approx = bands[0]
        details.extend(bands[1:])
    return details


def gradient_entropy_metric(img: torch.Tensor,
                            spatial_ndim: int = 2,
                            eps: Optional[float] = 1e-12) -> torch.Tensor:
    """
    Shannon entropy of the spatial-gradient magnitude.

    Args
    ----
    img : torch.Tensor
        Reconstructed image with shape (..., *im_size). Complex images are
        converted to magnitude.
    spatial_ndim : int
        Number of trailing spatial axes. Leading dims are treated as batch
        and are preserved in the output.
    eps : float
        Numerical floor for logs and normalization.

    Returns
    -------
    H : torch.Tensor
        Entropy with shape equal to the leading batch dims of `img`.
    """
    mag = img.abs()
    spatial_dims = tuple(range(-spatial_ndim, 0))
    grads = torch.gradient(mag, dim=spatial_dims)
    g2 = grads[0].square()
    for g in grads[1:]:
        g2 = g2 + g.square()
    grad = (g2 + eps).sqrt()
    p = grad / grad.sum(dim=spatial_dims, keepdim=True).clamp_min(eps)
    return (p * (p + eps).log()).sum(dim=spatial_dims)

def atkinson_entropy_metric(img: torch.Tensor,
                            spatial_ndim: int = 2,
                            eps: Optional[float] = 1e-12) -> torch.Tensor:
    """
    Atkinson entropy of the spatial-gradient magnitude.

    Args
    ----
    img : torch.Tensor
        Reconstructed image with shape (..., *im_size). Complex images are
        converted to magnitude.
    spatial_ndim : int
        Number of trailing spatial axes. Leading dims are treated as batch
        and are preserved in the output.
    eps : float
        Numerical floor for logs and normalization.

    Returns
    -------
    H : torch.Tensor
        Entropy with shape equal to the leading batch dims of `img`.
    """
    mag = img.abs()
    spatial_dims = tuple(range(-spatial_ndim, 0))
    nrm = mag.square().sum(dim=spatial_dims, keepdim=True).sqrt()
    mag_nrm = mag / nrm
    return -(mag_nrm * torch.log(mag_nrm + eps)).sum(dim=spatial_dims)
    grads = torch.gradient(mag, dim=spatial_dims)
    g2 = grads[0].square()
    for g in grads[1:]:
        g2 = g2 + g.square()
    grad = (g2 + eps).sqrt()
    p = grad / grad.sum(dim=spatial_dims, keepdim=True).clamp_min(eps)
    return (p * (p + eps).log()).sum(dim=spatial_dims)


def _gradient_magnitude(mag: torch.Tensor,
                        spatial_ndim: int,
                        eps: float) -> torch.Tensor:
    spatial_dims = _spatial_dims(spatial_ndim)
    grads = torch.gradient(mag, dim=spatial_dims)
    g2 = grads[0].square()
    for g in grads[1:]:
        g2 = g2 + g.square()
    return (g2 + eps).sqrt()


def normalized_gradient_squared_metric(img: torch.Tensor,
                                       spatial_ndim: int = 2,
                                       eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Normalized gradient squared, ``sum(g**2) / (sum(g)**2 + eps)``.

    ``g`` is the spatial-gradient magnitude. The score is scale-invariant and
    larger when gradient energy is spatially concentrated.
    """
    dims = _spatial_dims(spatial_ndim)
    g = _gradient_magnitude(img.abs(), spatial_ndim, eps)
    return g.square().sum(dim=dims) / (g.sum(dim=dims).square() + eps)


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    return kernel / kernel.sum()


def gaussian_smooth(mag: torch.Tensor,
                    sigma: float,
                    spatial_ndim: int = 2) -> torch.Tensor:
    """Separable Gaussian smoothing; ``sigma <= 0`` is a no-op."""
    if sigma <= 0:
        return mag
    if spatial_ndim != 2:
        raise ValueError('Gaussian smoothing is implemented for 2D images.')
    batch_shape = mag.shape[:-2]
    x = mag.reshape(-1, 1, *mag.shape[-2:])
    kernel = _gaussian_kernel1d(float(sigma), x.device, x.dtype)
    pad = kernel.numel() // 2
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='replicate'), kernel.view(1, 1, -1, 1))
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode='replicate'), kernel.view(1, 1, 1, -1))
    return x.reshape(*batch_shape, *mag.shape[-2:])


GAUSSIAN_ENTROPY_SIGMAS = (0.0, 0.5, 1.0, 2.0)


def gaussian_scale_gradient_entropy(img: torch.Tensor,
                                    sigmas: Sequence[float] = GAUSSIAN_ENTROPY_SIGMAS,
                                    spatial_ndim: int = 2,
                                    eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Existing gradient-entropy metric after Gaussian smoothing at each ``sigma``.

    Returns a stack with the sigma axis first, then the leading batch dims of
    ``img``. ``sigma=0`` is unsmoothed and matches ``gradient_entropy_metric``.
    This is distinct from ``multiscale_gradient_entropy_metric``, which pools.
    """
    mag = img.abs()
    return torch.stack([
        gradient_entropy_metric(gaussian_smooth(mag, sigma, spatial_ndim), spatial_ndim, eps)
        for sigma in sigmas
    ])


def average_edge_strength_metric(img: torch.Tensor,
                                 spatial_ndim: int = 2,
                                 eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Average edge strength (AES): mean Sobel-gradient magnitude.

    AES = mean sqrt((Sx * I)^2 + (Sy * I)^2) with the fixed 3x3 Sobel kernels

        Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    replicate padding, and no threshold (Gonzalez & Woods; the same Sobel AES
    used in MRI autofocus comparisons such as McGee et al., Med. Phys. 2000).
    Settings stay fixed across a sweep. Larger values indicate stronger edges.
    """
    if spatial_ndim != 2:
        raise ValueError('AES is defined with 2D Sobel kernels.')
    mag = img.abs()
    batch_shape = mag.shape[:-2]
    x = mag.reshape(-1, 1, *mag.shape[-2:])
    sobel_x = mag.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).reshape(1, 1, 3, 3)
    sobel_y = mag.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).reshape(1, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode='replicate')
    edge = (F.conv2d(xp, sobel_x).square() + F.conv2d(xp, sobel_y).square() + eps).sqrt()
    return edge.mean(dim=(-2, -1)).reshape(batch_shape)


def multiscale_gradient_entropy_metric(img: torch.Tensor,
                                       spatial_ndim: int = 2,
                                       eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Mean negative gradient entropy at up to three dyadic image scales.

    Pool magnitude before taking gradients, so coarse structure contributes
    alongside fine edges. Leading batch dimensions are preserved.
    """
    if spatial_ndim not in (1, 2, 3):
        raise ValueError('Multiscale entropy supports one to three spatial dimensions.')
    pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[spatial_ndim]
    mag = img.abs()
    batch_shape = mag.shape[:-spatial_ndim]
    scores = []
    for level in range(3):
        scores.append(gradient_entropy_metric(mag, spatial_ndim, eps))
        if level == 2 or min(mag.shape[-spatial_ndim:]) < 4:
            break
        pooled = pool(mag.reshape(-1, 1, *mag.shape[-spatial_ndim:]), 2)
        mag = pooled.reshape(*batch_shape, *pooled.shape[2:])
    return torch.stack(scores).mean(0)


def gradient_renyi_metric(img: torch.Tensor,
                          spatial_ndim: int = 2,
                          eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Negative order-two entropy of gradient magnitude; larger is sharper.

    log(sum(p**2)) uses the same natural-log units as gradient entropy but
    emphasizes strong edges more. The floor also keeps flat-patch gradients
    differentiable, matching gradient_entropy_metric.
    """
    dims = _spatial_dims(spatial_ndim)
    gradients = torch.gradient(img.abs(), dim=dims)
    energy = sum(part.square() for part in gradients) + eps
    total = energy.sqrt().sum(dim=dims).clamp_min(eps)
    return energy.sum(dim=dims).clamp_min(eps).log() - 2 * total.log()


def gradient_renyi_half_metric(img: torch.Tensor,
                              spatial_ndim: int = 2,
                              eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Negative order-1/2 gradient entropy, sensitive to weak diffuse edges.

    With p the normalized gradient magnitude, the score is
    -2*log(sum(sqrt(p))). Larger is better, with the same gradient floor as
    gradient_entropy_metric. Lowering the order increases the contribution
    of small probabilities; whether this improves autofocus requires testing.
    """
    dims = _spatial_dims(spatial_ndim)
    gradients = torch.gradient(img.abs(), dim=dims)
    magnitude = (sum(part.square() for part in gradients) + eps).sqrt()
    return (magnitude.sum(dim=dims).log()
            - 2 * magnitude.sqrt().sum(dim=dims).log())


def wavelet_renyi_half_metric(img: torch.Tensor,
                             spatial_ndim: int = 2,
                             eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Negative order-1/2 entropy of the decimated Haar detail magnitudes."""
    mag = img.abs()
    batch_shape = mag.shape[:-spatial_ndim]
    details = _haar_details(mag, spatial_ndim)
    if not details:
        return mag.new_zeros(batch_shape)
    coefficients = torch.cat([band.reshape(*batch_shape, -1) for band in details], dim=-1)
    # A smooth positive floor makes a flat patch uniform (minimum score).
    magnitude = (coefficients.square() + eps**2).sqrt()
    return magnitude.sum(-1).log() - 2 * magnitude.sqrt().sum(-1).log()


def undecimated_wavelet_entropy_metric(img: torch.Tensor,
                                      spatial_ndim: int = 2,
                                      eps: Optional[float] = 1e-12) -> torch.Tensor:
    """Haar detail entropy without decimation, using valid dilated filters.

    Evaluates every translation within a patch at each dyadic scale, reducing
    sensitivity to the decimated Haar grid. Valid filtering avoids introducing
    periodic patch boundaries; finite patch borders still affect the score.
    """
    mag = img.abs()
    batch_shape = mag.shape[:-spatial_ndim]
    dims = _spatial_dims(spatial_ndim)
    approx, dilation, details = mag, 1, []
    while min(approx.shape[-spatial_ndim:]) > dilation:
        bands = [approx]
        for dim in dims:
            next_bands = []
            for band in bands:
                first, second = [slice(None)] * band.ndim, [slice(None)] * band.ndim
                first[dim], second[dim] = slice(None, -dilation), slice(dilation, None)
                a, b = band[tuple(first)], band[tuple(second)]
                next_bands.extend(((a + b) * 2**-.5, (a - b) * 2**-.5))
            bands = next_bands
        approx = bands[0]
        details.extend(bands[1:])
        dilation *= 2
    if not details:
        return mag.new_zeros(batch_shape)
    coefficients = torch.cat([band.reshape(*batch_shape, -1) for band in details], dim=-1)
    magnitude = (coefficients.square() + eps**2).sqrt()
    p = magnitude / magnitude.sum(-1, keepdim=True)
    return (p * p.log()).sum(-1)


def laplacian_energy_metric(img: torch.Tensor,
                            spatial_ndim: int = 2,
                            eps: Optional[float] = 1e-12) -> torch.Tensor:
    """
    Energy of the discrete spatial Laplacian.

    Args
    ----
    img : torch.Tensor
        Reconstructed image with shape (..., *im_size). Complex images are
        converted to magnitude.
    spatial_ndim : int
        Number of trailing spatial axes. Leading dims are treated as batch
        and are preserved in the output.
    eps : float
        Numerical floor. Accepted for signature parity with the other metrics.

    Returns
    -------
    E : torch.Tensor
        Laplacian energy with shape equal to the leading batch dims of `img`.
        Larger values indicate stronger high-frequency content.
    """
    mag = img.abs()
    spatial_dims = _spatial_dims(spatial_ndim)
    lap = _discrete_laplacian(mag, spatial_ndim)
    return lap.square().sum(dim=spatial_dims)


def wavelet_sparsity_metric(img: torch.Tensor,
                            spatial_ndim: int = 2,
                            eps: Optional[float] = 1e-12) -> torch.Tensor:
    """
    Shannon sparsity of Haar wavelet detail coefficients.

    Args
    ----
    img : torch.Tensor
        Reconstructed image with shape (..., *im_size). Complex images are
        converted to magnitude.
    spatial_ndim : int
        Number of trailing spatial axes. Leading dims are treated as batch
        and are preserved in the output.
    eps : float
        Numerical floor for logs and normalization.

    Returns
    -------
    S : torch.Tensor
        Sparsity with shape equal to the leading batch dims of `img`. Larger
        values indicate more concentrated (sparser) high-frequency structure.
    """
    mag = img.abs()
    details = _haar_details(mag, spatial_ndim)
    batch_shape = mag.shape[:-spatial_ndim]
    if not details:
        return mag.new_zeros(batch_shape)

    w = torch.cat(
        [d.reshape(*batch_shape, -1).abs() for d in details],
        dim=-1,
    )
    p = w / w.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (p * (p + eps).log()).sum(dim=-1)


def peaky_curve_confidence(y: torch.Tensor,
                           x: Optional[torch.Tensor] = None,
                           dim: int = 0,
                           n_widths: int = 16,
                           eps: Optional[float] = 1e-12) -> torch.Tensor:
    """
    Confidence that a 1D slice of `y` is a smooth, interior, unimodal peak.

    Fits `a * exp(-((x - x0) / s)^2 / 2) + b` with `x0` at the discrete
    argmax, sweeping the width `s`. The score is the best Gaussian R^2
    among positive-amplitude fits. Flat, noisy, multi-modal, and
    frequency-wall curves score low.

    Args
    ----
    y : torch.Tensor
        Scores with the swept parameter along `dim`, e.g. (n_freqs, n_h, n_w).
    x : torch.Tensor, optional
        Sample locations along `dim`, shape (y.shape[dim],). Defaults to a
        uniform grid in [-1, 1].
    dim : int
        Axis of `y` to test.
    n_widths : int
        Number of log-spaced Gaussian widths to try.
    eps : float
        Numerical floor.

    Returns
    -------
    conf : torch.Tensor
        Confidence in [0, 1] with `dim` removed.
    """
    y = y.movedim(dim, 0)
    n = y.shape[0]
    batch_shape = y.shape[1:]
    yf = y.reshape(n, -1)
    dtype = yf.real.dtype if torch.is_complex(yf) else yf.dtype
    yf = yf.real.to(dtype)

    if x is None:
        xf = torch.linspace(-1, 1, n, device=y.device, dtype=dtype)
    else:
        xf = x.reshape(n).to(device=y.device, dtype=dtype)

    idx = yf.argmax(dim=0)
    x0 = xf[idx]
    xc = xf[:, None] - x0[None, :]
    span = (xf.max() - xf.min()).clamp_min(eps)
    dx = span / max(n - 1, 1)
    sigmas = torch.logspace(
        torch.log10(dx * 0.5),
        torch.log10(span * 0.6),
        n_widths,
        device=y.device,
        dtype=dtype,
    )
    g = torch.exp(-0.5 * (xc[None, :, :] / sigmas[:, None, None]).square())

    gtg = g.square().sum(dim=1)
    gt1 = g.sum(dim=1)
    gty = (g * yf[None]).sum(dim=1)
    y1 = yf.sum(dim=0)
    nf = float(n)
    det = (gtg * nf - gt1.square()).clamp_min(eps)
    a = (nf * gty - gt1 * y1) / det
    b = (gtg * y1 - gt1 * gty) / det
    yhat = a[:, None, :] * g + b[:, None, :]
    ss_res = (yf[None] - yhat).square().sum(dim=1)
    ss_tot = (yf - yf.mean(0, keepdim=True)).square().sum(0)
    r2 = (1 - ss_res / ss_tot.clamp_min(eps)).clamp(0, 1)
    r2 = torch.where((a > 0) & (ss_tot > eps), r2, torch.zeros_like(r2))
    conf = r2.max(dim=0).values
    interior = ((idx > 0) & (idx < n - 1)).to(conf.dtype)
    return (conf * interior).reshape(batch_shape)


@torch.no_grad()
def patch_ray_confidence(score_fn,
                         f: torch.Tensor,
                         t_max: float = 2.0,
                         n_t: int = 21) -> torch.Tensor:
    """
    Per-patch trust in a local autofocus solution `f`.

    Evaluates `score_fn` (higher-is-better) along the ray `t f` for
    t in [0, t_max] and returns ``peaky_curve_confidence`` of that curve.
    A unimodal peak at the Adam solution scores high; flat patches, noisy
    metrics, and solutions that ran to a wrap wall score low.

    Args
    ----
    score_fn : callable
        Maps coefficients shaped like `f` to a score per leading batch index.
    f : torch.Tensor
        Per-patch solutions, shape (G, P).
    t_max : float
        Ray goes from 0 through 1 (the solution) out to `t_max`.
    n_t : int
        Samples along the ray.
    """
    t = torch.linspace(0, t_max, n_t, device=f.device, dtype=f.real.dtype)
    scores = torch.stack([score_fn(tk * f) for tk in t], dim=0)
    return peaky_curve_confidence(scores, x=t, dim=0)


def to_iqa_input(img: torch.Tensor,
                 scale: float,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert reconstructed images to PyIQA NCHW RGB in ``[0, 1]``.

    Magnitude is divided by a shared positive ``scale`` (from the fully
    corrected image) and clamped. Grayscale is repeated to three identical
    channels. Leading dimensions are flattened into the batch axis.
    """
    mag = img.abs() if torch.is_complex(img) else img.real
    if mag.ndim < 2:
        raise ValueError('IQA input needs at least two spatial dimensions.')
    scale = float(scale)
    if not (scale > 0) or scale != scale:
        raise ValueError(f'IQA scale must be a positive finite value, got {scale}.')
    slices = mag.reshape(-1, *mag.shape[-2:]).to(dtype=torch.float32)
    rgb = (slices / scale).clamp(0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
    return rgb if device is None else rgb.to(device)


def _patch_clip_vision_model_alias():
    """PyIQA 0.1.16 FGResQ still uses CLIPVisionModel.vision_model.

    Transformers 5 flattened that module, so ``.encoder`` lives on the model
    itself. Alias ``vision_model`` to ``self`` before constructing FGResQ.
    """
    from transformers import CLIPVisionModel

    if getattr(CLIPVisionModel, '_gaim_vision_model_alias', False):
        return
    CLIPVisionModel.vision_model = property(lambda self: self)
    CLIPVisionModel._gaim_vision_model_alias = True


def _pyiqa_metric(name: str, device=None, **kwargs):
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError(
            "PyIQA is required for pretrained IQA wrappers. "
            "Install with: pip install 'pyiqa==0.1.16'"
        ) from exc
    if name in ('fgresq', 'fgresq_pair'):
        _patch_clip_vision_model_alias()
    metric = pyiqa.create_metric(name, device=device, **kwargs)
    metric.eval()
    return metric


def load_fgresq(device=None, **kwargs):
    """Load pretrained FGResQ once and reuse it for single-image and pairwise scoring."""
    return _pyiqa_metric('fgresq', device=device, **kwargs)


def load_arniqa(device=None, **kwargs):
    """Load pretrained ARNIQA (KonIQ regressor by default)."""
    return _pyiqa_metric('arniqa', device=device, **kwargs)


def _metric_device(metric) -> torch.device:
    if hasattr(metric, 'device'):
        return torch.device(metric.device)
    return next(metric.parameters()).device


def _metric_net(metric):
    return metric.net if hasattr(metric, 'net') else metric


@torch.no_grad()
def fgresq_metric(img: torch.Tensor,
                  metric,
                  scale: float) -> torch.Tensor:
    """Single-image FGResQ quality in ``[0, 1]``; higher is better.

    ``img`` is converted with ``to_iqa_input``; FGResQ then applies its own
    resize/center-crop and CLIP normalization. Do not pre-normalize.
    """
    scores = metric(to_iqa_input(img, scale, device=_metric_device(metric)))
    return scores.reshape(-1).cpu()


@torch.no_grad()
def arniqa_metric(img: torch.Tensor,
                  metric,
                  scale: float) -> torch.Tensor:
    """Single-image ARNIQA quality in ``[0, 1]``; higher is better.

    ``img`` is converted with ``to_iqa_input``; ARNIQA then applies ImageNet
    normalization and its half-resolution branch. Do not pre-normalize.
    """
    scores = metric(to_iqa_input(img, scale, device=_metric_device(metric)))
    return scores.reshape(-1).cpu()


def fgresq_pair_from_logits(quality0: torch.Tensor,
                            quality1: torch.Tensor,
                            compare_logits: torch.Tensor) -> dict:
    """Map FGResQ compare-head logits to unambiguous pairwise probabilities.

    PyIQA's ``fgresq_pair`` returns ``(quality0, quality1, rank, rank_prob)``.
    ``rank_prob`` is the softmax mass of the argmax class, not P(first better).

    Class indices from ``FGResQ.get_pair_rank`` in IQA-PyTorch 0.1.16:

    * ``0``: second image better
    * ``1``: first image better
    * ``2``: similar quality

    ``p_candidate_better`` is P(first argument better than the second).
    """
    probs = torch.softmax(compare_logits, dim=-1)
    rank = probs.argmax(dim=-1)
    rank_prob = probs.gather(-1, rank.unsqueeze(-1)).squeeze(-1)
    return {
        'quality0': quality0.reshape(-1).cpu(),
        'quality1': quality1.reshape(-1).cpu(),
        'rank': rank.reshape(-1).cpu(),
        'rank_prob': rank_prob.reshape(-1).cpu(),
        'p_candidate_better': probs[..., 1].reshape(-1).cpu(),
        'p_other_better': probs[..., 0].reshape(-1).cpu(),
        'p_similar': probs[..., 2].reshape(-1).cpu(),
    }


@torch.no_grad()
def fgresq_pair_metric(candidate: torch.Tensor,
                       other: torch.Tensor,
                       metric,
                       scale: float) -> dict:
    """FGResQ pairwise comparison with ``p_candidate_better`` for the first image.

    Uses the FGResQ network on an NR ``fgresq`` metric so one loaded model
    covers both single-image and pairwise scoring. Model preprocessing is
    applied once via ``forward_pair``.
    """
    net = _metric_net(metric)
    device = _metric_device(metric)
    x0 = to_iqa_input(candidate, scale, device=device)
    x1 = to_iqa_input(other, scale, device=device)
    quality0, quality1, logits = net.forward_pair(net.preprocess(x0), net.preprocess(x1))
    return fgresq_pair_from_logits(quality0, quality1, logits)
