"""
This file will contain useful functions to describe how phase evolves over the trajectory.

evolution = exp(-2j * pi * \sum_p f_p g_p(t))

g_p(t) are basis functions, f_p are the coefficients that need to be estimated.

By default, g_p(t) are scaled so that g_p.max() - g_p.min() = 1.
"""
import torch
from typing import Optional
from einops import rearrange

def compress_bases(bases: torch.Tensor,
                   num_bases: Optional[int] = None,
                   nrmse_thresh: Optional[float] = 1e-3) -> torch.Tensor:
    f"""
    Compress bases using an SVD
    
    Args
    ----
    bases: torch.Tensor
        bases tensor, shape (P, ...)
    num_bases: int
        if given, P' = num_bases
    nrmse_thresh: float
        if given, P' is the smallest number of bases such that the NRMSE between the original and compressed bases is less than nrmse_thresh
    
    Returns
    -------
    bases_new: torch.Tensor
        compressed bases tensor, shape (P', ...)
        P' < P
    """
    # Consts
    P = bases.shape[0]
    bases_flt = bases.reshape((P, -1))
    
    # SVD
    U, S, Vh = torch.linalg.svd(bases_flt, full_matrices=False)
    
    # Thresholding
    if num_bases is not None:
        bases_new = Vh[:num_bases]
    elif nrmse_thresh is not None:
        Sc = S.square().cumsum(dim=0)
        nrmses = (1 - Sc / Sc[-1]).sqrt()
        idxs = torch.argwhere(nrmses <= nrmse_thresh)
        idx = idxs.min()
        bases_new = Vh[:idx+1]
    else:
        raise ValueError("Either num_bases or nrmse_thresh must be given")
        
    # Center evolutions
    qs = torch.tensor([0.0, 1.0], device=bases_new.device)
    low, high = torch.quantile(bases_new, q=qs, dim=1)

    # Remove any indices with no variation
    denom = low.abs() + high.abs()
    idx_flat = (high - low).abs() / denom.clamp_min(1e-12) < 1e-6
    idx_flat = idx_flat | (denom == 0)
    bases_new = bases_new[~idx_flat]
    low = low[~idx_flat]
    high = high[~idx_flat]

    # Rescale
    scales = high - low
    bases_ofs = (high + low) / 2 / scales
    bases_new = bases_new / scales[:, None] - bases_ofs[:, None]
        
    # # Append exactly one flat term
    # ones = bases_new[:1] * 0 + 1
    # bases_new = torch.cat([ones, bases_new], dim=0)
    
    # Reshape
    return bases_new.reshape((bases_new.shape[0], *bases.shape[1:]))
        
        
        

def linear_time_bases(trj: torch.Tensor,
                      readout_dim: int = 0) -> torch.Tensor:
    """
    linear time only, useful for B0 modeling
    
    Args
    ----
    trj: torch.Tensor
        trajectory tensor, shape (..., D)
    readout_dim: int
        dimension of the readout dimension
    
    Returns
    -------
    g: torch.Tensor
        single basis functions, shape (1, ...)
    """
    shape = [1] * (trj.ndim - 1)
    shape[readout_dim] = -1
    g = torch.linspace(0, 1, trj.shape[readout_dim], 
                         dtype=trj.dtype, device=trj.device)
    g = g.reshape(*shape)
    
    return g[None, ...]

def polar_poly_fourier_bases(trj: torch.Tensor,
                             num_radial: int = 3,
                             num_angular: int = 1,
                             readout_dim: int = 0,
                             use_grad: bool = False) -> torch.Tensor:
    f"""
    Polynomial in pr, Fourier in pang, useful for spiral/radial gradient imperfections:
    cos(m * pang) * (pr ** n), sin(m * pang) * (pr ** n)
    
    Total number of bases K = num_radial * (num_angular * 2 + 1)
    
    Args
    ----
    trj: torch.Tensor
        trajectory tensor, shape (..., D)
    num_radial: int
        number of radial basis functions
    num_angular: int
        number of angular basis functions
    readout_dim: int
        dimension of the readout dimension
    
    Returns
    -------
    g: torch.Tensor
        basis functions, shape (P, ...,)
    """
    assert trj.shape[-1] == 2, "radial chebyshev bases only work in 2D"
     
    # Build pr, pang
    if use_grad:
        grad = trj.diff(dim=readout_dim)
        idx = torch.tensor([grad.shape[readout_dim] - 1], device=grad.device, dtype=torch.long)
        last = torch.index_select(grad, dim=readout_dim, index=idx)
        grad = torch.cat([grad, last], dim=readout_dim)
        pr = grad.norm(dim=-1)
    else:
        pr = trj.norm(dim=-1)
    pr /= pr.max() # in [0, 1]
    pang = torch.atan2(trj[..., 1], trj[..., 0]) # in [-pi, pi]
    
    # Polynomials in pr
    g_pr = [pr ** i for i in range(1, num_radial+1)]
    g_pr = torch.stack(g_pr, dim=-1) # shape (..., num_radial)
    
    # Fourier in pang
    g_sin = [torch.sin(i * pang) for i in range(1, num_angular+1)]
    g_cos = [torch.cos(i * pang) for i in range(0, num_angular+1)];
    g_cos[0] *= 2
    g_ang = torch.stack(g_sin + g_cos, dim=-1) # shape (..., num_angular * 2)
    g_ang /= 2 # in [-0.5, 0.5]
        
    # Combine
    g = g_pr[..., :, None] * g_ang[..., None, :]
    g = g.reshape(*trj.shape[:-1], -1).moveaxis(-1, 0)
    
    return g


def _cubic_bspline(u: torch.Tensor) -> torch.Tensor:
    u = u.abs()
    return torch.where(u < 1, (4 - 6 * u.square() + 3 * u**3) / 6,
                       (2 - u).clamp_min(0)**3 / 6)


def _hermitian_spline_spectrum(freq_hz: torch.Tensor,
                               n_knots: int,
                               max_freq_hz: float) -> torch.Tensor:
    """Even real cubic B-splines B_q(f) on an rFFT grid, shape (n_freq, Q).

    Each B_q is Hermitian (B_q(-f) = B_q(f)), so b_q = IFFT(B_q) is real.
    The spectrum is tapered to 0 over the last knot interval.
    """
    if n_knots < 3:
        raise ValueError('Need at least three spline knots')
    knot_hz = torch.linspace(0, max_freq_hz, n_knots, device=freq_hz.device, dtype=freq_hz.dtype)
    spacing = max_freq_hz / (n_knots - 1)
    positive = _cubic_bspline((freq_hz[:, None] - knot_hz) / spacing)
    negative = _cubic_bspline((freq_hz[:, None] + knot_hz) / spacing)
    basis = positive + negative
    basis[:, 0] *= 0.5
    taper = 0.5 * (1 + torch.cos(torch.pi * ((freq_hz - (max_freq_hz - spacing)) / spacing).clamp(0, 1)))
    basis = basis * taper[:, None]
    basis = torch.where(freq_hz[:, None] >= max_freq_hz, torch.zeros_like(basis), basis)
    return basis.to(torch.complex128)


def girf_bases(grad: torch.Tensor,
               dt: float,
               readout_dim: int = 0,
               num_splines: int = 10,
               max_freq_hz: float = 20e3,
               return_spectrum: bool = False):
    """
    GIRF spline bases for exp(-2j π sum_k f_k g_k(t)).

    Physical model: a real impulse response convolved with each gradient axis,

        exp(-j 2π sum_d (p_d * g_d)(t)),     p_d(f) = sum_q B_q(f) c_{q,d}.

    B_q(f) are cubic B-splines on [0, max_freq_hz], even-extended so they are
    Hermitian and b_q(t) = IFFT(B_q) is real. Complex c_q = a_q + j b_q then
    gives two real filters per spline: IFFT(B_q) and IFFT(j B_q). DC and
    Nyquist of the j B_q spectra are zeroed so the IR stays real.

    Each b_q is treated as a finite noncausal FIR (t = 0 at the center).
    Convolution with g_d is linear, not circular: both sequences are
    zero-padded to length N_t + N_ir - 1 before the FFT product, and the
    'same' length-N_t slice is kept (zero boundaries).

        g_{q,d,re}(t) = (b_q * g_d)(t)
        g_{q,d,im}(t) = (IFFT(j B_q) * g_d)(t)

    Output last axes are (2Q, D): Q real knots then Q imag knots, per gradient
    axis. Each (q, d) basis is scaled so max - min = 1.

    Args
    ----
    grad : torch.Tensor
        Gradient waveforms, shape (..., D), D in {2, 3}. Time is `readout_dim`.
    dt : float
        Dwell time in seconds.
    readout_dim : int
        Axis of `grad` that is time.
    num_splines : int
        Number of knots Q on [0, max_freq_hz].
    max_freq_hz : float
        Spline cutoff in Hz, at most Nyquist.
    return_spectrum : bool
        If True, also return the rFFT frequency axis and the B-spline
        spectra in the same peak-to-peak units as the time-domain bases.

    Returns
    -------
    g : torch.Tensor
        Bases with shape (2Q, D, ...). Time is still at `readout_dim`.
        ``g[..., :Q, d]`` is real(c_q) on axis d; ``g[..., Q:, d]`` is imag(c_q).
    freq_hz : torch.Tensor
        Only if ``return_spectrum``. rFFT frequencies in Hz, shape (F,).
    spline_fft : torch.Tensor
        Only if ``return_spectrum``. Complex B-spline spectra, shape (F, 2Q, D).
        lstsq coefficients of ``g`` apply directly:
        ``einsum('fqd,qds->sdf', spline_fft, coeffs)`` has shape (S, D, F).
    """
    if grad.ndim < 2:
        raise ValueError('grad must have a time axis and a trailing axis dimension')
    n_axes = grad.shape[-1]
    if n_axes not in (2, 3):
        raise ValueError('grad trailing dimension must be 2 or 3 axes')
    nyquist = 0.5 / dt
    if not 0 < max_freq_hz <= nyquist:
        raise ValueError('max_freq_hz must lie in (0, Nyquist]')

    n_time = grad.shape[readout_dim]
    n_ir = 1 << (2 * n_time - 1).bit_length()
    freq_hz = torch.fft.rfftfreq(n_ir, d=dt, device=grad.device, dtype=torch.float64)
    B = _hermitian_spline_spectrum(freq_hz, num_splines, max_freq_hz)
    # Real coeff uses B_q; imag coeff uses j B_q. rFFT Nyquist is real.
    basis_fft = torch.cat((B, 1j * B), dim=1)
    basis_fft[0, num_splines:] = 0
    basis_fft[-1, num_splines:] = 0

    # Finite IRs in FFT order, then centered so t=0 is at n_ir // 2.
    kernels = torch.fft.fftshift(torch.fft.irfft(basis_fft, n=n_ir, dim=0), dim=0)
    n_conv = n_time + n_ir - 1
    kernel_fft = torch.fft.rfft(kernels, n=n_conv, dim=0)
    grad_fft = torch.fft.rfft(grad.double(), n=n_conv, dim=readout_dim)
    broadcast = [1] * (grad.ndim + 1)
    broadcast[readout_dim] = kernel_fft.shape[0]
    broadcast[-1] = kernel_fft.shape[1]
    filtered = torch.fft.irfft(
        grad_fft.unsqueeze(-1) * kernel_fft.reshape(*broadcast),
        n=n_conv, dim=readout_dim,
    )
    filtered = filtered.narrow(readout_dim, n_ir // 2, n_time)
    bases = filtered.moveaxis(-1, -2)
    span = bases.flatten(end_dim=-3).amax(0) - bases.flatten(end_dim=-3).amin(0)
    scale = span.clamp_min(1e-12)
    bases = bases / scale
    bases = bases * (span > 1e-12).to(bases.dtype)
    bases = bases.to(dtype=grad.dtype)
    bases = rearrange(bases, '... q d -> q d ...')
    if not return_spectrum:
        return bases
    spline_fft = basis_fft.unsqueeze(-1) / scale.to(dtype=basis_fft.dtype)
    spline_fft = spline_fft * (span > 1e-12).to(dtype=spline_fft.dtype)
    return bases, freq_hz, spline_fft
