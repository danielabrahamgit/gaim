import torch

from mr_recon.linops import linop
from einops import einsum, rearrange
from typing import Optional
from tqdm import tqdm

from .optim import taylor_sph_optim, taylor_sph_lbfgs, taylor_patch_optim, lstsq_max_phase_wrap
from .patch import strided_patchify, interpolate_patch_grid
from .metrics import patch_ray_confidence

def _demod_recons(demods: torch.tensor,
                  ksp: torch.tensor,
                  recon: callable,
                  batch_size: Optional[int] = None) -> torch.tensor:
    """
    Args
    ----
    demods: torch.tensor
        Demodulation factors with shape (P, *trj_size)
    ksp: torch.tensor
        k-space data with shape (C, *trj_size)
    recon: callable
        Reconstruction function
        inputs:
            k-space data with shape (C, *trj_size)
        outputs:
            image with shape (*im_size)
    batch_size: Optional[int]
        Batch size for reconstruction. If None, no batching is performed.
            
    Returns
    -------
    imgs: torch.tensor
        Reconstructed images with shape (P, *im_size)
    """
    if batch_size is None:
        imgs = []
        for p in tqdm(range(demods.shape[0]), desc='P Recons'):
            imgs.append(recon(ksp * demods[p, ...]))
        imgs = torch.stack(imgs, dim=0)
    else:
        P = demods.shape[0]
        imgs = []
        for b1 in tqdm(range(0, P, batch_size), desc='P Recons'):
            b2 = min(b1 + batch_size, P)
            imgs_batch = recon(ksp[None,] * demods[b1:b2, None,])
            imgs.append(imgs_batch)
        imgs = torch.cat(imgs, dim=0)
    return imgs


def taylor_patch_pipeline(g_bases: torch.tensor,
                          phis: torch.tensor,
                          ksp: torch.tensor,
                          A_best: linop,
                          recon: callable,
                          discrim: callable,
                          patch_size=(100, 100),
                          stride=(50, 50),
                          n_iter: int = 5_000,
                          lr: float = 1e-2) -> torch.tensor:
    """
    Per-patch Taylor autofocus: each patch has its own f in R^P, same Adam
    update as taylor_sph_optim, one batched step over all patches.

    Phase model in a patch is sum_p f_p g_p(t). Those patch coefficients are
    then projected onto the spatial bases: f(r) ≈ F_φᵀ φ(r), with F_φ fit
    by confidence-weighted least squares at the patch anchors
    (pixels 0, stride, 2*stride, …). Alphas are F_φ g(t).

    Returns
    -------
    alphas : (B, *trj_size)
        Temporal coefficients F_φ g.
    confidence : (*im_size)
        Trust in [0, 1], interpolated from the patch grid.
    """
    P = g_bases.shape[0]
    x0 = recon(A_best, ksp)
    demods = g_bases * 2j * torch.pi
    recon_func = lambda k : recon(A_best, k)
    xs = _demod_recons(demods, ksp, recon_func,
                       batch_size=P//10)

    Px0 = strided_patchify(x0, patch_size, stride)
    Pxs = strided_patchify(xs, patch_size, stride)
    grid = Px0.shape[:-2]
    patch_hw = Px0.shape[-2:]
    G = 1
    for s in grid:
        G *= s
    Px0 = Px0.reshape(G, *patch_hw)
    Pxs = Pxs.reshape(P, G, *patch_hw)

    def xhat(fs):
        return Px0 + einsum(Pxs, fs + 0j, 'P G ..., G P -> G ...')

    fs = taylor_patch_optim(
        f_init=torch.zeros((G, P), dtype=x0.real.dtype, device=x0.device),
        g_bases=g_bases,
        loss_fn=lambda f: -discrim(xhat(f)),
        n_iter=n_iter,
        alpha_reg=1.5e-3,
        lr=lr,
    )
    fs_grid = fs.reshape(*grid, P).moveaxis(-1, 0)
    conf = patch_ray_confidence(lambda f: discrim(xhat(f)), fs)
    conf_img = interpolate_patch_grid(conf.reshape(*grid), tuple(x0.shape[-2:]), stride)

    # Anchors are the strided pixels (0, s, 2s, …), same grid as strided_patchify.
    steps = stride if isinstance(stride, (tuple, list)) else (stride,) * len(grid)
    phi_c = phis[(slice(None),) + tuple(slice(None, None, s) for s in steps)]
    Phi = phi_c.reshape(phi_c.shape[0], -1).T.double()
    Y = fs_grid.reshape(P, -1).T.double()
    w = conf.double().square().clamp_min(0)
    col = (Phi.square() * w[:, None]).sum(0).sqrt()
    active = col > 1e-12
    A = Phi[:, active] / col[active]
    gram = A.T @ (w[:, None] * A)
    gram.diagonal().add_(1e-4)
    F_phi = phis.new_zeros((phis.shape[0], P))
    F_phi[active] = (torch.linalg.solve(gram, A.T @ (w[:, None] * Y)) / col[active, None]).to(F_phi.dtype)
    alphas = einsum(F_phi, g_bases, 'B P, P ... -> B ...')    
    return alphas, conf_img

def taylor_sph_dc_pipeline(g_bases: torch.tensor,
                           phis: torch.tensor,
                           ksp: torch.tensor,
                           A_best: linop,
                           recon: callable,
                           n_iter: int = 5_000,
                           max_phase_wrap: float = 1.0,) -> torch.tensor:
    """
    Taylor pipeline for estimating the phase evolution.
    
    Args
    ----
    g_bases: torch.tensor
        Possible temporal phase bases functions with shape (P, *trj_size)
    phis: torch.tensor
        spherical harmonic bases with shape (B, *im_size)
    A_best: callable
        Best estimate of the forwad model 
        input: 
            image with shape (*im_size)
        output:
            tensor with shape (C, *trj_size)
    recon: callable
        Reconstruction operator
        inputs:
            linear operator
            k-space data with shape (C, *trj_size)
        output:
            image with shape (*im_size)
    n_iter: int
        Unused; kept for call compatibility with ``taylor_sph_pipeline``.
    max_phase_wrap: float
        Cap on per-basis wraps ``|alpha_b| * ||phi_b||_∞`` and on net
        phase ``|phi^T alpha|``. ``None`` drops the constraint.
    
    Returns
    -------
    F_phi: torch.tensor
        estimated phase coefficients with shape (B, P)
    alphas: torch.tensor
        estimated temporal evolution coefficients with shape (B, *trj_size)
    """
    # Consts
    P = g_bases.shape[0]
    B = phis.shape[0]
    
    # Recon each P + 1 times
    x0 = recon(A_best, ksp)
    demods = g_bases * 2j * torch.pi
    recon_func = lambda k : recon(A_best, k)
    xs = _demod_recons(demods, ksp, recon_func,
                       batch_size=P//10)
        
    # # Model for \hat{x}(f)
    # def xhat(F_phi):
    #     xbs = einsum(xs, F_phi + 0j, 'P ..., B P -> B ...')
    #     return x0 + einsum(xbs, phis + 0j, 'B ..., B ... -> ...')
    
    # Apply forward model to all phi weighted xs 
    xs_times_phi = einsum(xs, phis + 0j, 'P ..., B ... -> B P ...')
    xs_times_phi = rearrange(xs_times_phi, 'B P ... -> (B P) ...')  # (B P) ... -> (B P) ... 
    
    # kspace per xs
    ksp0 = A_best(x0)
    rnd_samples = torch.randperm(ksp.numel(), device=ksp.device)[:350_000]
    ksps = torch.zeros((P * B, len(rnd_samples)), dtype=ksp.dtype, device=ksp.device)
    pb_batch = 10
    for pb1 in tqdm(range(0, P*B, pb_batch), desc='Applying Forward Model'):
        pb2 = min(pb1 + pb_batch, P*B)
        ksps[pb1:pb2] = A_best(xs_times_phi[pb1:pb2]).reshape((pb2-pb1, -1))[:, rnd_samples]
    ksps = ksps.T
    ksp_targ = (ksp - ksp0).flatten()[rnd_samples]
    
    # ||ksps @ vec(F) - ksp_targ||^2 with |phi^T F g| <= max_phase_wrap.
    A = torch.cat([ksps.real, ksps.imag], dim=0)
    b = torch.cat([ksp_targ.real, ksp_targ.imag], dim=0)
    F_phi = lstsq_max_phase_wrap(A, b, phis, g_bases, max_phase_wrap)
    
    alphas = einsum(F_phi, g_bases, 'B P, P ... -> B ...')
    return F_phi, alphas

def taylor_sph_pipeline(g_bases: torch.tensor,
                        phis: torch.tensor,
                        ksp: torch.tensor,
                        A_best: linop,
                        recon: callable,
                        discrim: callable,
                        n_iter: int = 5_000,) -> torch.tensor:
    """
    Taylor pipeline for estimating the phase evolution.
    
    Args
    ----
    g_bases: torch.tensor
        Possible temporal phase bases functions with shape (P, *trj_size)
    phis: torch.tensor
        spherical harmonic bases with shape (B, *im_size)
    A_best: callable
        Best estimate of the forwad model 
        input: 
            image with shape (*im_size)
        output:
            tensor with shape (C, *trj_size)
    recon: callable
        Reconstruction operator
        inputs:
            linear operator
            k-space data with shape (C, *trj_size)
        output:
            image with shape (*im_size)
    discrim: callable
        Discriminator function
        inputs:
            image with shape (*im_size)
        output:
            scalar tensor
    
    Returns
    -------
    F_phi: torch.tensor
        estimated phase coefficients with shape (B, P)
    alphas: torch.tensor
        estimated temporal evolution coefficients with shape (B, *trj_size)
    """
    # Consts
    P = g_bases.shape[0]
    B = phis.shape[0]
    
    # Recon each P + 1 times
    x0 = recon(A_best, ksp)
    demods = g_bases * 2j * torch.pi
    recon_func = lambda k : recon(A_best, k)
    xs = _demod_recons(demods, ksp, recon_func,
                       batch_size=P//10)
        
    # Model for \hat{x}(f)
    def xhat(F_phi):
        xbs = einsum(xs, F_phi + 0j, 'P ..., B P -> B ...')
        return x0 + einsum(xbs, phis + 0j, 'B ..., B ... -> ...')
    
    # Optimize metric over limited small F
    F_phi_init = torch.zeros((B, P), dtype=phis.dtype, device=phis.device)
    # F_phi = taylor_sph_optim(F_phi_init, phis, g_bases, 
    #                          loss_fn=lambda x : -discrim(xhat(x)),
    #                          alpha_reg=3e-1,
    #                          n_iter=n_iter,
    #                          lr=1e-2)
    F_phi = taylor_sph_lbfgs(F_phi_init, phis, g_bases, 
                             loss_fn=lambda x : -discrim(xhat(x)),
                             max_phase_wrap=0.5,
                             penalty_weight=0.0,
                             alpha_reg=0.0,
                             n_iter=100,
                             lr=1)
    
    # Estimate temporal evolution coefficients
    alphas = einsum(F_phi, g_bases, 'B P, P ... -> B ...')
    
    return F_phi, alphas
    
    
    