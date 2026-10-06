"""Shared MRI setup for the single- and multi-patch BO experiments."""
import torch
from einops import einsum
from hofft.pipelines import svd_decomp_linop, hofft_decomp_linop
from hofft.decomp import hofft_params
from hofft.phase_coeffs import remove_linear_terms, rescale_phis_alphas
from mr_recon.linops import batching_params, sense_linop
from mr_recon.recons import CG_SENSE_recon
from gaim.metrics import gradient_entropy_metric, average_edge_strength_metric
from gaim.phase_expansions import girf_bases, compress_bases, polar_poly_fourier_bases, poly_time_bases


@torch.no_grad()
def load_bo_experiment(data_path="./data/tilt_spi_small", device="cuda"):
    # Load data
    torch_dev = torch.device(device)
    fpath = data_path
    trj = torch.load(f'{fpath}/trj.pt', map_location=torch_dev).type(torch.float32)
    ksp = torch.load(f'{fpath}/ksp.pt', map_location=torch_dev).type(torch.complex64)
    mps = torch.load(f'{fpath}/mps.pt', map_location=torch_dev).type(torch.complex64)
    evals = torch.load(f'{fpath}/evals.pt', map_location=torch_dev).type(torch.float32)
    dcf = torch.load(f'{fpath}/dcf.pt', map_location=torch_dev).type(torch.float32)
    phis = torch.load(f'{fpath}/phis.pt', map_location=torch_dev).type(torch.float32)
    alphas = torch.load(f'{fpath}/alphas.pt', map_location=torch_dev).type(torch.float32)
    im_size = mps.shape[1:]
    trj_size = trj.shape[:-1]
    B = phis.shape[0]
    C = mps.shape[0]
    dt = 2e-6
    fov = 0.22
    gamma_bar = 42.57e6
    hparams = hofft_params(kern_size=(3,)*2, os=1.25, L=15, 
                           cur_rank=500, max_als_iter=10, verbose=False)
    bparams = batching_params(coil_batch_size=C, field_batch_size=hparams.L)
    sense_kwargs = dict(max_eigen=1.0, max_iter=10, verbose=False)

    # Reduce eddy terms
    nsh = 16
    eddy_start = alphas.shape[0] - 16
    eddy_end = eddy_start + nsh
    eddy_idxs = slice(eddy_start, eddy_end)
    alphas = alphas[:eddy_end]
    phis = phis[:eddy_end]

    # Undersample
    R = 1
    alphas = alphas[:, :, ::R]
    trj = trj[:, ::R]
    dcf = dcf[:, ::R]
    ksp = ksp[:, :, ::R]
    trj_size = trj.shape[:-1]

    # Mask
    mask = (evals > 0.85).float()
    phis *= mask
    mps *= mask

    # Normalize
    phis, phis_mp, alphas, alphas_mp = rescale_phis_alphas(phis, alphas)
    for b in range(alphas.shape[0]):
        alphas[b] += alphas_mp[b]
        phis[b] += phis_mp[b]

    # Build linop for some guess of the eddy current terms
    def build_linop(alphas_eddy):
        if alphas_eddy.shape != alphas[eddy_idxs].shape:
            raise ValueError(f'Expected eddy alphas with shape {tuple(alphas[eddy_idxs].shape)}, '
                             f'got {tuple(alphas_eddy.shape)}')
        if not torch.isfinite(alphas_eddy).all():
            raise ValueError('Eddy alphas must be finite')
        alphas_tot = alphas.clone()
        alphas_tot[eddy_idxs] = alphas_eddy
        phis_tot, trj_dev, zeroth_order = remove_linear_terms(phis, alphas_tot, mask=mask)
        phis_tot *= mask
        trj_tot = trj + trj_dev
        # A_tot = sense_linop(trj_tot, mps, dcf, 
        #                     # nufft=nft, 
        #                     spatial_funcs=torch.ones_like(mps[:1]),
        #                     temporal_funcs=torch.exp(-2j * torch.pi * zeroth_order[None,]),
        #                     use_toeplitz=True,
        #                     bparams=bparams)
        # A_tot = svd_decomp_linop(phis_tot, alphas_tot, mps, trj_tot,
        #                         dcf=dcf, hparams=hparams,
        #                         spatial_mask=mask, bparams=bparams)
        # A_tot.temporal_funcs *= torch.exp(-2j * torch.pi * zeroth_order)
        A_tot = hofft_decomp_linop(phis_tot, alphas_tot, mps, trj_tot,
                                   dcf=dcf, hparams=hparams,
                                   spatial_mask=mask, bparams=bparams)
        A_tot.kern_weights *= torch.exp(-2j * torch.pi * zeroth_order)
        return A_tot

    # Build temporal phase bases
    # bases = polar_poly_fourier_bases(trj, num_radial=15, num_angular=1, use_grad=True)
    grad = trj.diff(dim=0) / dt / fov / gamma_bar
    grad = torch.cat([grad, grad[-1:]], dim=0)
    num_splines = 5
    max_freq_hz = 20e3
    bases, freq_hz, spline_fft = girf_bases(
        grad, dt, num_splines=num_splines, max_freq_hz=max_freq_hz, return_spectrum=True,
    )
    bases = bases.reshape((-1, *dcf.shape))
    bases_time = poly_time_bases(trj, num_polys=4)
    bases = torch.cat([bases, bases_time], dim=0)
    # bases = compress_bases(bases, num_bases=40)
    P = bases.shape[0]
    B = nsh

    # Setup gaim metrics
    discrim = lambda x : average_edge_strength_metric(x.abs(), spatial_ndim=2)
    # discrim = lambda x : gradient_entropy_metric(x.abs(), spatial_ndim=2)
    # discrim = lambda x : atkinson_entropy_metric(x.abs(), spatial_ndim=2)
    # discrim = lambda x : wavelet_sparsity_metric(x.abs(), spatial_ndim=2)
    recon = lambda linop, k : CG_SENSE_recon(linop, k, **sense_kwargs)

    # Recon without eddy currents
    A_noeddy = build_linop(torch.zeros_like(alphas[eddy_idxs]))
    img_noeddy = recon(A_noeddy, ksp)

    # Recon with perfect eddy current coefficients
    A_gt = build_linop(alphas[eddy_idxs])
    img_gt = recon(A_gt, ksp)

    # Recon function for a guessed f coefficent vector
    def phase_f(f):
        # Preserve the existing correction sign; the cache accepts either sign.
        return torch.exp(-2j * torch.pi * einsum(f, bases, 'P, P ... -> ...'))

    def recon_f(f):
        return recon(A_noeddy, ksp * phase_f(f))

    return dict(bases=bases, spatial_bases=phis[eddy_idxs], alphas_gt=alphas[eddy_idxs], mask=mask,
                build_linop=build_linop, recon=recon, ksp=ksp, trj_size=tuple(trj_size),
                img_noeddy=img_noeddy, img_gt=img_gt, phase_f=phase_f,
                recon_f=recon_f, recon_phase=lambda phase: recon(A_noeddy, ksp * phase),
                phase_one=torch.ones_like(dcf, dtype=ksp.dtype),)
