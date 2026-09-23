import torch
import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from hofft.utils import gen_grd, maxmin_indices, reduce_spatial, reduce_temporal, expand_spatial, normalize
from hofft.pipelines import svd_decomp_linop, rescale_phis_alphas
from hofft.decomp import hofft_params
from hofft.phase_coeffs import (
    remove_linear_terms,
    whiten_phis_alphas,     
    rescale_phis_alphas, 
    remove_empty_bases
)
from mr_recon.linops import sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import cufi_nufft
from mr_recon.utils import cvplot
from mr_recon.imperfections.sh import SH_BASES_FUNCTIONS
from mr_recon.multi_coil.calib import synth_cal
from mr_recon.multi_coil.coil_est import csm_from_espirit
from mr_recon.spatial import fourier_resize, spatial_phase_unwrap

from gaim.patch import strided_patchify
from gaim.metrics import gradient_entropy_metric, peaky_curve_confidence, wavelet_sparsity_metric
from gaim.phase_expansions import girf_bases, polar_poly_fourier_bases, compress_bases

from math import ceil
from einops import einsum

# Load data
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fpath = './data/tilt_spi_small'
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
hparams = hofft_params(kern_size=(3,)*2, os=1.25, L=30, cur_rank=500)
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
R = 2
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
    alphas_tot = alphas.clone()
    alphas_tot[eddy_idxs] = alphas_eddy
    phis_tot, trj_dev, zeroth_order = remove_linear_terms(phis, alphas_tot, mask=mask)
    phis_tot *= mask
    trj_tot = trj + trj_dev
    A_tot = svd_decomp_linop(phis_tot, alphas_tot, mps, trj_tot,
                            dcf=dcf, hparams=hparams, 
                            spatial_mask=mask, bparams=bparams)
    A_tot.temporal_funcs *= torch.exp(-2j * torch.pi * zeroth_order)
    return A_tot
    
# -------------------- Auto Calibrate Eddy Terms --------------------
# Build temporal phase bases
# bases = polar_poly_fourier_bases(trj, num_radial=15, num_angular=1, use_grad=True)
grad = trj.diff(dim=0) / dt / fov / gamma_bar
grad = torch.cat([grad, grad[-1:]], dim=0)
num_splines = 10
max_freq_hz = 9e3
bases, freq_hz, spline_fft = girf_bases(
    grad, dt, num_splines=num_splines, max_freq_hz=max_freq_hz, return_spectrum=True,
)
bases = bases.reshape((-1, *dcf.shape))
bases = compress_bases(bases, num_bases=20)
P = bases.shape[0]
B = nsh

# Setup gaim metrics 
discrim = lambda x : gradient_entropy_metric(x.abs(), spatial_ndim=2)
# discrim = lambda x : wavelet_sparsity_metric(x.abs(), spatial_ndim=2)
recon = lambda linop, k : CG_SENSE_recon(linop, k, **sense_kwargs)

# Iterative estimation of eddy currents
from gaim.pipelines import taylor_sph_pipeline, taylor_patch_pipeline, taylor_sph_dc_pipeline
alphas_auto = torch.zeros_like(alphas[eddy_idxs])
phis_eddy = phis[eddy_idxs].clone()
for k in range(3):
    print(f'Iteration {k+1}')
    
    # Build best linop 
    A_best = build_linop(alphas_auto)

    # Try to autocalibrate 
    alphas_dev, conf_eddy = taylor_patch_pipeline(bases, phis_eddy, ksp, A_best, 
                                                   n_iter=15_000,
                                                   recon=recon, 
                                                   discrim=discrim)
    # F_phi, alphas_dev = taylor_sph_pipeline(bases, phis_eddy, ksp, A_best, 
    #                                          n_iter=5_000,
    #                                          recon=recon, 
    #                                          discrim=discrim)
    # F_phi, alphas_dev = taylor_sph_dc_pipeline(bases, phis_eddy, ksp, A_best, 
    #                                             max_phase_wrap=0.2,
    #                                             recon=recon) 
    alphas_auto += alphas_dev
    torch.save(alphas_auto.cpu(), f'./tests/alphas_iter/{k}.pt')
    img_now = recon(build_linop(alphas_auto), ksp)
    recon_entropy = -discrim(img_now)
    print(f'recon entropy {recon_entropy.item():.6g}')
# alphas_auto = torch.load(f'./tests/alphas_iter/3.pt')

# -------------------- Reconstruct Images --------------------
# Recon image naive 
nft = cufi_nufft(im_size, oversamp=1.25, width=3)
A = sense_linop(trj, mps, dcf, nufft=nft,
                bparams=batching_params(coil_batch_size=C))
img_recon = CG_SENSE_recon(A, ksp, **sense_kwargs)

# Recon with all but eddy
A_noeddy = build_linop(torch.zeros_like(alphas[eddy_idxs]))
img_noeddy = CG_SENSE_recon(A_noeddy, ksp, **sense_kwargs)

# Recon ground truth
A_gt = build_linop(alphas[eddy_idxs])
img_gt = CG_SENSE_recon(A_gt, ksp, **sense_kwargs)

# Fit with auto calibrated eddy currents 
A_auto = build_linop(alphas_auto)
img_auto = CG_SENSE_recon(A_auto, ksp, **sense_kwargs)

# -------------------- Show Images --------------------
imgs = [img_recon, img_noeddy, img_gt, img_auto]
names = ['recon', 'no eddy', 'gt eddy', 'auto eddy']
for i in range(len(imgs)):
    imgs[i] = normalize(imgs[i].cpu(), imgs[0].cpu())
vmax = imgs[0].abs().median() + 3 * imgs[0].abs().std()
plt.figure(figsize=(14, 7))
for i, img in enumerate(imgs):
    plt.subplot(2, len(imgs), i+1)
    plt.title(names[i])
    plt.imshow(img.abs().cpu().rot90(), cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.subplot(2, len(imgs), i+1+len(imgs))
    plt.imshow(img.angle().cpu().rot90(), cmap='jet', 
               vmin=-torch.pi, vmax=torch.pi)
    plt.axis('off')
plt.tight_layout()

plt.figure(figsize=(14, 7))
k = ceil(nsh ** 0.5)
alphas_eddy_gt = alphas[eddy_idxs]
for b in range(0, nsh):
    plt.subplot(k, k, b + 1)
    plt.plot(alphas_eddy_gt[b, :, 0].cpu(), linewidth=2, alpha=0.1, color='blue')
    plt.plot(alphas_auto[b, :, 0].cpu(), color='red', linewidth=0.5)
plt.tight_layout()
plt.show()
