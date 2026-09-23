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
from gaim.metrics import (
    gradient_entropy_metric, 
    atkinson_entropy_metric, 
    peaky_curve_confidence, 
    wavelet_sparsity_metric,
)
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
# discrim = lambda x : atkinson_entropy_metric(x.abs(), spatial_ndim=2)
# discrim = lambda x : wavelet_sparsity_metric(x.abs(), spatial_ndim=2)
recon = lambda linop, k : CG_SENSE_recon(linop, k, **sense_kwargs)

# Recon without eddy currents
A_noeddy = build_linop(torch.zeros_like(alphas[eddy_idxs]))
img_noeddy = recon(A_noeddy, ksp)

# Recon with
A_gt = build_linop(alphas[eddy_idxs])
img_gt = recon(A_gt, ksp)

# Excract patches
patch_center = (75, 145)
W = 100
def grab_patch(img):
    patch = img[..., patch_center[0]-W//2:patch_center[0]+W//2, 
                     patch_center[1]-W//2:patch_center[1]+W//2]
    return patch
    # patch_rs = fourier_resize(patch, (500, 500))
    patch_rs = expand_spatial(patch, (500, 500))
    return (patch_rs / patch_rs.abs().max()) * patch.abs().max()
patch_noeddy = grab_patch(img_noeddy)
patch_gt = grab_patch(img_gt)

plt.figure(figsize=(10, 5))
vmin = 0
vmax = img_noeddy.abs().median() + 3 * img_noeddy.abs().std()
disc_noeddy = discrim(patch_noeddy)
disc_gt = discrim(patch_gt)
plt.subplot(1, 2, 1)
plt.title(f'{disc_noeddy / disc_gt:.2f}')
plt.imshow(patch_noeddy.abs().cpu(), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title(f'{disc_gt / disc_gt:.2f}')
plt.imshow(patch_gt.abs().cpu(), cmap='gray', vmin=0, vmax=vmax)
plt.axis('off')
plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.show()