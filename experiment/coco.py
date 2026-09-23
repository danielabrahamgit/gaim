import torch

import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from hofft.utils import gen_grd, maxmin_indices, reduce_spatial, reduce_temporal, expand_spatial
from hofft.pipelines import svd_decomp_linop, rescale_phis_alphas
from hofft.decomp import hofft_params
from hofft.phase_coeffs import (
    whiten_phis_alphas, 
    rescale_phis_alphas, 
    remove_empty_bases
)
from mr_recon.linops import sense_linop, batching_params
from mr_recon.recons import CG_SENSE_recon
from mr_recon.fourier import cufi_nufft
from mr_recon.utils import cvplot
from mr_recon.imperfections.sh import SH_BASES_FUNCTIONS

from gaim.patch import strided_patchify, interpolate_patch_grid
from gaim.metrics import gradient_entropy_metric, peaky_curve_confidence, wavelet_sparsity_metric, laplacian_energy_metric

from tqdm import tqdm
from einops import einsum
import numpy as np




def eval_sh_bases(xyz: torch.Tensor, order: int) -> torch.Tensor:
    """
    Evaluate real solid harmonics up to `order` at coordinates xyz.

    Args
    ----
    xyz : torch.Tensor
        Cartesian coordinates, shape (3, ...)
    order : int
        Maximum spherical-harmonic order.

    Returns
    -------
    bases : torch.Tensor
        Bases with shape ((order + 1)**2, ...)
    """
    n_terms = (order + 1) ** 2
    xyz_np = xyz.detach().cpu().numpy()
    terms = [np.asarray(SH_BASES_FUNCTIONS[k](xyz_np), dtype=np.float32)
             for k in range(n_terms)]
    return torch.as_tensor(np.stack(terms, axis=0), device=xyz.device, dtype=xyz.dtype)

# dataset = 'coco_spiral'
dataset = 'tilt_spi_invivo'

# Load data
torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fpath = '/local_mount/space/mayday/data/users/abrahamd/hofft'
trj = torch.load(f'{fpath}/data/{dataset}/trj.pt', map_location=torch_dev).type(torch.float32)
ksp = torch.load(f'{fpath}/data/{dataset}/ksp.pt', map_location=torch_dev).type(torch.complex64)
mps = torch.load(f'{fpath}/data/{dataset}/mps.pt', map_location=torch_dev).type(torch.complex64)
evals = torch.load(f'{fpath}/data/{dataset}/evals.pt', map_location=torch_dev).type(torch.float32)
dcf = torch.load(f'{fpath}/data/{dataset}/dcf.pt', map_location=torch_dev).type(torch.float32)
phis = torch.load(f'{fpath}/data/{dataset}/phis.pt', map_location=torch_dev).type(torch.float32)
alphas = torch.load(f'{fpath}/data/{dataset}/alphas.pt', map_location=torch_dev).type(torch.float32)
im_size = mps.shape[1:]
trj_size = trj.shape[:-1]
B = phis.shape[0]
C = mps.shape[0]
max_iter = 20
dt = 2e-6

# Mask
mask = (evals > 0.85).float()
phis *= mask
mps *= mask

# Remove empty bases
phis, alphas = remove_empty_bases(phis, alphas)

R = 3
alphas = alphas[:, :, ::R].squeeze()
dcf = dcf[:, ::R].squeeze()
trj = trj[:, ::R].squeeze()
ksp = ksp[:, :, ::R].squeeze()

# Print shapes
print(f'trj.shape: {trj.shape}')
print(f'mps.shape: {mps.shape}')
print(f'evals.shape: {evals.shape}')
print(f'dcf.shape: {dcf.shape}')
print(f'phis.shape: {phis.shape}')
print(f'alphas.shape: {alphas.shape}')

# Low-res image for phase estimate
ros = slice(None, trj.shape[0]//10)
nft = cufi_nufft(im_size, oversamp=1.25, width=3)
A = sense_linop(trj[ros], mps, dcf[ros], nufft=nft,
                bparams=batching_params(coil_batch_size=C))
img_low_res = CG_SENSE_recon(A, ksp[:, ros], 
                             max_eigen=1.0, max_iter=max_iter)

# Recon image naive 
A = sense_linop(trj, mps, dcf, nufft=nft,
                bparams=batching_params(coil_batch_size=C))
img_recon = CG_SENSE_recon(A, ksp, 
                           max_eigen=1.0, max_iter=max_iter)

# Recon with coco only
hparams = hofft_params(kern_size=(3,)*2, os=1.25, L=10, cur_rank=500)
A_coco = svd_decomp_linop(phis[:-1], alphas[:-1], mps, trj,
                          hparams=hparams,
                          bparams=batching_params(coil_batch_size=C))
img_coco = CG_SENSE_recon(A_coco, ksp, 
                         max_eigen=1.0, max_iter=max_iter)

# GAIM
freqs = torch.linspace(-100, 100, 20, device=torch_dev)
ts = torch.arange(alphas.shape[1], device=torch_dev) * dt

# # SVD decomp
# L = 10
# phase = torch.exp(-2j * torch.pi * freqs[:, None] * ts[None, :])
# U, S, V = torch.svd_lowrank(phase, q=L)
# freq_factors = (U[:, :L] * (S[:L] ** 0.5)).T
# temporal_factors = (V[:, :L].conj() * (S[:L] ** 0.5)).T
# imgs = [CG_SENSE_recon(A_coco, ksp * temporal_factors[l].conj(), 
#                         max_eigen=1.0, max_iter=max_iter) for l in range(L)]
# imgs = torch.stack(imgs, dim=0)
# imgs = einsum(imgs, freq_factors, 'L ..., L N -> N ...')


temporal_factors = torch.exp(-2j * torch.pi * freqs[:, None, None] * ts[None, None, :])
imgs = [CG_SENSE_recon(A_coco, ksp * temporal_factors[l].conj(), 
                        max_eigen=1.0, max_iter=max_iter) for l in range(len(freqs))]
imgs = torch.stack(imgs, dim=0)
# imgs = torch.zeros((20, *im_size), device=torch_dev, dtype=torch.complex64)

patch_size = (20,)*2
patch_stride = (10,)*2
patch_slices = tuple(slice(None, None, step) for step in patch_stride)
patches = strided_patchify(imgs, patch_size, stride=patch_stride)

# ge_scores = gradient_entropy_metric(patches, spatial_ndim=2, eps=1e-6)
ge_scores = wavelet_sparsity_metric(patches, spatial_ndim=2, eps=1e-6)
# ge_scores = laplacian_energy_metric(patches, spatial_ndim=2, eps=1e-6)

conf_patches = peaky_curve_confidence(ge_scores, x=freqs, dim=0)
conf_patches /= conf_patches.abs().max().clamp_min(1e-8)
idxs = torch.argmax(ge_scores, dim=0)
freq_patches = freqs[idxs]
# Keep score interpolation in the original pixel coordinate system. Choosing
# frequencies after interpolation avoids blending incompatible winning labels.
ge_scores_full = interpolate_patch_grid(ge_scores, im_size, stride=patch_stride)
b0_est = freqs[ge_scores_full.argmax(dim=0)] * mask
conf = interpolate_patch_grid(conf_patches, im_size, stride=patch_stride) * mask

# Weighted spherical-harmonic B0 fit, using conf as WLS weights
sh_order = 10
rs = gen_grd(im_size).to(device=torch_dev)
xyz = torch.stack(
    [rs[..., 0], rs[..., 1], torch.zeros_like(rs[..., 0])], dim=0)
bases = eval_sh_bases(xyz, sh_order)

# Fit only the measured patch anchors. Resizing the sparse estimates and then
# treating interpolated pixels as independent observations biases the fit.
sampled_bases = bases[(slice(None),) + patch_slices]
A = sampled_bases.reshape(bases.shape[0], -1).T
y = freq_patches.reshape(-1)
w = (conf_patches * mask[patch_slices]).reshape(-1).clamp_min(0)
w = w / w.mean().clamp_min(1e-8)
sqrt_w = w.sqrt()
col_ok = (A * sqrt_w[:, None]).abs().amax(dim=0) > 1e-8
A = A[:, col_ok]
coef = torch.zeros(bases.shape[0], device=torch_dev, dtype=A.dtype)
if col_ok.any():
    coef[col_ok] = torch.linalg.lstsq(A * sqrt_w[:, None], y * sqrt_w).solution
b0_sh = (bases * coef.reshape((-1,) + (1,) * (bases.ndim - 1))).sum(0)
b0_sh *= mask
print(f'SH order={sh_order}  WLS fit on {y.numel()} patch anchors (stride={patch_stride})')

kwargs = dict(cmap='jet', vmin=-100, vmax=100)
plt.figure(figsize=(12, 4))
plt.subplot(141)
plt.imshow(phis[-1].cpu().rot90(), **kwargs)
plt.axis('off')
plt.subplot(142)
plt.imshow(b0_est.cpu().rot90(), **kwargs)
plt.axis('off')
plt.subplot(143)
plt.imshow(b0_sh.cpu().rot90(), **kwargs)
plt.axis('off')
plt.subplot(144)
plt.imshow(conf.cpu().rot90(), cmap='magma', vmin=0, vmax=1)
plt.axis('off')
plt.tight_layout()

# plt.figure(figsize=(10,10))
# cmap = plt.cm.magma
# step = 1
# ge_show = ge_scores[:, ::step, ::step]
# conf_show = conf[::step, ::step]
# for i in range(ge_show.shape[1]):
#     for j in range(ge_show.shape[2]):
#         plt.subplot(ge_show.shape[1], ge_show.shape[2], i*ge_show.shape[2] + j + 1)
#         plt.plot(ge_show[:, i, j].cpu(), color=cmap(float(conf_show[i, j].cpu())))
#         plt.axis('off')
# plt.subplots_adjust(wspace=0.0, hspace=0.0)

phis_gaim = phis.clone()
alphas_gaim = alphas.clone()
phis_gaim[-1] = b0_sh
hparams.L = 50
A_gaim = svd_decomp_linop(phis_gaim, alphas_gaim, mps, trj,
                          hparams=hparams,
                          bparams=batching_params(coil_batch_size=C))
img_gaim = CG_SENSE_recon(A_gaim, ksp, 
                         max_eigen=1.0, max_iter=max_iter)


# Full recon
A_full = svd_decomp_linop(phis, alphas, mps, trj,
                          hparams=hparams,
                          bparams=batching_params(coil_batch_size=C))
img_full = CG_SENSE_recon(A_full, ksp, 
                         max_eigen=1.0, max_iter=max_iter)

breakpoint()

# Show images
imgs = [img_recon, img_coco, img_gaim, img_full]
plt.figure(figsize=(14, 7))
for i, img in enumerate(imgs):
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(img.abs().cpu().rot90(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, len(imgs), i+1+len(imgs))
    plt.imshow(img.angle().cpu().rot90(), cmap='jet', 
               vmin=-torch.pi, vmax=torch.pi)
    plt.axis('off')
plt.tight_layout()
plt.show()
