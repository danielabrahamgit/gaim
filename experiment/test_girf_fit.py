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

from gaim.phase_expansions import girf_bases, polar_poly_fourier_bases, compress_bases

from tqdm import tqdm
import numpy as np
from pathlib import Path

dataset = 'tilt_spi'

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
max_iter = 10
dt = 1e-6
fov = 0.22
gamma_bar = 42.57e6

# Mask
mask = (evals > 0.85).float()
phis *= mask
mps *= mask

# Normalize
# alphas[-12:] = 0
phis, phis_mp, alphas, alphas_mp = rescale_phis_alphas(phis, alphas)
for b in range(alphas.shape[0]):
    alphas[b] += alphas_mp[b]
    phis[b] += phis_mp[b]

# Fit to girf
alphas_sph = alphas[-16:]
grad = trj.diff(dim=0) / dt / fov / gamma_bar
grad = torch.cat([grad, grad[-1:]], dim=0)
alphas_start_zero = alphas_sph - alphas_sph[:, :1]
# bases_flat = polar_poly_fourier_bases(trj, num_radial=15, num_angular=1, use_grad=True)
# bases_flat = bases_flat.reshape((bases_flat.shape[0], -1)).T
num_splines = 10
max_freq_hz = 9e3
bases, freq_hz, spline_fft = girf_bases(
    grad, dt, num_splines=num_splines, max_freq_hz=max_freq_hz, return_spectrum=True,
)
bases_flat = bases.reshape((bases.shape[0] * bases.shape[1], -1)).T
bases_flat = compress_bases(bases_flat.T, num_bases=20).T
print(f'bases_flat.shape: {bases_flat.shape}')
# n_knot = bases.shape[-2]
# n_axes = bases.shape[-1]
# bases_flat = bases.reshape(-1, n_knot * n_axes)
target = alphas_start_zero.reshape((16, -1)).T
coeffs = torch.linalg.lstsq(bases_flat, target).solution
alphas_fit = (bases_flat @ coeffs).T.reshape((16, *trj_size))
# (2Q, D, 16) lstsq weights -> GIRF P_{s,d}(f), shape (16, D, F)
# girf_fft = torch.einsum(
#     'fqd,qds->sdf', spline_fft, coeffs.reshape(n_knot, n_axes, 16).to(spline_fft.dtype),
# )
# print(f'girf_fft.shape: {tuple(girf_fft.shape)}')
# alphas_fit = alphas_start_zero


# Remove linear terms
alphas[-16:] = alphas_fit
phis_new, trj_term, zeroth_order = remove_linear_terms(phis, alphas, mask=mask)
trj += trj_term
ksp *= torch.exp(2j * torch.pi * zeroth_order)
phis = phis_new * mask

# Remove empty bases
phis, alphas = remove_empty_bases(phis, alphas)

R = 2
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

# Recon image naive 
nft = cufi_nufft(im_size, oversamp=1.25, width=3)
A = sense_linop(trj, mps, dcf, nufft=nft,
                bparams=batching_params(coil_batch_size=C))
img_recon = CG_SENSE_recon(A, ksp, 
                           max_eigen=1.0, max_iter=max_iter)

# Recon 
hparams = hofft_params(kern_size=(3,)*2, os=1.25, L=30, cur_rank=500)
A_full = svd_decomp_linop(phis, alphas, mps, trj,
                          dcf=dcf,
                          hparams=hparams,
                          bparams=batching_params(coil_batch_size=C))
img_full = CG_SENSE_recon(A_full, ksp, 
                         max_eigen=1.0, max_iter=max_iter)

# Show images
imgs = [img_recon, img_full]
for i in range(len(imgs)):
    imgs[i] = normalize(imgs[i].cpu(), imgs[0].cpu())
vmax = imgs[0].abs().median() + 3 * imgs[0].abs().std()
plt.figure(figsize=(14, 7))
for i, img in enumerate(imgs):
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(img.abs().cpu().rot90(), cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.subplot(2, len(imgs), i+1+len(imgs))
    plt.imshow(img.angle().cpu().rot90(), cmap='jet', 
               vmin=-torch.pi, vmax=torch.pi)
    plt.axis('off')
plt.tight_layout()

# Show GIRF fit
p = 0
plt.figure(figsize=(14, 7))
for b in range(16):
    plt.subplot(4, 4, b+1)
    plt.plot(alphas_start_zero[b, :, p].cpu(), label='Ground truth', alpha=0.2, linewidth=2, color='C0')
    plt.plot(alphas_fit[b, :, p].cpu(), label=f'b={b} coeff fit', linewidth=0.75, color='red')
    plt.legend()
    # plt.xlim(18_000, 23_000)
plt.tight_layout()

# # GIRF transfer functions: magnitude and phase of each (SH, axis) entry
# band = freq_hz <= max_freq_hz
# freq_khz = freq_hz[band].detach().cpu().numpy() / 1e3
# girf_band = girf_fft[:, :, band].detach().cpu().numpy()
# axis_names = ['Gx', 'Gy', 'Gz'][:n_axes]

# fig_mag, axes_mag = plt.subplots(
#     16, n_axes, figsize=(4 * n_axes, 24), sharex=True, sharey=False,
#     squeeze=False, constrained_layout=True,
# )
# fig_mag.suptitle('GIRF magnitude')
# fig_ph, axes_ph = plt.subplots(
#     16, n_axes, figsize=(4 * n_axes, 24), sharex=True, sharey=False,
#     squeeze=False, constrained_layout=True,
# )
# fig_ph.suptitle('GIRF phase')
# for s in range(16):
#     for d in range(n_axes):
#         h = girf_band[s, d]
#         axes_mag[s, d].plot(freq_khz, np.abs(h), color='C0', linewidth=1)
#         axes_ph[s, d].plot(freq_khz, np.unwrap(np.angle(h)), color='C1', linewidth=1)
#         if s == 0:
#             axes_mag[s, d].set_title(axis_names[d])
#             axes_ph[s, d].set_title(axis_names[d])
#         if d == 0:
#             axes_mag[s, d].set_ylabel(f'SH {s}')
#             axes_ph[s, d].set_ylabel(f'SH {s}')
#         axes_mag[s, d].grid(True, alpha=0.3)
#         axes_ph[s, d].grid(True, alpha=0.3)
# for d in range(n_axes):
#     axes_mag[-1, d].set_xlabel('Frequency (kHz)')
#     axes_ph[-1, d].set_xlabel('Frequency (kHz)')


plt.show()
