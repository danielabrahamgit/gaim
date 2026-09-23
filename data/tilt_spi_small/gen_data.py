import torch
from hofft.utils import reduce_spatial
from math import ceil

# Load data
dataset = 'tilt_spi'
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
dt = 1e-6
fov = 0.22

# Reduce dataset size for faster debugging
ros = slice(0, 7_000, 2)
alphas = alphas[:, ros]
dcf = dcf[ros]
trj = trj[ros]
ksp = ksp[:, ros]
print(f'Old im_size: {im_size}')
im_size = (2 * (ceil(trj.abs().max().item()) * 2) // 2,) * 2
print(f'New im_size: {im_size}')
mps = reduce_spatial(mps, im_size)
phis = reduce_spatial(phis, im_size)
evals = reduce_spatial(evals, im_size)

# Save data
torch.save(alphas.cpu(), './data/tilt_spi_small/alphas.pt')
torch.save(phis.cpu(), './data/tilt_spi_small/phis.pt')
torch.save(evals.cpu(), './data/tilt_spi_small/evals.pt')
torch.save(dcf.cpu(), './data/tilt_spi_small/dcf.pt')
torch.save(trj.cpu(), './data/tilt_spi_small/trj.pt')
torch.save(ksp.cpu(), './data/tilt_spi_small/ksp.pt')
torch.save(mps.cpu(), './data/tilt_spi_small/mps.pt')