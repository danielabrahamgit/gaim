"""Run from the repository root: python tests/minimal_taylor_trust.py"""
from math import ceil
from pathlib import Path

import torch
import matplotlib as mpl
mpl.use('webagg')
import matplotlib.pyplot as plt

from gaim.bo_experiment import load_bo_experiment
from gaim.optim import taylor_trust

# Same data preparation, temporal bases, and random seed as test_taylor_trust.py.
torch.manual_seed(0)
torch.set_num_threads(4)
# e = load_bo_experiment(data_path="/local_mount/space/mayday/data/users/abrahamd/hofft/data/magnus_spi")
e = load_bo_experiment(data_path="./data/tilt_spi_small")
phis, bases, ksp = e['spatial_bases'], e['bases'], e['ksp']
build_linop, recon = e['build_linop'], e['recon']
output = Path('experiment/minimal_taylor_trust')
output.mkdir(parents=True, exist_ok=True)

# Keep the validated solver: L-BFGS inside each phase trust region, followed
# by a real reconstruction check and re-linearization after accepted updates.
result = taylor_trust(phis, bases, ksp, build_linop, recon,
                      outer_steps=5 * 0 + 60, inner_steps=25, radius=0.1,
                      max_radius=0.2*0 + 1.0, max_retries=4)
F, alphas_auto, img_auto = result['F'], result['alphas'], result['image']
# F is (B,P); contracting it with bases gives (B,*trj_size) alphas.
# img_auto comes from recon(build_linop(alphas_auto), ksp).
torch.save(alphas_auto.cpu(), output / 'alphas.pt')
torch.save(img_auto.cpu(), output / 'image.pt')
print(f'Metric: {result["initial_score"]:.6f} -> {result["final_score"]:.6f}', flush=True)

# img_gt = recon(build_linop(e['alphas_gt']), ksp)
# img_auto = recon(build_linop(alphas_auto), ksp)

# Compare reconstructions. Ground truth is used only for these plots.
imgs = [result['initial_image'], e['img_gt'], img_auto]
names = ['Uncorrected', 'Ground truth', 'Taylor trust-region reconstruction']
vmax = e['img_gt'].abs().max().item()
plt.figure(figsize=(14, 7))
for i, (img, name) in enumerate(zip(imgs, names)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img.abs().cpu(), cmap='gray', vmin=0, vmax=vmax)
    plt.title(name)
    plt.axis('off')
plt.tight_layout()
plt.savefig(output / 'comparison.png', dpi=160)

# Both alpha tensors use build_linop's scaling; compare trajectory 0 directly.
B = len(phis)
K = ceil(B ** 0.5)
plt.figure(figsize=(14, 7))
for b in range(B):
    plt.subplot(K, K, b + 1)
    plt.plot(e['alphas_gt'][b, :, 0].cpu(), color='tab:blue',
             linewidth=2, alpha=0.5, label='Ground truth')
    plt.plot(alphas_auto[b, :, 0].cpu(), color='tab:red', linewidth=1, label='Learned')
    plt.title(f'Alpha {b}')
    if b == 0:
        plt.legend()
plt.gcf().supxlabel('Readout sample (trajectory 0)')
plt.gcf().supylabel('Alpha coefficient')
plt.tight_layout()
plt.savefig(output / 'alphas.png', dpi=160)

# Plot metric history
scores = [result['history'][k]['score_before'] for k in range(len(result['history']))]
actual_gains = [result['history'][k]['actual_gain'] for k in range(len(result['history']))]
predicted_gains = [result['history'][k]['predicted_gain'] for k in range(len(result['history']))]
radii = [result['history'][k]['radius'] for k in range(len(result['history']))]
plt.figure(figsize=(14, 7))
plt.subplot(3, 1, 1)
plt.plot(scores, color='tab:blue', linewidth=2, alpha=0.5, label='Score')
plt.title('Score')
plt.subplot(3, 1, 2)
plt.plot(actual_gains, color='tab:red', linewidth=2, alpha=0.5, label='Actual gain')
plt.plot(predicted_gains, color='tab:green', linewidth=2, alpha=0.5, label='Predicted gain')
plt.title('Predicted gain')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(radii, color='tab:orange', linewidth=2, alpha=0.5, label='Radius')
plt.title('Radius')
plt.tight_layout()
plt.savefig(output / 'metric_history.png', dpi=160)
plt.show()
