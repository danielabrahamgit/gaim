"""Taylor self-calibration with a phase trust region (plans/taylor_plan.md).

Run in the MRI environment: python tests/test_taylor_trust.py
F has shape (B, P); the final alphas = F @ bases feed build_linop.
"""
import argparse
from math import ceil
from pathlib import Path

import torch

from gaim.metrics import gradient_entropy_metric
from gaim.optim import _lbfgs_direction, taylor_trust


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer-steps', type=int, default=5)
    parser.add_argument('--inner-steps', type=int, default=25)
    parser.add_argument('--radius', type=float, default=0.05, help='Incremental phase cap in cycles')
    parser.add_argument('--output', type=Path, default=Path('experiment/taylor_trust'))
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    from gaim.bo_experiment import load_bo_experiment
    e = load_bo_experiment()
    result = taylor_trust(e['spatial_bases'], e['bases'], e['ksp'], e['build_linop'], e['recon'],
                          outer_steps=args.outer_steps, inner_steps=args.inner_steps, radius=args.radius)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(result['alphas'].cpu(), args.output / 'alphas.pt')
    torch.save(result['image'].cpu(), args.output / 'image.pt')
    torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()},
               args.output / 'result.pt')
    print(f'Metric: {result["initial_score"]:.6f} -> {result["final_score"]:.6f}; '
          f'alphas.shape={tuple(result["alphas"].shape)}', flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = e['img_gt'].abs().max().item()
    for ax, image, title in zip(axes, [result['initial_image'], e['img_gt'], result['image']],
                                ['Uncorrected', 'Ground truth', 'Taylor trust-region reconstruction']):
        ax.imshow(image.abs().cpu(), cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=160)

    # Both alpha tensors already use build_linop's spatial-basis scaling.
    # Plot the first trajectory, matching the BO example, with no extra rescaling.
    B = result['alphas'].shape[0]
    K = ceil(B ** 0.5)
    fig, axes = plt.subplots(K, K, figsize=(14, 8), squeeze=False)
    for b, ax in enumerate(axes.flat):
        if b >= B:
            ax.axis('off')
            continue
        ax.plot(e['alphas_gt'][b, :, 0].detach().cpu(), color='tab:blue',
                linewidth=2, alpha=0.5, label='Ground truth')
        ax.plot(result['alphas'][b, :, 0].detach().cpu(), color='tab:red',
                linewidth=1, label='Learned')
        ax.set_title(f'Alpha {b}')
    axes.flat[0].legend()
    fig.supxlabel('Readout sample (trajectory 0)')
    fig.supylabel('Alpha coefficient')
    fig.tight_layout()
    fig.savefig(args.output / 'alphas.png', dpi=160)


if __name__ == '__main__':
    main()
