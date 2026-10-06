"""Run on a GPU: python tests/compare_taylor_hybrid.py --weights 1 10 30.

Sharpness is negative gradient entropy; DC is normalized squared error with
density compensation. Lambda=1 is the literal sum; 10 and 30 explore scale.
Hybrid runs and comparison images use the existing 10-step reconstruction.
The separate DC control retains its original 100-step calibration image solves.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from gaim.bo_experiment import load_bo_experiment
from gaim.metrics import gradient_entropy_metric
from gaim.taylor_hybrid import taylor_hybrid
from gaim.taylor_dc import taylor_dc


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--weights', nargs='+', type=float, default=[1, 10, 30])
    parser.add_argument('--outer-steps', type=int, default=60)
    parser.add_argument('--data-path', default='/local_mount/space/mayday/data/users/abrahamd/hofft/data/magnus_spi')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--reference', type=Path, default=None,
                        help='Reuse matching saved baselines; otherwise compute fresh baselines')
    args = parser.parse_args()
    args.output = args.output or Path(f'experiment/taylor_hybrid_{Path(args.data_path).name}')
    args.output.mkdir(parents=True, exist_ok=True)
    # Read before measure() writes metrics, including when reusing this folder.
    old_metrics = (json.loads((args.reference / 'metrics.json').read_text())
                   if args.reference is not None else None)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    e = load_bo_experiment(data_path=args.data_path)
    y, phi, g = e['ksp'], e['spatial_bases'], e['bases']

    def build(alphas):
        # Match the deterministic operator sampling used in the DC experiment.
        with torch.random.fork_rng(devices=[y.device.index or 0]):
            torch.manual_seed(0)
            return e['build_linop'](alphas)

    gt = e['recon'](build(e['alphas_gt']), y)
    torch.save(dict(image=gt.cpu(), alphas=e['alphas_gt'].cpu()), args.output / 'reference.pt')
    rows, images, alpha_sets = {}, {}, {}

    @torch.no_grad()
    def measure(name, alphas, image=None):
        A = build(alphas)
        if image is None:
            image = e['recon'](A, y)
        residual = A.forward(image) - y
        rows[name] = dict(
            sharpness=gradient_entropy_metric(image).item(),
            weighted_error=((A.dcf * residual.abs().square()).sum() / (A.dcf * y.abs().square()).sum()).item(),
            unweighted_error=(residual.abs().square().sum() / y.abs().square().sum()).item(),
            magnitude_nrmse=((image.abs() - gt.abs()).norm() / gt.abs().norm()).item(),
        )
        images[name], alpha_sets[name] = image.cpu(), alphas.cpu()
        print(name, rows[name], flush=True)
        (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')

    measure('Ground truth', e['alphas_gt'], gt)
    measure('Uncorrected', torch.zeros_like(e['alphas_gt']))
    # Reconstruct saved alphas under the current setup before using them as
    # controls, and fail if this no longer reproduces the earlier experiment.
    if args.reference is not None:
        for name, stem in [('Sharpness', 'sharpness'), ('Weighted DC', 'weighted_dc')]:
            reference = torch.load(args.reference / f'{stem}.pt', map_location=y.device, weights_only=False)
            measure(name, reference['alphas'])
            old_name = name if name == 'Sharpness' else 'Weighted DC standard recon'
            old_name = old_name if old_name in old_metrics else name
            for key in ['sharpness', 'weighted_error', 'magnitude_nrmse']:
                if abs(rows[name][key] - old_metrics[old_name][key]) > 1e-4:
                    raise ValueError(f'{name} baseline changed; omit --reference to compute fresh baselines')
    else:
        # Lambda=0 reproduces the original sharpness solver. Recompute both
        # controls when changing datasets, rather than transferring old alphas.
        sharp = taylor_hybrid(phi, g, y, build, e['recon'], dc_weight=0,
                               outer_steps=args.outer_steps)
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sharp.items()},
                   args.output / 'sharpness.pt')
        measure('Sharpness', sharp['alphas'], sharp['image'])

        class WeightedOperator:
            def __init__(self, alphas):
                self.A = build(alphas)
                self.sqrt_weight = self.A.dcf.sqrt()
                self.A.dcf = torch.ones_like(self.A.dcf)

            def forward(self, image):
                return self.sqrt_weight * self.A.forward(image)

            def adjoint(self, kspace):
                return self.A.adjoint(self.sqrt_weight * kspace)

        weight = build(torch.zeros_like(e['alphas_gt'])).dcf
        dc = taylor_dc(phi, g, weight.sqrt() * y, WeightedOperator,
                       outer_steps=args.outer_steps, cg_iterations=100)
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in dc.items()},
                   args.output / 'weighted_dc.pt')
        measure('Weighted DC', dc['alphas'])

    (args.output / 'config.json').write_text(json.dumps(dict(
        dataset=str(Path(args.data_path).resolve()), weights=args.weights,
        outer_steps=args.outer_steps, inner_steps=25,
        radius=0.1, max_radius=1.0, reconstruction='10-step DCF CG, cold start',
        objective='negative gradient entropy - lambda * normalized DCF squared residual',
        baseline_source=str(args.reference) if args.reference else 'fresh',
        dc_control_cg_iterations=100,
        seed=0, B=len(phi), P=len(g), kspace_shape=list(y.shape),
    ), indent=2) + '\n')

    for weight in args.weights:
        name, stem = f'Hybrid {weight:g}', f'hybrid_{weight:g}'

        def checkpoint(alphas, image, history):
            torch.save(dict(alphas=alphas.cpu(), image=image.cpu(), history=history),
                       args.output / f'{stem}.pt')

        result = taylor_hybrid(phi, g, y, build, e['recon'], dc_weight=weight,
                               outer_steps=args.outer_steps, callback=checkpoint)
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()},
                   args.output / f'{stem}.pt')
        measure(name, result['alphas'], result['image'])
        rows[name]['lambda'] = weight
        rows[name]['accepted_updates'] = sum(h['accepted'] for h in result['history'])
        rows[name]['final_objective'] = result['final_score']
        (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')

    names = list(images)
    fig, axes = plt.subplots(2, (len(names) + 1) // 2, figsize=(18, 9), squeeze=False)
    for ax, name in zip(axes.flat, names):
        ax.imshow(images[name].abs(), cmap='gray', vmin=0, vmax=gt.abs().max().item())
        ax.set_title(f"{name}\nNRMSE={rows[name]['magnitude_nrmse']:.3f}, DC={rows[name]['weighted_error']:.4f}")
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=160)

    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    for b, ax in enumerate(axes.flat):
        for name in ['Ground truth', 'Sharpness'] + [f'Hybrid {w:g}' for w in args.weights]:
            ax.plot(alpha_sets[name][b, :, 0], label=name, linewidth=1)
        ax.set_title(f'Alpha {b}')
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / 'alphas.png', dpi=160)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in names:
        row = rows[name]
        ax.scatter(row['weighted_error'], row['sharpness'])
        ax.annotate(name, (row['weighted_error'], row['sharpness']), xytext=(4, 4), textcoords='offset points')
    ax.set(xlabel='Normalized weighted k-space error (lower is better)',
           ylabel='Negative gradient entropy (higher is better)')
    fig.tight_layout()
    fig.savefig(args.output / 'tradeoff.png', dpi=160)


if __name__ == '__main__':
    main()
