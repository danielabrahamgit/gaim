"""GPU experiment: python tests/compare_taylor_dc.py --outer-steps 60.

Compare a fresh sharpness run with weighted/unweighted forward-Taylor fits.
Existing sharpness outputs are preserved; this writes a separate directory.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from gaim.bo_experiment import load_bo_experiment
from gaim.metrics import gradient_entropy_metric
from gaim.optim import taylor_trust
from gaim.taylor_dc import taylor_dc


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--outer-steps', type=int, default=60)
    parser.add_argument('--cg-iterations', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('experiment/taylor_dc_fixed_seed'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    e = load_bo_experiment(data_path='./data/tilt_spi_small')
    y, phi, g = e['ksp'], e['spatial_bases'], e['bases']
    (args.output / 'config.json').write_text(json.dumps(dict(
        dataset='tilt_spi_small', seed=0, kspace_shape=list(y.shape),
        B=len(phi), P=len(g), outer_steps=args.outer_steps,
        cg_iterations=args.cg_iterations, cg_tolerance=1e-5,
        initial_radius_cycles=0.1, max_radius_cycles=1.0,
        sharpness_cg_iterations=10, reference='ground-truth-alpha 10-step DCF reconstruction',
    ), indent=2) + '\n')
    raw_build = e['build_linop']

    def reproducible_build(alphas):
        # HOFFT fits an approximate operator using random samples. Keep those
        # samples fixed so trust ratios do not include fresh sampling noise.
        with torch.random.fork_rng(devices=[y.device.index or 0]):
            torch.manual_seed(0)
            return raw_build(alphas)

    e['build_linop'] = reproducible_build
    zero = torch.zeros_like(e['alphas_gt'])
    base = e['build_linop'](zero)
    dcf = base.dcf.clone()
    e['img_noeddy'] = e['recon'](base, y)
    e['img_gt'] = e['recon'](e['build_linop'](e['alphas_gt']), y)
    del base
    print(f'B={len(phi)} P={len(g)} kspace={tuple(y.shape)}', flush=True)

    # The library adjoint includes DCF. Disable that implicit weighting and
    # explicitly wrap sqrt(D) around A, ensuring a true adjoint in each case.
    class WeightedOperator:
        def __init__(self, alphas, weight):
            self.A = e['build_linop'](alphas)
            self.A.dcf = torch.ones_like(self.A.dcf)
            self.sqrt_weight = weight.sqrt()

        def forward(self, x):
            return self.sqrt_weight * self.A.forward(x)

        def adjoint(self, k):
            return self.A.adjoint(self.sqrt_weight * k)

    rows, images, alpha_sets = {}, {}, {}

    @torch.no_grad()
    def measure(name, alphas, image):
        A = e['build_linop'](alphas)
        residual = A.forward(image) - y
        rows[name] = dict(
            unweighted_error=(residual.abs().square().sum() / y.abs().square().sum()).item(),
            weighted_error=((dcf * residual.abs().square()).sum() / (dcf * y.abs().square()).sum()).item(),
            sharpness=gradient_entropy_metric(image).item(),
            magnitude_nrmse=((image.abs() - e['img_gt'].abs()).norm() / e['img_gt'].abs().norm()).item(),
        )
        images[name], alpha_sets[name] = image.detach().cpu(), alphas.detach().cpu()
        print(name, rows[name], flush=True)

    measure('Uncorrected', zero, e['img_noeddy'])
    measure('Ground truth', e['alphas_gt'], e['img_gt'])

    for name, weight in [('Unweighted DC', torch.ones_like(dcf)), ('Weighted DC', dcf)]:
        start = perf_counter()
        stem = name.lower().replace(' ', '_')
        build = lambda a: WeightedOperator(a, weight)

        def checkpoint(iteration, alphas, image, history):
            torch.save(dict(alphas=alphas.cpu(), image=image.cpu(), history=history),
                       args.output / f'{stem}.pt')

        result = taylor_dc(phi, g, weight.sqrt() * y, build,
                           outer_steps=args.outer_steps, cg_iterations=args.cg_iterations,
                           callback=checkpoint)
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()},
                   args.output / f'{stem}.pt')
        measure(name + ' initial LS', zero, result['initial_image'])
        measure(name, result['alphas'], result['image'])
        rows[name]['seconds'] = perf_counter() - start
        rows[name]['normal_residual'] = result['normal_residual']
        # Same default 10-step DCF reconstruction isolates the learned alphas
        # from differences caused by the image solver / weighting itself.
        standard = e['recon'](e['build_linop'](result['alphas']), y)
        measure(name + ' standard recon', result['alphas'], standard)
        (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')

    # Re-run sharpness on these exact samples: the user's saved output may
    # come from a different dataset. Keep all new outputs in this directory.
    torch.manual_seed(0)
    start = perf_counter()
    sharp = taylor_trust(phi, g, y, e['build_linop'], e['recon'],
                         outer_steps=args.outer_steps, inner_steps=25,
                         radius=0.1, max_radius=1.0, max_retries=4)
    torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sharp.items()},
               args.output / 'sharpness.pt')
    measure('Sharpness', sharp['alphas'], sharp['image'])
    rows['Sharpness']['seconds'] = perf_counter() - start
    (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')

    names = ['Uncorrected', 'Ground truth', 'Sharpness',
             'Unweighted DC', 'Weighted DC', 'Unweighted DC standard recon', 'Weighted DC standard recon']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, name in zip(axes.flat, names):
        ax.imshow(images[name].abs(), cmap='gray', vmin=0, vmax=e['img_gt'].abs().max().item())
        ax.set_title(f"{name}\nNRMSE={rows[name]['magnitude_nrmse']:.3f}")
        ax.axis('off')
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=160)
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    for b, ax in enumerate(axes.flat):
        for name in ['Ground truth', 'Sharpness', 'Unweighted DC', 'Weighted DC']:
            ax.plot(alpha_sets[name][b, :, 0], label=name, linewidth=1)
        ax.set_title(f'Alpha {b}')
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / 'alphas.png', dpi=160)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for stem, label in [('unweighted_dc', 'Unweighted'), ('weighted_dc', 'Weighted')]:
        history = torch.load(args.output / f'{stem}.pt', map_location='cpu', weights_only=False)['history']
        accepted = [h for h in history if h['accepted']]
        axes[0].plot([h['outer'] for h in accepted],
                     [h['reconstructed_error'] for h in accepted], label=label)
        axes[1].plot([h['outer'] for h in accepted], [h['ratio'] for h in accepted], label=label)
    axes[0].set(xlabel='Accepted outer iteration', ylabel='Normalized squared k-space error', yscale='log')
    axes[1].set(xlabel='Accepted outer iteration', ylabel='Actual / predicted fixed-image gain')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'history.png', dpi=160)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
