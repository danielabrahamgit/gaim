"""IQA versus phase-correction fraction on tilt_spi.

Reconstructs independently at alphas * (k/K) with the HOFFT/CG path from
tests/test_taylor.py. Pretrained FGResQ/ARNIQA are evaluation only.

Example (existing GPU allocation, repository root):
    srun --jobid=8710 --overlap /home/abrahamd/mambaforge/envs/mr_recon/bin/python \\
        experiment/iqa_sweep.py --device cuda --k 20 \\
        --output experiment/iqa_sweep_results
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from gaim.metrics import (
    arniqa_metric,
    average_edge_strength_metric,
    fgresq_metric,
    fgresq_pair_metric,
    gaussian_scale_gradient_entropy,
    GAUSSIAN_ENTROPY_SIGMAS,
    gradient_entropy_metric,
    load_arniqa,
    load_fgresq,
    normalized_gradient_squared_metric,
    to_iqa_input,
    wavelet_sparsity_metric,
)

GE_SIGMA_NAMES = tuple(
    f'ge_sigma_{sigma:g}'.replace('.', 'p') for sigma in GAUSSIAN_ENTROPY_SIGMAS
)
SCALAR_METRICS = (
    'gradient_entropy', 'wavelet_sparsity', 'fgresq', 'arniqa',
    'ngs', 'aes', *GE_SIGMA_NAMES,
    'dc_loss', 'dc_loss_normalized',
)
LOWER_BETTER = {'dc_loss', 'dc_loss_normalized'}


DEFAULT_DATA = Path('/local_mount/space/mayday/data/users/abrahamd/hofft/data/tilt_spi')
SCORE_TOL = 1e-4
PAIR_TIE_RANK = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=Path('experiment/iqa_sweep_results'))
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--shot-stride', type=int, default=2)
    parser.add_argument('--max-iter', type=int, default=10)
    parser.add_argument('--model-rank', type=int, default=50)
    parser.add_argument('--mask-threshold', type=float, default=0.85)
    parser.add_argument('--reconstructions', type=Path, default=None,
                        help='Reuse a saved reconstructions.pt instead of re-running HOFFT.')
    parser.add_argument('--skip-pretrained', action='store_true',
                        help='Reuse FGResQ/ARNIQA scores from output/scores.csv.')
    parser.add_argument('--dc-max-iter', type=int, default=50)
    parser.add_argument('--dc-tolerance', type=float, default=1e-8)
    return parser.parse_args()


def load_tilt_spi(data_dir, device, shot_stride, mask_threshold):
    """Load and preprocess tilt_spi exactly as tests/test_taylor.py."""
    from hofft.phase_coeffs import rescale_phis_alphas

    def load(name):
        return torch.load(data_dir / name, map_location=device)

    trj = load('trj.pt').float()
    ksp = load('ksp.pt').cfloat()
    mps = load('mps.pt').cfloat()
    evals = load('evals.pt').float()
    dcf = load('dcf.pt').float()
    phis = load('phis.pt').float()
    alphas = load('alphas.pt').float()
    mask = (evals > mask_threshold).float()
    phis = phis * mask
    mps = mps * mask
    phis, phis_mp, alphas, alphas_mp = rescale_phis_alphas(phis, alphas)
    for b in range(alphas.shape[0]):
        alphas[b] = alphas[b] + alphas_mp[b]
        phis[b] = phis[b] + phis_mp[b]
    alphas = alphas[:, :, ::shot_stride]
    dcf = dcf[:, ::shot_stride]
    trj = trj[:, ::shot_stride]
    ksp = ksp[:, :, ::shot_stride]
    return {
        'trj': trj, 'ksp': ksp, 'mps': mps, 'dcf': dcf, 'phis': phis,
        'alphas': alphas, 'mask': mask, 'im_size': mps.shape[1:],
    }


def reconstruct_hofft(data, alphas, max_iter, model_rank):
    """HOFFT + CG-SENSE used for the full correction in tests/test_taylor.py.

    Zero alphas drop every temporal basis inside HOFFT, so the k=0
    (no-correction) endpoint uses the same naive SENSE operator as
    tests/test_taylor.py.
    """
    from hofft.decomp import hofft_params
    from hofft.phase_coeffs import remove_linear_terms
    from hofft.pipelines import svd_decomp_linop
    from mr_recon.linops import batching_params
    from mr_recon.recons import CG_SENSE_recon

    if float(alphas.abs().max()) == 0:
        return reconstruct_naive(data, max_iter)

    phis = data['phis']
    mask = data['mask']
    mps = data['mps']
    phis_k, trj_term, zeroth_order = remove_linear_terms(phis, alphas, mask=mask)
    ksp_k = data['ksp'] * torch.exp(2j * torch.pi * zeroth_order)
    trj_k = data['trj'] + trj_term
    hparams = hofft_params(kern_size=(3,) * 2, os=1.25, L=model_rank, cur_rank=500)
    bparams = batching_params(coil_batch_size=mps.shape[0], field_batch_size=30)
    operator = svd_decomp_linop(
        phis_k, alphas, mps, trj_k, dcf=data['dcf'], hparams=hparams,
        spatial_mask=mask, bparams=bparams,
    )
    image = CG_SENSE_recon(operator, ksp_k, max_eigen=1.0, max_iter=max_iter)
    del operator
    return image


def reconstruct_naive(data, max_iter):
    from mr_recon.fourier import cufi_nufft
    from mr_recon.linops import batching_params, sense_linop
    from mr_recon.recons import CG_SENSE_recon

    mps = data['mps']
    nft = cufi_nufft(data['im_size'], oversamp=1.25, width=3)
    operator = sense_linop(
        data['trj'], mps, data['dcf'], nufft=nft,
        bparams=batching_params(coil_batch_size=mps.shape[0]),
    )
    return CG_SENSE_recon(operator, data['ksp'], max_eigen=1.0, max_iter=max_iter)


def make_unweighted_operator(data, alphas, model_rank):
    """A(alphas) with dcf=1 so the adjoint is the unweighted adjoint of A.

    k=0 uses naive SENSE (no phase). k>0 uses the same HOFFT encoding as
    tests/test_taylor.py, including zeroth-order demodulation of b.
    """
    from hofft.decomp import hofft_params
    from hofft.phase_coeffs import remove_linear_terms
    from hofft.pipelines import svd_decomp_linop
    from mr_recon.fourier import cufi_nufft
    from mr_recon.linops import batching_params, sense_linop

    ones = torch.ones_like(data['dcf'])
    mps = data['mps']
    if float(alphas.abs().max()) == 0:
        nft = cufi_nufft(data['im_size'], oversamp=1.25, width=3)
        operator = sense_linop(
            data['trj'], mps, ones, nufft=nft,
            bparams=batching_params(coil_batch_size=mps.shape[0]),
        )
        return operator, data['ksp']
    phis_k, trj_term, zeroth_order = remove_linear_terms(data['phis'], alphas, mask=data['mask'])
    ksp_k = data['ksp'] * torch.exp(2j * torch.pi * zeroth_order)
    trj_k = data['trj'] + trj_term
    hparams = hofft_params(kern_size=(3,) * 2, os=1.25, L=model_rank, cur_rank=500)
    operator = svd_decomp_linop(
        phis_k, alphas, mps, trj_k, dcf=ones, hparams=hparams,
        spatial_mask=data['mask'],
        bparams=batching_params(coil_batch_size=mps.shape[0], field_batch_size=30),
    )
    return operator, ksp_k


def minimized_data_consistency(operator, ksp, max_iter, tolerance):
    """Solve min_x ||A x - b||_2^2 with no extra regularization or DCF weights."""
    from mr_recon.algs import conjugate_gradient

    b_norm_sq = float(ksp.norm().square())
    ahb = operator.adjoint(ksp)
    residuals, iterates = conjugate_gradient(
        AHA=operator.normal,
        AHb=ahb,
        num_iters=max_iter,
        lamda_l2=0.0,
        tolerance=tolerance,
        return_resids=True,
        weights=None,
        verbose=True,
    )
    image = iterates[-1]
    loss = float((operator.forward(image) - ksp).norm().square())
    n_iter = len(residuals)
    return {
        'loss': loss,
        'loss_normalized': loss / max(b_norm_sq, 1e-30),
        'b_norm_sq': b_norm_sq,
        'n_iter': n_iter,
        'hit_max_iter': n_iter >= max_iter,
        'normal_residual': float(residuals[-1]) if residuals else float('nan'),
        'tolerance': tolerance,
        'max_iter': max_iter,
    }


def magnitude_nrmse(a, b, mask):
    a_mag, b_mag = a.abs() * mask, b.abs() * mask
    scale = (a_mag * b_mag).sum() / a_mag.square().sum().clamp_min(1e-20)
    return float((scale * a_mag - b_mag).norm() / b_mag.norm().clamp_min(1e-20))


def package_versions():
    import importlib.metadata as metadata

    names = [
        'pyiqa', 'torch', 'torchvision', 'transformers', 'timm',
        'huggingface_hub', 'accelerate', 'opencv-python-headless',
        'Pillow', 'numpy', 'scipy',
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import pyiqa
        versions['pyiqa_file'] = getattr(pyiqa, '__file__', None)
    except ImportError:
        versions['pyiqa_file'] = None
    return versions


def scalar_ordering(scores, fractions, tol=SCORE_TOL):
    scores = np.asarray(scores, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    finite = np.isfinite(scores)
    span = np.nanmax(scores) - np.nanmin(scores)
    scale = max(float(np.nanmax(np.abs(scores))), 1e-12)
    step_tol = tol * max(scale, 1.0) if scale >= 1 else tol * scale
    if not finite.all():
        spearman = None
    elif span <= step_tol:
        spearman = None
    else:
        rho = spearmanr(fractions, scores).statistic
        spearman = None if rho != rho else float(rho)
    diffs = np.diff(scores)
    n_adj = max(len(diffs), 1)
    return {
        'spearman_vs_fraction': spearman,
        'adjacent_decreases': int(np.sum(diffs < -step_tol)),
        'adjacent_ties': int(np.sum(np.abs(diffs) <= step_tol)),
        'adjacent_increases': int(np.sum(diffs > step_tol)),
        'best_fraction': float(fractions[int(np.argmax(scores))]),
        'best_score': float(scores[int(np.argmax(scores))]),
        'n_adjacent': int(n_adj),
        'all_finite': bool(finite.all()),
        'step_tolerance': float(step_tol),
    }


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output, fractions, mag, scale, slice_scores, adj_p, anc_p):
    names = [name for name in SCALAR_METRICS if name in slice_scores]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, name in zip(axes, names):
        series = np.asarray(slice_scores[name])
        if series.ndim == 1:
            series = series[None]
        for slice_idx, curve in enumerate(series):
            axis.plot(fractions, curve, marker='o', ms=3, label=f'slice {slice_idx}')
        ylabel = name
        if name in LOWER_BETTER:
            ylabel = f'{name} (lower better)'
        axis.set_xlabel('k/K')
        axis.set_ylabel(ylabel)
        axis.set_title(name)
    for axis in axes[len(names):]:
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'scores.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].plot(fractions[1:], np.mean(adj_p, axis=0), marker='o', ms=3)
    axes[0].axhline(0.5, color='0.6', lw=0.8)
    axes[0].set_xlabel('k/K of candidate')
    axes[0].set_ylabel('p_candidate_better')
    axes[0].set_title('adjacent: image_k vs image_{k-1}')
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].plot(fractions[1:], np.mean(anc_p, axis=0), marker='o', ms=3)
    axes[1].axhline(0.5, color='0.6', lw=0.8)
    axes[1].set_xlabel('k/K of candidate')
    axes[1].set_ylabel('p_candidate_better')
    axes[1].set_title('anchor: image_k vs image_0')
    axes[1].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(output / 'pairwise.png', dpi=140)
    plt.close(fig)

    n = len(mag)
    cols = min(7, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, axis in enumerate(axes):
        axis.axis('off')
        if idx >= n:
            continue
        axis.imshow(mag[idx].cpu(), cmap='gray', vmin=0, vmax=scale)
        axis.set_title(f'{idx}/{n - 1}', fontsize=8)
    fig.suptitle(f'shared display scale = {scale:.4g}')
    fig.tight_layout()
    fig.savefig(output / 'montage.png', dpi=120)
    plt.close(fig)


def write_report(output, args, versions, preprocessing, endpoint, ordering, pair_summary, wiring):
    def fmt_rho(value):
        return 'undefined' if value is None else f'{value:.3f}'

    lines = [
        '# IQA versus phase correction on tilt_spi',
        '',
        'Pretrained FGResQ and ARNIQA are not established clean-MRI distribution',
        'scores. A successful sweep shows sensitivity to this phase-corruption',
        'experiment, not general reliability for noise, aliasing, or anatomy.',
        '',
        '## Command',
        '',
        '```',
        f'srun --jobid=8710 --overlap /home/abrahamd/mambaforge/envs/mr_recon/bin/python \\',
        f'    experiment/iqa_sweep.py --device {args.device} --k {args.k} \\',
        f'    --data-dir {args.data_dir} --output {args.output}',
        '```',
        '',
        '## Versions',
        '',
    ]
    for name, version in versions.items():
        lines.append(f'- `{name}`: {version}')
    lines += [
        '',
        '## Preprocessing',
        '',
        f"- Dataset `{args.data_dir}` with shot stride {args.shot_stride}, "
        f"mask evals > {args.mask_threshold}, HOFFT L={args.model_rank}, "
        f"CG `{args.max_iter}` iterations.",
        f"- Shared magnitude scale from fully corrected image: `{preprocessing['scale']}` "
        f"(`median + 3*std`). Saved in `preprocessing.json`.",
        '- Grayscale repeated to RGB in `[0, 1]`; FGResQ/ARNIQA then apply their own',
        '  normalization. Gradient entropy and wavelet sparsity are scale-invariant',
        '  and are scored on magnitude.',
        '- `gradient_entropy_metric` and `wavelet_sparsity_metric` return `sum(p log p) = -H`,',
        '  so **higher is better**. NGS, AES, and Gaussian-scale entropy are also higher-is-better.',
        '  Data-consistency `L(k)` is **lower is better**.',
        '- NGS is `sum(g**2)/(sum(g)**2+eps)` on gradient magnitude. AES is the mean 3x3 Sobel',
        '  magnitude with fixed kernels and no threshold. Gaussian entropy uses the existing',
        f'  gradient-entropy metric after smoothing at σ = {list(GAUSSIAN_ENTROPY_SIGMAS)} px.',
        f'- Numerical tolerance for scalar ties/decreases: `{SCORE_TOL}`.',
        '- FGResQ pairwise `rank_prob` is the argmax-class probability, not P(candidate better).',
        '  Class `1` is first-image better; `p_candidate_better` uses that class.',
        '',
        '## Endpoints',
        '',
        f"- k=0 applies no phase correction (HOFFT drops empty bases at zero alphas, "
        f"so this is naive SENSE, matching `tests/test_taylor.py`). NRMSE vs naive SENSE: "
        f"{endpoint['nrmse_k0_vs_naive']:.4%}.",
        f"- k=K uses the full loaded alphas. NRMSE vs a second full-alpha recon: "
        f"{endpoint['nrmse_kK_repeat']:.4%}.",
        f"- Finite scores at every k: {endpoint['all_scores_finite']}. "
        f"Levels present: {endpoint['n_levels']} (expected {args.k + 1}).",
        '',
        '## Scalar ordering (per slice, then mean if multiple slices)',
        '',
        '| Metric | Spearman vs k/K | Adjacent decreases | Ties | Best fraction |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, stats in ordering.items():
        if name == 'follow_text':
            continue
        lines.append(
            f"| {name} | {fmt_rho(stats['spearman_vs_fraction'])} | "
            f"{stats['adjacent_decreases']} | {stats['adjacent_ties']} | "
            f"{stats['best_fraction']:.3f} |"
        )
    lines += [
        '',
        '## FGResQ pairwise',
        '',
        f"- Adjacent (image_k vs image_(k-1)): fraction preferring more-corrected "
        f"{pair_summary['adjacent_prefer_more_corrected']:.3f}, "
        f"tie fraction {pair_summary['adjacent_tie_fraction']:.3f}, "
        f"mean p_candidate_better {pair_summary['adjacent_mean_p']:.3f}.",
        f"- Versus uncorrected anchor image_0: mean p_candidate_better "
        f"{pair_summary['anchor_mean_p']:.3f}; "
        f"{pair_summary['anchor_saturation']}.",
        '',
        '## Data-consistency loss',
        '',
        f"- `L(k) = min_x ||A(alphas_k) x - b||_2^2` with unweighted adjoint (`dcf=1`),",
        f"  `lamda_l2=0`, CG max_iter={pair_summary.get('dc_max_iter', 'n/a')},",
        f"  tolerance={pair_summary.get('dc_tolerance', 'n/a')}.",
        f"- {pair_summary.get('dc_text', '')}",
        '',
        '## Wiring checks',
        '',
        f"- Identical images: rank={wiring.get('identical_rank')}, "
        f"p_similar={float(wiring.get('identical_p_similar') or float('nan')):.3f}.",
        f"- Swap consistency `|p(a>b) - p_other(b,a)|` = {float(wiring.get('swap_abs_diff') or float('nan')):.3e}.",
        '',
        '## Which metrics follow the sweep',
        '',
        ordering['follow_text'],
        '',
        'See `scores.csv`, `pairwise_adjacent.csv`, `pairwise_anchor.csv`,',
        '`scores.png`, `pairwise.png`, `montage.png`, and `results.pt`.',
        '',
    ]
    (output / 'report.md').write_text('\n'.join(lines))


@torch.no_grad()
def run(args):
    started = time.monotonic()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'Loading {args.data_dir} on {device}', flush=True)
    data = load_tilt_spi(args.data_dir, device, args.shot_stride, args.mask_threshold)
    alphas_original = data['alphas'].clone()
    fractions = [k / args.k for k in range(args.k + 1)]
    results_path = args.output / 'results.pt'
    prior = None
    if results_path.exists():
        prior = torch.load(results_path, map_location='cpu', weights_only=False)

    if args.reconstructions is not None:
        print(f'Reusing reconstructions from {args.reconstructions}', flush=True)
        saved = torch.load(args.reconstructions, map_location='cpu', weights_only=False)
        recon = saved['images']
        mag = saved['magnitude'] if 'magnitude' in saved else recon.abs()
        if torch.is_complex(mag):
            mag = mag.abs()
        scale = float(saved['scale'])
        if 'fractions' in saved:
            fractions = [float(v) for v in saved['fractions']]
            args.k = len(fractions) - 1
        naive = saved.get('naive')
        nrmse_k0 = float(prior['endpoint']['nrmse_k0_vs_naive']) if prior and 'endpoint' in prior else float('nan')
        nrmse_repeat = float(prior['endpoint']['nrmse_kK_repeat']) if prior and 'endpoint' in prior else float('nan')
    else:
        print('Endpoint reconstructions: naive SENSE, k=0, k=K', flush=True)
        naive = reconstruct_naive(data, args.max_iter).cpu()
        img0 = reconstruct_hofft(data, torch.zeros_like(alphas_original), args.max_iter, args.model_rank).cpu()
        imgK = reconstruct_hofft(data, alphas_original, args.max_iter, args.model_rank).cpu()
        imgK_repeat = reconstruct_hofft(data, alphas_original, args.max_iter, args.model_rank).cpu()
        nrmse_k0 = magnitude_nrmse(img0, naive, data['mask'].cpu())
        nrmse_repeat = magnitude_nrmse(imgK, imgK_repeat, data['mask'].cpu())
        print(f'k=0 vs naive NRMSE={nrmse_k0:.4%}; k=K repeat NRMSE={nrmse_repeat:.4%}', flush=True)

        images = [None] * (args.k + 1)
        images[0] = img0
        images[-1] = imgK
        for k in tqdm(range(1, args.k), desc='Correction sweep'):
            images[k] = reconstruct_hofft(
                data, alphas_original * fractions[k], args.max_iter, args.model_rank,
            ).cpu()
        recon = torch.stack([img.cpu() for img in images])
        mag = recon.abs()
        scale = float((mag[-1].median() + 3 * mag[-1].std()).clamp_min(1e-12))
        naive = naive.cpu()

    assert mag.shape[0] == args.k + 1
    n_slices = 1 if mag.ndim == 3 else mag.shape[1]
    slice_mags = mag[:, None] if mag.ndim == 3 else mag

    if args.reconstructions is None:
        image_dir = args.output / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
        for k, fraction in enumerate(fractions):
            torch.save({'k': k, 'fraction': fraction, 'image': recon[k]}, image_dir / f'k{k:02d}.pt')
        torch.save({
            'k': torch.arange(args.k + 1),
            'fractions': torch.tensor(fractions),
            'images': recon,
            'magnitude': mag,
            'naive': naive.abs() if naive is not None else mag[0],
            'scale': scale,
        }, args.output / 'reconstructions.pt')
        print(f'Saved {args.k + 1} reconstructions ({", ".join(f"k={k}" for k in range(args.k + 1))}) '
              f'to {image_dir} and reconstructions.pt', flush=True)

    scores_csv = args.output / 'scores.csv'
    skip_pretrained = args.skip_pretrained and scores_csv.exists()
    if skip_pretrained:
        print(f'Reusing pretrained scores from {scores_csv}', flush=True)
        rows = list(csv.DictReader(scores_csv.open()))
        fgresq_scores = torch.zeros((args.k + 1, n_slices))
        arniqa_scores = torch.zeros((args.k + 1, n_slices))
        for row in rows:
            fgresq_scores[int(row['k']), int(row['slice'])] = float(row['fgresq'])
            arniqa_scores[int(row['k']), int(row['slice'])] = float(row['arniqa'])
        if prior and 'wiring' in prior:
            wiring = prior['wiring']
        else:
            wiring = {'identical_rank': None, 'identical_p_similar': float('nan'),
                      'swap_abs_diff': float('nan'), 'input_range': []}
        fgresq = arniqa = None
    else:
        print('Loading pretrained IQA models', flush=True)
        fgresq = load_fgresq(device=device)
        arniqa = load_arniqa(device=device)
        rgb0 = to_iqa_input(slice_mags[0, 0], scale, device=device)
        rgbK = to_iqa_input(slice_mags[-1, 0], scale, device=device)
        ident = fgresq_pair_metric(slice_mags[0, 0], slice_mags[0, 0], fgresq, scale)
        forward = fgresq_pair_metric(slice_mags[-1, 0], slice_mags[0, 0], fgresq, scale)
        backward = fgresq_pair_metric(slice_mags[0, 0], slice_mags[-1, 0], fgresq, scale)
        wiring = {
            'identical_rank': int(ident['rank'][0]),
            'identical_p_similar': float(ident['p_similar'][0]),
            'swap_abs_diff': abs(float(forward['p_candidate_better'][0]) - float(backward['p_other_better'][0])),
            'input_range': [float(rgb0.min()), float(rgb0.max()), float(rgbK.min()), float(rgbK.max())],
        }
        print(f'Wiring: identical rank={wiring["identical_rank"]} p_similar={wiring["identical_p_similar"]:.3f}; '
              f'swap diff={wiring["swap_abs_diff"]:.3e}', flush=True)
        fgresq_scores = torch.zeros((args.k + 1, n_slices))
        arniqa_scores = torch.zeros((args.k + 1, n_slices))
        for k in range(args.k + 1):
            for s in range(n_slices):
                fgresq_scores[k, s] = fgresq_metric(slice_mags[k, s], fgresq, scale)
                arniqa_scores[k, s] = arniqa_metric(slice_mags[k, s], arniqa, scale)

    def stack_baseline(metric_fn):
        scores = torch.stack([metric_fn(slice_mags[k], spatial_ndim=2) for k in range(args.k + 1)])
        return scores[:, None] if scores.ndim == 1 else scores

    entropy = stack_baseline(gradient_entropy_metric)
    wavelet = stack_baseline(wavelet_sparsity_metric)
    ngs = stack_baseline(normalized_gradient_squared_metric)
    aes = stack_baseline(average_edge_strength_metric)
    gauss = torch.stack([
        gaussian_scale_gradient_entropy(slice_mags[k], spatial_ndim=2) for k in range(args.k + 1)
    ], dim=1)
    if gauss.ndim == 2:
        gauss = gauss[:, :, None]
    # gauss: (n_sigma, K+1, n_slices)

    print('Minimizing unweighted k-space data-consistency loss', flush=True)
    dc_rows = []
    dc_loss = torch.zeros((args.k + 1, n_slices))
    dc_norm = torch.zeros((args.k + 1, n_slices))
    for k, fraction in enumerate(tqdm(fractions, desc='DC least squares')):
        operator, ksp_k = make_unweighted_operator(
            data, alphas_original * fraction, args.model_rank)
        dc = minimized_data_consistency(operator, ksp_k, args.dc_max_iter, args.dc_tolerance)
        dc_loss[k, :] = dc['loss']
        dc_norm[k, :] = dc['loss_normalized']
        dc['k'] = k
        dc['fraction'] = fraction
        dc_rows.append(dc)
        del operator
        print(f'k={k}: L={dc["loss"]:.6g} L/||b||^2={dc["loss_normalized"]:.6g} '
              f'CG {dc["n_iter"]}/{args.dc_max_iter} ||AHA x - AHb||={dc["normal_residual"]:.3e}',
              flush=True)

    score_names = {
        'gradient_entropy': entropy,
        'wavelet_sparsity': wavelet,
        'fgresq': fgresq_scores,
        'arniqa': arniqa_scores,
        'ngs': ngs,
        'aes': aes,
        'dc_loss': dc_loss,
        'dc_loss_normalized': dc_norm,
    }
    for index, name in enumerate(GE_SIGMA_NAMES):
        score_names[name] = gauss[index]
    score_rows = []
    for k, fraction in enumerate(fractions):
        for s in range(n_slices):
            row = {'k': k, 'fraction': fraction, 'slice': s}
            for name, values in score_names.items():
                row[name] = float(values[k, s])
            score_rows.append(row)
    write_csv(args.output / 'scores.csv', score_rows,
              ['k', 'fraction', 'slice', *SCALAR_METRICS])
    write_csv(args.output / 'data_consistency.csv', dc_rows,
              ['k', 'fraction', 'loss', 'loss_normalized', 'b_norm_sq', 'n_iter',
               'hit_max_iter', 'normal_residual', 'tolerance', 'max_iter'])

    adj_rows, anc_rows = [], []
    adj_p = np.zeros((n_slices, args.k))
    anc_p = np.zeros((n_slices, args.k))
    adj_rank = np.zeros((n_slices, args.k), dtype=int)
    pair_fields = [
        'k', 'fraction', 'slice', 'kind', 'quality_candidate', 'quality_other',
        'rank', 'rank_prob', 'p_candidate_better', 'p_other_better', 'p_similar',
    ]
    if skip_pretrained and (args.output / 'pairwise_adjacent.csv').exists():
        adj_loaded = list(csv.DictReader((args.output / 'pairwise_adjacent.csv').open()))
        anc_loaded = list(csv.DictReader((args.output / 'pairwise_anchor.csv').open()))
        for row in adj_loaded:
            adj_p[int(row['slice']), int(row['k']) - 1] = float(row['p_candidate_better'])
            adj_rank[int(row['slice']), int(row['k']) - 1] = int(row['rank'])
        for row in anc_loaded:
            anc_p[int(row['slice']), int(row['k']) - 1] = float(row['p_candidate_better'])
        adj_rows, anc_rows = adj_loaded, anc_loaded
    else:
        for k in range(1, args.k + 1):
            for s in range(n_slices):
                adj = fgresq_pair_metric(slice_mags[k, s], slice_mags[k - 1, s], fgresq, scale)
                anc = fgresq_pair_metric(slice_mags[k, s], slice_mags[0, s], fgresq, scale)
                adj_p[s, k - 1] = float(adj['p_candidate_better'][0])
                anc_p[s, k - 1] = float(anc['p_candidate_better'][0])
                adj_rank[s, k - 1] = int(adj['rank'][0])
                def pack(src, kind):
                    return {
                        'k': k, 'fraction': fractions[k], 'slice': s, 'kind': kind,
                        'quality_candidate': float(src['quality0'][0]),
                        'quality_other': float(src['quality1'][0]),
                        'rank': int(src['rank'][0]),
                        'rank_prob': float(src['rank_prob'][0]),
                        'p_candidate_better': float(src['p_candidate_better'][0]),
                        'p_other_better': float(src['p_other_better'][0]),
                        'p_similar': float(src['p_similar'][0]),
                    }
                adj_rows.append(pack(adj, 'adjacent'))
                anc_rows.append(pack(anc, 'anchor'))
        write_csv(args.output / 'pairwise_adjacent.csv', adj_rows, pair_fields)
        write_csv(args.output / 'pairwise_anchor.csv', anc_rows, pair_fields)

    ordering = {}
    follow = []
    for name, values in score_names.items():
        per_slice = [scalar_ordering(values[:, s].numpy(), fractions) for s in range(n_slices)]
        rhos = [item['spearman_vs_fraction'] for item in per_slice]
        defined = [rho for rho in rhos if rho is not None]
        stats = dict(per_slice[0])
        stats['spearman_vs_fraction'] = None if not defined else float(np.mean(defined))
        stats['adjacent_decreases'] = int(np.mean([item['adjacent_decreases'] for item in per_slice]))
        stats['adjacent_ties'] = int(np.mean([item['adjacent_ties'] for item in per_slice]))
        stats['n_slices'] = n_slices
        stats['per_slice'] = per_slice
        stats['lower_better'] = name in LOWER_BETTER
        ordering[name] = stats
        rho = stats['spearman_vs_fraction']
        if rho is None:
            follow.append(f'{name}: Spearman undefined (constant or non-finite).')
        elif name in LOWER_BETTER:
            if rho <= -0.7 and stats['adjacent_increases'] <= max(1, args.k // 10):
                follow.append(f'{name}: decreases with correction (Spearman {rho:.3f}, lower is better).')
            elif rho < 0:
                follow.append(
                    f'{name}: weakly decreasing (Spearman {rho:.3f}, '
                    f'{stats["adjacent_increases"]} adjacent increases).'
                )
            else:
                follow.append(f'{name}: does not decrease with correction (Spearman {rho:.3f}).')
        elif rho >= 0.7 and stats['adjacent_decreases'] <= max(1, args.k // 10):
            follow.append(f'{name}: follows the sweep (Spearman {rho:.3f}).')
        elif rho > 0:
            follow.append(
                f'{name}: weakly increasing (Spearman {rho:.3f}, '
                f'{stats["adjacent_decreases"]} adjacent decreases); fails to be monotonic.'
            )
        else:
            follow.append(
                f'{name}: does not follow the sweep (Spearman {rho:.3f}).'
            )
    ordering['follow_text'] = ' '.join(follow)

    losses = np.asarray([row['loss'] for row in dc_rows], dtype=np.float64)
    rel_span = float((losses.max() - losses.min()) / max(abs(losses.mean()), 1e-30))
    dc_text = (
        f'CG hit the iteration cap at {sum(row["hit_max_iter"] for row in dc_rows)}/{len(dc_rows)} '
        f'levels (max_iter={args.dc_max_iter}, tolerance={args.dc_tolerance:g}); '
        f'reported L is then an upper bound on the true minimum. '
    )
    if rel_span < 1e-3:
        dc_text += (
            'L(k) is essentially constant across correction fractions '
            f'((max-min)/mean = {rel_span:.3g}), so the unregularized residual is '
            'uninformative for identifying the correct phase.'
        )
    else:
        dc_text += (
            f'L(k) relative range (max-min)/mean = {rel_span:.4g}. '
            f'Raw L and L/||b||_2^2 are in data_consistency.csv and scores.png.'
        )

    last_half = anc_p[:, anc_p.shape[1] // 2:]
    span = float(last_half.max() - last_half.min()) if last_half.size else 0.0
    saturation = (
        f'saturates in the second half (range {span:.3f})'
        if span < 0.05 else
        f'does not saturate in the second half (range {span:.3f}); treat as a ranking signal, not a calibrated gap'
    )
    pair_summary = {
        'adjacent_prefer_more_corrected': float(np.mean(adj_rank == 1)) if adj_rank.size else float('nan'),
        'adjacent_tie_fraction': float(np.mean(adj_rank == PAIR_TIE_RANK)) if adj_rank.size else float('nan'),
        'adjacent_mean_p': float(adj_p.mean()) if adj_p.size else float('nan'),
        'anchor_mean_p': float(anc_p.mean()) if anc_p.size else float('nan'),
        'anchor_saturation': saturation,
        'dc_max_iter': args.dc_max_iter,
        'dc_tolerance': args.dc_tolerance,
        'dc_text': dc_text,
    }

    all_finite = all(torch.isfinite(values).all() for values in score_names.values())
    endpoint = {
        'nrmse_k0_vs_naive': nrmse_k0,
        'nrmse_kK_repeat': nrmse_repeat,
        'all_scores_finite': bool(all_finite),
        'n_levels': int(mag.shape[0]),
    }
    versions = package_versions()
    preprocessing = {
        'scale': scale,
        'scale_rule': 'median + 3*std of fully corrected magnitude',
        'range': '[0, 1]',
        'channels': 'three identical grayscale channels',
        'orientation': 'native reconstruction, no rot90',
        'fgresq_preprocess': 'model resize 256, center crop 224, CLIP mean/std',
        'arniqa_preprocess': 'model ImageNet mean/std and 2x downsample branch',
        'score_tolerance': SCORE_TOL,
        'gradient_entropy_direction': 'higher is better; implementation is sum(p log p) = -H',
        'wavelet_sparsity_direction': 'higher is better; Haar-detail sum(p log p) = -H',
        'ngs': 'sum(g**2)/(sum(g)**2+eps); higher is more concentrated gradient energy',
        'aes': 'mean 3x3 Sobel magnitude; kernels fixed; no threshold; higher is stronger edges',
        'gaussian_entropy_sigmas_px': list(GAUSSIAN_ENTROPY_SIGMAS),
        'dc_loss': 'min_x ||A(alphas_k) x - b||_2^2, unweighted (dcf=1), no extra regularization; lower is better',
        'dc_max_iter': args.dc_max_iter,
        'dc_tolerance': args.dc_tolerance,
        'ngs': 'sum(g**2)/(sum(g)**2+eps); higher is more concentrated gradient energy',
        'aes': 'mean 3x3 Sobel magnitude; kernels fixed; no threshold; higher is stronger edges',
        'gaussian_entropy_sigmas_px': list(GAUSSIAN_ENTROPY_SIGMAS),
        'dc_loss': 'min_x ||A(alphas_k) x - b||_2^2, unweighted (dcf=1), no extra regularization; lower is better',
        'dc_max_iter': args.dc_max_iter,
        'dc_tolerance': args.dc_tolerance,
    }
    (args.output / 'preprocessing.json').write_text(json.dumps(preprocessing, indent=2) + '\n')
    (args.output / 'versions.json').write_text(json.dumps(versions, indent=2) + '\n')

    slice_score_np = {name: values.numpy().T for name, values in score_names.items()}
    save_plots(args.output, np.asarray(fractions), mag, scale, slice_score_np, adj_p, anc_p)
    write_report(args.output, args, versions, preprocessing, endpoint, ordering, pair_summary, wiring)
    torch.save({
        'fractions': torch.tensor(fractions),
        'images': recon,
        'magnitude': mag,
        'scale': scale,
        'scores': {name: values.cpu() for name, values in score_names.items()},
        'adjacent_p_candidate_better': torch.as_tensor(adj_p),
        'anchor_p_candidate_better': torch.as_tensor(anc_p),
        'wiring': wiring,
        'endpoint': endpoint,
        'versions': versions,
        'preprocessing': preprocessing,
        'naive': naive.abs().cpu() if naive is not None else mag[0].cpu(),
        'dc': dc_rows,
    }, args.output / 'results.pt')
    print(json.dumps({
        'elapsed_seconds': time.monotonic() - started,
        'endpoint': endpoint,
        'ordering': {k: v for k, v in ordering.items() if k != 'follow_text' and k != 'per_slice'},
        'pair_summary': pair_summary,
        'wiring': wiring,
        'output': str(args.output),
    }, indent=2, default=str), flush=True)
    print(f'Results saved to {args.output}', flush=True)


if __name__ == '__main__':
    run(parse_args())
