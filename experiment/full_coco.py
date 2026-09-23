"""Joint radial-phase autofocus for section 2.4.1 of latex/desc.tex.

Example (run from the repository root inside the existing GPU allocation):
    srun --jobid=8438 --overlap env PYTHONPATH=src:../hofft/src:../mr_recon/src \
        /home/abrahamd/mambaforge/envs/mr_recon/bin/python experiment/full_coco.py --device cuda

The stored spatial bases are known; NO stored alpha values enter autofocus.
Reconstruction uses the existing SENSE, HOFFT, and CG operators from coco.py.
Candidate synthesis approximates the converged linear inverse; its discrepancy
from direct finite-iteration CG is measured explicitly.
Outputs include fitted alphas, phase/image comparisons and numerical diagnostics.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gaim.metrics import (gradient_entropy_metric, gradient_renyi_metric, multiscale_gradient_entropy_metric,
                          peaky_curve_confidence, wavelet_sparsity_metric,
                          gradient_renyi_half_metric, wavelet_renyi_half_metric,
                          undecimated_wavelet_entropy_metric)
from gaim.patch import strided_patchify


FOCUS_METRICS = {'wavelet': wavelet_sparsity_metric,
                 'gradient': gradient_entropy_metric,
                 'renyi': gradient_renyi_metric,
                 'multiscale': multiscale_gradient_entropy_metric,
                 'gradient_half': gradient_renyi_half_metric,
                 'wavelet_half': wavelet_renyi_half_metric,
                 'undecimated': undecimated_wavelet_entropy_metric}


def radial_chebyshev(rho, degree):
    """Anchored Chebyshev polynomials on radius rho in [0, 1].

    g_j = (T_j(2*rho-1) - T_j(-1))/2, j=1,...,degree.
    Each basis is zero at radius zero and has maximum absolute value one
    over the radius interval. Thus |f_j| is its maximum phase in wraps.
    The constant polynomial is omitted: image sharpness cannot identify it.
    """
    if degree < 1:
        raise ValueError('degree must be positive')
    x = 2 * rho - 1
    previous, current = torch.ones_like(x), x
    terms = []
    for order in range(1, degree + 1):
        terms.append((current - (-1)**order) / 2)
        previous, current = current, 2 * x * current - previous
    return torch.stack(terms)


def interpolate_radius(values, rho):
    """Interpolate trailing, uniformly sampled radius axis (also complex)."""
    position = rho.clamp(0, 1) * (values.shape[-1] - 1)
    left = position.floor().long()
    right = (left + 1).clamp_max(values.shape[-1] - 1)
    fraction = position - left
    return values[..., left] * (1 - fraction) + values[..., right] * fraction


def truncated_svd(matrix, max_rank, tolerance):
    """Return a measured relative-Frobenius-error low-rank factorization."""
    q = min(max_rank + 8, min(matrix.shape))
    u, singular, v = torch.svd_lowrank(matrix, q=q, niter=3)
    energy = matrix.abs().double().square().sum()
    residual = (1 - singular.double().square().cumsum(0) / energy).clamp_min(0).sqrt()
    acceptable = torch.nonzero(residual <= tolerance).flatten()
    rank = min(int(acceptable[0]) + 1 if acceptable.numel() else max_rank, max_rank, q)
    left = u[:, :rank] * singular[:rank]
    right = v[:, :rank].mH
    error = float((matrix - left @ right).norm() / matrix.norm())
    while error > tolerance and rank < min(max_rank, q):
        rank = min(rank + 4, max_rank, q)
        left, right = u[:, :rank] * singular[:rank], v[:, :rank].mH
        error = float((matrix - left @ right).norm() / matrix.norm())
    return left, right, error


def make_dictionary(degree, wraps, grid_size, nodes, max_rank, tolerance, device):
    grid = torch.linspace(-wraps, wraps, grid_size, device=device)
    candidates = torch.cartesian_prod(*([grid] * degree)).reshape(-1, degree)
    radius_nodes = torch.linspace(0, 1, nodes, device=device)
    g_nodes = radial_chebyshev(radius_nodes, degree)
    # This is the CORRECTION dictionary: conjugate of the forward phase.
    dictionary = torch.exp(2j * torch.pi * (candidates @ g_nodes))
    left, right, error = truncated_svd(dictionary, max_rank, tolerance)
    return grid, candidates, g_nodes, left, right, error


class CocoRecon:
    """The uncorrected SENSE/CG reconstruction used in coco.py.

    Candidate SVD synthesis approximates the converged linear inverse. Its
    finite-CG discrepancy is measured against directly reconstructed candidates.
    """

    def __init__(self, trajectory, mps, dcf, steps=20):
        from mr_recon.linops import sense_linop, batching_params
        from mr_recon.fourier import cufi_nufft
        from mr_recon.recons import CG_SENSE_recon

        self.cg = CG_SENSE_recon
        self.steps, self.calls = steps, 0
        self.operator = sense_linop(
            trajectory.reshape(-1, trajectory.shape[-1]), mps, dcf.flatten(),
            nufft=cufi_nufft(mps.shape[1:], oversamp=1.25, width=3),
            bparams=batching_params(coil_batch_size=len(mps)))

    def __call__(self, data):
        self.calls += 1
        return self.cg(self.operator, data, max_eigen=1.0, max_iter=self.steps,
                       verbose=False, clear_gpu_mem=False)


def basis_reconstructions(recon, data, temporal, label):
    images = []
    started = time.monotonic()
    for index, modulation in enumerate(temporal):
        images.append(recon(data * modulation))
        if (index + 1) % 8 == 0 or index + 1 == len(temporal):
            print(f'{label}: {index + 1}/{len(temporal)} basis reconstructions '
                  f'({time.monotonic() - started:.1f}s)', flush=True)
    return torch.stack(images)


def search_patches(left, basis_images, candidates, grid, patch_size, stride,
                   batch_size=16, metric='wavelet'):
    metric_fn = FOCUS_METRICS[metric]
    score_batches = []
    for start in range(0, len(candidates), batch_size):
        images = (left[start:start + batch_size] @ basis_images.flatten(1)).reshape(
            -1, *basis_images.shape[1:])
        patches = strided_patchify(images, patch_size, stride)
        score_batches.append(metric_fn(patches, spatial_ndim=2, eps=1e-6))
    scores = torch.cat(score_batches)
    indices = scores.argmax(0)
    selected = candidates[indices].movedim(-1, 0).clone()
    degree, grid_size = candidates.shape[1], len(grid)
    grid_shape = (grid_size,) * degree + tuple(indices.shape)
    cube = scores.reshape(grid_shape)
    confidences, boundaries = [], []
    best_score = scores.gather(0, indices[None])[0]
    for axis in range(degree):
        # Profile scores over the remaining coordinates; use their peak shape
        # to distinguish usable patches from flat/ambiguous search responses.
        other_axes = tuple(i for i in range(degree) if i != axis)
        profile = cube.amax(dim=other_axes) if other_axes else cube
        confidences.append(peaky_curve_confidence(profile, x=grid, dim=0))
        offset = grid_size**(degree - axis - 1)
        coordinate = (indices // offset) % grid_size
        interior = (coordinate > 0) & (coordinate < grid_size - 1)
        boundaries.append(~interior)
        below = scores.gather(0, (indices - offset).clamp_min(0)[None])[0]
        above = scores.gather(0, (indices + offset).clamp_max(len(candidates) - 1)[None])[0]
        curvature = below - 2 * best_score + above
        shift = torch.where(interior & (curvature < -1e-8),
                            0.5 * (below - above) / curvature.clamp_max(-1e-8), 0.)
        selected[axis] += shift.clamp(-0.5, 0.5) * (grid[1] - grid[0])
    confidence = torch.stack(confidences).mean(0)
    return selected, confidence, torch.stack(boundaries), scores


def fit_spatial_maps(phis, patch_values, confidence, mask, stride, ridge=1e-4):
    """Fit f_k(r)=sum_b phi_b(r) C_bk at the measured patch anchors."""
    slices = tuple(slice(None, None, step) for step in stride)
    sampled = phis[(slice(None),) + slices].flatten(1).T.double()
    weight = (confidence * mask[slices]).flatten().double().clamp_min(0)
    if not bool(weight.sum() > 0):
        raise RuntimeError('No confident in-mask patches; inspect the metric and search range.')
    scale = (sampled.square() * weight[:, None]).sum(0).sqrt()
    active = scale > 1e-12
    design = sampled[:, active] / scale[active]
    normal = design.T @ (weight[:, None] * design)
    normal.diagonal().add_(ridge)
    rhs = design.T @ (weight[:, None] * patch_values.flatten(1).T.double())
    coefficients = phis.new_zeros((phis.shape[0], patch_values.shape[0]))
    coefficients[active] = (torch.linalg.solve(normal, rhs) / scale[active, None]).float()
    maps = (coefficients.T @ phis.flatten(1)).reshape(patch_values.shape[0], *mask.shape)
    return coefficients, maps, active


def synthesize_maps(maps, g_nodes, temporal_nodes, basis_images, batch_size=2048):
    flat = maps.flatten(1).T
    output = basis_images.new_empty(flat.shape[0])
    for start in range(0, len(flat), batch_size):
        correction = torch.exp(2j * torch.pi * (flat[start:start + batch_size] @ g_nodes))
        weights = correction @ temporal_nodes.mH
        output[start:start + batch_size] = (
            weights * basis_images.flatten(1)[:, start:start + batch_size].T).sum(1)
    return output.reshape(maps.shape[1:])


def make_phase_operator(phis, alphas, mps, trajectory, model_rank, coil_batch_size=None,
                        field_batch_size=1, cur_rank=500):
    """The native full-phase operator shared by CG and prediction diagnostics."""
    from hofft.decomp import hofft_params
    from hofft.pipelines import svd_decomp_linop
    from hofft.phase_coeffs import remove_empty_bases
    from mr_recon.linops import batching_params

    phis, alphas = remove_empty_bases(phis, alphas)
    hparams = hofft_params(kern_size=(3,) * (phis.ndim - 1), os=1.25,
                          L=model_rank, cur_rank=cur_rank, verbose=False)
    return svd_decomp_linop(
        phis, alphas, mps, trajectory.reshape(-1, trajectory.shape[-1]),
        hparams=hparams, bparams=batching_params(
            coil_batch_size=len(mps) if coil_batch_size is None else coil_batch_size,
            field_batch_size=field_batch_size))


def phase_reconstruction(phis, alphas, mps, trajectory, iterations, model_rank, label):
    """Mirror coco.py's full forward model and CG, including its DCF choice."""
    from mr_recon.recons import CG_SENSE_recon

    operator = make_phase_operator(phis, alphas, mps, trajectory, model_rank)

    def reconstruct(data):
        print(f'{label}: svd_decomp_linop (L={model_rank}) + CG_SENSE_recon ({iterations} iterations)', flush=True)
        return CG_SENSE_recon(operator, data, max_eigen=1.0, max_iter=iterations,
                              verbose=False, clear_gpu_mem=False)

    return reconstruct


@torch.enable_grad()
def refine_image_metric(coefficients, phis, g_nodes, temporal_nodes, basis_images,
                        initial_image, mask, patch_size, stride, wraps, iterations, metric):
    """Optimize the image prior directly, with no ground-truth alphas or image.

    Project the temporal dictionary to quadrature nodes once. Subsequent
    image/gradient evaluations need only pointwise phase and weighted sums,
    not NUFFTs or reconstructed candidate grids.
    """
    metric_fn = FOCUS_METRICS[metric]
    with torch.no_grad():
        nodes = torch.linspace(0, g_nodes.shape[1] - 1, min(256, g_nodes.shape[1]),
                               device=phis.device).long()
        g_small = g_nodes[:, nodes]
        projection = torch.linalg.pinv(temporal_nodes[:, nodes])
        projected_images = projection @ basis_images.flatten(1)
        scales = phis.flatten(1).abs().amax(1).clamp_min(1e-12)
        normalized_phis = phis.flatten(1) / scales[:, None]
        gradients = torch.gradient(initial_image.abs())
        structure = sum(part.square() for part in gradients).sqrt()
        weights = strided_patchify(structure, patch_size, stride).mean((-1, -2))
        weights = (weights / torch.quantile(weights, .9).clamp_min(1e-12)).clamp(0, 1)
        weights *= mask[::stride[0], ::stride[1]]
        weights /= weights.sum().clamp_min(1e-12)

    best_score, best_parameters = -float('inf'), None
    histories = []
    # Include zero as a second start: independently selected patch maxima may
    # fit together poorly as a single spatially consistent phase correction.
    for name, initial in (('patch', coefficients * scales[:, None]),
                          ('zero', torch.zeros_like(coefficients))):
        parameters = initial.detach().clone().requires_grad_()
        optimizer = torch.optim.Adam([parameters], lr=.04)
        history = []
        for iteration in range(iterations + 1):
            maps = parameters.T @ normalized_phis
            image = (torch.exp(2j * torch.pi * (maps.T @ g_small)).T
                     * projected_images).sum(0).reshape(mask.shape)
            scores = metric_fn(strided_patchify(image, patch_size, stride), spatial_ndim=2, eps=1e-6)
            score = (weights * scores).sum()
            penalty = 1e-4 * maps.square().mean() + .01 * (maps.abs() - wraps).clamp_min(0).square().mean()
            objective = score - penalty
            value = float(objective.detach())
            history.append(value)
            if value > best_score:
                best_score, best_parameters = value, parameters.detach().clone()
            if iteration == iterations:
                break
            optimizer.zero_grad()
            (-objective).backward()
            optimizer.step()
            if (iteration + 1) % 25 == 0:
                print(f'Image-prior refinement ({name}): {iteration + 1}/{iterations}, objective={value:.5f}', flush=True)
        histories.append({'start': name, 'objective': history})
    fitted = best_parameters / scales[:, None]
    return fitted, {'histories': histories, 'best_objective': best_score,
                    'criterion': 'Structure-weighted patch sparsity minus phase-size/range regularization; no ground truth.'}


def phase_errors(phis, estimated, truth, mask):
    """Full phase errors without allocating a pixels-by-time phase tensor."""
    spatial = phis[:, mask.bool()].double()
    gram = spatial @ spatial.T / spatial.shape[1]
    error, truth = (estimated - truth).double(), truth.double()
    mse = ((gram @ error) * error).sum(0).mean().clamp_min(0)
    energy = ((gram @ truth) * truth).sum(0).mean().clamp_min(1e-20)
    return {'rmse_wraps': float(mse.sqrt()), 'relative_l2': float((mse / energy).sqrt())}


def magnitude_error(image, reference, mask):
    x, y = image.abs()[mask.bool()].double(), reference.abs()[mask.bool()].double()
    scale = (x @ y) / (x @ x).clamp_min(1e-20)
    return float((scale * x - y).norm() / y.norm().clamp_min(1e-20))


def save_figures(output, images, phis, alphas, estimated, g, maps, confidence, mask):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), constrained_layout=True)
    reference = next(reversed(images.values())).abs()
    vmax = float(torch.quantile(reference[mask.bool()].cpu(), .99))
    for ax, (name, image) in zip(np.atleast_1d(axes), images.items()):
        # Display the same scale used by the gain-invariant image comparison.
        x, y = image.abs()[mask.bool()], reference[mask.bool()]
        scale = (x @ y) / x.square().sum().clamp_min(1e-20)
        ax.imshow((image.abs() * scale).cpu().rot90(), cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.axis('off')
    fig.savefig(output / 'images.png', dpi=160)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    # A single readout is plotted; all selected samples enter numeric errors.
    sample = np.arange(alphas.shape[1])
    for index, ax in enumerate(axes.flat):
        if index >= phis.shape[0]:
            ax.axis('off')
            continue
        extent = float(phis[index][mask.bool()].abs().max())
        ax.plot(sample, (alphas[index] * extent).cpu(), label='Known alpha')
        ax.plot(sample, (estimated[index] * extent).cpu(), '--', label='Estimated alpha')
        ax.set_title(f'Term {index}' + (' (unobservable)' if extent == 0 else ''))
        ax.set_xlabel('Flattened readout sample')
        ax.set_ylabel('Phase at max |phi| (wraps)')
    axes.flat[0].legend()
    fig.savefig(output / 'alphas.png', dpi=160)
    fig, axes = plt.subplots(1, maps.shape[0] + 1, figsize=(4 * (maps.shape[0] + 1), 4), constrained_layout=True)
    for index in range(maps.shape[0]):
        artist = axes[index].imshow((maps[index] * mask).cpu().rot90(), cmap='coolwarm')
        axes[index].set_title(f'f_{index + 1} (wraps)')
        axes[index].axis('off')
        fig.colorbar(artist, ax=axes[index])
    axes[-1].imshow(confidence.cpu().rot90(), cmap='magma', vmin=0, vmax=1)
    axes[-1].set_title('Patch confidence')
    axes[-1].axis('off')
    fig.savefig(output / 'parameter_maps.png', dpi=160)
    plt.close('all')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('/local_mount/space/mayday/data/users/abrahamd/hofft/data/coco_spiral'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'full_coco_results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--wraps', type=float, default=6.)
    parser.add_argument('--grid-size', type=int, default=17)
    parser.add_argument('--dictionary-nodes', type=int, default=2048)
    parser.add_argument('--rank', type=int, default=112)
    parser.add_argument('--svd-tolerance', type=float, default=1e-3)
    parser.add_argument('--search-iterations', type=int, default=20)
    parser.add_argument('--model-rank', type=int, default=50)
    parser.add_argument('--final-iterations', type=int, default=20)
    parser.add_argument('--refine-iterations', type=int, default=100)
    parser.add_argument('--patch-size', type=int, default=20)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--candidate-batch', type=int, default=16)
    parser.add_argument('--metric', choices=tuple(FOCUS_METRICS), default='wavelet')
    parser.add_argument('--shot-stride', type=int, default=3, help='Match coco.py: use every third readout.')
    parser.add_argument('--coils', type=int, default=0, help='Optional SVD coil compression; 0 keeps all channels.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def run(args):
    if args.grid_size < 3 or args.wraps <= 0 or args.shot_stride < 1:
        raise ValueError('Use grid-size >= 3, positive wraps and shot stride.')
    if args.final_iterations < 1 or args.refine_iterations < 0:
        raise ValueError('Use positive final iterations and nonnegative refinement iterations.')
    started = time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The coco.py reconstruction operators require CUDA; run inside the Slurm GPU allocation.')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    def load(name):
        return torch.load(args.data_dir / f'{name}.pt', map_location=device, weights_only=True)

    trj = load('trj').float()[:, ::args.shot_stride]
    data = load('ksp').cfloat()[:, :, ::args.shot_stride]
    mps, phis = load('mps').cfloat(), load('phis').float()
    mask = (load('evals') > .85).float()
    dcf = load('dcf').float()[:, ::args.shot_stride]
    phis, mps = phis * mask, mps * mask
    data = data.flatten(1)
    coil_energy = 1.
    if args.coils and args.coils < len(mps):
        values, vectors = torch.linalg.eigh(data @ data.mH)
        compression = vectors[:, -args.coils:].mH
        coil_energy = float(values[-args.coils:].sum() / values.sum())
        data = compression @ data
        mps = (compression @ mps.flatten(1)).reshape(args.coils, *mask.shape)
    rho = trj.norm(dim=-1).flatten()
    rho /= rho.max().clamp_min(1e-12)
    g = radial_chebyshev(rho, args.degree)
    print(f'coco_spiral: image={tuple(mask.shape)}, samples={data.shape[1]}, coils={len(mps)}, '
          f'coil energy={coil_energy:.5%}; radial degree={args.degree}', flush=True)

    grid, candidates, g_nodes, left, temporal_nodes, svd_error = make_dictionary(
        args.degree, args.wraps, args.grid_size, args.dictionary_nodes, args.rank,
        args.svd_tolerance, device)
    temporal = interpolate_radius(temporal_nodes, rho)
    print(f'{len(candidates)} candidate vectors -> {len(temporal)} basis reconstructions; '
          f'dictionary error={svd_error:.3%}', flush=True)
    if svd_error > args.svd_tolerance:
        print('The rank cap did not reach the requested SVD tolerance; increase --rank.', flush=True)
    recon = CocoRecon(trj, mps, dcf, steps=args.search_iterations)
    naive = recon(data)
    basis_images = basis_reconstructions(recon, data, temporal, 'Autofocus')
    # Check actual candidate modulation at real acquisition times, including
    # the conjugation convention and the interpolation of temporal factors.
    direct_errors, modulation_errors = [], []
    for index in (0, len(candidates)//2, len(candidates)-1):
        exact = torch.exp(2j * torch.pi * (candidates[index] @ g))
        modulation_errors.append(float((left[index] @ temporal - exact).norm() / exact.norm()))
        direct = recon(data * exact)
        synthesized = (left[index] @ basis_images.flatten(1)).reshape(mask.shape)
        direct_errors.append(float((synthesized - direct).norm() / direct.norm()))
    patch_size, stride = (args.patch_size,) * 2, (args.stride,) * 2
    selected, confidence, boundaries, scores = search_patches(
        left, basis_images, candidates, grid, patch_size, stride,
        args.candidate_batch, args.metric)
    coefficients, maps, active = fit_spatial_maps(phis, selected, confidence, mask, stride)
    patch_image = synthesize_maps(maps, g_nodes, temporal_nodes, basis_images)
    patch_coefficients = coefficients.clone()
    refinement = None
    if args.refine_iterations:
        coefficients, refinement = refine_image_metric(
            coefficients, phis, g_nodes, temporal_nodes, basis_images, naive, mask,
            patch_size, stride, args.wraps, args.refine_iterations, args.metric)
        maps = (coefficients.T @ phis.flatten(1)).reshape(args.degree, *mask.shape)
    estimated = coefficients @ g
    estimated_recon = phase_reconstruction(
        phis, estimated, mps, trj, args.final_iterations, args.model_rank, 'Estimated phase')
    estimated_image = estimated_recon(data)
    del estimated_recon
    print('Autofocus complete. Evaluating against stored alphas and reference image.', flush=True)

    # Ground truth is first accessed here: autofocus above is independent of it.
    original = load('alphas').float()
    if original.shape[2:] == (1,):
        original = original.expand(-1, -1, trj.shape[1])
    else:
        original = original[:, :, ::args.shot_stride]
    truth = (original - original[:, :1]).flatten(1)
    # A best radial projection separates model-family error from search error.
    radial_coefficients = torch.linalg.lstsq(g.T.double(), truth.T.double()).solution.T.float()
    radial_truth = radial_coefficients @ g
    # Use the ORIGINAL alphas here, exactly as coco.py does for its full model.
    oracle_recon = phase_reconstruction(
        phis, original.flatten(1), mps, trj, args.final_iterations, args.model_rank, 'Known phase')
    oracle = oracle_recon(data)
    del oracle_recon
    reference = load('img_gt').cfloat()
    images = {'Uncorrected': naive, 'Patch search (approx.)': patch_image, 'Radial autofocus': estimated_image,
              'Known-phase oracle': oracle, 'Stored reference': reference}
    alpha_metrics = []
    for index in range(phis.shape[0]):
        observable = bool(active[index])
        error = float((estimated[index] - truth[index]).norm() / truth[index].norm().clamp_min(1e-20)) if observable else None
        model_error = float((radial_truth[index] - truth[index]).norm() / truth[index].norm().clamp_min(1e-20)) if observable else None
        alpha_metrics.append({'term': index, 'observable': observable,
                              'relative_l2': error, 'radial_model_relative_l2': model_error})
    metrics = {
        'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'candidate_count': len(candidates), 'search_rank': len(temporal),
        'dictionary_relative_error': svd_error,
        'actual_modulation_relative_errors': modulation_errors,
        'direct_vs_synthesized_image_relative_errors': direct_errors,
        'coil_energy_retained': coil_energy,
        'phase': phase_errors(phis, estimated, truth, mask),
        'best_radial_phase': phase_errors(phis, radial_truth, truth, mask),
        'alpha_terms': alpha_metrics,
        'magnitude_relative_l2_vs_oracle': {name: magnitude_error(image, oracle, mask) for name, image in images.items()},
        'magnitude_relative_l2_vs_stored_reference': {name: magnitude_error(image, reference, mask) for name, image in images.items()},
        'boundary_fraction_per_parameter': boundaries[:, mask[::args.stride, ::args.stride].bool()].float().mean(1).tolist(),
        'confident_in_mask_patches': int(((confidence > 0) & mask[::args.stride, ::args.stride].bool()).sum()),
        'forward_model': {'operator': 'hofft.pipelines.svd_decomp_linop', 'L': args.model_rank, 'cur_rank': 500},
        'cg': {'implementation': 'mr_recon.recons.CG_SENSE_recon', 'search_iterations': args.search_iterations, 'final_iterations': args.final_iterations},
        'image_prior_refinement': refinement,
        'gauge': 'Alphas relative to each readout start; static phase is unidentifiable by magnitude autofocus.',
        'reconstruction': 'coco.py operators: sense_linop/cufi_nufft and CG_SENSE_recon for search; svd_decomp_linop and CG_SENSE_recon for estimated and oracle images.',
        'elapsed_seconds': time.monotonic() - started,
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    image_rows = '\n'.join(
        f'| {name} | {metrics["magnitude_relative_l2_vs_oracle"][name]:.2%} | '
        f'{metrics["magnitude_relative_l2_vs_stored_reference"][name]:.2%} |'
        for name in images)
    (output / 'report.md').write_text(
        '# Full coco radial-phase autofocus\n\n'
        'Final image quality is the success criterion. Stored alphas and the reference image '
        'are used only for evaluation, never for search or refinement.\n\n'
        f'The experiment uses {args.degree} radial Chebyshev functions, {len(mps)} coils, '
        f'and {trj.shape[1]} readout(s) at the original {tuple(mask.shape)} image size. '
        'Each g_k is zero at radius zero and has max absolute value one; f_k is in phase wraps.\n\n'
        '| Reconstruction | Magnitude NRMSE vs known-phase CG | Vs stored reference |\n'
        '|---|---:|---:|\n' + image_rows + '\n\n'
        'Errors use the evaluation mask and one optimal global magnitude scale. '
        'The known-phase image uses original, unmodified alphas with '
        '`svd_decomp_linop` and `CG_SENSE_recon`, matching `coco.py`.\n\n'
        f'{len(candidates)} candidate parameter vectors were synthesized from {len(temporal)} '
        f'basis reconstructions (dictionary error {svd_error:.3%}). '
        f'The largest checked discrepancy from direct candidate CG was {max(direct_errors):.3%}. '
        'Finite-iteration CG synthesis is an approximation, checked explicitly.\n\n'
        'Patch winners initialize the spatial fit. Optional refinement maximizes the '
        'structure-weighted patch image metric, with small phase-size/range penalties, '
        'from both that fit and zero. Final images use the existing full phase-aware '
        f'HOFFT operator (L={args.model_rank}, CUR rank=500) and {args.final_iterations} CG iterations.\n\n'
        'See `images.png`, `parameter_maps.png`, `results.pt`, and `metrics.json`. '
        '`alphas.png` and phase errors are diagnostics, not the optimization target.\n'
    )
    torch.save({'alphas_estimated': estimated.reshape_as(original).cpu(),
                'alphas_reference_zeroed': truth.reshape_as(original).cpu(),
                'radial_bases': g.reshape(args.degree, *trj.shape[:-1]).cpu(),
                'spatial_to_radial_coefficients': coefficients.cpu(),
                'patch_spatial_to_radial_coefficients': patch_coefficients.cpu(),
                'f_maps_wraps': maps.cpu(), 'patch_parameters': selected.cpu(),
                'patch_confidence': confidence.cpu(), 'patch_scores': scores.cpu(),
                'images': {name: image.cpu() for name, image in images.items()}}, output / 'results.pt')
    save_figures(output, images, phis, truth, estimated, g, maps, confidence, mask)
    print(json.dumps({key: metrics[key] for key in ('phase', 'best_radial_phase',
                     'magnitude_relative_l2_vs_oracle', 'boundary_fraction_per_parameter', 'elapsed_seconds')}, indent=2), flush=True)
    print(f'Results saved to {output}', flush=True)
    return metrics


if __name__ == '__main__':
    run(parse_args())
