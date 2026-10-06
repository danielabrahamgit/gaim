"""Build a single MRI encoding operator from an existing multi-patch BO result."""
import argparse
import json
from pathlib import Path

import torch

from gaim.metrics import gradient_entropy_metric
from gaim.multipatch_bo import patch_grid_indices
from test_bo_multipatch import cpu_tree, plot_results, reconstruct_spatial_solution
from debug_cross_score_winners import aligned_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--winners', type=Path, help='Optional cross-scored results.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    h = torch.load(args.run/'history.pt', weights_only=True, map_location='cpu')
    winners = torch.load(args.winners or args.run/'reconstructions.pt', weights_only=True, map_location='cpu')
    torch.manual_seed(h['settings']['seed'])
    torch.set_num_threads(4)
    from gaim.bo_experiment import load_bo_experiment
    e = load_bo_experiment()
    args.output.mkdir(parents=True, exist_ok=True)
    phi, scale = h['phi'].to(e['bases'].device), h['phi_scale'].to(e['bases'].device)
    indices, centers, shape = patch_grid_indices(e['img_noeddy'].shape,
        (h['settings']['patch_size'],)*2, h['settings']['stride'], e['bases'].device)
    torch.testing.assert_close(centers.cpu(), h['patch_centers'])
    expected_phi = e['spatial_bases'][:, centers[:, 0], centers[:, 1]].T.double()/scale
    torch.testing.assert_close(phi, expected_phi)
    coefficients = winners['delivered_f'].to(e['bases'])
    print('Fitting selected coefficients and building final encoding operator...', flush=True)
    result = reconstruct_spatial_solution(coefficients, e, phi, scale, h['diagnostics'][-1])
    patches = result['img_final'].flatten()[indices]
    baseline_patches = e['img_noeddy'].flatten()[indices]
    gt_patches = e['img_gt'].flatten()[indices]
    scores = gradient_entropy_metric(patches, spatial_ndim=2)
    baseline = gradient_entropy_metric(baseline_patches, spatial_ndim=2)
    fg = h['foreground_centers'].to(scores.device)
    summary = dict(alphas_shape=list(result['alphas'].shape),
        expected_alphas_shape=[e['spatial_bases'].shape[0], *e['trj_size']],
        source_run=str(args.run), source_winners=str(args.winners or args.run/'reconstructions.pt'),
        foreground_mean_final_gain=(scores-baseline)[fg].mean().item(),
        foreground_final_improved_patches=int(((scores>baseline+1e-6)&fg).sum()),
        baseline_aligned_nrmse=aligned_errors(baseline_patches, gt_patches)[fg].mean().item(),
        final_aligned_nrmse=aligned_errors(patches, gt_patches)[fg].mean().item(),
        field_reduced_chi2=result['final_field_reduced_chi2'].item())
    assert summary['alphas_shape'] == summary['expected_alphas_shape']
    torch.save(result['alphas'].cpu(), args.output/'alphas.pt')
    torch.save(cpu_tree(dict(**result, temporal_bases=e['bases'], phi_scale=scale,
        input_coefficients=coefficients, patch_scores=scores, baseline_scores=baseline,
        patch_centers=centers, img_noeddy=e['img_noeddy'], img_gt=e['img_gt'], summary=summary)),
        args.output/'final_reconstruction.pt')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_results(args.output, e, winners['magnitude_mosaic'], baseline.cpu(), winners,
        h['diagnostics'][-1], shape, final_image=result['img_final'], final_scores=scores.cpu())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
