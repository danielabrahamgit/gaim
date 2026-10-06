"""Check CG convergence and repeatability after compare_taylor_dc.py.

Run from the repository root in the MRI environment on a GPU.
"""
import argparse
import json
from pathlib import Path
import torch
from gaim.bo_experiment import load_bo_experiment
from gaim.taylor_dc import least_squares_image

torch.set_num_threads(4)
torch.manual_seed(0)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', type=Path, default=Path('experiment/taylor_dc_fixed_seed'))
args = parser.parse_args()
p = args.output
e = load_bo_experiment()
y = e['ksp']

def build(a):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(0)
        return e['build_linop'](a)

gt = e['recon'](build(e['alphas_gt']), y)
results = {}
for name in ['unweighted_dc', 'weighted_dc']:
    r = torch.load(p / (name + '.pt'), map_location='cuda', weights_only=False)
    A = build(r['alphas'])
    A2 = build(r['alphas'])
    repeat_error = ((A.forward(r['image']) - A2.forward(r['image'])).norm() / y.norm()).item()
    del A2
    weight = A.dcf.clone() if name == 'weighted_dc' else torch.ones_like(A.dcf)
    A.dcf = torch.ones_like(A.dcf)
    class W:
        def forward(self, x): return weight.sqrt() * A.forward(x)
        def adjoint(self, k): return A.adjoint(weight.sqrt() * k)
    data = weight.sqrt() * y
    x = r['image']
    before = ((W().forward(x)-data).norm()/data.norm()).square().item()
    polished, info = least_squares_image(W(), data, initial=x, max_iter=500, tolerance=1e-6)
    after = ((W().forward(polished)-data).norm()/data.norm()).square().item()
    results[name] = dict(before=before, after=after, repeat_operator_relative_difference=repeat_error,
        magnitude_nrmse_before=((x.abs()-gt.abs()).norm()/gt.abs().norm()).item(),
        magnitude_nrmse_after=((polished.abs()-gt.abs()).norm()/gt.abs().norm()).item(),
        **info)
    print(name, results[name], flush=True)
    (p/'cg_audit.json').write_text(json.dumps(results, indent=2)+'\n')
