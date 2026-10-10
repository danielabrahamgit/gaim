import pytest
import torch
from gaim.implicit_optim import implicit_objective_gradient, implicit_objective_trust


@pytest.mark.parametrize('samples', [5, 13])
@pytest.mark.parametrize('dc_weight', [0., .7, 3.])
def test_adjoint_gradient_against_dense_autograd(samples, dc_weight):
    torch.manual_seed(17)
    n=8
    matrix=torch.randn(samples,n,dtype=torch.complex128)
    phi=torch.randn(2,n,dtype=torch.float64)
    bases=torch.randn(3,samples,dtype=torch.float64)
    y=torch.randn(samples,dtype=torch.complex128)
    w=torch.linspace(0,2,samples,dtype=torch.float64)
    class Operator:
        dcf=w
        def __init__(self, a):self.matrix=matrix*torch.exp(-2j*torch.pi*(a.T@phi))
        def forward(self,x):return self.matrix@x
        def adjoint(self,y):return self.matrix.mH@(w*y)
    F=(.01*torch.randn(2,3,dtype=torch.float64)).requires_grad_()
    A=Operator(F@bases)
    x=torch.linalg.solve(A.matrix.mH@(w[:,None]*A.matrix)+.2*torch.eye(n,dtype=matrix.dtype),A.adjoint(y))
    metric=lambda z:-(z.abs().square()+.1).sqrt().sum()
    value=metric(x)/1.3-dc_weight*(w*(A.forward(x)-y).abs().square()).sum()/(w*y.abs().square()).sum()/.4
    exact,=torch.autograd.grad(value,F)
    actual=implicit_objective_gradient(A,x.detach(),y,phi,bases,metric,
             regularization=.2,cg_tolerance=1e-11,metric_scale=1.3,dc_weight=dc_weight,dc_scale=.4)
    torch.testing.assert_close(actual,exact,rtol=1e-7,atol=1e-7)


def test_optimization_is_monotone_and_returns_consistent_coefficients():
    torch.manual_seed(4)
    phi=torch.randn(2,6,dtype=torch.float64)
    bases=torch.randn(2,10,dtype=torch.float64)
    matrix=torch.randn(10,6,dtype=torch.complex128)
    class Operator:
        dcf=torch.linspace(.3,1.,10,dtype=torch.float64)
        def __init__(self,a):self.matrix=matrix*torch.exp(-2j*torch.pi*(a.T@phi))
        def forward(self,x):return self.matrix@x
        def adjoint(self,y):return self.matrix.mH@(self.dcf*y)
    y=Operator(.02*bases).forward(torch.randn(6,dtype=torch.complex128))
    result=implicit_objective_trust(phi,bases,y,Operator,lambda x:x.abs().sum()*0,
             dc_weight=1.,regularization=.1,cg_tolerance=1e-11,max_steps=12,verbose=False)
    assert result['final_score']>result['initial_score']
    accepted=[r for r in result['history'] if r['accepted']]
    assert all(r['actual_gain']>0 and r['phase_bound']<=r['radius']*(1+1e-6) for r in accepted)
    torch.testing.assert_close(result['alphas'],result['F']@bases)
