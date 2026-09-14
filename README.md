# GAIM

Generalized Auto-Focus for Imperfect Models in MRI.

## Install

From this directory:

```
pip install -e .
```

## What Does the GAIM Library Do?

Classical compressed sensing uses simple forward models and strong image priors to fill in missing k-space. That is limited by how tight the image prior is, and can hallucinate structure that is not in the data.

GAIM takes the opposite route. It uses aggressive, highly efficient forward models that sample k-space well, then treats remaining image artifacts as **model imperfections** rather than undersampling. Image priors are not used to invent missing measurements; they are used to **identify the physical forward model** under which the measurements already form a high-quality image.

MRI data follow

$$
\mathbf{b} = \mathbf{A}(\boldsymbol{\theta}_\text{true})\,\mathbf{x}_\text{true} + \mathbf{n}.
$$

For a candidate imperfection parameter $\boldsymbol{\theta}$, the least-squares reconstruction is

$$
\hat{\mathbf{x}}(\boldsymbol{\theta}) = \arg\min_{\mathbf{x}}\|\mathbf{A}(\boldsymbol{\theta})\,\mathbf{x} - \mathbf{b}\|_2^2.
$$

Data consistency alone cannot select $\boldsymbol{\theta}$, because a wrong model can fit $\mathbf{b}$ about as well as the true one. GAIM therefore searches with an image-domain metric $m(\mathbf{x})$ that scores clean images higher than artifacted ones:

$$
J(\boldsymbol{\theta}) = m(\hat{\mathbf{x}}(\boldsymbol{\theta})), \qquad
\boldsymbol{\theta}_\text{GAIM} = \arg\max_{\boldsymbol{\theta}\in\boldsymbol{\Theta}} J(\boldsymbol{\theta}).
$$

The library is organized around the three design choices this search depends on:

1. **Parameterization** — how $\boldsymbol{\theta}\in\boldsymbol{\Theta}$ describes a family of model imperfections while staying low-dimensional.
2. **Metric** — which $m(\mathbf{x})$ best separates reconstructions from correct vs. incorrect forward models.
3. **Search** — how to evaluate $J(\boldsymbol{\theta})$ with as few expensive reconstructions as possible.

## Usage

Provide a parameterized encoding $\mathbf{A}(\boldsymbol{\theta})$, k-space data $\mathbf{b}$, an image metric $m$, and a set of candidate parameters:

```python
from gaim import gaim, negative_gradient_entropy

def encode(params):
    # Return a linop A(θ) with .forward / .adjoint (and optionally .normal)
    return make_forward_model(params)

result = gaim(
    encode,
    measurements,
    metric=negative_gradient_entropy,
    param_candidates=thetas,
)
print(result.params)   # θ_GAIM
print(result.image)    # x̂(θ_GAIM)
```
