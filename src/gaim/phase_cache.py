"""Incremental phase-subspace reconstruction, as described in plans/faster_bo.md."""

import math

import torch


class PhaseBasisRecon:
    """Cache a fixed linear reconstruction map applied to phase vectors.

    ``reconstruct(v)`` must return the complex reconstruction (or just its
    patch) of ``kspace * v`` for a fixed encoding operator and fixed k-space.
    A query outside the cached span adds one orthogonal residual and performs
    one reconstruction. Other queries only project and synthesize.

    ``tolerance`` bounds relative *phase* projection error, not image or metric
    error. Synthesis is exact in the span only for a linear reconstruction;
    truncated CG introduces an additional approximation even on cache misses.
    Keep a separate cache for every encoding operator / dataset. Full-image
    responses can be shared by all patches of that image.
    Storage grows with the number of linearly independent residuals.
    """

    def __init__(self, reconstruct, tolerance=0.01):
        if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError("tolerance must be finite and in [0, 1)")
        self.reconstruct = reconstruct
        self.tolerance = tolerance
        self.basis = None  # Rows store b_l (not their conjugates).
        self.images = None
        self.queries = 0
        self.reconstructions = 0
        self.hits = 0
        self.last_relative_residual = 0.0

    @property
    def rank(self):
        return 0 if self.basis is None else self.basis.shape[0]

    @torch.no_grad()
    def seed(self, phase, image):
        """Reuse an already computed reconstruction, e.g. the zero correction."""
        if self.rank:
            raise ValueError("Only an empty cache can be seeded")
        norm = torch.linalg.vector_norm(phase)
        if not torch.isfinite(phase).all() or norm == 0:
            raise ValueError("phase must be finite and nonzero")
        self.basis = (phase.flatten() / norm).unsqueeze(0).clone()
        self.images = (image / norm).unsqueeze(0).clone()

    @torch.no_grad()
    def project(self, phase):
        """Enrich as needed and return synthesis weights in the current basis.

        If later queries grow the cache, pad these weights with trailing zeros.
        Separating projection from synthesis permits batched, patch-specific
        synthesis from shared full-image responses.
        """
        flat = phase.flatten()
        norm = torch.linalg.vector_norm(flat)
        if not torch.isfinite(flat).all() or norm == 0:
            raise ValueError("phase must be finite and nonzero")
        if self.rank and (flat.numel() != self.basis.shape[1]
                          or flat.device != self.basis.device
                          or flat.dtype != self.basis.dtype):
            raise ValueError("phase shape, device and dtype must match the cache")
        self.queries += 1
        residual = flat.clone()
        coeffs = flat.new_zeros(self.rank)
        # Two-pass Gram-Schmidt limits loss of orthogonality in complex64.
        for _ in range(2):
            if self.rank:
                correction = self.basis.conj() @ residual
                coeffs += correction
                residual -= correction @ self.basis
        residual_norm = torch.linalg.vector_norm(residual)
        eta = (residual_norm / norm).item()
        self.last_relative_residual = eta
        numerical_floor = 10 * torch.finfo(flat.real.dtype).eps
        if not self.rank or eta > max(self.tolerance, numerical_floor):
            new_basis = residual / residual_norm
            # Keep the RHS at the query's scale: unit-norm phase vectors can
            # cause premature stopping in solvers with absolute tolerances.
            image = self.reconstruct((new_basis * norm).reshape(phase.shape)) / norm
            self.reconstructions += 1
            self.basis = (new_basis.unsqueeze(0) if self.basis is None else
                          torch.cat([self.basis, new_basis.unsqueeze(0)]))
            self.images = (image.unsqueeze(0) if self.images is None else
                           torch.cat([self.images, image.unsqueeze(0)]))
            coeffs = torch.cat([coeffs, residual_norm.reshape(1)])
        else:
            self.hits += 1
        return coeffs

    @torch.no_grad()
    def __call__(self, phase):
        coeffs = self.project(phase)
        return (coeffs @ self.images.flatten(1)).reshape(self.images.shape[1:])

    def stats(self):
        return dict(rank=self.rank, queries=self.queries,
                    reconstructions=self.reconstructions, cache_hits=self.hits,
                    tolerance=self.tolerance,
                    last_relative_residual=self.last_relative_residual)
