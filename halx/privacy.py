"""Privacy-preserving inference layer of HALX.

Two mechanisms, each with a precisely scoped guarantee:

1. ``MondrianGeneralizer`` -- multidimensional k-anonymous generalization
   (LeFevre et al., ICDE 2006) of a chosen set of quasi-identifiers (QIs).
   Every generalized QI vector that leaves the layer is shared by at least
   ``k`` training records, so a feature store or model log that keeps the
   generalized inputs cannot single out an individual flow by its QIs.

2. ``DPStandardizer`` -- the per-feature mean/std statistics that the
   preprocessing pipeline needs (and which are normally shipped with a model)
   are released under (epsilon, delta)-differential privacy with the Gaussian
   mechanism, accounted with zero-concentrated DP (Bun & Steinke, TCC 2016).

Scope: neither mechanism makes the trained network weights differentially
private; that would require DP-SGD and is out of scope (see paper, Sec. VI).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Public, data-independent transform
# --------------------------------------------------------------------------
def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


# --------------------------------------------------------------------------
# Differentially private standardization
# --------------------------------------------------------------------------
def rho_from_eps_delta(eps: float, delta: float) -> float:
    """Largest rho such that rho-zCDP implies (eps, delta)-DP.

    Uses eps = rho + 2*sqrt(rho*ln(1/delta)) (Bun & Steinke, Prop. 1.3).
    """
    L = math.log(1.0 / delta)
    s = (-2 * math.sqrt(L) + math.sqrt(4 * L + 4 * eps)) / 2.0
    return s * s


class DPStandardizer:
    """Standardize features with (eps, delta)-DP mean and variance estimates.

    Inputs are assumed already mapped by the public ``signed_log1p`` transform
    and are clipped to [-C, C]. Under replace-one adjacency with public n the
    L2 sensitivity of a clipped sum is 2C and of a sum of squares is C^2.
    The total budget rho is split evenly over the 2d released statistics.
    eps=None disables noise (non-private baseline).
    """

    def __init__(self, eps: float | None = 1.0, delta: float = 1e-5,
                 clip: float = 25.0, seed: int = 0):
        self.eps, self.delta, self.clip, self.seed = eps, delta, clip, seed

    def fit(self, X: np.ndarray) -> "DPStandardizer":
        rng = np.random.default_rng(self.seed)
        Xc = np.clip(X, -self.clip, self.clip)
        n, d = Xc.shape
        s1 = Xc.sum(0)
        s2 = (Xc ** 2).sum(0)
        if self.eps is not None:
            rho_q = rho_from_eps_delta(self.eps, self.delta) / (2 * d)
            sig1 = (2 * self.clip) / math.sqrt(2 * rho_q)
            sig2 = (self.clip ** 2) / math.sqrt(2 * rho_q)
            s1 = s1 + rng.normal(0, sig1, d)
            s2 = s2 + rng.normal(0, sig2, d)
            self.noise_std_mean_ = sig1 / n
        else:
            self.noise_std_mean_ = 0.0
        mu = s1 / n
        var = np.maximum(s2 / n - mu ** 2, 1e-4)
        self.mean_, self.std_ = mu, np.sqrt(var)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.clip(X, -self.clip, self.clip) - self.mean_) / self.std_
        return np.clip(Z, -10, 10).astype(np.float32)


# --------------------------------------------------------------------------
# Mondrian multidimensional k-anonymity
# --------------------------------------------------------------------------
@dataclass
class _Node:
    dim: int = -1
    thr: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: np.ndarray | None = None  # leaf representative (median)
    size: int = 0


class MondrianGeneralizer:
    """Strict Mondrian partitioning on the QI columns of a training matrix.

    ``fit`` builds a k-anonymous partition tree of the training QIs (every
    leaf holds >= k records); ``transform`` routes any record (train or
    unseen) to its leaf and replaces its QIs with the leaf median, so every
    released QI vector equals that of >= k training records.
    k <= 1 disables generalization.
    """

    def __init__(self, qi_idx: list[int], k: int = 10):
        self.qi_idx, self.k = list(qi_idx), int(k)

    def fit(self, X: np.ndarray) -> "MondrianGeneralizer":
        if self.k <= 1 or not self.qi_idx:
            self.root_ = None
            return self
        Q = X[:, self.qi_idx].astype(np.float64)
        self.span_ = np.ptp(Q, axis=0) + 1e-12
        self.n_leaves_ = 0
        self.root_ = self._build(Q, np.arange(len(Q)))
        return self

    def _build(self, Q, idx):
        sub = Q[idx]
        node = _Node(size=len(idx))
        if len(idx) >= 2 * self.k:
            widths = np.ptp(sub, axis=0) / self.span_
            for dim in np.argsort(-widths):
                if widths[dim] <= 0:
                    break
                col = sub[:, dim]
                thr = np.median(col)
                lm = col <= thr
                if lm.all():  # median equals max -> split strictly below
                    lm = col < thr
                nl = int(lm.sum())
                if nl >= self.k and len(idx) - nl >= self.k:
                    node.dim = int(dim)
                    node.thr = float(thr if not (col <= thr).all() else np.nextafter(thr, -np.inf))
                    node.left = self._build(Q, idx[lm])
                    node.right = self._build(Q, idx[~lm])
                    return node
        node.value = np.median(sub, axis=0)
        self.n_leaves_ += 1
        return node

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            return X
        X = X.copy()
        Q = X[:, self.qi_idx]
        out = np.empty_like(Q)
        self._route(self.root_, Q, np.arange(len(Q)), out)
        X[:, self.qi_idx] = out
        return X

    def _route(self, node, Q, idx, out):
        if node.value is not None:
            out[idx] = node.value
            return
        m = Q[idx, node.dim] <= node.thr
        if m.any():
            self._route(node.left, Q, idx[m], out)
        if (~m).any():
            self._route(node.right, Q, idx[~m], out)

    def min_group_size(self, X: np.ndarray) -> int:
        """Smallest equivalence-class size among records of X (audit helper)."""
        if self.root_ is None:
            return 1
        G = self.transform(X)[:, self.qi_idx]
        _, counts = np.unique(G, axis=0, return_counts=True)
        return int(counts.min())


class PrivacyLayer:
    """Composite layer: Mondrian k-anonymity on QIs -> DP standardization."""

    def __init__(self, qi_idx, k=10, eps=1.0, delta=1e-5, seed=0):
        self.gen = MondrianGeneralizer(qi_idx, k)
        self.std = DPStandardizer(eps, delta, seed=seed)

    def fit(self, X):
        # DP statistics are computed on the *pre-generalization* values: the
        # Mondrian partition depends on the whole training set, so computing
        # them after generalization would void the per-record sensitivity
        # bound of DPStandardizer.
        self.gen.fit(X)
        self.std.fit(X)
        return self

    def transform(self, X):
        return self.std.transform(self.gen.transform(X))
