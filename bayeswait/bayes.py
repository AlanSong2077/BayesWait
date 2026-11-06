"""Hierarchical Bayesian runtime model.

    y = log T,  x = log(workload scale)  (x = 0 for commands without a scale)
    y = beta_0 + beta_1 x + beta_2 x^2 + eps            local power law with curvature (fixed start-up cost
                                                       bends the log-log curve)
    eps ~ Student-t(nu_obs, 0, s2)                     heavy-tailed noise (scale mixture of normals)
    beta_node ~ N(beta_parent, s2 D_d)                 nested levels: root -> ... -> exact command
    s2_node ~ scaled-Inv-chi2(nu0, s2_depth)           noise variance shrunk to the pooled value of its depth
    a node's past observations are discounted by delta_d per newer observation (local-level dynamics)

D_d is diagonal in the node's own centred coordinates (level at the node's mean scale, slope, curvature): the
level deviates with variance 1/kappa_d, slope and curvature with 1/kappa_b and 1/kappa_c; the most specific
level (one command, one scale) deviates only in level. Each node is a conjugate Normal-Inverse-Gamma update
whose prior is its parent's posterior, so it follows its parent while it has little data and itself once it
has much (credibility weighting). The posterior predictive of log T is Student-t.

Common history heuristics are special cases: delta = 1 -> mean of the history, delta -> 0 -> last value, 0 < delta < 1 ->
EMA; beta_1 = 1, 2 -> linear / quadratic growth; per-family LOO model choice -> shrinkage of slope and curvature
to the parent's. Hyper-parameters are set by maximising the log predictive density of held-out early calls.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats

P = 3   # features (1, x, x^2)

def _phi(x): return np.array([1.0, x, x * x])

class HB:
    def __init__(self, kappa=(1.0,), delta=(1.0,), kappa_b=1.0, kappa_c=10.0, nu0=4.0, nu_obs=4.0,
                 mu0=math.log(1000.0), s20=4.0, scaled=True):
        self.scaled = scaled                         # False: no command has a scale -> scalar (level-only) model
        self.kappa = list(kappa); n = len(self.kappa)
        self.delta = list(delta) if len(delta) == n else [delta[0]] * n
        self.kappa_b, self.kappa_c, self.nu0, self.nu_obs = kappa_b, kappa_c, nu0, nu_obs
        self.m0 = np.array([mu0, 0.0, 0.0]); self.s20 = s20
        self.st = {}                                   # node -> [XtX (3x3), Xty (3), yty]
        self.pool = [[1.0, math.log(s20)] for _ in range(n)]   # per depth: running mean of log squared residual

    # ---------------------------------------------------------------- internals
    def _s2p(self, d):
        """pooled noise variance of depth d, geometric-mean estimator (E log chi2_1 = -1.2704)"""
        return math.exp(self.pool[d][1] / self.pool[d][0] + 1.2704)

    def _D(self, d, last, xb):
        A = np.array([[1.0, -xb, xb * xb], [0.0, 1.0, -2.0 * xb], [0.0, 0.0, 1.0]])
        dev = np.diag([1.0 / self.kappa[d], 0.0 if last else 1.0 / self.kappa_b, 0.0 if last else 1.0 / self.kappa_c])
        return A @ dev @ A.T

    def _walk(self, chain, x):
        """posterior of every node on the chain: list of (mean, cov factor, s2, W)"""
        m, Pi, s2, W = self.m0, np.diag([100.0, 100.0, 100.0]), self.s20, 0.0
        out, n = [], len(chain)
        for d, node in enumerate(chain):
            s = self.st.get(node)
            seen = s is not None and s[0][0, 0] > 0
            xb = s[0][0, 1] / s[0][0, 0] if seen else x
            C = Pi + self._D(d, d == n - 1, xb)
            if not seen: Pi = C                                    # unseen node: parent + node deviation
            else:
                XtX, Xty, yty = s; W = XtX[0, 0]
                Ci = np.linalg.inv(C); Pr = Ci + XtX; Pi = np.linalg.inv(Pr)
                mp = Pi @ (Ci @ m + Xty)
                ss = max(yty + m @ Ci @ m - mp @ Pr @ mp, 0.0)
                m, s2 = mp, (self.nu0 * self._s2p(d) + ss) / (self.nu0 + W)
            out.append((m, Pi, s2, W))
        return out

    # ---------------------------------------------------------------- API
    def _walk1(self, chain):
        m, Pi, s2, W = self.m0[0], 100.0, self.s20, 0.0; out = []
        for d, node in enumerate(chain):
            s = self.st.get(node); C = Pi + 1.0 / self.kappa[d]
            if s is None or s[0] <= 0: Pi = C
            else:
                W, Sy, Syy = s; Pr = 1.0 / C + W; Pi = 1.0 / Pr; mp = Pi * (m / C + Sy)
                ss = max(Syy + m * m / C - mp * mp * Pr, 0.0); m, s2 = mp, (self.nu0 * self._s2p(d) + ss) / (self.nu0 + W)
            out.append((m, Pi, s2, W))
        return out

    def _update1(self, chain, y):
        post = self._walk1(chain)
        for d, node in enumerate(chain):
            s = self.st.get(node); w = 1.0
            if s is None: s = self.st[node] = [0.0, 0.0, 0.0]
            elif s[0] > 0:
                m, Pi, s2, W = post[d]; r2 = (y - m) ** 2 / (1.0 + Pi)
                pl = self.pool[d]; pl[0] = 0.999 * pl[0] + 1.0; pl[1] = 0.999 * pl[1] + math.log(max(r2, 1e-12))
                w = (self.nu_obs + 1.0) / (self.nu_obs + r2 / s2); dl = self.delta[d]
                s[0] *= dl; s[1] *= dl; s[2] *= dl
            s[0] += w; s[1] += w * y; s[2] += w * y * y

    def predict(self, chain, x=0.0):
        """chain: node keys from the root to the most specific. -> (loc, scale, dof) of the Student-t of log T"""
        if not self.scaled:
            m, Pi, s2, W = self._walk1(chain)[-1]; return float(m), math.sqrt(s2 * (1.0 + Pi)), self.nu0 + W
        m, Pi, s2, W = self._walk(chain, x)[-1]; f = _phi(x)
        return float(f @ m), math.sqrt(s2 * (1.0 + f @ Pi @ f)), self.nu0 + W

    def update(self, chain, x, y):
        if not self.scaled: return self._update1(chain, y)
        post = self._walk(chain, x); f = _phi(x)
        for d, node in enumerate(chain):
            s = self.st.get(node); w = 1.0
            if s is None: s = self.st[node] = [np.zeros((P, P)), np.zeros(P), 0.0]
            elif s[0][0, 0] > 0:
                m, Pi, s2, W = post[d]
                r2 = (y - f @ m) ** 2 / (1.0 + f @ Pi @ f)
                pl = self.pool[d]; pl[0] = 0.999 * pl[0] + 1.0; pl[1] = 0.999 * pl[1] + math.log(max(r2, 1e-12))
                w = (self.nu_obs + 1.0) / (self.nu_obs + r2 / s2)      # t-noise: posterior precision weight
                dl = self.delta[d]; s[0] *= dl; s[1] *= dl; s[2] *= dl
            s[0] += w * np.outer(f, f); s[1] += w * y * f; s[2] += w * y * y

    @staticmethod
    def logpdf(y, loc, scale, dof): return float(stats.t.logpdf(y, dof, loc, scale))

def fuse(pred, y_prior, s2_prior):
    """combine the predictive (loc, scale, dof) with an independent noisy estimate of log T (e.g. an LLM prior)
    of variance s2_prior, by precision weighting"""
    loc, sc, dof = pred; v = sc * sc
    w = s2_prior / (v + s2_prior)
    return w * loc + (1 - w) * y_prior, math.sqrt(v * s2_prior / (v + s2_prior)), dof

def keller_profile(loc, scale, dof, horizon_ms=6e8, first_ms=5.0):
    """(t grid, N_1(t)) with N_1 = integral of sqrt(h(t)/2); for check cost c the cumulative checking density is
    N_1 / sqrt(c)"""
    t = np.geomspace(first_ms, horizon_ms, 6000); z = (np.log(t) - loc) / scale
    f = stats.t.pdf(z, dof) / (scale * t); S = stats.t.sf(z, dof)
    n = np.sqrt(np.maximum(f / np.maximum(S, 1e-300), 0.0) / 2.0)
    return t, np.concatenate([[0.0], np.cumsum(0.5 * (n[1:] + n[:-1]) * np.diff(t))])

def keller_checks(loc, scale, dof, c, profile=None):
    """Keller (1974): for delay cost 1 per ms and cost c per check the optimal checking density is
    n(t) = sqrt(h(t) / 2c), h = hazard rate of the predictive runtime distribution; the k-th check is placed
    where the cumulative density N(t) reaches k - 1/2."""
    t, N1 = profile if profile is not None else keller_profile(loc, scale, dof)
    N = N1 / math.sqrt(c); ks = np.arange(0.5, N[-1], 1.0)
    return list(np.interp(ks, N, t)) if len(ks) else [float(math.exp(loc))]
