# Theory notes

## 1. Problem
A tool call starts at time 0 and finishes at an unknown time T > 0. The agent can only learn whether it has
finished by checking; a check at time t costs c (the price of one agent round, in the same unit as time) and
reveals 1{T <= t}. A schedule is an increasing sequence t_1 < t_2 < ...; the call is detected at the first
t_k >= T. Objective: minimise

    J = E[ t_K - T ] + c E[K],          K = min{k : t_k >= T}.

Given a predictive distribution F of T (density f, survival S = 1 - F, hazard h = f / S) this is the classical
optimal-checking problem (Barlow, Hunter & Proschan 1963).

## 2. Estimation: hierarchical Bayesian model
y = log T, x = log(scale) (0 if none). For a command with hierarchy path v_0 (root) -> ... -> v_L (exact):

    y = phi(x)' beta_{v_L} + eps,  phi(x) = (1, x, x^2),  eps ~ t_{nu_obs}(0, s2)
    beta_{v_d} | beta_{v_{d-1}} ~ N(beta_{v_{d-1}}, s2 D_d)

D_d is diagonal in the node's centred coordinates (level at the node's mean scale, slope, curvature) with
entries 1/kappa_d, 1/kappa_b, 1/kappa_c; the exact level deviates only in level. Each node holds discounted
sufficient statistics (sum of w phi phi', w phi y, w y^2), multiplied by delta at every new observation of that
node (power discounting, as in West & Harrison's discount dynamic linear models). The t-noise is handled as a
scale mixture: an observation enters with weight w = (nu_obs + 1) / (nu_obs + r^2 / s2), r its one-step-ahead
standardised residual. Walking down the path, each node's Normal-Inverse-Gamma posterior is the next node's
prior; an unseen node inherits its parent's posterior plus its deviation covariance. The noise scale of a node
is shrunk towards the pooled one-step-ahead residual variance of its depth (estimated by exp(mean log r^2 +
1.2704); 1.2704 = -E[log chi2_1] makes the estimator unbiased for Gaussian residuals and robust to outliers).
Posterior predictive: log T ~ t_nu(mu, s), nu = nu0 + effective sample size.

Special cases (the usual history heuristics): delta = 1 -> shrunk mean; delta -> 0 -> last value; 0 < delta < 1
-> EMA; slope 1 / 2 -> linear / quadratic growth in the input size; kappa_d -> infinity -> pooled estimate of the
parent level. Hyper-parameters (kappa_d per level, kappa_b, kappa_c, delta, nu0, nu_obs) maximise the log
predictive density of the tuning calls (empirical Bayes).

Component ablations (experiments/results/components_*.txt): the hierarchy and the discount are essential on
real traces, the t-noise on prediction accuracy, the curvature on workloads with a fixed start-up cost.

## 3. Decision: Keller's checking density
Keller (1974) relaxes the schedule to a smooth checking density n(t) (checks per unit time). If the job ends at
t, the expected delay is about 1 / (2 n(t)) and the number of checks so far is N(t) = int_0^t n, so

    J[n] ~ int_0^inf f(t) [ 1/(2 n(t)) + c N(t) ] dt = int_0^inf [ f(t) / (2 n(t)) + c S(t) n(t) ] dt.

Pointwise minimisation gives

    n*(t) = sqrt( h(t) / (2c) ),        h = f / S.                                    (Keller)

The k-th check is placed where N*(t) = k - 1/2. The schedule is closed form given the predictive distribution.

## 4. Properties

**Proposition 1 (scale equivariance).** Let log T ~ t_nu(mu, s). The Keller times satisfy
t_k(mu, s, nu, c) = e^mu * tau_k(s, nu, c e^-mu): the schedule depends on the job only through its typical
duration e^mu, its log-scale uncertainty s and tail index nu, and on the cost only through the dimensionless
ratio c / e^mu.
*Proof.* Substituting t = e^mu u, h_mu(t) = h_0(u) e^-mu, so n*(t) dt = sqrt(h_0(u) / (2 c e^-mu)) du.

**Proposition 2 (tail spacing).** For large t the Keller gap Delta(t) = 1 / n*(t) satisfies

    log-t predictive:      Delta(t) ~ sqrt( 2 c t (log t - mu) / nu ),
    log-normal (nu -> inf): Delta(t) ~ s sqrt( 2 c t / (log t - mu) ).

*Proof.* With z = (log t - mu)/s the t density satisfies g(z) ~ K z^-(nu+1) and its survival K z^-nu / nu, so
h(t) = g(z) / (s t S) ~ nu / (t (log t - mu)). For the normal, Mills' ratio gives h(t) ~ (log t - mu) /
(s^2 t). Insert in Delta = sqrt(2c / h). (Numerically verified: ratio to the asymptote 1.01 at t = 10^7 ms.)

**Corollary (why the classical heuristics fail in the tail).** The optimal gap grows like sqrt(t) up to a log
factor. A fixed interval d has gap O(1): on a long job it checks O(T / d) times instead of O(sqrt(T)). Exponential
backoff has gap proportional to t: its detection delay on a long job is O(T) instead of O(sqrt(c T log T)).
Polling every 100 ms after a point estimate is a fixed interval and inherits its failure (observed: 5-100 checks
per call on real traces).

**Robustness (empirical).** The exact optimum (dynamic programme on the predictive) is within a few percent
of Keller under the model (median gap +0.6% at c = 1 s, +2.4% at c = 5 s; +10-20% at c <= 200 ms), but on real
data it is much worse, because it trusts the predictive tail; Keller's hazard-based spacing degrades gracefully
when the tail is misspecified (experiments/results/ablate_*.txt).
