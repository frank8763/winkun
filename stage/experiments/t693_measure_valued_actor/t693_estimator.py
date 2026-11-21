"""Detached Gaussian mean credit, before clipping/float32 command mapping.

No simulator, model, optimizer or learned Q is used in this module. The two
estimators agree in expectation for bounded conditional returns; no variance
ordering or finite Actor improvement follows from that identity.
"""
import math
import numpy as np

IDS = ['dev_task6_00', 'dev_task6_02', 'dev_task6_03', 'dev_task6_14',
       't502_field_sector0_offset1', 't502_field_sector2_offset1',
       't502_field_sector5_offset1', 't502_field_sector7_offset2']
SEED = 69320261121
SLOTS = 1024
SCALE = 512
SIGMA = .05


def slots():
    """Draw before inspecting lengths or outcomes; never resample invalid slots."""
    return np.random.default_rng([SEED, 0]).integers(0, SLOTS, size=(8, 4))


def draws(task, episode):
    """Separate action streams and common independent suffix stream per pair.

    MVD row axes: coordinate, plus/minus, action-coordinate.
    Score row axes: pair, first/second iid draw, action-coordinate.
    Both have four suffixes. Shared nuisance innovations are allowed because
    each derivative measure still has the required marginal distribution.
    """
    mvd = np.zeros((2, 2, 2), dtype=np.float64)
    radii = []
    for d in range(2):
        rng = np.random.default_rng([SEED, 1, task, episode, d])
        u = float(rng.random())
        radius = math.sqrt(-2 * math.log1p(-u))
        nuisance = float(rng.normal())
        mvd[d, :, 1-d] = nuisance
        mvd[d, :, d] = [radius, -radius]
        radii.append(radius)
    score = np.random.default_rng([SEED, 2, task, episode]).normal(size=(2, 2, 2))
    suffix = [[SEED, 3, task, episode, pair] for pair in range(2)]
    return dict(mvd=mvd, score=score, radii=np.asarray(radii), suffix_seeds=suffix)


def command(mu, innovation):
    return np.clip(np.asarray(mu, np.float64) + SIGMA*np.asarray(innovation), -1, 1).astype(np.float32)


def credit(method, outcomes, action_draws):
    y = np.asarray(outcomes, np.float64)
    z = np.asarray(action_draws, np.float64)
    assert y.shape == (2, 2) and z.shape == (2, 2, 2)
    assert np.isfinite(y).all() and np.isfinite(z).all()
    assert np.isin(y, [-1., 0., 1.]).all()
    if method == 'mvd':
        assert z[0, 0, 0] >= 0 and z[1, 0, 1] >= 0
        for d in range(2):
            assert z[d, 0, d] == -z[d, 1, d]
            assert z[d, 0, 1-d] == z[d, 1, 1-d]
        return (y[:, 0]-y[:, 1])/(SIGMA*math.sqrt(2*math.pi))
    assert method == 'score'
    return (((y[:, 0]-y[:, 1])[:, None] *
             (z[:, 0]-z[:, 1]))/(2*SIGMA)).mean(axis=0)


def integrate(f, lower, upper, breaks=()):
    """Deterministic split Gauss-Legendre integration, no stochastic fixture."""
    x, w = np.polynomial.legendre.leggauss(128)
    edges = [lower] + sorted(v for v in breaks if lower < v < upper) + [upper]
    return sum(float(w @ f((a+b)/2 + (b-a)*x/2))*(b-a)/2
               for a, b in zip(edges[:-1], edges[1:]))


def algebra_contract():
    errors = []
    normal = lambda z: np.exp(-z*z/2)/math.sqrt(2*math.pi)
    rayleigh = lambda r: r*np.exp(-r*r/2)
    assert abs(integrate(rayleigh, 0, 12)-1) < 1e-12
    for mu, sigma in [(.17, .4), (.95, .2), (-.91, .3)]:
        # Constants, polynomial expectations, clipping and a discontinuous
        # bounded event all refer to the same frozen test function of action.
        lo, hi = (-1-mu)/sigma, (1-mu)/sigma
        phi = lambda x: math.exp(-x*x/2)/math.sqrt(2*math.pi)
        cdf = lambda x: .5*math.erfc(-x/math.sqrt(2))
        fixtures = [
            ('constant', lambda a: np.ones_like(a), 0., []),
            ('linear', lambda a: a, 1., []),
            ('quadratic', lambda a: a*a, 2*mu, []),
            ('clipped', lambda a: a.clip(-1, 1), cdf(hi)-cdf(lo), [-1., 1.]),
            ('threshold', lambda a: (a >= .2).astype(float), phi((.2-mu)/sigma)/sigma, [.2]),
        ]
        for name, f, expected, edges in fixtures:
            plus = [(a-mu)/sigma for a in edges]
            minus = [(mu-a)/sigma for a in edges]
            jd = (integrate(lambda r: f(mu+sigma*r)*rayleigh(r), 0, 12, plus) -
                  integrate(lambda r: f(mu-sigma*r)*rayleigh(r), 0, 12, minus))/(sigma*math.sqrt(2*math.pi))
            sf = integrate(lambda z: f(mu+sigma*z)*z*normal(z)/sigma,
                           -12, 12, plus)
            error = max(abs(jd-expected), abs(sf-expected))
            assert error < 1e-10, (name, mu, jd, sf, expected)
            errors.append(dict(function=name, mu=mu, sigma=sigma, error=error))
    for n in [1, 20, 512, 1024]:
        values = np.arange(1, n+1, dtype=float)/1024
        masked = np.zeros(SLOTS);masked[:n] = values
        assert abs((SLOTS/SCALE)*masked.mean()-values.sum()/SCALE) < 1e-12
    d = draws(0, 0)
    assert np.array_equal(d['mvd'], draws(0, 0)['mvd'])
    assert not np.array_equal(d['mvd'], draws(0, 1)['mvd'])
    for method in ['mvd', 'score']:
        assert np.array_equal(credit(method, np.ones((2, 2)), d[method]), np.zeros(2))
    expected = np.array([2., -1.])/(SIGMA*math.sqrt(2*math.pi))
    assert np.array_equal(credit('mvd', [[1, -1], [0, 1]], d['mvd']), expected)
    # A near-boundary pair may map to duplicate commands. Keep those draws.
    assert command([2, 2], [.1, -.1]).dtype == np.float32
    assert np.array_equal(command([2, 2], [.1, -.1]), [1, 1])
    return dict(gaussian_derivative_cases=errors,
                fixed1024_masked512_scaling=True, deterministic_independent_stream_namespaces=True,
                constant_return_zero_credit=True, signed_coordinates_checked=True,
                clipping_float32_mapping=True, no_variance_or_performance_claim=True)
