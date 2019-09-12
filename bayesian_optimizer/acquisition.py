from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import cdist
import numpy as np


# Basin hopping is carried over from 1D testing. It will need to be modified to conform to new formatting before it
#  can be used.
def basin_hopping(X, acquisition_func, bounds, x0=None, acquisition_func_args=None, **kwargs):
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": ((bounds.xmin, bounds.xmax),)}
    if acquisition_func_args:
        minimizer_kwargs['args'] = acquisition_func_args
    if not x0:
        x0 = np.array(np.random.uniform(low=bounds.xmin, high=bounds.xmax))

    res = basinhopping(acquisition_func, x0=(*x0,), T=5.0, stepsize=4,
                       accept_test=bounds, minimizer_kwargs=minimizer_kwargs)
    print('optimizer result, new x:', res['x'])
    if np.any(np.abs(X - res['x']) < np.abs(bounds.xmax - bounds.xmin) / 1e4):
        print('Optimizer found existing point. Selecting next sample at random.')
        result = np.array(np.random.uniform(low=bounds.xmin, high=bounds.xmax))
    else:
        result = res['x']

    return result


def brute_force(X, acquisition_func, bounds, scale, points=100,
                acquisition_func_args=None, samples=1, hierarchy=False, **kwargs):
    # Batches over exploitation parameter
    # Samples are selections in a batch

    lines = []
    for min, max in zip(bounds.xmin, bounds.xmax):
        lines.append(np.linspace(min, max, points))
    mesh = np.meshgrid(*lines)
    # Organize input into objective function as
    mesh = np.array([ar.flatten() for ar in mesh]).T
    # Send scaled mesh to model
    aspirants = acquisition_func(scale.transform(mesh), *acquisition_func_args)

    if not hierarchy:
        if 'T' in kwargs:
            final_candidates, w = parameterized_selection(X, mesh, aspirants, samples, T=kwargs['T'])
        else:
            final_candidates, w = parameterized_selection(X, mesh, aspirants, samples)
    else:
        final_candidates = hierarchical_selection(mesh, aspirants, samples)

    return final_candidates


def parameterized_selection(X, candidates, quality, n, T=0.1):
    # T: Scaling factor for weights, set smaller for more exploitation, larger for more exploration

    # Collect distance of candidates to all points
    choices = np.vstack([X, candidates])
    distances = np.sum(cdist(choices, choices, metric='euclidean'), axis=1).reshape(choices.shape[0], -1)

    # Calculate normalized distance metric
    var_dist = distances[X.shape[0]:] / np.std(distances)
    var_dist = var_dist / np.max(var_dist)

    # Use weights to choose candidates to pass
    quality -= np.min(quality)
    quality /= np.max(quality)
    weight = np.exp(var_dist * quality / T)
    weight /= np.sum(weight)

    selection_indices = np.random.choice(candidates.shape[0], n, p=weight.flatten(), replace=False)

    return candidates[selection_indices, :], np.array(var_dist * quality).flatten()


def hierarchical_selection(candiates, quality, n):
    selection_indices = np.argpartition(quality.flatten(), -n)[-n:]

    return candiates[selection_indices, :], None

class MyBounds(object):
    def __init__(self, xmin=[-1., ], xmax=[1., ]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

# TODO: generalize to arbitray dimensiosn


def return_ucb(x, k, model):
    mu, sigma = model.predict(x, full_cov=False)

    return mu - k * sigma
