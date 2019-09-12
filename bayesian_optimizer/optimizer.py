import GPy
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler

from acquisition import brute_force, return_ucb

# Defaults
default_kernel = GPy.kern.RBF
default_model = GPy.models.GPRegression

# TODO: Make a custom plotting function since we supply already normalized data to GPy
#       making GPy's plot slightly less useful
# TODO: NEXT put in a less complicated choice metric and test


class BayesianOptimizer:
    # should be (samples, parameters)
    def __init__(self, dims, model=default_model, kernel=default_kernel,
                 model_args={},
                 kernel_args={'variance': 1., 'lengthscale': 1.}, scale=True):
        # data
        self._X = None
        self._Y = None
        # current max result
        self._f_star = None
        self._f_star_index = None

        self.model_args = model_args
        self.kernel_args = kernel_args

        self._model = model
        self.model = None
        self.kernel = kernel(dims, **kernel_args)
        self.scale = scale

        self.xi = 3.0

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        model = self._model(self.X, self.Y, self.kernel, **self.model_args)
        model.optimize_restarts(num_restarts=10, verbose=False, parallel=True, num_processes=4)

        self.model = model

        return model

    def choose(self, bounds, samples=1, batches=1, acquisition_func=None, acquisition_func_args=None, **kwargs):
        if not acquisition_func:
            acquisition_func = return_ucb
            acquisition_func_args = [1.5, self.model]

        if batches > 1:
            exploitation = np.linspace(0.25, 2.5, batches)
        final_candidates = []

        for i in range(batches):
            if batches > 1:
                acquisition_func_args[0] = exploitation[i]

            candidates = brute_force(self.X, acquisition_func, bounds, self._transformer_x,
                                     acquisition_func_args=acquisition_func_args,
                                     samples=samples, **kwargs)
            # real_coordinates = self._transformer_x.inverse_transform(candidates)

            final_candidates.append(candidates)

        return final_candidates

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, x):
        if self.scale:
            self._transformer_x = MinMaxScaler(feature_range=(-1, 1)).fit(x)
            self._X = self._transformer_x.transform(x)
        else:
            self._X = x

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, y):
        if self.scale:
            self._transformer_y = MinMaxScaler(feature_range=(-1, 1)).fit(y)
            self._Y = self._transformer_y.transform(y)
        else:
            self._Y = y
        self._f_star_index = np.argmax(self.Y)
        self._f_star = self.Y[self._f_star_index]

    def get_model(self, xmin, xmax, points=1000):
        """
        Return Mean and SD from current GPR model.
        :param xmin: Tuple with minimum value in each dimension
        :param xmax: Tuple with maximum value in each dimension
        :param points: Integer number of points to use in the mesh along each dimension
        :return: Array, Array: Mesh used to evaluate and the mean values
        """
        lines = []
        for min, max in zip(xmin, xmax):
            lines.append(np.linspace(min, max, points))
        vals = np.meshgrid(*lines)
        vals = np.array([ar.flatten() for ar in vals]).T
        x_T = self._transformer_x.transform(vals)
        y_T_m, _ = self.model.predict(x_T, full_cov=False)

        return vals, self._transformer_y.inverse_transform(y_T_m)

    # TODO: This isn't working with the new code forms yet
    def _return_ei(self, x, exploitation=3.0):
        x = np.array(x).reshape(-1, 1)

        mean, std = self._model.predict(x, full_cov=False)
        pdf = np.exp(-(mean - self._f_star - self.xi) ** 2 / (2 * std ** 2)) / np.sqrt(2 * np.pi * std ** 2)
        cdf = erf((mean - self._f_star - self.xi) / std)

        EI = mean - self._f_star * cdf + std * pdf

        return -EI
