import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.gaussian_process as gp
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from sklearn.neighbors import KernelDensity

class GHGPredictor:

    bics_list_ = None

    def __init__(self):
        self.df_ = None
        self.X_ = None
        self.y_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.X_test_ = None
        self.y_test_ = None
        self.models_ = dict()
        self.models_2_ = dict()

    @classmethod
    def load_bics_list(cls, PATH):
        cls.bics_list_ = pd.read_csv(PATH, index_col="Index")

    def load_data(self, PATH):
        self.df_ = pd.read_csv(PATH, index_col="Ticker")

    def train_test_split(self):
        pass

    def compare_intensity_distribution(self, years, params=None, bics=None, pct=False):
        if not params:
            params = {
                "bins": 20,
                "alpha": 0.5,
                "histtype": "step",
                "range": [0, 0.4]
            }

        # store the data based on YEAR
        _memo = []
        for year in years:
            conditions = (self.df_['Year'] == year) & (self.df_['BICS_4'] == bics if bics else 1)
            _memo.append(self.df_.loc[conditions, "Intensity"])

        # plot the histogram(s) for inputed year(s)
        color = ["red", "blue", "orange", "green", "pink"]
        for i, year in enumerate(_memo):
            # set the weights if y is displayed in percentage form
            weights = (np.ones(len(year)) / len(year)) if pct else None
            plt.hist(year, edgecolor=color[i], weights=weights, **params)

        plt.legend(years)
        plt.xlabel("Intensity")
        ylabel = ("%" if pct else "#") + " of companies"
        plt.ylabel(ylabel)
        plt.title(bics if bics else "ALL")

    @ignore_warnings(category=ConvergenceWarning)
    def train_models(self, nrestarts=5):
        if self.bics_list_ is not None:
            #
            _kernel = gp.kernels.RBF([50, 50, 50], [(1e-5, 100), (1e-5, 1e6), (1e-5, 100)])
            __kernel = gp.kernels.ConstantKernel()
            ___kernel = gp.kernels.WhiteKernel()
            for _, bics in self.bics_list_.iterrows():
                _sector = bics[0]
                print(_sector)
                self.models_[_sector] = gp.GaussianProcessRegressor(kernel=_kernel*__kernel+___kernel, n_restarts_optimizer=nrestarts, alpha=0.1,
                                                                    normalize_y=True)
                _df = self.df_[self.df_['BICS_4'] == _sector]
                _X, _y = np.log(_df[['Revenue', 'Market Cap', "GHG Scope 1P"]].values), np.log(_df["Intensity"].values)
                self.models_[_sector].fit(_X, _y)

    def train_models_2(self, fit_intercept=True):
        if self.bics_list_ is not None:
            quantiles = np.arange(0.01, 1, 0.01)
            for _, bics in self.bics_list_.iterrows():
                _sector = bics[0]
                _df = self.df_[self.df_['BICS_4'] == _sector]
                for q in quantiles:
                    self.models_2_[_sector, q] = QuantileRegressor(quantile=q, alpha=0.1, solver="highs", fit_intercept=fit_intercept)
                    _X, _y = _df['GHG Scope 1P'].values[:, np.newaxis], _df["GHG Scope 1C"].values
                    # _X, _y = _df[['Revenue', "GHG Scope 1P"]].values, _df["GHG Scope 1C"].values
                    self.models_2_[_sector, q].fit(_X, _y)

    @staticmethod
    def _plot_distribution(_mean, _std, ci=0.95, y_true=None, ax=None):
        # X-axis range
        _lower = np.exp(_mean - 3 * _std)
        _upper = np.exp(_mean + 3 * _std)
        _x = np.linspace(_lower, _upper, 1000)

        # plot lognormal
        _scale = np.exp(_mean)
        _params = {
            "s": _std,
            "scale": _scale
        }

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 10))

        # predicted distribution
        ax.plot(_x, stats.lognorm.pdf(_x, **_params))

        # predicted value (median)
        ax.vlines(_scale, 0, stats.lognorm.pdf(_scale, **_params), color='red')

        # true value
        if y_true is not None:
            ax.scatter(y_true, stats.lognorm.pdf(y_true, **_params), s=50, color="darkorange")

        # confidence interval
        _tmp = stats.norm.ppf(1 - (1 - ci) / 2, _mean, _std) - _mean
        _lb, _ub = np.exp(_mean - _tmp), np.exp(_mean + _tmp)
        ax.vlines(_lb, 0, stats.lognorm.pdf(_lb, **_params), color='lightblue', linestyles='dashed')
        ax.vlines(_ub, 0, stats.lognorm.pdf(_ub, **_params), color='lightblue', linestyles='dashed')

        # plot setting
        ax.set_ylim(bottom=0)
        ax.set_xticks([_lb, _scale, _ub])
        ax.set_xlabel("GHG Scope 1 Emission Intensity (mt/$1000)")
        ax.set_yticks([])


    @staticmethod
    def _table_distribution(_mean, _std):
        res = []
        for ci in (0.8, 0.6, 0.4, 0.2):
            _tmp = stats.norm.ppf(1 - (1 - ci) / 2, _mean, _std) - _mean
            res.append(np.exp(_mean - _tmp))
            res.append(np.exp(_mean + _tmp))
        res.append(np.exp(_mean))
        return sorted(res)

    def predict(self, X, y_true=None, ci=0.95, plot=True, ax=None):

        _X = np.array(X)
        _sector = _X[2]
        _mean, _std = self.models_[_sector].predict(np.log(_X[[0, 1, 3]].astype(np.float64)).reshape(1, -1),
                                                    return_std=True)
        _mean = _mean[0]
        _std = _std[0]

        if plot:
            self._plot_distribution(_mean, _std, ci, y_true, ax)
            return _mean, _std
        else:
            return self._table_distribution(_mean, _std)

    def predict_2(self, X, y_true=None, plot=True, ax=None, bw=None):

        quantiles = np.arange(0.01, 1, 0.01)
        _X = np.array(X)
        _sector = _X[2]

        _res = []
        for q in quantiles:
            # _res.append(self.models_2_[_sector, q].predict(_X[0].astype(np.float64).reshape(1, -1))[0])
            _res.append(self.models_2_[_sector, q].predict([[float(_X[3])]])[0])

        scope = _res[-1] - _res[0]

        bw = bw if bw else scope / 20

        kde = KernelDensity(bandwidth=bw)
        X = np.array(_res)[:, None]
        _res = sorted(_res)
        kde.fit(X)
        _X_ = np.linspace(_res[0] - 3 * bw, _res[-1] + 3 * bw, 1000)
        log_prob = kde.score_samples(_X_[:, None])

        if plot:
            ax.fill_between(_X_ / float(_X[0]) * 1000, np.exp(log_prob), alpha=0.5)
            ax.set_yticks([])

    # def train_models_2(self, fit_intercept=True):
    #     if self.bics_list_ is not None:
    #         quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #         for _, bics in self.bics_list_.iterrows():
    #             _sector = bics[0]
    #             _df = self.df_[self.df_['BICS_4'] == _sector]
    #             for q in quantiles:
    #                 self.models_2_[_sector, q] = QuantileRegressor(quantile=q, alpha=0.1, solver="highs", fit_intercept=fit_intercept)
    #                 _X, _y = _df['GHG Scope 1P'].values[:, np.newaxis], _df["GHG Scope 1C"].values
    #                 # _X, _y = _df[['Revenue', "GHG Scope 1P"]].values, _df["GHG Scope 1C"].values
    #                 self.models_2_[_sector, q].fit(_X, _y)

    # def predict_2(self, X, y_true=None, plot=True, ax=None):
    #
    #     quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     _X = np.array(X)
    #     _sector = _X[2]
    #
    #     _res = []
    #     for q in quantiles:
    #         # _res.append(self.models_2_[_sector, q].predict(_X[0].astype(np.float64).reshape(1, -1))[0])
    #         _res.append(self.models_2_[_sector, q].predict([[float(_X[3])]])[0])
    #
    #     _res = [0] + _res + [2*_res[-1] - _res[-2]]
    #     _res = np.array(_res) / float(_X[0]) * 1000
    #
    #     if plot:
    #         _x, _y = [], []
    #         for l, r in zip(_res[:-1], _res[1:]):
    #             if r - l >= 1e-10:
    #                 h = 0.1 / (r - l)
    #             else:
    #                 h = _y[-1]
    #             _x.append(l + (r - l) / 2)
    #             _y.append(h)
    #
    #         _x = [2*_x[0] - _x[1]] + _x + [2*_x[-1] - _x[-2]]
    #         _y = [0] + _y + [0]
    #
    #         print(_x)
    #
    #         ax.set_xlim([_x[0], _x[-1]])
    #         _fit = interp1d(_x, _y, kind='linear')
    #         _x_ = np.linspace(_x[0], _x[-1], 5000)
    #         ax.plot(_x_, _fit(_x_), color='blue')
    #         ax.vlines(_x[1], 0, _y[1], color='lightblue', linestyles='dashed')
    #         ax.vlines(_res[5], 0, _fit(_res[5]), color='red')
    #         if y_true and _x[-1] > y_true > _x[0]:
    #             ax.scatter(y_true, _fit(y_true), color='orange')
    #         ax.vlines(_x[10], 0, _y[10], color='lightblue', linestyles='dashed')
    #         ax.set_yticks([])
    #         ax.set_xticks([_x[1], _res[5], _x[10]])
    #         ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #
    #         _, y_end = ax.get_ylim()
    #         ax.set_ylim([0, y_end])