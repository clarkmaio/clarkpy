from dataclasses import dataclass
import pandas as pd
from xgboostlss.model import XGBoostLSS, DMatrix
from xgboostlss.distributions.Gaussian import Gaussian
import numpy as np


@dataclass
class GaussianXGBLSS(XGBoostLSS):

    def __post_init__(self):
        self._init_model()
        return

    def _init_model(self):
        self = super(GaussianXGBLSS, self)(
                        Gaussian(
                            stabilization='None',
                            response_fn='exp',
                            loss_fn='nll'
                        )
                    )

    def fit(self, X, y):
        dtrain = DMatrix(data=X, label=y)
        super(GaussianXGBLS, self).fit(dtrain)
        return


    def predict(X, pred_type: str = 'samples', n_samples: int = 1000):
        '''
        :param pred_type: choose between 'samples', 'quantiles', 'parameters'
        :param n_samples:
        :return:
        '''

        dtest = DMatrix(data=X)
        pred = super(GaussianXGBLS, self).predict(dtest, pred_type=pred_type, n_samples=n_samples)
        return pred

    

if __name__ == '__main__':


    N = 1000
    X = np.linspace(0, 4, N)
    y = np.exp(X) + np.random.randn(N) * np.exp(X/1.5)


    mdl = GaussianXGBLSS()
    mdl.fit(X=X.reshape(-1, 1), y=y)
