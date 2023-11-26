
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Iterable, Dict, Union

import optuna
import sklearn
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error



def scorer_hub(loss):
    if isinstance(loss, sklearn.metrics._scorer._PredictScorer):
        return loss
    elif loss in ('mae', 'mean_absolute_error'):
        return make_scorer(mean_absolute_error)
    elif loss in ('mse', 'mean_squared_error'):
        return make_scorer(mean_squared_error)
    else:
        return make_scorer(loss)


@dataclass
class HyperParameter:
    name: str
    type: str
    low: Union[float, int] = None
    high: Union[float, int] = None
    choices: Any = None

    def __post_init__(self):
        assert self.type in ('int', 'float', 'log_float', 'categorical'), "Type input must be a string, choose between: 'int', 'float', 'log_float', 'categorical'"
        assert (self.choices is not None) or ((self.low is not None) and (self.high is not None)), 'low/high or choices inputs must be not None'

@dataclass
class BayesianGridSearch:
    '''
    Class to perform Bayesian gridsearch

    :param model_class: model class that will be used to create instance. MUST have .fit, .predict methods
    :param X:
    :param y:
    :param loss: loss metric. It can be a string if implemented in scorer_hub or a function loss(y_pred, y_true) -> float.
    :param random_state: random state that will be passed to KFold. If a integer value all the trials iterations will score performance on the same KFold split
    :param optim_parameters: list of HyperParameter instance. These are the variable that will be optimized
    :param model_parameters: fixed arguments that will be passed to model constructor
    :param shuffle: True to shuffle data when splitting in folds
    :param n_jobs: parameter passed to optuna study.optimize
    :param random_state: random state passed to KFold class. Set to a number to get always the same split across optimisation steps
    '''


    model_class: Any
    X: Any
    y: Any
    loss: Any
    optim_parameters: Iterable[HyperParameter]
    model_parameters: Dict = field(default_factory=dict)
    n_folds: int = 4
    shuffle: bool = False
    n_jobs: int = 1
    random_state: int = None


    def __post_init__(self):
        self.loss = scorer_hub(self.loss)


    def tune(self, n_trials: int = 100, timeout: float = None) -> optuna.study.Study:

        study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs, timeout=timeout)
        return study


    def objective(self, trial: optuna.trial.Trial):

        # Initialize model
        optim_parameters = self.build_optim_parameters_dict(trial=trial)
        model = self.model_class(**{**optim_parameters, **self.model_parameters})

        # Score is mean of cross validations score
        cv = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        cross_val_results = cross_val_score(estimator=model, X=self.X, y=self.y, n_jobs=1, cv=cv, scoring=self.loss)

        return np.mean(cross_val_results)


    def build_optim_parameters_dict(self, trial: optuna.trial.Trial) -> Dict:

        optim_parameters_dict = {}
        for param in self.optim_parameters:
            if param.type == 'int':
                optim_parameters_dict[param.name] = trial.suggest_int(name=param.name, low=param.low, high=param.high)
            elif param.type == 'float':
                optim_parameters_dict[param.name] = trial.suggest_float(name=param.name, low=param.low, high=param.high)
            elif param.type == 'log_float':
                optim_parameters_dict[param.name] = trial.suggest_float(name=param.name, low=param.low, high=param.high, log=True)
            elif param.type == 'categorical':
                optim_parameters_dict[param.name] = trial.suggest_categorical(name=param.name, choices=param.choices)

        return optim_parameters_dict



if __name__ == '__main__':

    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    X, y = make_regression(n_samples=1000, n_features=5)

    model_class = RandomForestRegressor


    # Params
    hyper_parameters = [
        HyperParameter(name='n_estimators', type='int', low=10, high=500),
        HyperParameter(name='min_samples_split', type='float', low=0., high=1.),
        HyperParameter(name='max_features', type='categorical', choices=['sqrt', 'log2']),
    ]

    # Run grid search
    bayesian_gs = BayesianGridSearch(model_class=model_class, X=X, y=y, loss='mae',
                                   optim_parameters=hyper_parameters,
                                   n_folds=2, shuffle=True, n_jobs = 1)
    study = bayesian_gs.tune(n_trials = 10)

    print('Best parameters', study.best_params)
    best_model = RandomForestRegressor(**study.best_params)
    best_model.fit(X, y)
    print(best_model)








