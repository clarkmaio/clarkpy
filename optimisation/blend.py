import cvxpy as cvx
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Iterable
import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

@dataclass
class Result:
    weights: np.ndarray
    intercet: float
    value: float

@dataclass
class Blend:
    convex: bool = True
    fit_intercept: bool = True
    lower_bound: Union[float, List[float]] = 0.
    upper_bound: Union[float, List[float]] = 1.
    lp_norm: int = 1

    def __post_init__(self):
        self.__intercept = None
        self.__coef = None
        self.__constraints = None
        self.__variable: cvx.Variable = None
        self.__is_trained: bool = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], sample_weights: Iterable = None) -> Result:
        '''
        Fit the model
        '''

        # Check input
        assert X.shape[0] == y.shape[0], 'X and y must have the same number of rows'
        assert X.shape[1] > 0, 'X must have at least one column'

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values


        # Define variables
        n = X.shape[1] + self.fit_intercept
        self.__variable = cvx.Variable(n)

        # Build yhat
        y_hat = self._build_yhat(X)

        # Define objective
        if sample_weights is None:
            objective = cvx.Minimize(cvx.norm(y_hat - y, p=self.lp_norm))
        else:
            objective = cvx.Minimize(cvx.norm(cvx.multiply(sample_weights, y_hat - y), p=self.lp_norm))

        # Define constraints
        self.__constraints = self._build_constraints()

        # Solve problem
        problem = cvx.Problem(objective=objective, constraints=self.__constraints)
        problem.solve()

        # Return result
        if self.fit_intercept:
            self.__intercept = self.__variable.value[0]
            self.__coef = self.__variable.value[1:]
        else:
            self.__intercept = 0
            self.__coef = self.__variable.value

        self.__is_trained = True
        return Result(weights=self.__coef, intercet = self.__intercept, value=problem.value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict y
        '''

        if self.fit_intercept:
            return X @ self.__coef + self.__intercept
        else:
            return X @ self.__coef

    def _build_constraints(self):
        constraints = []

        if self.lower_bound is not None:
            if self.fit_intercept:
                constraints.append(self.__variable[1:] >= self.lower_bound)
            else:
                constraints.append(self.__variable >= self.lower_bound)

        if self.upper_bound is not None:
            if self.fit_intercept:
                constraints.append(self.__variable[1:] <= self.upper_bound)
            else:
                constraints.append(self.__variable <= self.upper_bound)

        if self.convex:
            if self.fit_intercept:
                constraints.append(cvx.sum(self.__variable[1:]) == 1)
            else:
                constraints.append(cvx.sum(self.__variable) == 1)

        return constraints

    def _build_yhat(self, X: np.ndarray) -> np.ndarray:
        '''
        Build yhat
        '''
        if self.fit_intercept:
            y_hat = X @ self.__variable[1:] + self.__variable[0]
        else:
            y_hat = X @ self.__variable

        return y_hat

    def save(self, path: str) -> None:
        '''
        Save model
        '''

        model_params = {
            '__intercept': self.__intercept,
            '__coef': self.__coef,
            '__constraints': self.__constraints,
            '__is_trained': self.__is_trained,

            'convex': self.convex,
            'fit_intercept': self.fit_intercept,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'lp_norm': self.lp_norm

        }

        # Save pickle
        with open(path, 'wb') as f:
            pickle.dump(model_params, f)


    @staticmethod
    def load(path: str):
        '''
        Load model
        '''

        with open(path, 'rb') as f:
            model_params = pickle.load(f)

        # Create instance and initialize
        blend_mdl = Blend()
        for key, item in model_params.itemize():
            blend_mdl.__setattr__(name=key, value=item)
        return blend_mdl


    @property
    def coef_(self):
        return self.__coef

    @property
    def intercept_(self):
        return self.__intercept

    @property
    def constraints_(self):
        return self.__constraints


    def diagnostic_plot(self, x: Iterable, models: Union[np.ndarray, pd.DataFrame], target: Iterable, model_names: Iterable = None) -> None:
        '''
        Create a diagnostic plot to visualize models, bias and weights of Blend
        :param x: feature used as x axis. A.e. for timeseries it should be time
        :param models: np.ndarray containing model predictions
        :param target: target
        :param blend: Fitted blend class
        '''

        assert self.__is_trained

        if isinstance(models, pd.DataFrame):
            self.diagnostic_plot(x=x, models=models.values, target=target, model_names=models.columns)
            return

        if model_names is None:
            model_names = [f'Model {i}' for i in range(models.shape[1])]

        y_hat = self.predict(models)
        n_models = models.shape[1]

        # Prepare grid plot
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('Blend diagnostic', fontweight='bold')

        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=1)
        ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2, rowspan=1)
        ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
        ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)

        # Model plot
        ax1.plot(x, target, 'o', color='black', label='Target')
        for i in range(n_models):
            ax1.plot(x, models[:, i], 'o', label=model_names[i], alpha=.2, color=f'C{i}')
        ax1.plot(x, y_hat, 'x', color='red', label='Blend')
        ax1.grid(linestyle=':')
        ax1.legend(loc='upper left')
        ax1.set_title('Model', fontweight='bold')

        # Bias plot
        for i in range(n_models):
            ax2.plot(x, target - models[:, i], 'o', label=f'Bias {model_names[i]}', alpha=.2, color=f'C{i}')
        ax2.plot(x, target - y_hat, 'x', color='red', label='Bias Blend')
        ax2.grid(linestyle=':')
        ax2.legend(loc='upper left')
        ax2.set_title('Residuals', fontweight='bold')

        # Weights bar plot
        ax3.grid(linestyle=':')
        ax3.bar(x=[model_names[i] for i in range(n_models)] + ['Intercept'],
                height=np.concatenate([result.weights, [result.intercet]]),
                color=[f'C{i}' for i in range(n_models)] + ['white'], edgecolor='black')
        ax3.set_title('Coefficients', fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Compute performance
        metrics = ['MAE', 'MAPE', 'BIAS']
        performance_table = pd.DataFrame(np.nan, columns=metrics, index=list(model_names) + ['Blend'])
        for idx, m in enumerate(performance_table.index):
            if m == 'Blend':
                performance_table.loc[m, 'MAE'] = mean_absolute_error(y_true = target, y_pred=y_hat)
                performance_table.loc[m, 'MAPE'] = mean_absolute_percentage_error(y_true=target, y_pred=y_hat)
                performance_table.loc[m, 'BIAS'] = np.mean(target-y_hat)
            else:
                performance_table.loc[m, 'MAE'] = mean_absolute_error(y_true=target, y_pred=models[:, idx])
                performance_table.loc[m, 'MAPE'] = mean_absolute_percentage_error(y_true=target, y_pred=models[:, idx])
                performance_table.loc[m, 'BIAS'] = np.mean(target - models[:, idx])
        performance_table = performance_table.round(1)


        # Performance plot
        table = ax4.table(performance_table.values, cellLoc = 'center', colLabels = metrics, rowLabels = performance_table.index, loc = 'center',
                          rowColours = ['lightgray'] * performance_table.shape[0],
                          colColours = ['lightblue'] * performance_table.shape[0])

        for (row, col), cell in table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        table.auto_set_column_width(col=list(range(performance_table.shape[1])))

        ax4.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    # Generate data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.randn(100) * .1


    # Build model mockup
    arrays = [(y * .5 + np.random.randn(100) * .1).reshape(-1, 1)-1,
              (y * 1.7 + np.random.randn(100) * .1).reshape(-1, 1)-1,
              (np.random.randn(100) * .5).reshape(-1, 1)]
    X = np.concatenate(arrays, axis=1)
    X = pd.DataFrame(X, columns=['A', 'B', 'C'])

    # Fit model
    model = Blend(fit_intercept=True, convex=True, lp_norm=1)
    result = model.fit(X, y)
    y_hat = model.predict(X)

    # Print results
    print('Coeff solution', result.weights)
    print('Intercept', result.intercet)
    print('Optimal value', result.value)

    plt.ion()
    model.diagnostic_plot(x=x, models=X, target=y)
    plt.show(block=True)
