
import pandas as pd
import numpy as np
from typing import Iterable, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from dataclasses import dataclass, field

import torch
from torch.nn import Module, Linear, ReLU, Identity, ModuleList, Dropout
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD, Adam, RMSprop
from torchviz import make_dot

from ml.loss import CRPSLoss, AveragePinballLoss, loss_hub, PinballLoss


#matplotlib.use('Agg')


def optimizer_hub(optimizer: Union[str], parameters, **kwargs):
    if isinstance(optimizer, str):
        if optimizer.lower() == 'sgd':
            return SGD(parameters, **kwargs)
        elif optimizer.lower() == 'adam':
            return Adam(parameters, **kwargs)
        elif optimizer.lower() == 'rmsprop':
            return RMSprop(parameters, **kwargs)
        else:
            raise ValueError('Optimizer not implemented')

def to_torch_tensor(X, y=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    '''Whatever are X, y convert them to torch tensors'''

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    elif isinstance(X, pd.DataFrame):
        X = torch.from_numpy(X.values).float()

    if y is not None:
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        elif isinstance(y, pd.DataFrame):
            y = torch.from_numpy(y.values).float()

    if y is None:
        return X
    else:
        return X, y



class MLP(Module):
    def __init__(self,
                 input_shape: int,
                 hidden_layer_units: Iterable[int] = [100, 1],
                 hidden_activation: Iterable[Module] = [ReLU(), Identity()],
                 dropout_layers: Union[Iterable[float], None] = None):

        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_layer_units = hidden_layer_units
        self.hidden_activation = hidden_activation
        self.dropout_layers = dropout_layers

        self._parameters_check()
        self.build_layers()


    def build_layers(self):
        '''Build all the modules needed to define the forward method'''
        self._layers = self._build_layers(input_shape = self.input_shape, hidden_layers_units = self.hidden_layer_units)
        self._activations = self._build_activations(hidden_activation = self.hidden_activation)
        self._dropouts = self._build_dropouts(dropout_layers = self.dropout_layers)


    def _build_layers(self, input_shape, hidden_layers_units: Iterable[int]) -> ModuleList:
        '''
        Build ModuleList of linear layers
        '''
        layers = ModuleList()
        units = [input_shape] + hidden_layers_units
        for input, output in zip(units[:-1], units[1:]):
            layers.append(Linear(input, output))
        return layers

    def _build_activations(self, hidden_activation: Iterable[Module]) -> ModuleList:
        '''
        Build ModuleList of activation functions
        '''
        activations = ModuleList()
        for activation in hidden_activation:
            if activation is None:
                activations.append(Identity())
            else:
                activations.append(activation)
        return activations

    def _build_dropouts(self, dropout_layers: Union[Iterable[float], None]) -> ModuleList:
        '''
        Build ModuleList of dropout layers
        '''

        dropout = ModuleList()
        if dropout_layers is None:
            for _ in self.hidden_layer_units:
                dropout.append(Dropout(0.0))
        else:
            for dropouta_rate in dropout_layers:
                dropout.append(Dropout(dropouta_rate))
        return dropout

    def plot_summary(self):
        sample_input = torch.randn(1, self.input_shape)
        graph = make_dot(self(sample_input), params=dict(self.named_parameters()))
        graph.render(filename='neural_net_graph', format='png', cleanup=True)



    def forward(self, x):
        for layer, activation, dropout in zip(self._layers, self._activations, self._dropouts):
            x = layer(x)
            x = activation(x)
            x = dropout(x)
        return x

    def predict(self, X):
        return self(X)

    def _parameters_check(self):
        assert isinstance(self.hidden_layer_units, Iterable), 'hidden_layer_units must be an iterable'
        assert isinstance(self.hidden_activation, Iterable), 'hidden_activation must be an iterable'
        assert len(self.hidden_layer_units) == len(self.hidden_activation), 'Number of hidden layers and hidden activations must be the same'

@dataclass
class NeuralNetModel:
    hidden_layer_units: Iterable = field(default_factory=lambda: [100])
    hidden_activation: Iterable[Module] = field(default_factory=lambda: [ReLU()])
    dropout_layers: Union[Iterable[float], None] = None

    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = 'Adam'
    loss: Union[str, Module] = 'mse'

    def __post_init__(self):
        self.loss_function = loss_hub(self.loss)

    def fit(self, X, y):

        X_tensor, y_tensor = to_torch_tensor(X, y)

        self.mlp = MLP(input_shape = X_tensor.shape[1],
                       hidden_layer_units=self.hidden_layer_units,
                       hidden_activation=self.hidden_activation,
                       dropout_layers=self.dropout_layers)

        self.optimizer_class = optimizer_hub(optimizer=self.optimizer,
                                             parameters=self.mlp.parameters(),
                                             lr=self.learning_rate)

        # Prepare dataloader
        self.dataset = TensorDataset(X_tensor, y_tensor)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Optimize
        self.loss_history = self.optimization_core(dataloader=self.dataloader)

        return self


    def fine_tune(self, X, y, epochs=100, batch_size=32, learning_rate=0.01):
        assert hasattr(self, 'mlp'), 'Model must be fitted before fine tuning'
        return

    def optimization_core(self, dataloader) -> pd.DataFrame:
        '''
        Run optimization looping on dataloader
        Return optimization history as dataframe
        '''

        loss_history = pd.DataFrame(np.nan, columns=['loss'],
                                         index=pd.Index(range(1, self.epochs + 1), name='epoch'))

        # Loop on dataloader splitting with epochs and bathces
        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}/{self.epochs}')
            for X_batch, y_batch in pbar:
                self.optimizer_class.zero_grad()

                y_pred = self.mlp(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                loss.backward()
                self.optimizer_class.step()

                pbar.set_postfix({'loss': loss.item()})

            loss_history.loc[epoch, 'loss'] = loss.item()
        return loss_history

    def predict(self, X):
        X_tensor = to_torch_tensor(X)
        y_pred = self.mlp(X_tensor)


        # Return y_pred in the same format as X
        if isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            return y_pred.detach().numpy()
        else:
            return y_pred

    def plot_summary(self):
        self.mlp.plot_summary()





if __name__ == '__main__':
    # Simple resgression example
    import matplotlib.pyplot as plt

    N = 1000
    X = np.linspace(0, 10, N).reshape(-1, 1)
    y = np.sin(X) * X + 0.2*np.random.normal(0, 2, N).reshape(-1, 1)

    X = pd.DataFrame(X, columns=['x'])
    y = pd.DataFrame(y, columns=['y'])

    neuralnet = NeuralNetModel(hidden_layer_units=[50, 50, 50, 9],
                               hidden_activation=[ReLU(), ReLU(), ReLU(), Identity()],
                               dropout_layers=[0, 0, 0, 0],
                               epochs=600, batch_size=512,
                               learning_rate=0.01, optimizer='Adam',
                               loss=AveragePinballLoss(quantiles=[.01, .2, .3, .4, .5, .6, .7, .8, .99]))

    neuralnet.fit(X, y)

    pred = neuralnet.predict(X)
    plt.scatter(X, y, alpha=0.1)
    plt.plot(X, pred, alpha=1)
    plt.show()


