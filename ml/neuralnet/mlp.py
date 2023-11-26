
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
                 layers_units: Iterable[int] = [100, 1],
                 activations: Iterable[Module] = [ReLU(), Identity()],
                 dropout_layers: Union[Iterable[float], None] = None):

        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.layers_units = layers_units
        self.activations = activations
        self.dropout_layers = dropout_layers

        self._parameters_check()
        self.build_layers()


    def build_layers(self):
        '''Build all the modules needed to define the forward method'''
        self._layers = self._build_layers(input_shape = self.input_shape, layers_units = self.layers_units)
        self._activations = self._build_activations(activations = self.activations)
        self._dropouts = self._build_dropouts(dropout_layers = self.dropout_layers)


    def _build_layers(self, input_shape, layers_units: Iterable[int]) -> ModuleList:
        '''
        Build ModuleList of linear layers
        '''
        layers = ModuleList()
        units = [input_shape] + layers_units
        for input, output in zip(units[:-1], units[1:]):
            layers.append(Linear(input, output))
        return layers

    def _build_activations(self, activations: Iterable[Module]) -> ModuleList:
        '''
        Build ModuleList of activation functions
        '''
        activations_modulelist = ModuleList()
        for activation in activations:
            if activation is None:
                activations_modulelist.append(Identity())
            else:
                activations_modulelist.append(activation)
        return activations_modulelist

    def _build_dropouts(self, dropout_layers: Union[Iterable[float], None]) -> ModuleList:
        '''
        Build ModuleList of dropout layers
        '''

        dropout = ModuleList()
        if dropout_layers is None:
            for _ in self.layers_units:
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
        assert isinstance(self.layers_units, Iterable), 'layers_units must be an iterable'
        assert isinstance(self.activations, Iterable), 'activations must be an iterable'
        assert len(self.layers_units) == len(self.activations), 'Number of hidden layers and hidden activations must be the same'


    def freeze_layer_weights(self, layer_index: Union[Iterable[int], int]):
        '''
        Freeze layer weights
        layer_index
        '''
        if isinstance(layer_index, int):
            layer_index = [layer_index]
        for i in layer_index:
            for param in self._layers[i].parameters():
                param.requires_grad = False


    def unfreeze_layer_weights(self):
        '''
        Unfreeze all layer weights
        layer_index
        '''
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = True

@dataclass
class NeuralNetModel:
    layers_units: Iterable = field(default_factory=lambda: [100])
    activations: Iterable[Module] = field(default_factory=lambda: [ReLU()])
    dropout_layers: Union[Iterable[float], None] = None

    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = 'Adam'
    loss: Union[str, Module] = 'mse'
    verbose: bool = False

    def __post_init__(self):
        self.mlp = None
        self.loss_function = loss_hub(self.loss)
        self.is_trained = False

    def fit(self, X, y, epochs: int = None, batch_size: int = None, learning_rate: float = None, shuffle: bool = False):
        '''
        Just train
        :param epochs: number of epochs to train. If not specified the init value will be used.
        :param batch_size: batch size. If not specified the init value will be used.
        :param learning_rate: learning rate. If not specified the init value will be used.
        :param shuffle: shuffle data
        '''


        # Update train parameters if needed
        self.update_train_params(epochs, batch_size, learning_rate)

        # Format X, y to torch tensors and build dataloader
        X_tensor, y_tensor = to_torch_tensor(X, y)
        self.dataset = TensorDataset(X_tensor, y_tensor)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle)

        # Build core model
        if not self.is_trained:
            # Rebuild (and rinitialize) model only if it is not trained yet. Important for fine tuning
            self.mlp = MLP(input_shape = X_tensor.shape[1],
                           layers_units=self.layers_units,
                           activations=self.activations,
                           dropout_layers=self.dropout_layers)
            self.is_trained = True

        self.optimizer_class = optimizer_hub(optimizer=self.optimizer,
                                             parameters=self.mlp.parameters(),
                                             lr=self.learning_rate)

        # Optimize
        self.loss_history = self.optimization_core(dataloader=self.dataloader)


    def fine_tune(self, X, y, freeze_layer_index: Union[int, Iterable[int]] = 0, epochs: int = None, batch_size: int = None, learning_rate: float = None, shuffle: bool = False):
        '''
        Fine tune a trained model.
        :param freeze_layer_index: index of the layer to freeze. Weights of this layer will not be updated during fine tuning.
                                   Usually the first layers are freezed.
                                   It can be an integer or a list of integers that will refer to layer index.
        :param epochs: number of epochs to train. If not specified the init value will be used.
        :param batch_size: batch size. If not specified the init value will be used.
        :param learning_rate: learning rate. If not specified the init value will be used.
        :param shuffle: shuffle data
        '''


        assert hasattr(self, 'mlp') and (self.mlp is not None), 'Model must be fitted before fine tuning'

        # Freeze
        self.mlp.freeze_layer_weights(layer_index=freeze_layer_index)

        # Train
        self.fit(X=X, y=y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, shuffle=shuffle)

        # Unfreeze
        self.mlp.unfreeze_layer_weights()


    def optimization_core(self, dataloader) -> pd.DataFrame:
        '''
        Run optimization looping on dataloader
        Return optimization history as dataframe
        '''

        loss_history = pd.DataFrame(np.nan, columns=['loss'],
                                         index=pd.Index(range(1, self.epochs + 1), name='epoch'))

        # Loop on dataloader splitting with epochs and bathces
        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}/{self.epochs}', disable=not self.verbose)
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


    def _update_learning_rate(self, learning_rate: float = None):
        if learning_rate is not None:
            self.learning_rate = learning_rate

    def _update_batch_size(self, batch_size: int = None):
        if batch_size is not None:
            self.batch_size = batch_size

    def _update_epochs(self, epochs: int = None):
        if epochs is not None:
            self.epochs = epochs

    def update_train_params(self, epochs: int = None, batch_size: int = None, learning_rate: float = None):
        self._update_learning_rate(learning_rate)
        self._update_batch_size(batch_size)
        self._update_epochs(epochs)


    def plot_summary(self):
        self.mlp.plot_summary()





if __name__ == '__main__':

    # Build data
    N_train = 1000
    X = np.random.randn(N_train, 2)
    X[:, 0] = np.linspace(0, 10, N_train)
    y = 1 * X[:, [0]] * np.sin(X[:, [0]]) + 1.1 * np.random.normal(0, 2, N_train).reshape(-1, 1)

    # Build data
    N_train = 1000
    X = np.linspace(0, 10, N_train).reshape(-1, 1)
    y = 1 * X * np.sin(X) + 1.1 * np.random.normal(0, 2, N_train).reshape(-1, 1)

    model = NeuralNetModel(layers_units=[100, 100, 100],
                            activations=[ReLU(), ReLU(), Identity()],
                            dropout_layers=None,
                            epochs=500,
                            batch_size=512,
                            learning_rate=0.01,
                            optimizer='Adam',
                            loss='crps',
                            verbose=True)

    model.fit(X, y)
    y_pred = model.predict(X)

    model.loss_history.plot()
    plt.show()




    plt.scatter(X[:, 0], y)
    plt.plot(X[:, 0], y_pred, color='orange', alpha=0.1)
    plt.show()

