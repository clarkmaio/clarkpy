import torch
from torch.nn import Module, Linear, ReLU, MSELoss, Identity
from torch import nn
import pandas as pd
import numpy as np
from typing import Iterable, Union
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from torch.optim import SGD, Adam, RMSprop

matplotlib.use('TkAgg')

@dataclass
class NeuralNetModel:
    hidden_layer_units: Iterable = field(default_factory=lambda: [100])
    hidden_activation: Iterable = field(default_factory=lambda: ['ReLU'])
    output_size: int = 1
    output_activation: Union[str, None] = None

    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = 'SGD'
    loss: str = 'MSELoss'


    def __post_init__(self):
        self.mlp = MLP(hidden_layer_units=self.hidden_layer_units,
                      hidden_activation=self.hidden_activation,
                      output_size=self.output_size,
                      output_activation=self.output_activation,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      learning_rate=self.learning_rate,
                      optimizer=self.optimizer,
                      loss=self.loss)

    def loss_hub(self, loss):

        if isinstance(loss, str):
            if loss.lower() == 'crps':
                return CRPSLoss()
            elif loss.lower() == 'pinball':
                return AveragePinballLoss()
            elif loss.lower() in ('mseloss', 'mse'):
                return MSELoss
            else:
                raise ValueError('Loss function not implemented')
        else:
            return loss

    def fit(self, X, y):
        self.mlp.build_layers(shape_X=X.shape[1])
        self.loss = self.loss_hub(self.loss)
        self.optimizer_class = SGD(self.mlp.parameters(), lr=self.learning_rate)


        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_history = pd.DataFrame(np.nan, columns=['epoch', 'loss'], index=range(self.epochs))
        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch))
            self.optimizer_class.zero_grad()
            y_pred = self.mlp(X)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()

            self.loss_history.loc[epoch, 'epoch'] = epoch
            self.loss_history.loc[epoch, 'loss'] = loss.item()

        return self




@dataclass(eq=False)
class MLP(Module):
    hidden_layer_units: Iterable = field(default_factory = lambda : [100])
    hidden_activation: Iterable = field(default_factory = lambda: ['ReLU'])
    output_size: int = 1
    output_activation: Union[str, None] = None

    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = 'SGD'
    loss: str = 'MSELoss'


    def __post_init__(self):
        return

    def build_layers(self, shape_X):
        self.layers = []
        self.activation = []

        # First Layer
        self.layers.append(Linear(shape_X, self.hidden_layer_units[0]))

        # Hidden Layers
        for units in range(len(self.hidden_layer_units)-1):
            self.layers.append(Linear(self.hidden_layer_units[i], self.hidden_layer_units[i+1]))

        # Output Layer
        self.layers.append(Linear(self.hidden_layer_units[-1], self.output_size))

        # Activations
        for i in range(len(self.hidden_activation)):
            self.activation.append(self.activation_hub(self.hidden_activation[i]))
        self.activation.append(self.activation_hub(self.output_activation))


        return

    def activation_hub(self, activation):
        if isinstance(activation, str):
            return getattr(nn, activation)()
        elif activation is None:
            return Identity()
        else:
            return activation

    def loss_hub(self, loss):

        if isinstance(loss, str):
            if loss.lower() == 'crps':
                return CRPSLoss()
            elif loss.lower() == 'pinball':
                return AveragePinballLoss()
            elif loss.lower() in ('mseloss', 'mse'):
                return MSELoss
            else:
                raise ValueError('Loss function not implemented')

        else:
            return loss


    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation[i](x)
        return x

    def predict(self, X):
        return self(X)



# Pinball Loss Function
class PinballLoss(Module):
    def __init__(self, tau):
        super(PinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y_pred, y):
        return torch.mean(torch.max(self.tau*(y-y_pred), (self.tau-1)*(y-y_pred)))

# Compute loss function as average of a set of PinPallLoss functions
class AveragePinballLoss(Module):
    def __init__(self, taus=np.arange(0.1, 1, 0.1)):
        self.taus = taus
        super(AveragePinballLoss, self).__init__()
        self.losses = [PinballLoss(tau) for tau in taus]

    def forward(self, y_pred, y):
        loss = 0
        for i, tau in enumerate(self.taus):
            loss += self.losses[i](y_pred[:, i], y)
        return loss/len(self.taus)


class CRPSLoss(Module):
    def __init__(self):
        super(CRPSLoss, self).__init__()

    def forward(self, y_pred, y):

        # Compute 1/2 * E[|y_pred - y_pred'|]
        mix_loss = 0
        for i in range(y_pred.shape[1]):
            xx = y_pred[:, i]
            xx_expanded = xx.unsqueeze(0).expand(len(xx), -1)
            mix_loss += 0.5*torch.mean(torch.abs(xx_expanded - xx_expanded.t()))

            xx =  y_pred[0, :]
            xx_expanded = xx.unsqueeze(0).expand(len(xx), -1)
            mix_loss += 0.5*torch.mean(torch.abs(xx_expanded - xx_expanded.t()))


        mae_loss = torch.mean(torch.abs(y_pred - y), dim=1)

        return torch.mean(mae_loss - mix_loss)


if __name__ == '__main__':
    # Simple resgression example
    import matplotlib.pyplot as plt

    N = 1000
    X = np.linspace(0, 10, N)
    y = np.sin(X) * X + np.random.normal(0, 2, N)

    X = X.reshape((X.shape[0], 1))

    # Convert to torch tensors
    X_tot = torch.from_numpy(X).float()
    y_tot = torch.from_numpy(y.reshape((y.shape[0], 1))).float()

    import torch
    from torch import nn

    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.hidden = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.output = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.hidden(x)
            x = self.relu(x)
            x = self.output(x)
            return x


    model = SimpleNet(1, 100, 1)
    # Define a loss function and an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Number of epochs
    epochs = 1000

    # Train the model
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tot)
        loss = criterion(outputs, y_tot)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')



    plt.scatter(X_tot.flatten(), y_tot.flatten())
    plt.plot(X_tot.flatten(), model(X_tot).detach().numpy(), color='red')
    plt.show()

    #plt.plot(X_tot.flatten(), model(X_tot).detach().numpy()[:, 0])
    for i in range(9):
        plt.plot(X_tot.flatten(), model(X_tot).detach().numpy()[:, i])
