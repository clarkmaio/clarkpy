import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import torch

def plot_weights(named_parameters: List[Tuple[str, torch.Tensor]],
                 annot: bool = False, annot_kws: dict = {'size': 3},
                 subplot_title: bool = False, title: str = None,
                 figsize: Tuple[int, int]= None):
    """
    This function is used to visualize the weights of a neural network's layers as heatmaps.

    :param named_parameters (List[Tuple[str, torch.Tensor]]): A list of tuples where each tuple contains the name of the layer and the layer's weights in the form of a PyTorch tensor.
    The function creates a matplotlib figure with a number of subplots equal to the number of layers in the neural network. For each layer, the function extracts the weights, converts them to a numpy array, and displays them as a heatmap using the seaborn library. The heatmap uses a diverging color palette to highlight positive and negative weights.
    This function is useful for visualizing and understanding how the model's weights change during training or fine-tuning.
    """
    N_LAYERS = len(named_parameters)
    cmap = sns.diverging_palette(-100, 100, as_cmap=True)

    fig, ax = plt.subplots(1, N_LAYERS, figsize=figsize)

    for i, (name, weights) in enumerate(named_parameters):
        weights_numpy = weights.detach().numpy()
        sns.heatmap(data=weights_numpy,
                    annot=annot, annot_kws=annot_kws,
                    fmt='.2f', cmap=cmap, cbar=False,
                    linewidths=1, ax=ax[i],
                    xticklabels=False, yticklabels=False,
                    square=True)
        if subplot_title:
            ax[i].set_title(name)

    if title:
        fig.suptitle(title, fontweight='bold')

    #plt.tight_layout(pad=2.0)