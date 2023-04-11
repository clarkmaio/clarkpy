import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable

def spider_chart(labels: Iterable, values: Iterable, title: str = None):
    '''
    Create a spider chart
    :param labels:
    :param values:
    :return:
    '''


    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Compute pie slices
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # ripete il primo valore della lista di angoli alla fine
    angles = np.concatenate((angles, [angles[0]]))

    # Create plot
    ax.plot(angles, np.concatenate((values, [values[0]])), 'o-', linewidth=2)
    ax.fill(angles, np.concatenate((values, [values[0]])), alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, max(values))

    ax.set_title(title)
    plt.show()





if __name__ == '__main__':
    labels = ['Abilità 1', 'Abilità 2', 'Abilità 3', 'Abilità 4', 'Abilità 5']
    values = [4, 3, 5, 2, 4]
    plt.ion()
    spider_chart(labels, values)
