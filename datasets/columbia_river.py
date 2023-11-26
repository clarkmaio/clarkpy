import pandas as pd
from typing import Dict

LINK = 'https://github.com/clarkmaio/datasets/raw/main/columbia_river.parquet'


DESCR = '''
        Columbia river dataset
        '''



def load_columbia_river() -> pd.DataFrame:
    df = pd.read_parquet(LINK)
    return df



if __name__ == '__main__':

    # Example
    df = load_columbia_river()
    df.query('discharge_dellas > 0', inplace=True)

    import matplotlib.pyplot as plt
    plt.scatter(x=df['discharge_dellas'], y=df['total_hydro_columbia'], alpha=0.1, color='blue', edgecolors='k')
    plt.show()

