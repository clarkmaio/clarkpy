import pandas as pd
from typing import Dict

LINK = 'https://raw.githubusercontent.com/clarkmaio/datasets/main/radio_plays.csv'


DESCR = '''
        Radio plays dataset
        -------------------
        
        Description:
            Dataset containing the number of plays of songs (identified by TrackId variable) from radio stations (identified by StationId variable).
            The granularity is daily
        
        Columns:
            - Day [datetime64[ns]]: valuedate (daily granularity)
            - Plays [int]: number of plays
            - TrackId [str]:
            - StationId [str]:
            
            
        Summary statistics:
        
                    Plays
        count  12271.000000
        mean       1.062668
        std        0.281863
        min        0.000000
        25%        1.000000
        50%        1.000000
        75%        1.000000
        max       11.000000      
        '''


def load_radio_plays() -> Dict:
    '''Load data'''
    df = pd.read_csv(LINK)
    df['Day'] = pd.to_datetime(df['Day'])

    dataset = {'DESCR': DESCR, 'data': df, 'columns': df.columns}
    return dataset



if __name__ == '__main__':
    dataset  = load_radio_plays()