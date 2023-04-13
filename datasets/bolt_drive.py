import pandas as pd
from typing import Dict
from __init__ import load_pickle_from_url

LINK = 'https://github.com/clarkmaio/datasets/raw/main/bolt_drive.parquet'

DESCR = '''
        Bolt drive dataset
        -------------------------
        
        Description:
            In this dataset you will find data of Bolt rides in Tallinn.
            Each sample correspond to a different ride.
            For each sample you will find starting position, ending position and value of the ride in EUR (namely the cost of the ride service).
        
        Columns:
            - start_time [datetime64[ns]]
            - start_lat [float]
            - start_lng [float]
            - end_lat [float]
            - end_lng [float]
            - ride_value [float]
        
        Summary statistics:
        
                    start_lat  start_lng    end_lat    end_lng  ride_value
        count  627210.00  627210.00  627210.00  627210.00   627210.00
        mean       59.43      24.74      59.40      24.72        2.26
        std         0.02       0.06       1.39       1.65       44.89
        min        59.32      24.51     -37.82    -122.45        0.11
        25%        59.42      24.71      59.42      24.71        0.55
        50%        59.43      24.74      59.43      24.74        1.06
        75%        59.44      24.77      59.44      24.77        1.71
        max        59.57      24.97      61.55     144.97     3172.70
                
        
        
        '''


def load_bolt_drive() -> Dict:
    df = pd.read_parquet(LINK)
    df['start_time'] = pd.to_datetime(df['start_time'])

    dataset = {'data': df, 'DESCR': DESCR, 'columns': df.columns}
    return dataset


if __name__ == '__main__':
    dataset = load_bolt_drive()