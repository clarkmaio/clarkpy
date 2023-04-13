from datetime import datetime
import pandas as pd
from dataclasses import dataclass


@dataclass
class Calendar:

    @staticmethod
    def load_calendar(start_date: datetime, end_date: datetime, freq: str = 'D') -> pd.DataFrame:
        '''Build calendar dataframe'''

        timerange = pd.date_range(start=start_date, end=end_date, freq=freq)

        index = pd.Index(timerange, name='valuedate')
        calendar = pd.DataFrame(index=index)

        calendar = Calendar.__add_features(calendar)
        return calendar

    @staticmethod
    def __add_features(calendar: pd.DataFrame) -> pd.DataFrame:

        calendar['day'] = calendar.index.day

        return calendar

if __name__ == '__main__':

    calendar = Calendar.load_calendar(datetime(2020, 1, 1), end_date=datetime(2020, 12, 31, 23), freq='H')