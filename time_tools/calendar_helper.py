from datetime import datetime
import pandas as pd
from dataclasses import dataclass


@dataclass
class CalendarHelper:

    @staticmethod
    def load_calendar(start_date: datetime, end_date: datetime, freq: str = 'D') -> pd.DataFrame:
        '''Build calendar dataframe'''

        timerange = pd.date_range(start=start_date, end=end_date, freq=freq)

        index = pd.Index(timerange, name='valuedate')
        calendar = pd.DataFrame(index=index)

        calendar = CalendarHelper.__add_features(calendar)
        return calendar

    @staticmethod
    def __add_features(calendar: pd.DataFrame) -> pd.DataFrame:
        '''Create all boring calendar features'''

        calendar['hour'] = calendar.index.hour
        calendar['day'] = calendar.index.day
        calendar['doy'] = calendar.index.dayofyear
        calendar['weekday'] = calendar.index.weekday
        calendar['woy'] = calendar.index.isocalendar()['week']
        calendar['month'] = calendar.index.dayofyear
        calendar['quarter'] = calendar.index.quarter


        return calendar





        return calendar

if __name__ == '__main__':

    calendar = CalendarHelper.load_calendar(datetime(2020, 1, 1), end_date=datetime(2020, 12, 31, 23), freq='H')