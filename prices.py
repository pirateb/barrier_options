import datetime
from typing import List


# n business day
def business_day(date: datetime.datetime, bus_days):
    from pandas.tseries.offsets import BDay

    return (date - BDay(bus_days))\
        .to_pydatetime()\
        .date()


# get prices from yahoo data
def get_prices(symbols: List[str],
               start_date: datetime.datetime = business_day(datetime.date.today(), 30),
               end_date: datetime.datetime = business_day(datetime.date.today(), 1)):
    ## TODO: par
    import pandas_datareader as pdr
    import pandas as pd

    df = pd.DataFrame()
    for s in symbols:
        print("Downloading {0}".format(s))
        # df_tmp = pdr.get_data_yahoo(symbols=s, start=start_date, end=end_date)
        df_tmp = pdr.get_data_google(symbols=s, start=start_date, end=end_date)
        # adjusted?
        df[s] = df_tmp["Close"]
        # df[s] = df_tmp["Adj Close"]

    return df
