


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def read_data(vix_path, bond_path, fedfunds_path):
    try:
        vix_price = pd.read_csv(vix_path)
        bond_yield = pd.read_csv(bond_path,
                skiprows=[x for x in range(1,1462)])
        fedfunds_rate = pd.read_csv(fedfunds_path)
    
    except FileNotFoundError as f:
        print('file not found', f)
    return vix_price,bond_yield,fedfunds_rate

vix_price, bond_yield, fedfunds_rate = read_data(
    vix_path=r'C:\Users\PCadmin\Downloads\VIX_History.csv'
    ,bond_path=r'C:\Users\PCadmin\Downloads\bond_prices2.csv'
    ,fedfunds_path=r'C:\Users\PCadmin\Downloads\fedfundsrate.csv')



vix_price.rename(columns={ vix_price.columns[0] : 'Date',
                          vix_price.columns[4] : 'VIX 5-Day Returns'}
                         ,inplace=True)
vix_price.drop(columns=['OPEN','HIGH','LOW',], inplace=True)


# changing dtype of dates from str to datetime64[ns] for time series calculations

def convert_datetime(dates_1,dates_2):
    new_dates_1 = pd.to_datetime(dates_1)
    new_dates_2 = pd.to_datetime(dates_2)
    return new_dates_1, new_dates_2


vix_price['Date'], bond_yield['observation_date'] = convert_datetime(
            dates_1=vix_price['Date'],
            dates_2=bond_yield['observation_date'])

# -----renaming columns, merging datasets------

bond_yield['Federal Funds Rate'] = fedfunds_rate['DFF']
bond_yield.rename(columns={bond_yield.columns[0]: 'Date',
                           bond_yield.columns[1]: '10-Year Treasury Bond Yield'},
                  inplace=True)
jan5th_yield = vix_price['VIX 5-Day Returns'].pct_change(periods=3) *100
vix_price['VIX 5-Day Returns'] = vix_price['VIX 5-Day Returns'].pct_change(periods=5) *100
vix_price.iloc[3,1] = jan5th_yield.iloc[3]

mixed_df = bond_yield.merge(vix_price, on='Date', how='inner') 



