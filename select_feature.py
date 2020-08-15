import pandas as pd
import numpy as np
from get_feature import get_all_feature, get_PCA_factors
from get_data import get_test



def CalReturn(series):
    '''cal return
    '''
    return series.pct_change()

def VolNormalize(series, rolling_window=252):
    return series/series.rolling(rolling_window).mean()

def DFLag(DF, lag_i):
    DF = DF.copy().shift(lag_i)
    DF.columns = [x+'_lag'+str(lag_i) for x in DF.columns]
    return DF

def RangeCal(DF, name):
    return (DF['High_'+name]-DF['Low_'+name])/DF['Open_'+name]

def Select_Feature(data, feature, n_lag=3):
    '''
    process return calcluation
    vol normalization
    important assets
    technical signals
    lag all above data in range(n_lag)
    '''
    # clean data
    data = data.T.dropna(thresh=150).T.ffill()
    feature = feature.ffill().T.dropna().T
    # cal return 
    return_cols = ['Close_^HSI', 'Close_^N225'] 
    DF_return = pd.DataFrame([CalReturn(data[i]) for i in return_cols]).T
    DF_return.columns = [i+'_return' for i in return_cols]
    # normalize vol
    vol_cols = ['Vol._Gold', 'Vol._Silver ','Vol._Natural Gas','Vol._US Corn ','Vol._US Wheat ','Vol._SP500Futures','Volume_^HSI','Volume_^N225']
    DF_vol = pd.DataFrame([VolNormalize(data[i]) for i in vol_cols]).T
    # range
    range_cols = ['Yield2y','Yield10y','Yield30y','Gold', 'Silver ','Natural Gas','US Corn ','US Wheat ','SP500Futures','^HSI','^N225','VIX','Dollar_index',
#                  'FTSE100',
                  ]
    DF_range = pd.DataFrame([RangeCal(data, name) for name in range_cols]).T
    DF_range.columns = ['Range_'+i for i in range_cols]
    # important raw data
    keep_cols = [
            'Change_Yield2y', 'Price_Yield10y', 'Change_Yield30y', 
            'Change_Dollar_index',
            'Change_Gold','Change_Silver ',
            'Vol._Crude Oil WTI',  'Change_Crude Oil WTI',
            'Vol._Natural Gas', 'Change_Natural Gas',
            'Vol._US Corn ', 'Change_US Corn ',
            'Vol._US Wheat ', 'Change_US Wheat '
            'Change %_FTSE100',
            'Vol._SP500Futures', 'Change %_SP500Futures', 
            'Volume_^HSI',
            'Volume_^N225', 
            'Change_VIX', 'SMB', 'HML', 'RMW', 'CMA', 'RF'
            ]
    keep_cols = set(keep_cols) & set(data.columns)
    DF_keep = data.copy()[keep_cols]
    # technical signals
    term = ['SMA', 'vol', 'ATR', 'RSI', 'CCI', 'OC_spread', 'LH_spread']
    technical = ['Yield2y_', 'Yield10y_', 'Yield30y_', 'Dollar_index_', 'Gold_', 'Silver _',
                 'Crude Oil WTI_', 'Natural Gas_', 'US Corn _', 'US Wheat _', 'FTSE100_', 'SP500Futures_',
                 'VIX_' ]
    tec_col = [technical[i]+term[j] for j in range(len(term)) for i in range(len(technical))]
    tec_col = set(tec_col) & set(feature.columns)
    DF_feature = feature[tec_col]
    DF_feature.index = pd.to_datetime(DF_feature.index)
    # combined data
    DF_all = pd.concat([DF_return, DF_vol, DF_keep, DF_range], axis=1)
    DF_all = pd.merge(DF_all, DF_feature, left_index=True, right_index=True, how='left')
    # lag 
    DF_lagged = pd.concat([DF_all] + [DFLag(DF_all, i) for i in range(1,n_lag)], axis=1)
    DF_lagged = DF_lagged.iloc[254:,:]
    # nan handling
    temp = DF_lagged.isna().sum().sort_values(ascending=False)
    drop_cols = temp[temp>50].index
    DF_output = DF_lagged[set(DF_lagged.columns)-set(drop_cols)]
    print('Dropped features:',drop_cols)
    return DF_output.ffill()

def RollingCategory(series, n):
    vol = series.std()
    mu = series.mean()
    t1 = mu+vol*n
    t2 = mu-vol*n
    if series[-1]>=t1:
        return 1
    elif series[-1]<=t2:
        return -1
    else:
        return 0

def CategorizeY(y,method='roll_vol'):
    '''
    catagorize y
    method: 
        equal: threshold for y
        roll_vol: rolling vol
        normal: total sample threshold
    '''
    if method == 'equal':
        n = len(y)
        t1 = y.nlargest(n//3).min()
        t2 = y.nsmallest(n//3).max()
        z = y.copy()
        z.loc[:] = 'mid'
        z[y>t1] = 'up'
        z[y<t2] = 'down'
        return z
    elif method == 'normal':
        n=0.5
        vol = y.std()
        mu = y.mean()
        t1 = mu+vol*n
        t2 = mu-vol*n
        z = y.copy()
        z.loc[:] = 'mid'
        z[y>t1] = 'up'
        z[y<t2] = 'down'
        z.value_counts()/len(z)
        return z
    elif method == 'roll_vol':
        rolling_n = 50
        std_n = 1
        z = y.rolling(rolling_n).apply(lambda x: RollingCategory(x, std_n ))#RollingCategory(x, std_n))
        c = z.copy()
        c.loc[:] = 'mid'
        c[z==1] = 'up'
        c[z==-1] = 'down'
#        z.value_counts()/len(z)
        return c

def Feature_Inputs(st = '1998-01-02', ed = '2019-12-31', SPget=False, y_method = 'roll_vol',PCA_includ=False):
    data, Y = get_test(st,ed,parameters='All')
    feature = get_all_feature(False, start =st, end=ed)
#    PCAfeature = get_PCA_factors(data,n=120,components=20)
    # cal X and Y
    x = Select_Feature(data, feature, n_lag=3).dropna() # data before 2000 and after 2020 all drop due to missing data
#    if PCA_includ:
#        PCAfeature.index = pd.to_datetime(PCAfeature.index)
#        x = pd.merge(x, PCAfeature, )
    y = CategorizeY(Y,method=y_method).loc[x.index]
    if SPget:
        return x, y, Y
    return x, y

if __name__ == '__main__':
    # get data
#    st = '1998-01-02'
#    ed = '2019-12-31'
#    data, Y = get_test(st,ed,parameters='All')
#    feature = get_all_feature(False, start =st, end=ed)
#    # cal X and Y
#    x = GetKhorX(data, feature, n_lag=3).dropna() # data before 2000 and after 2020 all drop due to missing data
#    y = CategorizeY(Y,method='roll_vol').loc[x.index]
    x, y = Feature_Inputs(st = '1998-01-02', ed = '2019-12-31')

