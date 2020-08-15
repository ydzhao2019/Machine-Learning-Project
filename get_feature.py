import numpy as np
import pandas as pd
import talib as ta
from sklearn.decomposition import PCA

def get_asset(data): #get asset name
    assetList = []
    for col in data.columns:
        if col.startswith('Price_'):
            assetList.append(col[6:])
    return assetList

def get_SMA(data,n=20): #n days SMA
    assetList = get_asset(data)    
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'_SMA']=ta.SMA(data['Price_'+asset].dropna(),n)
    df = df.dropna(how='all')
    return df

def get_return(data,n=1): #return
    assetList = get_asset(data)
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'_'+str(n)+'day_return']=data['Price_'+asset].dropna().pct_change(n)
    df = df.dropna(how='all')
    return df

#volatility
#------------------------------------------------------------
def get_vol(data,n=20): #n days vol
    assetList = get_asset(data) 
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'_vol']=ta.SMA(data['Price_'+asset].dropna(),n)
    df = df.dropna(how='all')
    return df

def get_ATR(data,n=14): #n days ATR
    assetList = get_asset(data) 
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'_ATR']=ta.ATR(data['High_'+asset].dropna(),data['Low_'+asset].dropna(),data['Price_'+asset].dropna(),n)
    df = df.dropna(how='all')
    return df

def get_ATR_status(data,n1=10,n2=50): #n1 days ATR vs n2 days ATR, ATR(n1)>ATR(n2): high vol 1; otherwise: low vol 0
    atr1 = get_ATR(data,n1)
    atr2 = get_ATR(data,n2)
    atr_diff = atr1-atr2
    df = pd.DataFrame(0,columns=atr_diff.columns,index=atr_diff.index)
    df[atr_diff>0]=1
    df[atr_diff<0]=0
    df[np.isnan(atr_diff)]=np.nan
    df.columns = [i+'_status' for i in df.columns]
    df = df.dropna(how='all')
    return df

def get_ATR_status_change(data,n1=10,n2=50): # ATR status change low to high 1; high to low -1
    atr_status = get_ATR_status(data,n1,n2)
    df = atr_status.diff()
    df.columns = [i+'_change' for i in df.columns]
    df = df.dropna(how='all')
    return df

#trend
#------------------------------------------------------------

def get_RSI(data,n=14): #n days RSI
    assetList = get_asset(data) 
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'RSI']=ta.RSI(data['Price_'+asset].dropna(),n)
    df = df.dropna(how='all')
    return df

def get_RSI_status(data,n=14): #n days RSI status: oversell: RSI<30 1; overbuy:>70  -1; otherwise 0
    rsi = get_RSI(data,n)
    df = pd.DataFrame(0,columns=rsi.columns,index=rsi.index)
    df[rsi<30]=1
    df[rsi>70]=-1
    df[np.isnan(rsi)]=np.nan
    df.columns = [i+'_status' for i in df.columns]
    df = df.dropna(how='all')
    return df

def get_RSI_status_change(data,n=14): #n days RSI status change: long:1; short:-1
    rsi_status = get_RSI_status(data,n)
    df = rsi_status.diff()
    df.columns = [i+'_change' for i in df.columns]
    df = df.dropna(how='all')
    return df

def get_CCI(data,n=14): #n days CCI
    assetList = get_asset(data) 
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'_CCI']=ta.CCI(data['High_'+asset].dropna(),data['Low_'+asset].dropna(),data['Price_'+asset].dropna(),n)
    df = df.dropna(how='all')
    return df

def get_CCI_status(data,n=14): #n days CCI status: long: CCI>+100 1; short:<-100 -1; otherwise 0
    cci = get_CCI(data,n)
    df = pd.DataFrame(0,columns= cci.columns,index=cci.index)
    df[cci>100]=1
    df[cci<-100]=-1
    df[np.isnan(cci)]=np.nan
    df.columns = [i+'_status' for i in df.columns]
    df = df.dropna(how='all')
    return df

def get_CCI_status_change(data,n=14): #n days RSI status change: long:1; short:-1
    cci_status = get_CCI_status(data,n)
    df = cci_status.diff()
    df.columns = [i+'_change' for i in df.columns]
    df = df.dropna(how='all')
    return df

#spread
#------------------------------------------------------------
def get_open_to_close_spread(data): #close - open
    assetList = get_asset(data) 
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'_OC_spread'] = data['Price_'+asset]-data['Open_'+asset]
    df = df.dropna(how='all')
    return df

def get_low_to_high_spread(data): #high - low
    assetList = get_asset(data) 
    df = pd.DataFrame()
    for asset in assetList:
        df[asset+'_LH_spread'] = data['High_'+asset]-data['Low_'+asset]
    df = df.dropna(how='all')
    return df

def get_cross_asset_price_spread(data): #asset1 price - asset2 price
    assetList = get_asset(data) 
    df = pd.DataFrame()
    for i in range(len(assetList)):
        for j in range(len(assetList)):
            if i!=j:
                df[assetList[i]+'-'+assetList[j]] = data['Price_'+assetList[i]]-data['Price_'+assetList[j]]
    df = df.dropna(how='all')
    return df

def get_all_feature(cross_asset_spread=True,full=True,start='1994-01-04',end='2020-06-19'):
    data = pd.read_csv('output/full.csv',index_col=0)
    df = []
    df.append(get_SMA(data))
    df.append(get_return(data,1))
    df.append(get_return(data,5))
    df.append(get_return(data,10))
    df.append(get_return(data,30))
    df.append(get_return(data,60))
    df.append(get_vol(data))
    df.append(get_ATR(data))
    df.append(get_ATR_status(data))
    df.append(get_ATR_status_change(data))
    df.append(get_RSI(data))
    df.append(get_RSI_status(data))
    df.append(get_RSI_status_change(data))
    df.append(get_CCI(data))
    df.append(get_CCI_status(data))
    df.append(get_CCI_status_change(data))
    df.append(get_open_to_close_spread(data))
    df.append(get_low_to_high_spread(data))
    if cross_asset_spread:
        if full:
            df.append(get_cross_asset_price_spread(data))
        else:
            bond = pd.read_csv('output/Bond.csv',index_col=0)
            commodity = pd.read_csv('output/Commodity.csv',index_col=0)
            equity = pd.read_csv('output/Equity.csv',index_col=0)
            fx = pd.read_csv('output/FX.csv',index_col=0)
            df.append(get_cross_asset_price_spread(bond))
            df.append(get_cross_asset_price_spread(commodity))
            df.append(get_cross_asset_price_spread(equity))
            df.append(get_cross_asset_price_spread(fx))
    df=pd.concat(df,axis=1,sort=False)
    df=df.sort_index()
    df=df['1994-01-04':].ffill().dropna(axis=1)
    df=df.loc[start:end]
    return df

def get_PCA_factors(data,n=120,components=20):
    new_factor = []
    index=[]
    explained_var = []
    for i in range(n,len(data.index)):
        X = data[data.index[i-n]:data.index[i]]
        pca = PCA(n_components=components)
        pca.fit(X)
        X_new = pca.transform(X)
        new_factor.append(X_new[-1])
        index.append(data.index[i])
        explained_var.append(sum(pca.explained_variance_ratio_))
    df = pd.DataFrame(new_factor,index=index)
    df.columns = ['PC'+str(int(i)+1) for i in df.columns]
    explained_var = pd.DataFrame(explained_var,index=index)
    explained_var.columns=['Explained Var']
    return df, explained_var 


#example
if __name__ == '__main__':
    #feature
    df1 = get_all_feature() #all feature for the whole time period
    df2 = get_all_feature(False) #all feature withour cross asset spread for the whole time period
    df3 = get_all_feature(True,False) #all feature with cross asset spread in different section for the whole time period
    df4 = get_all_feature(start ='2019-01-01',end='2020-01-12') #all feature from start to end 
    df5 = get_all_feature(False,start ='2019-01-01',end='2020-01-01') #all feature withour cross asset spread from start to end 
    df6 = get_all_feature(True,False,start ='2019-01-01',end='2020-01-01') #all feature with cross asset spread in different section from start to end 
    #pca factors 
    pca_factor, explained_var =get_PCA_factors(df1) #default rolling:120, default components:20, explain var is a series of rolling explained var
    pca_factor, explained_var =get_PCA_factors(df1,120,20)
