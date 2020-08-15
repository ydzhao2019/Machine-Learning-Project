import numpy as np
import pandas as pd
import os
    
path1='data/'
path2='output/'

notnum=[]
def change_to_num(x):
    global notnum
    if type(x)==str:
        x=x.replace(',','')
        if x[-1]=='M':
            return float(x[:-1])*1e6
        elif x[-1]=='B':
            return float(x[:-1])*1e9
        elif x[-1]=='K':
            return float(x[:-1])*1e3  
        elif x[-1]=='%':
            return float(x[:-1].replace(',',''))*1e-2
        elif x=='-':
            return np.nan
        return float(x)
    else:
        if x==x:
            return x
        else:
            notnum+=[str(x)]       
            
drop=pd.read_csv(path1+'drop_list.csv').applymap(lambda x:x[2:-1]).values[:,0]

def get_Bond(tocsv=False):
    Treasury_Yield_2Y=pd.read_csv(path1+'Bond/2Y_Treasury_Yield.csv',index_col=0,parse_dates=True)
    Treasury_Yield_2Y['Change']=Treasury_Yield_2Y['Change %'].map(change_to_num)
    Treasury_Yield_2Y.drop(columns=['Change %'],inplace=True)
    Treasury_Yield_2Y.columns=np.array(Treasury_Yield_2Y.columns)+'_Yield2y'
    Treasury_Yield_2Y=Treasury_Yield_2Y.iloc[::-1,:]
    Treasury_Yield_10Y=pd.read_csv(path1+'Bond/10Y_Treasury_Yield.csv',index_col=0,parse_dates=True)
    Treasury_Yield_10Y['Change']=Treasury_Yield_10Y['Change %'].map(change_to_num)
    Treasury_Yield_10Y.drop(columns=['Change %'],inplace=True)
    Treasury_Yield_10Y.columns=np.array(Treasury_Yield_10Y.columns)+'_Yield10y'
    Treasury_Yield_10Y=Treasury_Yield_10Y.iloc[::-1,:]
    Treasury_Yield_30Y=pd.read_csv(path1+'Bond/30Y_Treasury_Yield.csv',index_col=0,parse_dates=True)
    Treasury_Yield_30Y['Change']=Treasury_Yield_30Y['Change %'].map(change_to_num)
    Treasury_Yield_30Y.drop(columns=['Change %'],inplace=True)
    Treasury_Yield_30Y.columns=np.array(Treasury_Yield_30Y.columns)+'_Yield30y'
    Treasury_Yield_30Y=Treasury_Yield_30Y.iloc[::-1,:]
    Bond=pd.concat([Treasury_Yield_2Y,Treasury_Yield_10Y,Treasury_Yield_30Y],axis=1,join='outer')
    if tocsv==True:
        Bond.to_csv(path2+'Bond.csv')
    return Bond
    
def get_FX(tocsv=False):
    Dollar_index=pd.read_csv(path1+'FX/US Dollar Index Historical Data.csv',index_col=0,parse_dates=True)
    Dollar_index['Change']=Dollar_index['Change %'].map(change_to_num)
    Dollar_index.drop(columns=['Change %'],inplace=True)
    Dollar_index.columns=np.array(Dollar_index.columns)+'_Dollar_index'
    Dollar_index=Dollar_index.iloc[::-1,:]
    FX=Dollar_index
    if tocsv==True:
        FX.to_csv(path2+'FX.csv')
    return FX
    
def get_Equity(tocsv=False):
    df={}
    for info in os.listdir('data/Equity/'):
        if (info[-3:] == 'csv') and (info not in drop):
            domain = os.path.abspath(r'data/Equity/') #获取文件夹的路径
            data = pd.read_csv(os.path.join(domain,info),index_col=0,parse_dates=True)
            df[info]=data
    for key in df.keys():
        data=df[key]
        data_num=data.applymap(change_to_num)
        data_num.columns=np.array(data_num.columns)+'_'+key.split()[0].split('.')[0]
        df[key]=data_num
    Equity=pd.concat([df[key] for key in df.keys()],axis=1,join='outer')
    # print(np.unique(notnum)) #data中存在的缺失数据以什么表示
    if tocsv==True:
        Equity.to_csv(path2+'Equity.csv')
    return Equity
    
def get_Commodity(tocsv=False):
    folder=['PRECIOUS METALS','INDUSTRIAL METALS','ENERGY','AGRICULTURE']
    PRECIOUS_METALS,INDUSTRIAL_METALS,ENERGY,AGRICULTURE=np.nan,np.nan,np.nan,np.nan
    mid_df=[PRECIOUS_METALS,INDUSTRIAL_METALS,ENERGY,AGRICULTURE]
    for i in range(len(folder)):
        df={}
        f=folder[i]
        for info in os.listdir('data/Commodity/'+f):
            if (info[-3:] == 'csv') and (info not in drop):
                domain = os.path.abspath(r'data/Commodity/'+f) #获取文件夹的路径
                data = pd.read_csv(os.path.join(domain,info),index_col=0,parse_dates=True)
                df[info]=data
        for key in df.keys():
            data=df[key]
            data['Vol.']=data['Vol.'].map(change_to_num)
            data['Change']=data['Change %'].map(change_to_num)
            data.drop(columns=['Change %'],inplace=True)
            data[['Price','Open','High','Low']] = data[['Price','Open','High','Low']].applymap(change_to_num)
            data.columns=np.array(data.columns)+'_'+key[:-28]
            df[key]=data
        mid_df[i]=pd.concat([df[key] for key in df.keys()],axis=1,join='outer')
    Commodity=pd.concat(mid_df,axis=1,join='outer')
    if tocsv==True:
        Commodity.to_csv(path2+'Commodity.csv')
    return Commodity
    
def get_Other(tocsv=False):
    VIX=pd.read_csv(path1+'Other\CBOE Volatility Index Historical Data_2.csv',index_col=0,parse_dates=True)
    VIX['Vol.']=VIX['Vol.'].map(change_to_num)
    VIX['Change']=VIX['Change %'].map(change_to_num)
    VIX.drop(columns=['Change %'],inplace=True)
    VIX.columns=np.array(VIX.columns)+'_VIX'
    FF5=pd.read_csv(path1+'Other\F-F_Research_Data_5_Factors_2x3_daily.csv',index_col=0,parse_dates=True)
    Other=pd.concat([VIX,FF5],axis=1,join='outer')
    if tocsv==True:
        Other.to_csv(path2+'Other.csv')
    return Other
    
def get_all_data(tocsv=False):
    Bond=get_Bond()
    FX=get_FX()
    Commodity=get_Commodity()
    Equity=get_Equity()
    Other=get_Other()
    full=pd.concat([Bond,FX,Commodity,Equity,Other],axis=1,join='outer')
    if tocsv==True:
        full.to_csv(path2+'full.csv')
    return full

def full_clean():
    full=get_all_data()
    full['Y']=full['Close_^SP500'].dropna()/full['Close_^SP500'].dropna().shift(1)-1
    index=full['Y'].dropna().index
    full_na=full.fillna(method='ffill')
    full_na.drop(columns=['Open_^SP500', 'High_^SP500','Low_^SP500', 'Close_^SP500', 'Adj Close_^SP500', 'Volume_^SP500',
                      'vwretd_SP500return(dvd)', 'vwretx_SP500return(dvd)',
                      'ewretd_SP500return(dvd)', 'ewretx_SP500return(dvd)',
                      'totval_SP500return(dvd)', 'sprtrn_SP500return(dvd)'],inplace=True)
    #drop所有和SP500index的数据
    full_na.iloc[:,:-1]=full_na.iloc[:,:-1].shift(1)
    #X均为shift一期
    full_na=full_na.loc[index,:]
    #保留对应index日期的数据
    full_na.dropna(axis=1,how='all',inplace=True)
    #drop 全为nan的列
    return full_na

def get_test(st_time:str, ed_time:str, parameters='All'):
    full_na=full_clean()
    index=full_na.index
    if st_time in index:
        if ed_time in index:
            df = full_na[st_time:ed_time]
        else:
            for day in range(int(ed_time[-2:]),0,-1):
                new_ed_time = ed_time[:-2]+str(day)
                if new_ed_time in index:
                    break
            df = full_na[st_time:new_ed_time]
    else:
        for day in range(int(st_time[-2:]),0,-1):
            new_st_time = st_time[:-2]+str(day)
            if new_st_time in index:
                break
        if ed_time in index:
            df = full_na[new_st_time:ed_time]
        else:
            for day in range(int(ed_time[-2:]),0,-1):
                new_ed_time = ed_time[:-2]+str(day)
                if new_ed_time in index:
                    break        
            df = full_na[new_st_time:new_ed_time]        
    if parameters == 'All':
        return df.iloc[:,:-1], df['Y']
    else:
        return df[parameters], df['Y']
    

if __name__ == '__main__':
    cols = ['Price_VIX','Close_SPY','Price_Yield2y']
    X, Y = get_test('2000-01-04','2020-06-19',
#                      parameters=['Price_Yield2y', 'Open_Yield2y', 'High_Yield2y', 'Low_Yield2y','Change_Yield2y', 'Price_Yield10y', 'Open_Yield10y','High_Yield10y'],
                      parameters='All'
                      )
    print(X)
