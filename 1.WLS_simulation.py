import pdb
import numpy as np
import pandas as pd
import xarray as xr
import warnings; warnings.simplefilter("ignore")
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from itertools import combinations

inpath = './input/'
outpath = './output/'

runlist = ['NF_ALL_25','NF_WET_25','NF_DRY_25','PF_ALL_25','PF_WET_25','PF_DRY_25']

grid=runlist[0][-2:]

def Standardized(Data):
    SDataFrame = pd.DataFrame()
    for para in Data.columns.tolist():
        if np.isin(para,tlist_05+tlist_95):continue
        if np.isin(para,tlist_50):
            u = np.mean(pd.concat([Data[para],Data[para+'_05'],Data[para+'_95']],axis=0))
            sita = np.std(pd.concat([Data[para],Data[para+'_05'],Data[para+'_95']],axis=0))

            SDataFrame[para] = (Data[para] - u) / sita
            SDataFrame[para+'_05'] = (Data[para+'_05'] - u) / sita
            SDataFrame[para+'_95'] = (Data[para+'_95'] - u) / sita
        else:
            u = np.mean(Data[para])
            sita = np.std(Data[para])
            SDataFrame[para] = (Data[para] - u) / sita
    return SDataFrame


def wlssim(data,model,trait):
    Y = data[trait]
    Weights = 1/(data[trait+'_95'] - data[trait+'_05'])
    X = data[model]
    X = sm.add_constant(X)
    WLS = sm.WLS(Y,X,weights=Weights).fit()
    RMSE = mean_squared_error(Y,WLS.predict(),sample_weight=Weights,squared=False)/Y.mean()
    Modeldatatmp = pd.DataFrame(
        np.array([str(trait),len(Y),WLS.aic,WLS.rsquared,RMSE,str(model)]).reshape(1,-1),
        columns=['Trait','Datanum','AIC','R2','RMSE','Model'])
    return Modeldatatmp,WLS


envlist= ['TEMP','RNET','VPD','Psum','PET','AI','Ms','Ga','Psurf']
biolist=['RD','LAI','VOD','Age','Hc']
plist = envlist+biolist

tlist_50 = ['C','g1','gpmax','P50x','P50s']  
tlist_05=['C_05','g1_05','gpmax_05','P50x_05','P50s_05']
tlist_95=['C_95','g1_95','gpmax_95','P50x_95','P50s_95']
tlist = tlist_05 +tlist_50+tlist_95

datatmp = pd.read_csv(inpath + 'forcing'+grid+'.csv')
data = datatmp.dropna(subset=plist+tlist+['PT'])
print(len(data))

Modellist = []
for count1 in range(1,len(envlist)+1):
    for count2 in range(1,len(biolist)+1):
        Modellist = Modellist +[list(env)+list(bio) 
            for env in combinations(envlist,count1)
            for bio in combinations(biolist,count2)]

selectresult = pd.DataFrame()

for i,tree in enumerate(['NF','PF']):
    datatmp = pd.read_csv(inpath + 'forcing'+grid+'.csv')
    datatmp=datatmp[datatmp['PT']==tree]
    dataTMP = datatmp.dropna(subset=plist+tlist+['PT'])
    for trait in tlist_50:
        for i,modeltmp in enumerate(Modellist):
            tmp,_=wlssim(dataTMP,modeltmp,trait)
            if i == 0 : 
                select_tmp = tmp.copy()
                AIC_min = tmp.AIC.values
            if AIC_min >= tmp.AIC.values : 
                select_tmp = tmp.copy()
                AIC_min = tmp.AIC.values
            print('Modeltype:',str(trait),i+1,'of',len(Modellist))
        select_tmp['PT'] = tree    
        selectresult = pd.concat([selectresult,select_tmp])
MODELinfo = selectresult.set_index(['Trait','PT'])

for i,MODE in enumerate(runlist): 
    result= pd.DataFrame()
    tree = MODE[0:2]
    wetdry = MODE[3:6]

    if wetdry == 'ALL':forcing = data[data[pt]==tree] 
    else:forcing = data[(data[pt]==tree)&(data[AIindex]==wetdry)]
    for trait in tlist_50:

        model= list(MODELinfo['Model'][trait][tree].split("', '"))
        model[0] = model[0].strip("['");model[-1] = model[-1].strip("']")
        
        MODELtmp,WLS = wlssim(forcing,model,trait)
        print(MODELtmp)

        _,WLS_std = wlssim(Standardized(forcing[tlist+plist]),model,trait)
        params = WLS_std.params

        X = Standardized(forcing[model])
        X = sm.add_constant(X)
        beta_data = pd.DataFrame()
        beta_data['Variable'] = X.columns
        beta_data['beta'] = params.values
        MODELtmp['beta'] = [beta_data['beta'].tolist()]       
        result = pd.concat([result,MODELtmp])

    if wetdry == 'ALL':
        print(MODE)
        print(forcing[['lat','lon']])
        print(len(forcing))

    result.to_csv(outpath +'/Modelresult_'+MODE+'.csv',index=False)



