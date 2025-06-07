import pdb
import glob
import path
import numpy as np
import pandas as pd
import xarray as xr
import warnings; warnings.simplefilter("ignore")
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from itertools import combinations

runlist = ['NF_ALL_25','PF_ALL_25']
grid=runlist[0][-2:]

yearlist=range(2024,2100)
startyear=yearlist[0]
stopyear=yearlist[-1]
saveyear=startyear-yearlist[0]

inpath = './input/'
outpath = './output/'
cmippath = './data/CMIP/cmip_yearly/'

envlist= ['TEMP','RNET','VPD','Psum','PET','AI','Ms','Ga','Psurf']
biolist=['RD','LAI','VOD','Age','Hc']
plist = envlist+biolist

tlist_50 = ['C','g1','gpmax','P50x','P50s']                     
tlist_05=['C_05','g1_05','gpmax_05','P50x_05','P50s_05']     
tlist_95=['C_95','g1_95','gpmax_95','P50x_95','P50s_95']
tlist = tlist_05 +tlist_50+tlist_95

cmiplist = ['TEMP','RNET','Psum','VPD','Psurf']

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


tmp = pd.read_csv(inpath + 'forcing'+grid+'.csv').dropna(subset=plist+tlist+['PT'])
dfcmip=pd.concat([pd.read_csv(filelist).assign(Year=yearlist[i]) for i,filelist in tqdm(enumerate(sorted(glob.glob(cmippath +'*.csv'))))])

for percentage in [35,40,45]:
    for runindex in range(len(runlist)):
        result= pd.DataFrame()
        MODE=runlist[runindex]
        tree=MODE[0:2]

        MODEinfo = pd.read_csv(outpath +'/Modelresult_'+MODE+'.csv').set_index('Trait')
        print(MODEinfo)

        Traitmean = pd.DataFrame(columns=['Year','C','g1','gpmax','P50x','P50s'])
        Traitmean['Year'] = range(startyear,stopyear+1)
        Traitmean = Traitmean.set_index('Year')

        data = tmp[tmp['PT']==tree][tmp['PT_'+tree+'%']>=percentage]

        for trait in tqdm(tlist_50):
            model= list(MODEinfo['Model'][trait].split("', '"))
            model[0] = model[0].strip("['");model[-1] = model[-1].strip("']")

            MODELtmp,WLS = wlssim(data,model,trait)
            result = pd.concat([result,MODELtmp])

            predictlist = list(set(model) & set(cmiplist))
            restlist = list(set(model) - set(cmiplist))
            df_predict = dfcmip[['lat', 'lon','Year'] + predictlist]
            df_same = data[['lat', 'lon'] + restlist]
            df_year = pd.merge(df_same,df_predict,on=['lat','lon'],how='inner').set_index(['lat', 'lon','Year']).dropna()

            WLS_output = WLS.predict(sm.add_constant(df_year[model])).reset_index().groupby('Year').mean().drop(columns=['lat','lon']).values
            Traitmean[trait] = WLS_output[startyear-stopyear-1:]
            
        if percentage == 40: Traitmean.to_csv(outpath +'Predictresult_'+runlist[runindex]+'.csv')
        else:Traitmean.to_csv(outpath +str(percentage) +'/Predictresult_'+runlist[runindex]+'.csv')