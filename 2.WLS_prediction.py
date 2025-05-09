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
startyear=yearlist[0] #2024
stopyear=yearlist[-1] #2099
saveyear=startyear-yearlist[0]

##==========================================================  Prediction =================================================================================
inpath = './input/'
outpath = './output/'
cmippath = './data/CMIP/cmip_yearly/'

envlist= ['TEMP','RNET','VPD','Psum','PET','AI','Ms','Ga','Psurf']  #ENV factors
biolist=['RD','LAI','VOD','Age','Hc']                               #BIO factors
plist = envlist+biolist

tlist_50 = ['C','g1','gpmax','P50x','P50s']                     
tlist_05=['C_05','g1_05','gpmax_05','P50x_05','P50s_05']     
tlist_95=['C_95','g1_95','gpmax_95','P50x_95','P50s_95']
tlist = tlist_05 +tlist_50+tlist_95

cmiplist = ['TEMP','RNET','Psum','VPD','Psurf']                     #Variables predicted by CMIP6

### Wls simulation
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

## Reading forcing data and CMIP6 data
tmp = pd.read_csv(inpath + 'forcing'+grid+'.csv').dropna(subset=plist+tlist+['PT'])
dfcmip=pd.concat([pd.read_csv(filelist).assign(Year=yearlist[i]) for i,filelist in tqdm(enumerate(sorted(glob.glob(cmippath +'*.csv'))))])

for runindex in range(len(runlist)):
    result= pd.DataFrame()
    MODE=runlist[runindex]
    tree=MODE[0:2]          # NF and PF

    ### Read model-selection result
    MODEinfo = pd.read_csv(outpath +'/Modelresult_'+MODE+'.csv').set_index('Trait')
    print(MODEinfo)

    Traitmean = pd.DataFrame(columns=['Year','C','g1','gpmax','P50x','P50s'])
    Traitmean['Year'] = range(startyear,stopyear+1)
    Traitmean = Traitmean.set_index('Year')

    data = tmp[tmp['PT']==tree]

    for trait in tqdm(tlist_50):
        model= list(MODEinfo['Model'][trait].split("', '"))
        model[0] = model[0].strip("['");model[-1] = model[-1].strip("']")

        MODELtmp,WLS = wlssim(data,model,trait)
        result = pd.concat([result,MODELtmp])

        predictlist = list(set(model) & set(cmiplist)) # Variables which are predicted by cmip6 in the selected model
        restlist = list(set(model) - set(cmiplist))    # Variables which are not predicted by cmip6 in the selected model
        df_predict = dfcmip[['lat', 'lon','Year'] + predictlist]
        df_same = data[['lat', 'lon'] + restlist]

        df_year = pd.merge(df_same,df_predict,on=['lat','lon'],how='inner').set_index(['lat', 'lon','Year']).dropna()

        print('')
        print('Point num for model: ',MODEinfo['Datanum'][trait])
        print('Point num for prediction: ',len(df_year)/len(sorted(glob.glob(cmippath +'*.csv'))))
        # Check whether the count of predicted pixels correspond to that used for building the WLS model 

        WLS_output = WLS.predict(sm.add_constant(df_year[model])).reset_index().groupby('Year').mean().drop(columns=['lat','lon']).values
        Traitmean[trait] = WLS_output[startyear-stopyear-1:]
        
    Traitmean.to_csv(outpath +'/Predictresult_'+runlist[runindex]+'.csv')