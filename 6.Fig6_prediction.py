import pdb
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import warnings; warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from tqdm import tqdm
import csv
import glob
import string
from pylab import *
plt.rcParams['font.sans-serif'] = ['Times New Roman']
font_it={'family': 'serif', 'style': 'italic', 'fontname': 'Times New Roman'}
font_normal={'family': 'serif', 'fontname': 'Times New Roman'}
alphalist = list(string.ascii_lowercase)

inpath = './output/'
outpath = './Figure/'

runlist = ['NF_ALL_25','PF_ALL_25']

envlist= ['TEMP','RNET','VPD','Psum','PET','AI','Ms','Ga','Psurf']
biolist=['RD','LAI','VOD','Age','Hc']
plist = envlist+biolist

tlist_50 = ['C','g1','gpmax','P50x','P50s']                     
tlist_05=['C_05','g1_05','gpmax_05','P50x_05','P50s_05']     
tlist_95=['C_95','g1_95','gpmax_95','P50x_95','P50s_95']

cmiplist = ['TEMP','RNET','Psum','VPD']

startyear=2024
stopyear=2100
yearlist=range(startyear,stopyear)

tlist= ['C','g1','gpmax','P50x','P50s']
traitname = [r'$\mathregular{C}$  $\mathregular{(mm}$ $\mathregular{hr^{-1}}$ $\mathregular{MPa^{-1})}$',
             r'$\mathregular{g_1}$  $\mathregular{(kPa^{1/2})}$',
             r'$\mathregular{g_{p,max}}$  $\mathregular{(mm}$ $\mathregular{hr^{-1}}$ $\mathregular{MPa^{-1})}$',
             r'$\mathregular{P50_x}$  $\mathregular{(-MPa)}$',
             r'$\mathregular{P50_s}$  $\mathregular{(-MPa)}$']
landlist = ['NF','PF']
colorlist = ['#75c475','#ffc167']
zorderlist = [2,1]


tmp = pd.read_csv('./input/forcing25.csv').dropna(subset=plist+tlist+['PT'])
cmippath = './data/CMIP/cmip_cdf_yearly/'
dfcmip=pd.concat([pd.read_csv(filelist).assign(Year=yearlist[i]) for i,filelist in tqdm(enumerate(sorted(glob.glob(cmippath +'*.csv'))))])
dfcmip=dfcmip.drop(columns='time')

def line_withconst(x,k,b): 
    return k*x+b

def wlssim(data,model,trait):
    Y = data[trait]
    Weights = 1/(data[trait+'_95'] - data[trait+'_05'])
    X = data[model]
    X = sm.add_constant(X)
    WLS = sm.WLS(Y,X,weights=Weights).fit()
    RMSE = mean_squared_error(Y,WLS.predict(),sample_weight=Weights,squared=False)/Y.mean()
    para = str(list(WLS.params.values))
    para_std = str(list(WLS.bse.values))
    Modeldatatmp = pd.DataFrame(
        np.array([str(trait),len(Y),WLS.aic,WLS.rsquared,RMSE,str(model),para,para_std]).reshape(1,-1),
        columns=['Trait','Datanum','AIC','R2','RMSE','Model','Para','Para_std'])
    return Modeldatatmp,WLS 

def drawpic(trait):
    global Line_info
    Line_info = pd.DataFrame(columns=['Lands','k','const','r2','p'])
    Line_info['Lands'] = landlist[:len(runlist)]

    for runindex in range(len(runlist)):
        MODE=runlist[runindex]
        tree=MODE[0:2]
        Predictresult = pd.read_csv(inpath+'Predictresult_'+MODE+'.csv')
        year = np.arange(startyear,stopyear)
        mean = Predictresult[trait][startyear-stopyear:]
        y1 = mean
        x1 = sm.add_constant(year)
        OLS = sm.OLS(y1,x1).fit()

        Line_info['k'][runindex] = OLS.params[1]
        Line_info['const'][runindex] = OLS.params[0]
        Line_info['r2'][runindex] = OLS.rsquared
        Line_info['p'][runindex] = OLS.pvalues[0]

        year1 = year
        mean1 = mean
        ax1 = sns.lineplot(x=year1,y=Line_info['k'][runindex] * year1 + Line_info['const'][runindex],color=colorlist[runindex],linewidth=2.5,zorder=zorderlist[runindex])
        ax1 = sns.scatterplot(x=year1,y=mean1,s=20,color=colorlist[runindex],alpha=0.7)

def bootstrapping(trait,bootstrapping_size):
    for runindex in range(len(runlist)):
        result= pd.DataFrame()
        MODE=runlist[runindex]
        tree=MODE[0:2]

        MODEinfo = pd.read_csv(inpath +'/Modelresult_'+MODE+'.csv').set_index('Trait')
        Traitmean = pd.DataFrame(columns=['Year','C','g1','gpmax','P50x','P50s'])
        Traitmean['Year'] = range(startyear,stopyear+1)
        Traitmean = Traitmean.set_index('Year')

        data = tmp[tmp['PT']==tree]
        model= list(MODEinfo['Model'][trait].split("', '"))
        model[0] = model[0].strip("['");model[-1] = model[-1].strip("']")

        MODELtmp,WLS = wlssim(data,model,trait)

        result = pd.concat([result,MODELtmp])

        restlist = list(set(model) - set(cmiplist))
        predictlist = list(set(model) & set(cmiplist))

        if runindex ==2 : 
            data_pf = tmp[tmp['PT']=='PF']
            df_same = data_pf[['lat', 'lon'] + restlist]
        else: df_same = data[['lat', 'lon'] + restlist] 

        point = df_same[['lat','lon']]
        point['point'] = 1
        dfcmip_mean =  dfcmip.merge(point,on=['lat','lon'],how='inner').dropna(subset=['point']).groupby('Year').mean().drop(columns=['lat','lon','point'])

        df_predict = dfcmip_mean[predictlist]
        df_predict[restlist] = df_same[restlist].mean()
        df_predict['const'] = 1
        df_predict = df_predict[WLS.params.index]

        beta_mean = WLS.params.values
        beta_std = WLS.bse.values
        beta_cov_matrix = WLS.cov_params()
        beta_random_list = np.random.multivariate_normal(beta_mean, beta_cov_matrix, size=bootstrapping_size)
        bootstrapping_df = df_predict @ beta_random_list.T
        
        year = np.arange(startyear,stopyear)
        for i,samplenum in enumerate(bootstrapping_df.columns.tolist()):
            bootstrapping_sample = bootstrapping_df[samplenum][startyear-stopyear:]
            ax1 = sns.regplot(x=year,y=bootstrapping_sample,color=colorlist[runindex],ci=None,scatter_kws={'alpha':0},line_kws={'alpha':0.01})

fig = plt.figure(figsize=[20,13])
for k,trait in tqdm(enumerate(tlist)):
    ax1 = plt.subplot(2,3,k+1)
    bootstrapping(trait,200)
    drawpic(trait)
    ax1.set_title(str(traitname[k]),rotation=0,fontsize=26,pad=12)

    if trait == 'C':  yplotinfo = [4,16,4]
    if trait == 'g1': yplotinfo = [2,10,2]  
    if trait == 'gpmax': yplotinfo = [2,8,2]
    if trait == 'P50x':  yplotinfo = [0,5,1]
    if trait == 'P50s': yplotinfo = [0,1.2,0.4]

    ylim_down,ylim_up,delta_y = yplotinfo
    if trait == 'C':  R2plotinfo = [2019,ylim_down]
    if trait == 'g1': R2plotinfo = [2019,ylim_down]  
    if trait == 'gpmax': R2plotinfo = [2019,ylim_down]
    if trait == 'P50x':  R2plotinfo = [2019,ylim_down]
    if trait == 'P50s': R2plotinfo = [2019,ylim_down]

    y_ticks = np.arange(ylim_down,ylim_up+delta_y/2,delta_y)
    y_labels = ['%.1f'%num for num in y_ticks]
    ax1.set(xlabel='',xlim=(2014,2110),xticks=[2024,2040,2060,2080,2100],xticklabels=['2024','2040','2060','2080','2100'])
    plt.yticks(font='Times New Roman',rotation=0,fontsize = 20)
    plt.xticks(font='Times New Roman',fontsize = 20)

    xlim_down = 2014
    xlim_up = 2110
    picnum_y_position = ylim_up + (ylim_up-ylim_down)/15
    picnum_x_position = xlim_down - (xlim_up-xlim_down)/10
    ax1.text(picnum_x_position,picnum_y_position,alphalist[k]+')',fontsize=26,fontdict=font_normal,weight='bold')
    text_x_position,text_y_position = R2plotinfo
    line_y_space = (ylim_up - ylim_down)/15

    ax1.set(ylabel='',ylim=(ylim_down,ylim_up), yticks=y_ticks,yticklabels=y_labels)
    for i,Line_info_text in enumerate(landlist):
        r2 = Line_info['r2'][i]
        p = Line_info['p'][i]

        if r2<0 : continue
        line_y_position = text_y_position + (2-i) * line_y_space

        ax1.text(np.array(text_x_position),np.array(line_y_position),f'$\mathregular{{R^2= {"%.2f"%r2.round(2)}}}$,',
           fontdict=font_normal,fontsize=17,color=colorlist[i],weight='bold')
        ax1.text(np.array(text_x_position)+24,np.array(line_y_position),r'p',fontdict=font_it,fontsize=17,color=colorlist[i],weight='bold')
        if p < 0.01 :
           ax1.text(np.array(text_x_position)+27,np.array(line_y_position),f'$\mathregular{{< 0.01}}$',
               fontdict=font_normal,fontsize=17,color=colorlist[i],weight='bold')
        else: 
           ax1.text(np.array(text_x_position)+27,np.array(line_y_position),f'$\mathregular{{= {"%.2f"%p.round(2)}}}$',
               fontdict=font_normal,fontsize=17,color=colorlist[i],weight='bold')

ax1 = sns.lineplot(x=np.arange(2000,2002),y= -1,color=colorlist[0],linewidth=2, label='NF')
ax1 = sns.lineplot(x=np.arange(2000,2002),y= -1,color=colorlist[1],linewidth=2, label='PF')
plt.subplots_adjust(wspace =0.2,hspace=0.25)
plt.legend(fontsize=24,frameon=False,bbox_to_anchor=(1.75,0.55))
fig.savefig(outpath + 'Figure 6.jpg',dpi=300,bbox_inches='tight')






