import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import warnings; warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import statsmodels.api as sm
import csv
import os
import string
from pylab import *
import matplotlib.font_manager as font_manager
from matplotlib.patches import FancyArrow
plt.rcParams['font.sans-serif'] = ['Times New Roman']
font_it={'family': 'serif', 'style': 'italic', 'fontname': 'Times New Roman'}
font_normal={'family': 'serif', 'fontname': 'Times New Roman'}
alphalist = list(string.ascii_lowercase)

runlist = ['NF_ALL_25','NF_WET_25','NF_DRY_25','PF_ALL_25','PF_WET_25','PF_DRY_25']

inpath = './output/'
outpath = './Figure/'

envlist= ['TEMP','RNET','VPD','Psum','Psurf','PET','AI','Ms','Ga']
biolist=['RD','LAI','VOD','Age','Hc']
plist = envlist+biolist

plistname=[ r'$\mathregular{T_a}$',
            r'$\mathregular{R_n}$',
            r'$\mathregular{VPD}$',
            r'$\mathregular{P_r}$',
            r'$\mathregular{P_s}$',
            r'$\mathregular{PET}$',
            r'$\mathregular{DI}$',
            r'$\mathregular{M_s}$',
            r'$\mathregular{G_a}$',
            r'$\mathregular{D_r}$',
            r'$\mathregular{LAI}$',
            r'$\mathregular{VOD}$',
            r'$\mathregular{A_t}$',
            r'$\mathregular{H_c}$']
            
tlist = ['C','g1','gpmax','P50x','P50s']
traitname = [r'$\mathregular{C}$',
             r'$\mathregular{g_1}$',
             r'$\mathregular{g_{p,max}}$',
             r'$\mathregular{P50_x}$',
             r'$\mathregular{P50_s}$']

traitshape = ['o','*','s','^','v']
traitsize  = [200,500,150,250,250]          
traitcolor = ['#FF0000']*5
AIcolor = ['#008800','#000000']
markAI=['s','^']
AIname=['DI ≤ 1.5','DI > 1.5']

tick_label_size = 14

def pvalue_plot(ax,data_df,pvalue_df,heatmap_colorchange):
    for i in range(data_df.shape[0]):
        for j in range(data_df.shape[1]):
            data_point =data_df[i][j]
            pvalue_point = pvalue_df[i][j]
            if pvalue_point < 0.05:
                if pvalue_point > 0.01: 
                    star='*'
                    if np.abs(data_point) >= heatmap_colorchange: 
                        ax.text(j + 0.8, i + 0.35, star, ha='center', va='center', color='white', fontsize=8)
                    else:ax.text(j + 0.8, i + 0.35, star, ha='center', va='center', color='black', fontsize=8)
                elif pvalue_point > 0.001:
                    star='**'
                    if np.abs(data_point) >= heatmap_colorchange: 
                        ax.text(j + 0.8, i + 0.35, star, ha='center', va='center', color='white', fontsize=8)
                    else:ax.text(j + 0.8, i + 0.35, star, ha='center', va='center', color='black', fontsize=8)
                else:
                    star='***'
                    if np.abs(data_point) >= heatmap_colorchange: 
                        ax.text(j + 0.8, i + 0.35, star, ha='center', va='center', color='white', fontsize=8)
                    else:ax.text(j + 0.8, i + 0.35, star, ha='center', va='center', color='black', fontsize=8)

mr_NF_ALL = pd.read_csv(inpath + 'Modelresult_'+runlist[0]+'.csv').set_index('Trait') #NF_ALL
mr_NF_WET = pd.read_csv(inpath + 'Modelresult_'+runlist[1]+'.csv').set_index('Trait') #NF_WET
mr_NF_DRY = pd.read_csv(inpath + 'Modelresult_'+runlist[2]+'.csv').set_index('Trait') #NF_DRY
mr_PF_ALL = pd.read_csv(inpath + 'Modelresult_'+runlist[3]+'.csv').set_index('Trait') #PF_ALL
mr_PF_WET = pd.read_csv(inpath + 'Modelresult_'+runlist[4]+'.csv').set_index('Trait') #PF_WET
mr_PF_DRY = pd.read_csv(inpath + 'Modelresult_'+runlist[5]+'.csv').set_index('Trait') #PF_DRY


#Fig 4
fig = plt.figure(figsize=[12,12])

xlim_down = 0
xlim_up = 1
ylim_down = 0
ylim_up = 1
picnum_y_position = ylim_up + (ylim_up-ylim_down)/20
picnum_x_position = xlim_down - (xlim_up-xlim_down)/10 *1.3

ax1 = plt.subplot(2,2,1)
ax1.text(picnum_x_position,picnum_y_position,alphalist[0]+')',fontsize=20,fontdict=font_normal,weight='bold')
for traitnum,trait in enumerate(tlist):
    data = {'x': [mr_NF_ALL['R2'][trait]], 'y': [mr_PF_ALL['R2'][trait]]}
    df = pd.DataFrame(data)
    ax1 = sns.scatterplot(x='x',y='y',s=traitsize[traitnum],data = df,color=traitcolor[traitnum],marker=traitshape[traitnum],alpha=0.5)

ax1.set(ylim=(0,1),yticks=[0,0.2,0.4,0.6,0.8,1.0],yticklabels=['0','0.2','0.4','0.6','0.8','1.0'],
xlim=(0,1),xticks=[0,0.2,0.4,0.6,0.8,1.0],xticklabels=['0','0.2','0.4','0.6','0.8','1.0'])
plt.yticks(font='Times New Roman',rotation=0,fontsize=15)
plt.xticks(font='Times New Roman',fontsize=15)
plt.ylabel('PF $\mathregular{R^2}$',font='Times New Roman',fontsize=18)
plt.xlabel('NF $\mathregular{R^2}$',font='Times New Roman',fontsize=18)

x = np.arange(0,1,0.001)
plt.plot(x,x,'-.',color='#737373',linewidth=2)
arrow = FancyArrow(0.6, 0.6, 0.2, -0.2, width=0.005, head_width=0.03,color='#737373')
ax1.add_patch(arrow)
ax1.text(0.75,0.28,'NF Better',fontsize=17,color='#000000')



ax1 = plt.subplot(2,2,2)
ax1.text(picnum_x_position,picnum_y_position,alphalist[1]+')',fontsize=20,fontdict=font_normal,weight='bold')
for traitnum,trait in enumerate(tlist):
    data = {'x': [mr_NF_WET['R2'][trait]], 'y': [mr_PF_WET['R2'][trait]]}
    df = pd.DataFrame(data)
    ax1 = sns.scatterplot(x='x',y='y',s=traitsize[traitnum],data = df,color=AIcolor[0],marker=traitshape[traitnum],alpha=0.5)
    data = {'x': [mr_NF_DRY['R2'][trait]], 'y': [mr_PF_DRY['R2'][trait]]}
    df = pd.DataFrame(data)
    ax1 = sns.scatterplot(x='x',y='y',s=traitsize[traitnum],data = df,color=AIcolor[1],marker=traitshape[traitnum],alpha=0.5)

ax1.set(ylim=(0,1),yticks=[0,0.2,0.4,0.6,0.8,1.0],yticklabels=['0','0.2','0.4','0.6','0.8','1.0'],
xlim=(0,1),xticks=[0,0.2,0.4,0.6,0.8,1.0],xticklabels=['0','0.2','0.4','0.6','0.8','1.0'])
plt.yticks(font='Times New Roman',rotation=0,fontsize=15)
plt.xticks(font='Times New Roman',fontsize=15)
plt.ylabel('',font='Times New Roman',fontsize=18)
plt.xlabel('NF $\mathregular{R^2}$',font='Times New Roman',fontsize=18)

x = np.arange(0,1,0.001)
plt.plot(x,x,'-.',color='#737373',linewidth=2)
arrow = FancyArrow(0.2, 0.6, 0.2, 0.2, width=0.005, head_width=0.03,color='#737373')
ax1.add_patch(arrow)
ax1.text(0.35,0.88,'Dry Better',fontsize=17,color='#000000')



ax1 = plt.subplot(2,2,3)
ax1.text(picnum_x_position,picnum_y_position,alphalist[2]+')',fontsize=20,fontdict=font_normal,weight='bold')
for traitnum,trait in enumerate(tlist):
    data = {'x': [mr_NF_ALL['RMSE'][trait]], 'y': [mr_PF_ALL['RMSE'][trait]]}
    df = pd.DataFrame(data)
    ax1 = sns.scatterplot(x='x',y='y',s=traitsize[traitnum],data = df,color=traitcolor[traitnum],marker=traitshape[traitnum],alpha=0.5,label=traitname[traitnum])

ax1.set(ylim=(0,1),yticks=[0,0.2,0.4,0.6,0.8,1.0],yticklabels=['0','0.2','0.4','0.6','0.8','1.0'],
xlim=(0,1),xticks=[0,0.2,0.4,0.6,0.8,1.0],xticklabels=['0','0.2','0.4','0.6','0.8','1.0'])
plt.yticks(font='Times New Roman',rotation=0,fontsize=15)
plt.xticks(font='Times New Roman',fontsize=15)
plt.ylabel('PF $\mathregular{RMSE}$',font='Times New Roman',fontsize=18)
plt.xlabel('NF $\mathregular{RMSE}$',font='Times New Roman',fontsize=18)

x = np.arange(0,1,0.001)
plt.plot(x,x,'-.',color='#737373',linewidth=2)
arrow = FancyArrow(0.6, 0.6, -0.2, 0.2, width=0.005, head_width=0.03,color='#737373')
ax1.add_patch(arrow)
ax1.text(0.3,0.87,'NF Better',fontsize=17,color='#000000')
ax1.legend(fontsize=17,frameon=False,loc=4,handletextpad=0.5)


ax1 = plt.subplot(2,2,4)
ax1.text(picnum_x_position,picnum_y_position,alphalist[3]+')',fontsize=20,fontdict=font_normal,weight='bold')
for traitnum,trait in enumerate(tlist):
    data = {'x': [mr_NF_WET['RMSE'][trait]], 'y': [mr_PF_WET['RMSE'][trait]]}
    df = pd.DataFrame(data)
    ax1 = sns.scatterplot(x='x',y='y',s=traitsize[traitnum],data = df,color=AIcolor[0],marker=traitshape[traitnum],alpha=0.5)
    data = {'x': [mr_NF_DRY['RMSE'][trait]], 'y': [mr_PF_DRY['RMSE'][trait]]}
    df = pd.DataFrame(data)
    ax1 = sns.scatterplot(x='x',y='y',s=traitsize[traitnum],data = df,color=AIcolor[1],marker=traitshape[traitnum],alpha=0.5)

ax1.set(ylim=(0,1),yticks=[0,0.2,0.4,0.6,0.8,1.0],yticklabels=['0','0.2','0.4','0.6','0.8','1.0'],
xlim=(0,1),xticks=[0,0.2,0.4,0.6,0.8,1.0],xticklabels=['0','0.2','0.4','0.6','0.8','1.0'])
plt.yticks(font='Times New Roman',rotation=0,fontsize=15)
plt.xticks(font='Times New Roman',fontsize=15)
plt.ylabel('',font='Times New Roman',fontsize=18)
plt.xlabel('NF $\mathregular{RMSE}$',font='Times New Roman',fontsize=18)

x = np.arange(0,1,0.001)
plt.plot(x,x,'-.',color='#737373',linewidth=2)
arrow = FancyArrow(0.5, 0.9, -0.2, -0.2, width=0.005, head_width=0.03,color='#737373')
ax1.add_patch(arrow)
ax1.text(0.15,0.6,'Dry Better',fontsize=17,color='#000000')

patch0 = mpatches.Patch(color=(int(AIcolor[0][1:3], 16)/255,int(AIcolor[0][3:5], 16)/255,int(AIcolor[0][5:7], 16)/255,0.5), label=AIname[0])
patch1 = mpatches.Patch(color=(int(AIcolor[1][1:3], 16)/255,int(AIcolor[1][3:5], 16)/255,int(AIcolor[1][5:7], 16)/255,0.5), label=AIname[1])
plt.legend(handles=[patch0,patch1],fontsize=17,frameon=False,loc=4,handlelength=1.1)

plt.subplots_adjust(wspace =0.15,hspace=0.22)
fig.savefig(outpath +'Figure 4.jpg',dpi=300,bbox_inches = 'tight')


#Fig 5
fig = plt.figure(figsize = (15,7))
ax1 = plt.subplot(211)
data_ax1 = pd.DataFrame(columns={'Variable':['const']+plist}).set_index('Variable').to_xarray()
pvalue_ax1 = pd.DataFrame(columns={'Variable':['const']+plist}).set_index('Variable').to_xarray()
for trait in tlist:
    model = list(mr_NF_ALL['Model'][trait].split("', '"))
    model[0] = model[0].strip("['");model[-1] = model[-1].strip("']")
    beta = mr_NF_ALL['beta'][trait].strip("[").strip("]")
    beta = np.array([float(num_str) for num_str in beta.split(',')])
    df_tmp = pd.DataFrame({'Variable':['const']+model,trait:beta}).set_index('Variable').to_xarray()
    data_ax1 = xr.merge([data_ax1,df_tmp])
    pvalue = mr_NF_ALL['pvalue'][trait].strip("[").strip("]")
    pvalue = np.array([float(num_str) for num_str in pvalue.split(',')])
    df_tmp = pd.DataFrame({'Variable':['const']+model,trait:pvalue}).set_index('Variable').to_xarray()
    pvalue_ax1 = xr.merge([pvalue_ax1,df_tmp])
data_ax1=data_ax1.to_dataframe().reindex(index=['const']+plist)
pvalue_ax1 = pvalue_ax1.to_dataframe().reindex(index=['const']+plist)
heatmap_data_ax1 = data_ax1[1:15].values.T
heatmap_pvalue_ax1 = pvalue_ax1[1:15].values.T
ax1 = sns.heatmap(heatmap_data_ax1,annot = True,fmt='.2f',square = True,annot_kws={'size':10,'family':'Times New Roman'},
    cbar=True,cmap = 'RdGy_r',vmin=-1, vmax=1.5,center=0,cbar_kws={'pad':0.015,'ticks':[-1,-0.5,0,0.5,1,1.5],'format': '%.2f'})
pvalue_plot(ax1,heatmap_data_ax1,heatmap_pvalue_ax1,0.65)
ax1.set(xlabel='',xticklabels='') 
ax1.set(ylabel='',yticklabels=traitname)
plt.yticks(font='Times New Roman',rotation=0,fontsize = tick_label_size)
plt.xticks(font='Times New Roman',fontsize = tick_label_size)


ax3 = plt.subplot(212)  
data_ax3 = pd.DataFrame(columns={'Variable':['const']+plist}).set_index('Variable').to_xarray()
pvalue_ax3 = pd.DataFrame(columns={'Variable':['const']+plist}).set_index('Variable').to_xarray()
for trait in tlist:
    model = list(mr_PF_ALL['Model'][trait].split("', '"))
    model[0] = model[0].strip("['");model[-1] = model[-1].strip("']")
    beta = mr_PF_ALL['beta'][trait].strip("[").strip("]")
    beta = np.array([float(num_str) for num_str in beta.split(',')])
    df_tmp = pd.DataFrame({'Variable':['const']+model,trait:beta}).set_index('Variable').to_xarray()
    data_ax3 = xr.merge([data_ax3,df_tmp])
    pvalue = mr_PF_ALL['pvalue'][trait].strip("[").strip("]")
    pvalue = np.array([float(num_str) for num_str in pvalue.split(',')])
    df_tmp = pd.DataFrame({'Variable':['const']+model,trait:pvalue}).set_index('Variable').to_xarray()
    pvalue_ax3 = xr.merge([pvalue_ax3,df_tmp])
data_ax3=data_ax3.to_dataframe().reindex(index=['const']+plist)
pvalue_ax3 = pvalue_ax3.to_dataframe().reindex(index=['const']+plist)
heatmap_data_ax3 = data_ax3[1:15].values.T
heatmap_pvalue_ax3 = pvalue_ax3[1:15].values.T
ax3 = sns.heatmap(heatmap_data_ax3,annot = True,fmt='.2f',square = True,annot_kws={'size':10,'family':'Times New Roman'},
    cbar=True,cmap = 'RdGy_r',vmin=-1, vmax=1.5,center=0,cbar_kws={'pad':0.015,'ticks':[-1,-0.5,0,0.5,1,1.5],'format': '%.2f'})
pvalue_plot(ax3,heatmap_data_ax3,heatmap_pvalue_ax3,0.65)
ax3.set(xlabel='',xticklabels=plistname)
ax3.set(ylabel='',yticklabels=traitname)
plt.yticks(font='Times New Roman',rotation=0,fontsize = tick_label_size)
plt.xticks(font='Times New Roman',fontsize = tick_label_size)


ax1.axvline(0,ymin=0,ymax=1,color='black',linewidth=0.5,clip_on=False)
ax1.axvline(len(plist),ymin=0,ymax=1,color='black',linewidth=0.5,clip_on=False)
ax1.axhline(0,xmin=0,xmax=1,color='black',linewidth=0.5,clip_on=False)
ax1.axhline(len(tlist),xmin=0,xmax=1,color='black',linewidth=0.5,clip_on=False)
ax3.axvline(0,ymin=0,ymax=1,color='black',linewidth=0.5,clip_on=False)
ax3.axvline(len(plist),ymin=0,ymax=1,color='black',linewidth=0.5,clip_on=False)
ax3.axhline(0,xmin=0,xmax=1,color='black',linewidth=0.5,clip_on=False)
ax3.axhline(len(tlist),xmin=0,xmax=1,color='black',linewidth=0.5,clip_on=False)

ax1.axvline(-1.2,ymin=0.5/len(tlist),ymax=(len(tlist)-0.5)/len(tlist),color='black',linewidth=0.7,clip_on=False)
ax1.axhline(0.5           ,xmin=-1.2/len(plist),xmax=-1.1/len(plist),color='black',linewidth=0.7,clip_on=False)
ax1.axhline(len(tlist)-0.5,xmin=-1.2/len(plist),xmax=-1.1/len(plist),color='black',linewidth=0.7,clip_on=False)
ax1.text(-1.7,len(tlist)/2+0.08,'NF',fontsize=tick_label_size,fontdict=font_normal)

ax3.axvline(-1.2,ymin=0.5/len(tlist),ymax=(len(tlist)-0.5)/len(tlist),color='black',linewidth=0.7,clip_on=False)
ax3.axhline(0.5           ,xmin=-1.2/len(plist),xmax=-1.1/len(plist),color='black',linewidth=0.7,clip_on=False)
ax3.axhline(len(tlist)-0.5,xmin=-1.2/len(plist),xmax=-1.1/len(plist),color='black',linewidth=0.7,clip_on=False)
ax3.text(-1.7,len(tlist)/2+0.08,'PF',fontsize=tick_label_size,fontdict=font_normal)

ax3.axhline(5.7,xmin=0.5/len(plist), xmax=8.5/len(plist),              color='black',linewidth=0.7,clip_on=False)
ax3.axhline(5.7,xmin=9.5/len(plist), xmax=(len(plist)-0.5)/len(plist), color='black',linewidth=0.7,clip_on=False)
ax3.axvline(0.5,            ymin=-0.7/len(tlist),ymax=-0.6/len(tlist),color='black',linewidth=0.7,clip_on=False)
ax3.axvline(8.5,            ymin=-0.7/len(tlist),ymax=-0.6/len(tlist),color='black',linewidth=0.7,clip_on=False)
ax3.axvline(9.5,            ymin=-0.7/len(tlist),ymax=-0.6/len(tlist),color='black',linewidth=0.7,clip_on=False)
ax3.axvline(len(plist)-0.5, ymin=-0.7/len(tlist),ymax=-0.6/len(tlist),color='black',linewidth=0.7,clip_on=False)
ax3.text(4.1,                  6.1,'ENV',fontsize=tick_label_size,fontdict=font_normal)
ax3.text((len(plist)+9)/2-0.4, 6.1,'BIO',fontsize=tick_label_size,fontdict=font_normal)

plt.subplots_adjust(hspace=0.12)
fig.savefig(outpath + 'Figure 5.jpg',dpi=300,bbox_inches = 'tight')