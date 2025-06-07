import pdb
import numpy as np
import pandas as pd
import seaborn as sns
import warnings; warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from openpyxl import load_workbook
plt.rcParams['font.sans-serif'] = ['Times New Roman']
font_it={'family': 'serif', 'style': 'italic', 'fontname': 'Times New Roman'}
font_normal={'family': 'serif', 'fontname': 'Times New Roman'}
colorlist = ['#969696','#6BAED6','#74C476','#FEB24C','#74C476','#FEB24C']
y_order_list = ['SH','NF','PF']

inpath = './input/'
outpath = './Figure/'
Sitedata = inpath + 'Table S1.xlsx'
sheetname = ['MCS','SCS']

data = pd.read_excel(Sitedata,sheet_name=sheetname[0])
data = data[['ID','OldID','Group','Species','Lat','Lon','TC','P50','Ks']]

data['P50'] = -1*data['P50']
data.loc[data['TC']=='NFA','TC']='NF'
data.loc[data['TC']=='NFG','TC']='NF'
data.loc[data['TC']=='PFA','TC']='PF'
data.loc[data['TC']=='PFG','TC']='PF'

#Fig 2

fig = plt.figure(figsize = [15,6], dpi = 300)

datatmp = data[['TC','P50']].dropna()
ax1 = plt.subplot(121)
ax1 = sns.violinplot(y=datatmp['TC'],x=datatmp['P50'],palette=colorlist[1:],width=0.6,inner=None,
    linewidth=0,orient ='h',order=y_order_list,zorder=1)
ax1 = sns.boxplot(y=datatmp['TC'],x=datatmp['P50'],palette=colorlist[1:],width=0.15,linecolor='black',showcaps=False,
    linewidth=1.5,fliersize=4,orient ='h',order=y_order_list,zorder=10)
data_experiment = data[['TC','P50']][-7:].dropna()

ax1.set(xlim=(-1.5,8),xticks=[0,2,4,6,8],xticklabels=['0.0','2.0','4.0','6.0','8.0'])
plt.xlabel(r'$\mathregular{P50_x}$ $\mathregular{(-MPa)}$',fontsize = 20)
plt.ylabel('')
plt.yticks(font='Times New Roman',rotation=0,fontsize = 20)
plt.xticks(font='Times New Roman',fontsize = 15)
ax1.text(7.05,-0.2,f'a'+')',fontsize=20,fontdict=font_normal,weight='bold')

datatmp = data[['TC','Ks']].dropna()
ax2 = plt.subplot(122)
ax2=sns.violinplot(y=datatmp['TC'],x=datatmp['Ks'],palette=colorlist[1:],width=0.6,inner=None,
    linewidth=0,orient ='h',order=y_order_list)
ax2=sns.boxplot(y=datatmp['TC'],x=datatmp['Ks'],palette=colorlist[1:],width=0.15,linecolor='black',showcaps=False,
    linewidth=1.5,fliersize=4,orient ='h',order=y_order_list)
data_experiment = data[['TC','Ks']][-7:].dropna()


ax2.set(xlim=(-3,20),xticks=[0,5,10,15,20],xticklabels=['0.0','5.0','10.0','15.0','20.0'])
plt.xlabel(r'$\mathregular{K_s}$ (kg $\mathregular{m^{-1}}$ $\mathregular{s^{-1}}$ $\mathregular{MPa^{-1}}$)',fontsize = 20)
plt.ylabel('')
ax2.set(yticklabels=['']*3)
plt.xticks(font='Times New Roman',fontsize = 15)
ax2.text(18.2,-0.2,f'b'+')',fontsize=20,fontdict=font_normal,weight='bold')

plt.subplots_adjust(wspace =0.05,hspace=0.1)
fig.savefig(outpath+'Figure 2.jpg',dpi=300,bbox_inches='tight')


#Fig 3

def sim(data):
    x1=np.log(data.P50)/np.log(10)
    x2=sm.add_constant(x1)
    y1=np.log(data.Ks)/np.log(10)
    try:tmp=sm.OLS(y1,x2).fit();slope_tmp,r2_tmp,p_tmp = tmp.params[1],tmp.rsquared, tmp.pvalues.P50
    except:slope_tmp,r2_tmp,p_tmp = None,None,None
    return x1,y1,slope_tmp,r2_tmp,p_tmp

def text_pic(xlim_down,xlim_up,ylim_down,ylim_up,picnum,title,R2x=-9,R2y=-9):
    delta_y = (ylim_up-ylim_down)/15
    delta_x = (xlim_up-xlim_down)/28
    ax1.text(R2x,R2y,f'$\mathregular{{R = {"%.2f"%np.sqrt(r2).round(2)}}}$',fontdict=font_normal,fontsize=13)
    ax1.text(R2x,R2y-delta_y,r'p',fontdict=font_it,fontsize=13)
    if pv < 0.01 :
        ax1.text(R2x+delta_x,R2y-delta_y,f'$\mathregular{{< 0.01}}$',fontdict=font_normal,fontsize=13)
    else: 
        ax1.text(R2x+delta_x,R2y-delta_y,f'$\mathregular{{= {"%.2f"%pv.round(2)}}}$',fontdict=font_normal,fontsize=13)

    plt.ylabel(r'$\mathregular{K_s}$ $\mathregular{log_{10}}$ (kg $\mathregular{m^{-1}}$ $\mathregular{s^{-1}}$ $\mathregular{MPa^{-1}}$)',font='Times New Roman',fontsize=16)
    plt.xlabel(r'$\mathregular{P50_x}$ $\mathregular{log_{10}}$ $\mathregular{(-MPa)}$',font='Times New Roman',fontsize=16)
    plt.yticks(font='Times New Roman',fontsize=13)
    plt.xticks(font='Times New Roman',fontsize=13)
    plt.title(title,fontsize=16)
    picnum_y_position = ylim_up + (ylim_up-ylim_down)/10*0.4
    picnum_x_position = xlim_down - (xlim_up-xlim_down)/10*2.3
    ax1.text(picnum_x_position,picnum_y_position,picnum+')',fontsize=16,fontdict=font_normal,weight='bold')
    return

data = data.dropna(axis=0,subset=['ID','TC','P50','Ks'])
SB=data[data.TC=='SH']
NF=data[data.TC=='NF']
PF=data[data.TC=='PF']

fig=plt.figure(figsize=(13,8))
x2,y2,k,r2,pv=sim(data)
ax1 = plt.subplot(231)
sns.regplot(x=x2,y=y2,ci=95,color=colorlist[0],scatter_kws={'s':14})
ax1.set(xlim=(-1.2,1),xticks=np.arange(-1,1.1,0.5),xticklabels=['-1.0','-0.5','0.0','0.5','1.0'],
     ylim=(-1.5,1.5),yticks=np.arange(-1.5,1.6,0.5),yticklabels=['-1.5','-1.0','-0.5','0.0','0.5','1.0','1.5'])
text_pic(-1.2,1,-1.5,1.5,'a','Entire Dataset',-1.05,-0.5)

x2,y2,k,r2,pv=sim(NF)
ax1 = plt.subplot(232)
sns.regplot(x=x2,y=y2,ci=95,color=colorlist[2],scatter_kws={'s':14})
ax1.set(xlim=(-1.2,1),xticks=[-1,-0.5,0,0.5,1],xticklabels=['-1.0','-0.5','0.0','0.5','1.0'],
    ylim=(-1.5,1.5),yticks=[-1.5,-1,-0.5,0,0.5,1,1.5],yticklabels=['-1.5','-1.0','-0.5','0.0','0.5','1.0','1.5'])
text_pic(-1.2,1,-1.5,1.5,'b','NF (Across Sites)',-1,-1)

x3,y3,k,r2,pv=sim(PF)
ax1 = plt.subplot(233)
sns.regplot(x=x3,y=y3,ci=95,color=colorlist[3],scatter_kws={'s':14})
ax1.set(xlim=(-1,1),xticks=[-1,-0.5,0,0.5,1],xticklabels=['-1.0','-0.5','0.0','0.5','1.0'],
  ylim=(-1.5,1.5),yticks=[-1.5,-1,-0.5,0,0.5,1,1.5],yticklabels=['-1.5','-1.0','-0.5','0.0','0.5','1.0','1.5'])
text_pic(-1,1,-1.5,1.5,'c','PF (Across Sites)',-0.9,1)

x1,y1,k,r2,pv=sim(SB)
ax1 = plt.subplot(234)
sns.regplot(x=x1,y=y1,ci=95,color=colorlist[1],scatter_kws={'s':14})
ax1.set(xlim=(-1,1),xticks=[-1,-0.5,0,0.5,1],xticklabels=['-1.0','-0.5','0.0','0.5','1.0'],
    ylim=(-1,1.5),yticks=[-1,-0.5,0,0.5,1,1.5],yticklabels=['-1.0','-0.5','0.0','0.5','1.0','1.5'])
text_pic(-1,1,-1,1.5,'d','SH (Across Sites)',-0.6,-0.4)

dataS = pd.read_excel(Sitedata,sheet_name=sheetname[1])
dataS = dataS.sort_values(by=['Species'],axis=0)
dataS['TC'].replace({'PFA':'PF (Within Sites)','PFG':'PF (Within Sites)','NFA':'NF (Within Sites)','NFG':'NF (Within Sites)'},inplace=True)
dataS['P50'] *= -1

data = dataS.dropna(axis=0,subset=['TC','P50','Ks'])
data = data.dropna(axis=0,subset='ID')
NF=data[data.TC=='NF (Within Sites)']
PF=data[data.TC=='PF (Within Sites)']

x2,y2,k,r2,pv=sim(NF);print('e. NF (Within Sites):','k:',k.round(2),'r2:',r2.round(2),'p:',pv.round(2))
ax1 = plt.subplot(235)
sns.regplot(x=x2,y=y2,ci=95,color=colorlist[2],scatter_kws={'s':14})
ax1.set(xlim=(-1.2,0.7),xticks=[-1.0,-0.5,0,0.5],xticklabels=['-1.0','-0.5','0.0','0.5'],
    ylim=(-1,1),yticks=[-1,-0.5,0,0.5,1],yticklabels=['-1.0','-0.5','0.0','0.5','1.0'])
text_pic(-1.2,1,-1,1,'e','NF (Within Sites)',-1,-0.5)

x3,y3,k,r2,pv=sim(PF);print('f. PF (Within Sites):','k:',k.round(2),' r2:',r2.round(2),'p:',pv.round(2))
ax1 = plt.subplot(236)
sns.regplot(x=x3,y=y3,ci=95,color=colorlist[3],scatter_kws={'s':14})
ax1.set(xlim=(-1,1),xticks=[-1,-0.5,0,0.5,1],xticklabels=['-1.0','-0.5','0.0','0.5','1.0'],
  ylim=(-2,1),yticks=[-2,-1,0,1],yticklabels=['-2.0','-1.0','0.0','1.0'])
text_pic(-1,1,-2,1,'f','PF (Within Sites)',-0.75,0)

plt.subplots_adjust(wspace =0.3,hspace=0.36)
fig.savefig(outpath + 'Figure 3.jpg',dpi=300,bbox_inches='tight')