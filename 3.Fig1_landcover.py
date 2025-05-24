import pdb
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gp
import seaborn as sns
import warnings; warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import rioxarray
import gc
from memory_profiler import profile
gc.collect()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['patch.linewidth'] = 1

inpath = './data/'
Sitedata = './input/Table S1.xlsx'
outpath = './Figure/'

latmin=17.875
latmax=53.875
lonmin=73.125
lonmax=135.125

def add_north(ax, labelsize=12, loc_x=0.08, loc_y=1.05, width=0.028,height=0.06, pad=0.18):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen*(loc_x - width*.5), miny + ylen*(loc_y - pad)]
    right = [minx + xlen*(loc_x + width*.5), miny + ylen*(loc_y - pad)]
    top = [minx + xlen*loc_x, miny + ylen*(loc_y - pad + height)]
    center = [minx + xlen*loc_x, left[1] + (top[1] - left[1])*.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen*loc_x,
            y=miny + ylen*(loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontproperties='Times New Roman',
            weight='bold')
    ax.add_patch(triangle)

forest1 = rioxarray.open_rasterio(inpath + 'FOREST/2005forest_cheng.tif').sel(band=1)
forest1 = forest1.rio.reproject('EPSG:4326')
forest1 = forest1.rename({'x':'lon','y':'lat'})
natural = xr.where((forest1 > 0) & (forest1 <= 15) == 1, 4, 0)
planted = xr.where(forest1 >= 30, 5, 0)
land = xr.concat((natural,planted),'lands').sum(dim='lands')
lands = xr.where(~np.isnan(land),land, 0)
del forest1, natural, planted, land

chinashape = gp.read_file(inpath+"SHAPE/china_country.shp")
nine_dot_line = gp.read_file(inpath+"SHAPE/china_nine_dotted_line.shp")
fig = plt.figure(figsize=(8, 10))
proj = ccrs.PlateCarree()

leftlon, rightlon, lowerlat, upperlat = (72, 136.5, 2, 55)
ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=proj)
ax1.set_extent([leftlon, rightlon, lowerlat, upperlat],crs=ccrs.PlateCarree())
ax1.set_xticks(np.arange(80, 135, 10), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(5, 56, 10), crs=ccrs.PlateCarree())
ax1.tick_params(axis='both', which='major', labelsize=12, direction='in', length=5, width=0.7)
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.add_geometries(chinashape.geometry, crs=ccrs.PlateCarree(), edgecolor='k', facecolor='none', lw=0.5)
ax1.add_geometries(nine_dot_line.geometry, crs=ccrs.PlateCarree(), edgecolor='k', facecolor='none', lw=0.5)
colordict = ['#FFFFFF','#FFFFFF','#FFFFFF','#FFFFFF','#74C476','#FEB24C']
ax = plt.gca()
add_north(ax)

left, bottom, width, height = 0.52, 0.365, 0,0
ax2 = fig.add_axes([left, bottom, width, height], projection=ccrs.PlateCarree())

data = pd.read_excel(Sitedata,sheet_name='All Sites')
lon=np.array(data['Lon'])
lat=np.array(data['Lat'])
ax1.scatter(lon,lat,12,color = '#252525',marker='v',transform=proj)
del data
data = pd.read_excel(Sitedata,sheet_name='Comparison Sites')
lon=np.array(data['Lon'])
lat=np.array(data['Lat'])
ax1.scatter(lon,lat,12,color = '#0044bb',marker='v',transform=proj)
data = data[data['ID']>500]
lon=np.array(data['Lon'])
lat=np.array(data['Lat'])
ax1.scatter(lon,lat,12,color = '#FF0000',marker='v',transform=proj)


patch4 = mpatches.Patch(color=colordict[4], label="NF")
patch5 = mpatches.Patch(color=colordict[5], label="PF")
scatter1 = ax2.scatter(0,0,32,color = '#252525',marker='v',transform=proj,label='Multi-species comparison sites\nfrom Meta-analysis')
scatter2 = ax2.scatter(0,0,32,color = '#0044bb',marker='v',transform=proj,label='Same-species comparison sites\nfrom Ye (2021) and Shangguan et al. (2022)')
scatter3 = ax2.scatter(0,0,32,color = '#FF0000',marker='v',transform=proj,label='Same-species comparison sites\nbased on field data from this study')
ax1.legend(handles=[patch4,patch5],fontsize=12,frameon=False,bbox_to_anchor=(0.97,0.142))
plt.legend(handles=[scatter1,scatter2,scatter3],fontsize=12,frameon=False)

ax1.pcolormesh(lands.lon,lands.lat,lands,zorder=0,transform=ccrs.PlateCarree(),cmap=mcolors.ListedColormap(colordict))
del lands

plt.savefig(outpath + "Figure 1.jpg", dpi=300, bbox_inches='tight')
gc.collect()