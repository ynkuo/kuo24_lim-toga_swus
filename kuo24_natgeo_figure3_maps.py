#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:37:50 2024

@author: yk545
"""

from scipy.io import savemat
import numpy as np
import netCDF4 as nc
from os.path import dirname, join as pjoin
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.stats
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import sys
sys.path.append("/Users/yk545/Documents/Research/Codes/Function_YNKuo/")
from function_YNKuo import *

fpath = '/Users/yk545/Documents/Research/Data/landmask/'
f_mask = '/Users/yk545/Documents/Research/Data/CESM2LE/cesm2le_landmask.nc'
ds = nc.Dataset(f_mask)
landmask = ds['landmask'][:,:]
maskocean_cesm = np.ones_like(landmask)
maskocean_cesm[landmask==1] = np.nan
lat_cesm = ds['lat'][:]
lon_cesm = ds['lon'][:]
ds.close()

nyr_trd = 34
var1 = 'psl_trd_djfmam'
var2 = 'prect_trd_djfmam'
fpath = '/Users/yk545/Documents/Research/Data/TOGA/'
f1 = 'cam6_toga_ersstv5climSSTSIC_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f1)
mat_contents = sio.loadmat(mat_fname)
prect_df = mat_contents[var2][:,:,:]

f2 = 'cam6_toga_ersstv5climSSTSIC_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f2)
mat_contents = sio.loadmat(mat_fname)
psl_df = mat_contents[var1][:,:,:]

### reading F2000 case
f1 = 'cam6_toga_f2000_ersstv5_clim_lim60trd35yr_2Ktrd_DJFMAM_pr_psl_ens01-20.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f1)
mat_contents = sio.loadmat(mat_fname)
prect_elnino = (mat_contents['prect_elnino'][:,:,:] - mat_contents['prect_ctrl'][:,:,:])/35 #/35: converting the unit of forced response as delta(prect) per year
prect_2k = (mat_contents['prect_2k'][:,:,:] - mat_contents['prect_ctrl'][:,:,:])/35
psl_elnino = (mat_contents['psl_elnino'][:,:,:] - mat_contents['psl_ctrl'][:,:,:])/100/35
psl_2k = (mat_contents['psl_2k'][:,:,:] - mat_contents['psl_ctrl'][:,:,:] )/100/35


### Reading PRECT
prect_lim = np.empty((10,192,288)) * np.nan;
psl_lim = np.empty((10,192,288)) * np.nan;

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_lim60_prect_trd_seasonal_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
prect_lim[0:10,:,:] = mat_contents[var2][:,:,:]
del data_dir, mat_fname, mat_contents

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_lim60_psl_trd_seasonal_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
psl_lim[0:10,:,:] = mat_contents[var1][:,:,:]
del data_dir, mat_fname, mat_contents


###### PLOTTING
savepath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_LIM_SWUS_figure/'

nyr_trd = 10
lon2dcesm, lat2dcesm = np.meshgrid(lon_cesm,lat_cesm)
tick_font_size = 14
title_font_size = 15
domain = [157,260,16,57]
clim = 5
clev = 1
Lev = np.arange(-clim,clim+0.1,clev)
Levp = np.arange(-clim,clim+0.1,clev/2)
Lev_psl = np.arange(0.1,10,0.3)
Lev_psl_n = np.arange(-6.1,-0.09,0.3)

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax4 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax4.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_df,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_df,np.nanmean(prect_df,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_df,0)) * np.NAN
S2N[agr>0.67] = 1
density= 3
hc = ax4.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_df,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')

density = 1
agr = ens_agree_ensmean(psl_df,np.nanmean(psl_df,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_df,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax4.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax4.set_title('Radiatively Forced ${psl}$, ${pr}$ with ${SST}$ fixed (10)', fontsize=title_font_size)
ax4.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax4.add_feature(cfeature.BORDERS, edgecolor='gray')
ax4.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax4.set_extent(domain)
# SWUS box
ax4.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax4.set_rasterized(True)
plt.savefig(savepath+'figure3_df_psl_trd_red_blue.eps',dpi=300)
plt.show()


### readig CAM6
fpath = '/Users/yk545/Documents/Research/Data/CESM2_SF/'
f1 = 'cesm2le_aaer_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f1)
mat_contents = sio.loadmat(mat_fname)
prect_aaer = mat_contents[var2][:,:,:]

f2 = 'cesm2le_aaer_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f2)
mat_contents = sio.loadmat(mat_fname)
psl_aaer = mat_contents[var1][:,:,:]

f2 = 'cesm2le_aaer_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f2)
mat_contents = sio.loadmat(mat_fname)
tos_aaer = mat_contents['tos_trd_djfmam'][:,:,:]

f2 = 'cesm2le_xaer_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f2)
mat_contents = sio.loadmat(mat_fname)
tos_xaer = mat_contents['tos_trd_djfmam'][:,:,:]

f1 = 'cesm2le_xaer_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f1)
mat_contents = sio.loadmat(mat_fname)
prect_xaer = mat_contents[var2][:,:,:]

f2 = 'cesm2le_xaer_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+f2)
mat_contents = sio.loadmat(mat_fname)
psl_xaer = mat_contents[var1][:,:,:]


fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax4 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax4.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_aaer,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_aaer,np.nanmean(prect_aaer,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_aaer,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax4.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_aaer,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_aaer,np.nanmean(psl_aaer,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_aaer,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax4.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax4.set_title('Anthropogenic Aerosols (AAER) Forced ${psl}$, ${pr}$ (20)', fontsize=title_font_size)
ax4.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax4.add_feature(cfeature.BORDERS, edgecolor='gray')
ax4.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax4.set_extent(domain)
# SWUS box
ax4.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax4.set_rasterized(True)
plt.savefig(savepath+'figure3_aaer_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax4 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax4.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_xaer,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_xaer,np.nanmean(prect_xaer,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_xaer,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax4.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_xaer,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_xaer,np.nanmean(psl_xaer,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_xaer,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax4.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax4.set_title('xAAER Forced ${psl}$, ${pr}$ (10)', fontsize=title_font_size)
ax4.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax4.add_feature(cfeature.BORDERS, edgecolor='gray')
ax4.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax4.set_extent(domain)
# SWUS box
ax4.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax4.set_rasterized(True)
plt.savefig(savepath+'figure3_xaer_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax4 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax4.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_elnino,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_elnino,np.nanmean(prect_elnino,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_elnino,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax4.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_elnino,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_elnino,np.nanmean(psl_elnino,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_elnino,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax4.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax4.set_title('El Niño-like (LIM) Tropical ${SST}$ Forced ${psl}$, ${pr}$ (20)', fontsize=title_font_size)
ax4.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax4.add_feature(cfeature.BORDERS, edgecolor='gray')
ax4.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax4.set_extent(domain)
# SWUS box
ax4.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax4.set_rasterized(True)
plt.savefig(savepath+'figure3_elnino_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax4 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax4.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_2k,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_2k,np.nanmean(prect_2k,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_2k,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax4.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_2k,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_2k,np.nanmean(psl_2k,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_2k,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax4.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax4.set_title('Uniform Tropical 2K ${SST}$ Forced ${psl}$, ${pr}$ (20)', fontsize=title_font_size)
ax4.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax4.add_feature(cfeature.BORDERS, edgecolor='gray')
ax4.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax4.set_extent(domain)
# SWUS box
ax4.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax4.set_rasterized(True)
plt.savefig(savepath+'figure3_2k_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax4 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax4.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*(np.nanmean(prect_lim,0) - (np.nanmean(prect_elnino,0)+np.nanmean(prect_df,0))),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
prect_nl = np.empty((100,192,288)) * np.NAN
ind1 = np.random.choice(np.arange(len(prect_lim)), size=100)
ind2 = np.random.choice(np.arange(len(prect_elnino)), size=100)
ind3 = np.random.choice(np.arange(len(prect_df)), size=100)
prect_nl = prect_lim[ind1,:,:] - (prect_elnino[ind2,:,:] + prect_df[ind3,:,:])
agr = ens_agree_ensmean(prect_nl,\
                        (np.nanmean(prect_lim,0) - (np.nanmean(prect_elnino,0)+np.nanmean(prect_df,0))),\
                        threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_2k,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax4.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)

ctr_plt = (np.nanmean(psl_lim,0) - (np.nanmean(psl_elnino,0)+np.nanmean(psl_df,0)))
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax4.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
del ind1, ind2, ind3, agr, S2N
psl_nl = np.empty((100,192,288)) * np.NAN
ind1 = np.random.choice(np.arange(len(prect_lim)), size=100)
ind2 = np.random.choice(np.arange(len(prect_elnino)), size=100)
ind3 = np.random.choice(np.arange(len(prect_df)), size=100)
psl_nl = psl_lim[ind1,:,:] - (psl_elnino[ind2,:,:] + psl_df[ind3,:,:])
agr = ens_agree_ensmean(psl_nl,\
                        (np.nanmean(psl_lim,0) - (np.nanmean(psl_elnino,0)+np.nanmean(psl_df,0))),\
                        threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_2k,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax4.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
del ind1, ind2, ind3, agr, S2N
ax4.set_title('Nonlinearity: El Niño-like ${SST}$ & RF Forced ${psl}$, ${pr}$', fontsize=title_font_size)
ax4.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax4.add_feature(cfeature.BORDERS, edgecolor='gray')
ax4.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax4.set_extent(domain)
# SWUS box
ax4.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax4.set_rasterized(True)
#plt.savefig(savepath+'figure3_elnino_rf_nonlinearity_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()
#savemat("Kuo24_NatGeo_nonlinear_elnino.mat", {'psl_nl':psl_nl,'prect_nl':prect_nl})
#del psl_nl, prect_nl
