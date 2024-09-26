#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:43:44 2024

@author: yk545
"""

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


vname1 = 'tos_trd_djfmam'
vname2 = 'prect_trd_djfmam'
vname3 = 'psl_trd_djfmam'

f_mask = '/Users/yk545/Documents/Research/Data/CESM2LE/cesm2le_landmask.nc'
ds = nc.Dataset(f_mask)
landmask = ds['landmask'][:,:]
maskocean_cesm = np.ones_like(landmask)
maskocean_cesm[landmask==1] = np.nan
lat_cesm = ds['lat'][:]
lon_cesm = ds['lon'][:]
ds.close()

fpath = '/Users/yk545/Documents/Research/Data/CESM2LE/'
fname = 'cesm2le_smbb_swus_prect_tas_sm_npi_nino34_tos_wp_1950_2050.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
nino34_cesm = np.empty((100,)) * np.nan;
nino34_cesm[0:50], pstd = lintrd(np.arange(34),DJFMAMmean(mat_contents['nino34'][:,30*12:30*12+35*12].T))
fname = 'cesm2le_cmip6_swus_prect_tas_sm_npi_nino34_1950_2050.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
nino34_cesm[50:100], pstd = lintrd(np.arange(34),DJFMAMmean(mat_contents['nino34'][:,30*12:30*12+35*12].T))

select_nino34 = nino34_cesm[nino34_cesm>np.percentile(nino34_cesm,90)]
select_nino34_l = nino34_cesm[nino34_cesm<np.percentile(nino34_cesm,10)]

fname = 'cesm2le_smbb_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
temp = np.empty((100,192,288)) * np.nan
temp[0:50,:,:] = mat_contents[vname1][:,:,:]
fname = 'cesm2le_cmip6_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
temp[50:100,:,:] = mat_contents[vname1][:,:,:]
tos_cesm = np.empty((len(select_nino34),192,288)) * np.nan
tos_cesm_l = np.empty((len(select_nino34_l),192,288)) * np.nan
for i in range(len(select_nino34)):
#    ind = select_nino34[i]
    ind = np.where(nino34_cesm==select_nino34[i])[0]
    tos_cesm[i,:,:] = temp[ind,:,:]
    ind = np.where(nino34_cesm==select_nino34_l[i])[0]
    tos_cesm_l[i,:,:] = temp[ind,:,:]
del temp, data_dir, mat_fname, mat_contents

fname = 'cesm2le_smbb_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
temp = np.empty((100,192,288)) * np.nan
temp[0:50,:,:] = mat_contents[vname2][:,:,:]
fname = 'cesm2le_cmip6_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
temp[50:100,:,:] = mat_contents[vname2][:,:,:]
prect_cesm = np.empty((len(select_nino34),192,288)) * np.nan
prect_cesm_l = np.empty((len(select_nino34_l),192,288)) * np.nan
for i in range(len(select_nino34)):
#    ind = select_nino34[i]
    ind = np.where(nino34_cesm==select_nino34[i])[0]
    prect_cesm[i,:,:] = temp[ind,:,:]
    ind = np.where(nino34_cesm==select_nino34_l[i])[0]
    prect_cesm_l[i,:,:] = temp[ind,:,:]
del temp, data_dir, mat_fname, mat_contents


fname = 'cesm2le_smbb_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
temp = np.empty((100,192,288)) * np.nan
temp[0:50,:,:] = mat_contents[vname3][:,:,:]
fname = 'cesm2le_cmip6_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
temp[50:100,:,:] = mat_contents[vname3][:,:,:]
psl_cesm = np.empty((len(select_nino34),192,288)) * np.nan
psl_cesm_l = np.empty((len(select_nino34_l),192,288)) * np.nan
for i in range(len(select_nino34)):
#    ind = select_nino34[i]
    ind = np.where(nino34_cesm==select_nino34[i])[0]
    psl_cesm[i,:,:] = temp[ind,:,:]
    ind = np.where(nino34_cesm==select_nino34_l[i])[0]
    psl_cesm_l[i,:,:] = temp[ind,:,:]
del temp, data_dir, mat_fname, mat_contents

strlat_nino34 = 91
endlat_nino34 = 101
strlon_nino34 = 152
endlon_nino34 = 193
fpath = '/Users/yk545/Documents/Research/Data/TOGA/'
var1 = 'psl_trd_djfmam'
var2 = 'prect_trd_djfmam'
nyr_trd = 10

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

fname = 'sst_ersstv5_1854_2020_cesm2_192x288.nc'
ds = nc.Dataset(fpath+fname)
nino34_ersst, pstd = lintrd(np.arange(34),\
                            DJFMAMmean(area_weighted_mean(ds['sst'][1512:1512+35*12,:,:], \
                            np.ones((192,288)), lat_cesm, lon_cesm, \
                            strlat_nino34, endlat_nino34, strlon_nino34, endlon_nino34)))
ds.close()

fname = 'sst_LIM60_1958_2017_cesm2_192x288.nc'
ds = nc.Dataset(fpath+fname)
tos_lim ,pstd = lintrd(np.arange(34),DJFMAMmean(ds['SST'][264:264+35*12,:,:]))
nino34_lim, pstd = lintrd(np.arange(34),\
                            DJFMAMmean(area_weighted_mean(ds['SST'][264:264+35*12,:,:], \
                            np.ones((192,288)), lat_cesm, lon_cesm, \
                            strlat_nino34, endlat_nino34, strlon_nino34, endlon_nino34)))
ds.close()

prect_lim_l = np.empty((10,192,288)) * np.nan;
psl_lim_l = np.empty((10,192,288)) * np.nan;

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_lim67_prect_trd_seasonal_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
prect_lim_l[0:10,:,:] = mat_contents[var2][:,:,:]
del data_dir, mat_fname, mat_contents

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_lim67_psl_trd_seasonal_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
psl_lim_l[0:10,:,:] = mat_contents[var1][:,:,:]
del data_dir, mat_fname, mat_contents

fname = 'sst_LIM67_1958_2017_cesm2_192x288.nc'
ds = nc.Dataset(fpath+fname)
tos_lim_l ,pstd = lintrd(np.arange(34),DJFMAMmean(ds['SST'][264:264+35*12,:,:]))
nino34_lim_l, pstd = lintrd(np.arange(34),\
                            DJFMAMmean(area_weighted_mean(ds['SST'][264:264+35*12,:,:], \
                            np.ones((192,288)), lat_cesm, lon_cesm, \
                            strlat_nino34, endlat_nino34, strlon_nino34, endlon_nino34)))
ds.close()

fpath = '/Users/yk545/Documents/Research/Data/CESM2_PiControl/'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cesm2_pictrl_top97p5ptile_nino34trd_tos_psl_prect_trd_map_34year_djfmam.mat')
mat_contents = sio.loadmat(mat_fname)
prect_pi = mat_contents['prect_trd'][:,:,:]
psl_pi = mat_contents['psl_trd'][:,:,:]
tos_pi = mat_contents['tos_trd'][:,:,:]

fpath = '/Users/yk545/Documents/Research/Data/CESM2_PiControl/'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cesm2_pictrl_low2p5ptile_nino34trd_tos_psl_prect_trd_map_34year_djfmam.mat')
mat_contents = sio.loadmat(mat_fname)
prect_pi_l = mat_contents['prect_trd'][:,:,:]
psl_pi_l = mat_contents['psl_trd'][:,:,:]
tos_pi_l = mat_contents['tos_trd'][:,:,:]

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cesm2_pictrl_npi_trd_swus_prect_trd_nino34_tos_trd_34yr_djfmam_overlapped.mat')
mat_contents = sio.loadmat(mat_fname)
nino34_pi = mat_contents['nino34_trd'][0,:]


savepath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_LIM_SWUS_figure/'

flabel = 18; ftick = 16
nyr_trd = 10
lon2dcesm, lat2dcesm = np.meshgrid(lon_cesm,lat_cesm)

fig = plt.figure(facecolor = 'none')
fig.set_size_inches(21,4)
tick_font_size = 14
title_font_size = 16
clim = 0.3
clev = 0.03
Lev = np.arange(-clim,clim+0.01,clev)
domain = [1,359,-28,28]
ax2 = fig.add_subplot(1,3,1,projection=ccrs.PlateCarree(central_longitude=-180))
ax2.set_aspect('auto')
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*np.nanmean(tos_pi_l,0)*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax2.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
ax2.add_feature(cfeature.LAND, facecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax2.set_extent(domain)
## Nino34
ax2.add_patch(mpatches.Rectangle(xy=[lon_cesm[152], lat_cesm[91]], width=lon_cesm[192]-lon_cesm[152],\
                                 height=lat_cesm[100]-lat_cesm[91],edgecolor = 'yellow',\
                                 linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax2 = fig.add_subplot(1,3,2,projection=ccrs.PlateCarree(central_longitude=-180))
ax2.set_aspect('auto')
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*np.nanmean(tos_cesm_l,0)*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax2.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
ax2.add_feature(cfeature.LAND, facecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax2.set_extent(domain)
## Nino34
ax2.add_patch(mpatches.Rectangle(xy=[lon_cesm[152], lat_cesm[91]], width=lon_cesm[192]-lon_cesm[152],\
                                 height=lat_cesm[100]-lat_cesm[91],edgecolor = 'yellow',\
                                 linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax1 = fig.add_subplot(1,3,3,projection=ccrs.PlateCarree(central_longitude=-180))
ax1.set_aspect('auto')
cs = ax1.contourf(lon_cesm, lat_cesm, nyr_trd*tos_lim_l*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax1.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
ax1.add_feature(cfeature.LAND, facecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax1.set_extent(domain)
## Nino34
ax1.add_patch(mpatches.Rectangle(xy=[lon_cesm[152], lat_cesm[91]], width=lon_cesm[192]-lon_cesm[152],\
                                 height=lat_cesm[100]-lat_cesm[91],edgecolor = 'yellow',\
                                 linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
fig.set_rasterized(False)
plt.savefig(savepath+'figure2_cesm2_lanina_sst_trd.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'none')
fig.set_size_inches(21,4)
tick_font_size = 14
title_font_size = 16
clim = 0.5
clev = 0.05
Lev = np.arange(-clim,clim+0.01,clev)
domain = [1,359,-28,28]
ax2 = fig.add_subplot(1,3,1,projection=ccrs.PlateCarree(central_longitude=-180))
ax2.set_aspect('auto')
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*np.nanmean(tos_pi,0)*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax2.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
ax2.add_feature(cfeature.LAND, facecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax2.set_extent(domain)
## Nino34
ax2.add_patch(mpatches.Rectangle(xy=[lon_cesm[152], lat_cesm[91]], width=lon_cesm[192]-lon_cesm[152],\
                                 height=lat_cesm[100]-lat_cesm[91],edgecolor = 'yellow',\
                                 linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax2 = fig.add_subplot(1,3,2,projection=ccrs.PlateCarree(central_longitude=-180))
ax2.set_aspect('auto')
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*np.nanmean(tos_cesm,0)*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax2.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
ax2.add_feature(cfeature.LAND, facecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax2.set_extent(domain)
## Nino34
ax2.add_patch(mpatches.Rectangle(xy=[lon_cesm[152], lat_cesm[91]], width=lon_cesm[192]-lon_cesm[152],\
                                 height=lat_cesm[100]-lat_cesm[91],edgecolor = 'yellow',\
                                 linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax1 = fig.add_subplot(1,3,3,projection=ccrs.PlateCarree(central_longitude=-180))
ax1.set_aspect('auto')
cs = ax1.contourf(lon_cesm, lat_cesm, nyr_trd*tos_lim*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax1.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
ax1.add_feature(cfeature.LAND, facecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax1.set_extent(domain)
## Nino34
ax1.add_patch(mpatches.Rectangle(xy=[lon_cesm[152], lat_cesm[91]], width=lon_cesm[192]-lon_cesm[152],\
                                 height=lat_cesm[100]-lat_cesm[91],edgecolor = 'yellow',\
                                 linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
fig.set_rasterized(False)
plt.savefig(savepath+'figure2_cesm2_elnino_sst_trd.eps',dpi=300)
plt.show()


fig = plt.figure(facecolor = 'none')
fig.set_size_inches(21,4)
tick_font_size = 14
title_font_size = 16
clim = 5
clev = 1
Lev = np.arange(-clim,clim+0.1,clev)
Levp = np.arange(-clim,clim+0.1,clev/2)
Lev_psl = np.arange(0.1,10,0.3)
Lev_psl_n = np.arange(-6.1,-0.09,0.3)
domain = [157,260,16,57]
ax3 = fig.add_subplot(1,3,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax3.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_pi_l,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_pi_l,np.nanmean(prect_pi_l,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_pi_l,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax3.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_pi_l,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax3.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax3.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')

density = 1
agr = ens_agree_ensmean(psl_pi_l,np.nanmean(psl_pi_l,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_pi_l,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax3.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax3.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax3.add_feature(cfeature.BORDERS, edgecolor='gray')
ax3.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax3.set_extent(domain)
# SWUS box
ax3.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax2 = fig.add_subplot(1,3,2,projection=ccrs.Robinson(central_longitude=-115))
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_cesm_l,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_cesm_l,np.nanmean(prect_cesm_l,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_cesm_l,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax2.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_cesm_l,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax2.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax2.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')

density = 1
agr = ens_agree_ensmean(psl_cesm_l,np.nanmean(psl_cesm_l,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_cesm_l,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax2.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax2.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax2.add_feature(cfeature.BORDERS, edgecolor='gray')
ax2.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax2.set_extent(domain)
# SWUS box
ax2.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax1 = fig.add_subplot(1,3,3,projection=ccrs.Robinson(central_longitude=-115))
cs = ax1.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_lim_l,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_lim_l,np.nanmean(prect_lim_l,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_lim_l,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax1.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_lim_l,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax1.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax1.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_lim_l,np.nanmean(psl_lim_l,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_lim_l,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax1.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('blue')
    collection.set_linewidth(0.)
ax1.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax1.add_feature(cfeature.BORDERS, edgecolor='gray')
ax1.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax1.set_extent(domain)
# SWUS box
ax1.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

fig.set_rasterized(True)
plt.savefig(savepath+'figure2_cesm2_lanina_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()


fig = plt.figure(facecolor = 'none')
fig.set_size_inches(21,4)
tick_font_size = 14
title_font_size = 16
clim = 5
clev = 1
Lev = np.arange(-clim,clim+0.1,clev)
Levp = np.arange(-clim,clim+0.1,clev/2)
domain = [157,260,16,57]
ax3 = fig.add_subplot(1,3,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax3.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_pi,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_pi,np.nanmean(prect_pi,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_pi,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax3.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_pi,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax3.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax3.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')

density = 1
agr = ens_agree_ensmean(psl_pi,np.nanmean(psl_pi,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_pi,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax3.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax3.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax3.add_feature(cfeature.BORDERS, edgecolor='gray')
ax3.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax3.set_extent(domain)
# SWUS box
ax3.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax2 = fig.add_subplot(1,3,2,projection=ccrs.Robinson(central_longitude=-115))
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_cesm,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_cesm,np.nanmean(prect_cesm,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_cesm,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax2.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_cesm,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax2.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax2.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')

density = 1
agr = ens_agree_ensmean(psl_cesm,np.nanmean(psl_cesm,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_cesm,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax2.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax2.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax2.add_feature(cfeature.BORDERS, edgecolor='gray')
ax2.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax2.set_extent(domain)
# SWUS box
ax2.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

ax1 = fig.add_subplot(1,3,3,projection=ccrs.Robinson(central_longitude=-115))
cs = ax1.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_lim,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_lim,np.nanmean(prect_lim,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_lim,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax1.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_lim,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax1.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax1.contour(lon_cesm, lat_cesm, nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')

density = 1
agr = ens_agree_ensmean(psl_lim,np.nanmean(psl_lim,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_lim,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax1.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax1.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(category='cultural',\
    name='admin_1_states_provinces_lines',\
    scale='50m',\
    facecolor='none')
ax1.add_feature(cfeature.BORDERS, edgecolor='gray')
ax1.add_feature(states_provinces,\
                edgecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 16}
gl.ylabel_style = {'size': 16}
ax1.set_extent(domain)
# SWUS box
ax1.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))

fig.set_rasterized(True)
plt.savefig(savepath+'figure2_cesm2_elnino_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()