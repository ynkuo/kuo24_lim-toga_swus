#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:20:01 2024

@author: yk545
"""

### Importing the library
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

### Reading landmasks
f_mask = '/Users/yk545/Documents/Research/Data/landmask/cesm2le_landmask_g025grid.nc'
ds = nc.Dataset(f_mask)
maskland_cmip6 = ds['landmask'][:,:]
maskocean = np.ones_like(maskland_cmip6)
maskocean[maskland_cmip6==1] = np.nan
lat_cmip6 = ds['lat'][:]
lon_cmip6 = ds['lon'][:]
ds.close()

fpath = '/Users/yk545/Documents/Research/Data/landmask/'
f_mask = '/Users/yk545/Documents/Research/Data/CESM2LE/cesm2le_landmask.nc'
ds = nc.Dataset(f_mask)
landmask = ds['landmask'][:,:]
maskocean_cesm = np.ones_like(landmask)
maskocean_cesm[landmask==1] = np.nan
lat_cesm = ds['lat'][:]
lon_cesm = ds['lon'][:]
ds.close()

### Reading Obs.
nyr_trd = 34
dof = nyr_trd-2
var1 = 'psl_trd_djfmam'
var2 = 'prect_trd_djfmam'
var1_std = 'psl_trd_std_djfmam'
var2_std = 'prect_trd_std_djfmam'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  '/Users/yk545/Documents/Research/Data/Observation/gpcc_prect_trd_seasonal_1980_2014_cesm2grid_v2022.mat')
mat_contents = sio.loadmat(mat_fname)
trd_obs_pr = mat_contents[var2][:,:]
StuT = trd_obs_pr/mat_contents[var2_std][:,:]
del data_dir, mat_fname, mat_contents
p_obs_pr = np.empty((192,288)) * np.nan
p_obs_pr[np.abs(StuT)>scipy.stats.t.ppf(0.975,df = dof)] = 1
del StuT

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  '/Users/yk545/Documents/Research/Data/ERA5/era5_psl_trd_seasonal_1980_2014_cesm2grid.mat')
mat_contents = sio.loadmat(mat_fname)
trd_obs_psl = mat_contents[var1][:,:]
StuT = trd_obs_psl/mat_contents[var1_std][:,:]
del data_dir, mat_fname, mat_contents
p_obs_psl = np.empty((192,288)) * np.nan
p_obs_psl[np.abs(StuT)>scipy.stats.t.ppf(0.975,df = dof)] = 1
del StuT

### Reading CMIP6
vname1 = 'tos_trd_djfmam'
vname2 = 'prect_trd_djfmam'
vname3 = 'psl_trd_djfmam'
fpath = '/Users/yk545/Documents/Research/Data/CMIP6/'
fname = 'cmip6_r1i1p1f1_djfmam_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
tos_cmip6 = mat_contents[vname1][:,:,:]
del data_dir, mat_fname, mat_contents
fname = 'cmip6_r1i1p1f1_djfmam_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_cmip6 = mat_contents[vname2][:,:,:]
del data_dir, mat_fname, mat_contents
fname = 'cmip6_r1i1p1f1_djfmam_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
psl_cmip6 = mat_contents[vname3][:,:,:]
del data_dir, mat_fname, mat_contents

### Reading ACCESS-ESM1-5
fname = 'cmip6_access-esm1-5_djfmam_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
tos_access = mat_contents[vname1][:,:,:]
del data_dir, mat_fname, mat_contents
fname = 'cmip6_access-esm1-5_djfmam_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_access = mat_contents[vname2][:,:,:]
del data_dir, mat_fname, mat_contents
fname = 'cmip6_access-esm1-5_djfmam_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
psl_access = mat_contents[vname3][:,:,:]
del data_dir, mat_fname, mat_contents

### Reading CESM2-LE
fpath = '/Users/yk545/Documents/Research/Data/CESM2LE/'
fname = 'cesm2le_smbb_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
tos_cesm = np.empty((100,192,288)) * np.nan
tos_cesm[0:50,:,:] = mat_contents[vname1][:,:,:]
del data_dir, mat_fname, mat_contents
fname = 'cesm2le_cmip6_tos_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
tos_cesm[50:100,:,:] = mat_contents[vname1][:,:,:]
del data_dir, mat_fname, mat_contents

fname = 'cesm2le_smbb_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_cesm = np.empty((100,192,288)) * np.nan
prect_cesm[0:50,:,:] = mat_contents[vname2][:,:,:]
del data_dir, mat_fname, mat_contents
fname = 'cesm2le_cmip6_prect_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_cesm[50:100,:,:] = mat_contents[vname2][:,:,:]
del data_dir, mat_fname, mat_contents

fname = 'cesm2le_smbb_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
psl_cesm = np.empty((100,192,288)) * np.nan
psl_cesm[0:50,:,:] = mat_contents[vname3][:,:,:]
del data_dir, mat_fname, mat_contents
fname = 'cesm2le_cmip6_psl_trd_seasonal_1980_2014.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
psl_cesm[50:100,:,:] = mat_contents[vname3][:,:,:]
del data_dir, mat_fname, mat_contents

### readig LIM-SSTs
fpath = '/Users/yk545/Documents/Research/Data/TOGA/'
## reading ERSSTv5
fname = 'sst_ersstv5_1854_2020_cesm2_192x288.nc'
ds = nc.Dataset(fpath+fname)
trd_ersst,pstd = lintrd(np.arange(34),DJFMAMmean(ds['sst'][1512:1512+35*12,:,:]))
ds.close()

var1 = 'psl_trd_djfmam'
var2 = 'prect_trd_djfmam'
prect_ersst = np.empty((20,192,288)) * np.nan;
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_ersstv5_prect_trd_seasonal_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
prect_ersst[0:10,:,:] = mat_contents[var2][:,:,:]
del data_dir, mat_fname, mat_contents
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'access_toga_ersstv5_prect_trd_seasonal_1980_2014_gCESM.mat')
mat_contents = sio.loadmat(mat_fname)
prect_ersst[10:20,:,:] = mat_contents[var2][:,:,:]
del data_dir, mat_fname, mat_contents

psl_ersst = np.empty((20,192,288)) * np.nan;
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_ersstv5_psl_trd_seasonal_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
psl_ersst[0:10,:,:] = mat_contents[var1][:,:,:]
del data_dir, mat_fname, mat_contents
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'access_toga_ersstv5_psl_trd_seasonal_1980_2014_gCESM.mat')
mat_contents = sio.loadmat(mat_fname)
psl_ersst[10:20,:,:] = mat_contents[var1][:,:,:]
del data_dir, mat_fname, mat_contents


###### PLOTTING
savepath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_LIM_SWUS_figure/'

nyr_trd = 10
lon2d, lat2d = np.meshgrid(lon_cmip6,lat_cmip6)
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
ax3 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax3.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*trd_obs_pr,\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
density = 3
hc = ax3.contourf(lon_cesm,lat_cesm, p_obs_pr*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = trd_obs_psl/100
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax3.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax3.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
hc = ax3.contourf(lon_cesm,lat_cesm, p_obs_psl,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax3.set_title('Observed ${psl}$, ${pr}$ Trends (ERA5/GPCC)', fontsize=title_font_size)
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
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax3.set_extent(domain)
# SWUS box
ax3.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax3.set_rasterized(True)
plt.savefig(savepath+'figure1_obs_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax4 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax4.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_ersst,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_ersst,np.nanmean(prect_ersst,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_ersst,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax4.contourf(lon_cesm,lat_cesm, S2N*landmask,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_ersst,0)
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
agr = ens_agree_ensmean(psl_ersst,np.nanmean(psl_ersst,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_ersst,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax4.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax4.set_title('TOGA with ERSSTv5 ${psl}$, ${pr}$ Trends (20)', fontsize=title_font_size)
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
plt.savefig(savepath+'figure1_toga_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax2 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*landmask*np.nanmean(prect_cesm,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_cesm,np.nanmean(prect_cesm,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_ersst,0)) * np.NAN
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
cc = ax2.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax2.contour(lon_cesm,lat_cesm,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_cesm,np.nanmean(psl_cesm,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_ersst,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax2.contourf(lon_cesm,lat_cesm, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax2.set_title('CESM2-LE ${psl}$, ${pr}$ Trends (100)', fontsize=title_font_size)
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
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax2.set_extent(domain)
# SWUS box
ax2.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax2.set_rasterized(True)
plt.savefig(savepath+'figure1_cesm2le_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()


fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax1 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax1.contourf(lon_cmip6, lat_cmip6, nyr_trd*maskland_cmip6*np.nanmean(prect_cmip6,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_cmip6,np.nanmean(prect_cmip6,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_cmip6,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax1.contourf(lon_cmip6,lat_cmip6, S2N*maskland_cmip6,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_cmip6,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax1.contour(lon_cmip6,lat_cmip6,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax1.contour(lon_cmip6,lat_cmip6,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_cmip6,np.nanmean(psl_cmip6,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_cmip6,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax1.contourf(lon_cmip6,lat_cmip6, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax1.set_title('CMIP6 ${psl}$, ${pr}$ Trends (17)', fontsize=title_font_size)
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
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax1.set_extent(domain)
# SWUS box
ax1.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax1.set_rasterized(True)
plt.savefig(savepath+'figure1_cmip6_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()    

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax1 = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=-115))
cs = ax1.contourf(lon_cmip6, lat_cmip6, nyr_trd*maskland_cmip6*np.nanmean(prect_access,0),\
          transform=ccrs.PlateCarree(), cmap=cm.BrBG,\
              levels = Levp, extend='both')
cs.set_clim(-clim,clim)
agr = ens_agree_ensmean(prect_access,np.nanmean(prect_access,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(prect_access,0)) * np.NAN
S2N[agr>0.67] = 1
density = 3
hc = ax1.contourf(lon_cmip6, lat_cmip6, S2N*maskland_cmip6,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'.',density*'.'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('black')
    collection.set_linewidth(0.)
ctr_plt = np.nanmean(psl_access,0)
ctr_plt_n = np.empty_like(ctr_plt) * np.nan;
ctr_plt_n[ctr_plt<0] = ctr_plt[ctr_plt<0]
cc = ax1.contour(lon_cmip6,lat_cmip6,nyr_trd*ctr_plt_n,\
                 levels = Lev_psl_n,\
                 transform=ccrs.PlateCarree(),colors='blue')
ctr_plt_p = np.empty_like(ctr_plt) * np.nan;
ctr_plt_p[ctr_plt>0] = ctr_plt[ctr_plt>0]
cc = ax1.contour(lon_cmip6,lat_cmip6,nyr_trd*ctr_plt_p,\
                 levels = Lev_psl,\
                 transform=ccrs.PlateCarree(),colors='red')
density = 1
agr = ens_agree_ensmean(psl_access,np.nanmean(psl_access,0),threadhold = 0)
S2N = np.empty_like(np.nanmean(psl_access,0)) * np.NAN
S2N[agr>0.67] = 1
hc = ax1.contourf(lon_cmip6,lat_cmip6, S2N,\
          transform=ccrs.PlateCarree(), colors='none',\
          hatches=[density*'/',density*'/'])
for i, collection in enumerate(hc.collections):
    collection.set_edgecolor('k')
    collection.set_linewidth(0.)
ax1.set_title('ACCESS-ESM1.5-LE ${psl}$, ${pr}$ Trends (40)', fontsize=title_font_size)
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
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax1.set_extent(domain)
# SWUS box
ax1.add_patch(mpatches.Rectangle(xy=[lon_cesm[188], lat_cesm[130]], width=22.5, height=9.42,edgecolor = 'navy',\
                             linewidth = 2,facecolor='none',transform=ccrs.PlateCarree()))
ax1.set_rasterized(True)
plt.savefig(savepath+'figure1_accessesm_pr_psl_trd_red_blue.eps',dpi=300)
plt.show()    

fig = plt.figure()
fig = plt.figure(facecolor = 'w')
fig.set_size_inches(1.5,6)
cax2 = fig.add_axes([0.05, 0.1, 0.3,0.8])
Lev = np.arange(-clim,clim+0.01,clev)
cbr = fig.colorbar(cs,cax = cax2, fraction=0.08,orientation = 'vertical',\
                 ticks = Lev)
cbr.ax.tick_params(labelsize=12)
cbr.set_label(size=16, \
                  label='mm/month per decade')
cax2.set_rasterized(True)
plt.savefig(savepath+'figure1_pr_colorbar.svg',dpi=300)
plt.show()

fig = plt.figure()
fig = plt.figure(facecolor = 'w')
fig.set_size_inches(8,1.3)
cax2 = fig.add_axes([0.1, 0.5, 0.8,0.25])
Lev = np.arange(-clim,clim+0.01,clev)
cbr = fig.colorbar(cs,cax = cax2, fraction=0.08,orientation = 'horizontal',\
                 ticks = Lev)
cbr.ax.tick_params(labelsize=14)
cbr.set_label(size=18, \
                  label='mm/month per decade')
cax2.set_rasterized(True)
plt.savefig(savepath+'figure1_pr_colorbar_horizontal.svg',dpi=300)
plt.show()

### Plotting SSTs
clim = 0.5
clev = 0.05
Lev = np.arange(-clim,clim+0.01,clev)
domain = [1,359,-28,28]
fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax3 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=-180))
ax3.set_aspect('auto')
cs = ax3.contourf(lon_cesm, lat_cesm, nyr_trd*trd_ersst*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax3.set_title('Observed ${SST}$ Trend (ERSSTv5)', fontsize=title_font_size)
ax3.coastlines(color = 'gray')
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
ax3.add_feature(cfeature.LAND, facecolor='gray')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter(number_format = '.0f')
gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1, color='gray', alpha=0.5)
gl.xlabels_top = False
gl.xlines = False
gl.yformatter = lat_formatter
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax3.set_extent(domain)
ax3.set_rasterized(True)
plt.savefig(savepath+'figure1_obs_sst_trd.eps', dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax2 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=-180))
ax2.set_aspect('auto')
cs = ax2.contourf(lon_cesm, lat_cesm, nyr_trd*np.nanmean(tos_cesm,0)*maskocean_cesm,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax2.set_title('CESM2-LE ${SST}$ Trend (100)', fontsize=title_font_size)
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
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax2.set_extent(domain)
ax2.set_rasterized(True)
plt.savefig(savepath+'figure1_cesm2le_sst_trd.eps',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax1 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=-180))
ax1.set_aspect('auto')
cs = ax1.contourf(lon_cmip6, lat_cmip6, nyr_trd*np.nanmean(tos_cmip6,0)*maskocean,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax1.set_title('CMIP6 ${SST}$ Trend (17)', fontsize=title_font_size)
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
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax1.set_extent(domain)
ax1.set_rasterized(True)
plt.savefig(savepath+'figure1_cmip6_sst_trd.eps',dpi=300)

plt.show()

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(6,3.5)
ax1 = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=-180))
ax1.set_aspect('auto')
cs = ax1.contourf(lon_cmip6, lat_cmip6, nyr_trd*np.nanmean(tos_access,0)*maskocean,\
          transform=ccrs.PlateCarree(), cmap=cm.RdYlBu_r,\
              levels = Lev, extend='both')
cs.set_clim(-clim,clim)
ax1.set_title('ACCESS-ESM1.5-LE ${SST}$ Trend (40)', fontsize=title_font_size)
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
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax1.set_extent(domain)
ax1.set_rasterized(True)
plt.savefig(savepath+'figure1_accessesm_sst_trd.eps',dpi=300)

plt.show()

fig = plt.figure()
fig = plt.figure(facecolor = 'w')
fig.set_size_inches(1.5,6)
cax2 = fig.add_axes([0.05, 0.1, 0.3,0.8])
Lev = np.arange(-clim,clim+0.01,clev*2)
cbr = fig.colorbar(cs,cax = cax2, fraction=0.08,orientation = 'vertical',\
                 ticks = Lev)
cbr.ax.tick_params(labelsize=12)
cbr.set_label(size=16, \
                  label='K per decade')
cax2.set_rasterized(True)
plt.savefig(savepath+'figure1_sst_colorbar.svg',dpi=300)

plt.show()

fig = plt.figure()
fig = plt.figure(facecolor = 'w')
fig.set_size_inches(8,1.3)
cax2 = fig.add_axes([0.1, 0.5, 0.8,0.25])
Lev = np.arange(-clim,clim+0.01,clev*2)
cbr = fig.colorbar(cs,cax = cax2, fraction=0.08,orientation = 'horizontal',\
                 ticks = Lev)
cbr.ax.tick_params(labelsize=14)
cbr.set_label(size=18, \
                  label='K per decade')
cax2.set_rasterized(True)
plt.savefig(savepath+'figure1_sst_colorbar_horizontal.svg',dpi=300)
plt.show()
