#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:08:10 2023

@author: yk545
"""

import numpy as np
import netCDF4 as nc
from os.path import dirname, join as pjoin
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import scipy.stats
from cartopy.util import add_cyclic_point
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import sys
sys.path.append("/Users/yk545/Documents/Research/Codes/Function_YNKuo/")
from function_YNKuo import *

savepath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_LIM_SWUS_figure/'

nyr = 34

#### Loading data
fpath = '/Users/yk545/Documents/Research/Data/Observation/'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'obs_swus_prect_ts_sm_1980_2020.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam = DJFMAMmean(mat_contents['prect'][0,0:35*12])
sm_mam, sm_jja, sm_son, sm_djf = seasonalM(mat_contents['sm'][0,0:35*12])
ts_mam, ts_jja, ts_son, ts_djf = seasonalM(mat_contents['ts'][0,0:35*12]+273.15)

fpath = '/Users/yk545/Documents/Research/Data/GOGA/'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_goga_ersstv5_swus_prect_tas_sm_1980_2016.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_goga = DJFMAMmean(mat_contents['prect'][:,0:35*12].T)
sm_mam_goga, sm_jja_goga, sm_son_goga, sm_djf_goga = seasonalM(mat_contents['sm'][:,0:35*12].T)
ts_mam_goga, ts_jja_goga, ts_son_goga, ts_djf_goga = seasonalM(mat_contents['tas'][:,0:35*12].T)

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'access_goga_ersstv5_swus_prect_tas_sm_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_access = DJFMAMmean(mat_contents['prect'][:,:].T)
sm_mam_access, sm_jja_access, sm_son_access, sm_djf_access = seasonalM(mat_contents['sm'][:,:].T)
ts_mam_access, ts_jja_access, ts_son_access, ts_djf_access = seasonalM(mat_contents['tas'][:,:].T)

fpath = '/Users/yk545/Documents/Research/Data/TOGA/'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_ersstv5_swus_prect_tas_sm_1980_2016.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_ersst = DJFMAMmean(mat_contents['prect'][:,0:35*12].T)
sm_mam_ersst, sm_jja_ersst, sm_son_ersst, sm_djf_ersst = seasonalM(mat_contents['sm'][:,0:35*12].T)
ts_mam_ersst, ts_jja_ersst, ts_son_ersst, ts_djf_ersst = seasonalM(mat_contents['tas'][:,0:35*12].T)
    
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_lim60_swus_prect_tas_sm_1980_2016.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_lim60 = DJFMAMmean(mat_contents['prect'][:,0:35*12].T)
sm_mam_lim60, sm_jja_lim60, sm_son_lim60, sm_djf_lim60 = seasonalM(mat_contents['sm'][:,0:35*12].T)
ts_mam_lim60, ts_jja_lim60, ts_son_lim60, ts_djf_lim60 = seasonalM(mat_contents['tas'][:,0:35*12].T)

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'cam6_toga_lim67_swus_prect_tas_sm_1980_2016.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_lim67 = DJFMAMmean(mat_contents['prect'][:,0:35*12].T)
sm_mam_lim67, sm_jja_lim67, sm_son_lim67, sm_djf_lim67 = seasonalM(mat_contents['sm'][:,0:35*12].T)
ts_mam_lim67, ts_jja_lim67, ts_son_lim67, ts_djf_lim67 = seasonalM(mat_contents['tas'][:,0:35*12].T)

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'access_toga_ersstv5_swus_prect_tas_sm_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_ersst_a = DJFMAMmean(mat_contents['prect'][:,].T)
sm_mam_ersst_a, sm_jja_ersst_a, sm_son_ersst_a, sm_djf_ersst_a = seasonalM(mat_contents['sm'][:,:].T)
ts_mam_ersst_a, ts_jja_ersst_a, ts_son_ersst_a, ts_djf_ersst_a = seasonalM(mat_contents['tas'][:,:].T)

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'access_toga_lim60_swus_prect_tas_sm_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_lim60_a = DJFMAMmean(mat_contents['prect'][:,:].T)
sm_mam_lim60_a, sm_jja_lim60_a, sm_son_lim60_a, sm_djf_lim60_a = seasonalM(mat_contents['sm'][:,:].T)
ts_mam_lim60_a, ts_jja_lim60_a, ts_son_lim60_a, ts_djf_lim60_a = seasonalM(mat_contents['tas'][:,:].T)

data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+'access_toga_lim67_swus_prect_tas_sm_1980_2014.mat')
mat_contents = sio.loadmat(mat_fname)
prect_djfmam_lim67_a = DJFMAMmean(mat_contents['prect'][:,:].T)
sm_mam_lim67_a, sm_jja_lim67_a, sm_son_lim67_a, sm_djf_lim67_a = seasonalM(mat_contents['sm'][:,:].T)
ts_mam_lim67_a, ts_jja_lim67_a, ts_son_lim67_a, ts_djf_lim67_a = seasonalM(mat_contents['tas'][:,:].T)


### For DJFMAM PRECT vs JJA SM TS
Ttrd = np.arange(nyr)

precttrd_amip = np.empty((20,)) * np.nan; precttrd_O = np.empty((20,)) * np.nan
precttrd_E = np.empty((20,)) * np.nan; precttrd_L = np.empty((20,)) * np.nan

sv = 0
prect_obs = scale_data(prect_djfmam,scale = sv)
precttrd_amip[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(prect_djfmam_goga,scale = sv))
precttrd_amip[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(prect_djfmam_access,scale = sv))
precttrd_O[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(prect_djfmam_ersst,scale = sv))
precttrd_E[0:10],trdstd = lintrd(Ttrd,scale_data(prect_djfmam_lim60,scale = sv))
precttrd_L[0:10],trdstd = lintrd(Ttrd,scale_data(prect_djfmam_lim67,scale = sv))
precttrd_O[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(prect_djfmam_ersst_a,scale = sv))
precttrd_E[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(prect_djfmam_lim60_a,scale = sv))
precttrd_L[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(prect_djfmam_lim67_a,scale = sv))

tastrd_amip = np.empty((20,)) * np.nan; tastrd_O = np.empty((20,)) * np.nan
tastrd_E = np.empty((20,)) * np.nan; tastrd_L = np.empty((20,)) * np.nan

sv = 0
ts_obs = scale_data(ts_jja[1:len(ts_jja)],scale = sv)
tastrd_amip[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_goga[1:len(ts_jja_goga),:],scale = sv))
tastrd_amip[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_access[1:len(ts_jja_access),:],scale = sv))
tastrd_O[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_ersst[1:len(ts_jja_ersst),:],scale = sv))
tastrd_E[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_lim60[1:len(ts_jja_lim60),:],scale = sv))
tastrd_L[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_lim67[1:len(ts_jja_lim67),:],scale = sv))
tastrd_O[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_ersst_a[1:len(ts_jja_ersst_a),:],scale = sv))
tastrd_E[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_lim60_a[1:len(ts_jja_lim60_a),:],scale = sv))
tastrd_L[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(ts_jja_lim67_a[1:len(ts_jja_lim67_a),:],scale = sv))

smtrd_amip = np.empty((20,)) * np.nan; smtrd_O = np.empty((20,)) * np.nan
smtrd_E = np.empty((20,)) * np.nan; smtrd_L = np.empty((20,)) * np.nan

sv = 1 ## minus mean and divided by std

sm_obs = scale_data(sm_jja[1:len(sm_jja)],scale = sv)
smtrd_amip[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_goga[1:len(sm_jja_goga),:],scale = sv))
smtrd_amip[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_access[1:len(sm_jja_access),:],scale = sv))
smtrd_O[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_ersst[1:len(sm_jja_ersst),:],scale = sv))
smtrd_E[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_lim60[1:len(sm_jja_lim60),:],scale = sv))
smtrd_L[0:10],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_lim67[1:len(sm_jja_lim67),:],scale = sv))
smtrd_O[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_ersst_a[1:len(sm_jja_ersst_a),:],scale = sv))
smtrd_E[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_lim60_a[1:len(sm_jja_lim60_a),:],scale = sv))
smtrd_L[10:20],trdstd = lintrd(Ttrd[0:34],scale_data(sm_jja_lim67_a[1:len(sm_jja_lim67_a),:],scale = sv))

precttrd_obs, precttrdstd_obs = lintrd(Ttrd,prect_obs)
tstrd_obs, tstrdstd_obs = lintrd(Ttrd,ts_obs)
smtrd_obs, smtrdstd_obs = lintrd(Ttrd,sm_obs)


###### Reading PiCtrl data

fpath = '/Users/yk545/Documents/Research/Data/CESM2_PiControl/'
fname = 'cesm2_pictrl_djfmam_nino34_swus_pr_jja_swus_sm_tas_34yr_overlapped.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_pi_cesm2 = mat_contents['prect_trd_djfmam'][0,:]
tas_pi_cesm2 = mat_contents['tas_trd_jja'][0,:]
sm_pi_cesm2 = mat_contents['sm_trd_jja'][0,:]


fpath = '/Users/yk545/Documents/Research/Data/CMIP6/'
fname = 'cmip6_access-esm1-5_piControl_r1i1p1f1_swus_prect_tas_sm_tos_nino34_trd.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_pi_access = mat_contents['prect_trd_djfmam'][0,:]
tas_pi_access = mat_contents['tas_trd_jja'][0,:]
sm_pi_access = mat_contents['sm_trd_jja'][0,:]

fname = 'cmip6_piControl_r1i1p1f1_swus_prect_tas_sm_wp_tos_nino34_trd.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_pi = mat_contents['prect_trd_djfmam'][:,:]
prect_pi[:,1] = np.nan; prect_pi[:,4] = np.nan;
tas_pi = mat_contents['tas_trd_jja'][:,:]
tas_pi[:,1] = np.nan; tas_pi[:,4] = np.nan;
sm_pi = mat_contents['sm_trd_jja'][:,:]
sm_pi[:,1] = np.nan; sm_pi[:,4] = np.nan;

fname = 'cmip6_r1i1p1f1_swus_prect_tas_sm_nino34_1950_2100.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect = DJFMAMmean(mat_contents['prect'][30*12:(30+35)*12,:,2])
temp, sm, temp1, temp2 = seasonalM(mat_contents['sm'][30*12:(30+35)*12,:,2])
temp, tas, temp1, temp2 = seasonalM(mat_contents['tas'][30*12:(30+35)*12,:,2])

sm = scale_data(sm[1:len(ts_jja_goga),:],scale = 1)
prect = scale_data(prect,scale = 0)
tas = scale_data(tas[1:len(ts_jja_goga),:],scale = 0)

Ttrd = np.arange(nyr)
tas_trd_cmip6,trdstd = lintrd(Ttrd,tas)
sm_trd_cmip6,trdstd = lintrd(Ttrd,sm)
prect_trd_cmip6,trdstd = lintrd(Ttrd,prect)



tas_trd_cesm2le = np.empty((100,)) * np.nan
sm_trd_cesm2le = np.empty((100,)) * np.nan
prect_trd_cesm2le = np.empty((100,)) * np.nan

fpath = '/Users/yk545/Documents/Research/Data/CESM2LE/'
fname = 'cesm2le_smbb_swus_prect_tas_sm_npi_nino34_1950_2050.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_cesm = DJFMAMmean(mat_contents['prect'][:,30*12:(30+35)*12].T)
temp, sm_cesm, temp1, temp2 = seasonalM(mat_contents['sm'][:,30*12:(30+35)*12].T)
temp, tas_cesm, temp1, temp2 = seasonalM(mat_contents['tas'][:,30*12:(30+35)*12].T)

sm = scale_data(sm_cesm[1:len(ts_jja_goga),:],scale = 1)
prect = scale_data(prect_cesm,scale = 0)
tas = scale_data(tas_cesm[1:len(ts_jja_goga),:],scale = 0)

tas_trd_cesm2le[0:50],trdstd = lintrd(Ttrd,tas)
sm_trd_cesm2le[0:50],trdstd = lintrd(Ttrd,sm)
prect_trd_cesm2le[0:50],trdstd = lintrd(Ttrd,prect)

fpath = '/Users/yk545/Documents/Research/Data/CESM2LE/'
fname = 'cesm2le_cmip6_swus_prect_tas_sm_npi_nino34_1950_2050.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_cesm = DJFMAMmean(mat_contents['prect'][:,30*12:(30+35)*12].T)
temp, sm_cesm, temp1, temp2 = seasonalM(mat_contents['sm'][:,30*12:(30+35)*12].T)
temp, tas_cesm, temp1, temp2 = seasonalM(mat_contents['tas'][:,30*12:(30+35)*12].T)

sm = scale_data(sm_cesm[1:len(ts_jja_goga),:],scale = 1)
prect = scale_data(prect_cesm,scale = 0)
tas = scale_data(tas_cesm[1:len(ts_jja_goga),:],scale = 0)

tas_trd_cesm2le[50:100],trdstd = lintrd(Ttrd,tas)
sm_trd_cesm2le[50:100],trdstd = lintrd(Ttrd,sm)
prect_trd_cesm2le[50:100],trdstd = lintrd(Ttrd,prect)

fpath = '/Users/yk545/Documents/Research/Data/CMIP6/'
fname = 'access-esm1-5_40member_swus_prect_tas_sm_nino34_tos_wp_npi_1950_2100.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_access = DJFMAMmean(mat_contents['prect'][30*12:(30+35)*12,:,2])
temp, sm_access, temp1, temp2 = seasonalM(mat_contents['sm'][30*12:(30+35)*12,:,2])
temp, tas_access, temp1, temp2 = seasonalM(mat_contents['tas'][30*12:(30+35)*12,:,2])

sm = scale_data(sm_access[1:len(ts_jja_goga),:],scale = 1)
prect = scale_data(prect_access,scale = 0)
tas = scale_data(tas_access[1:len(ts_jja_goga),:],scale = 0)

tas_trd_access,trdstd = lintrd(Ttrd,tas)
sm_trd_access,trdstd = lintrd(Ttrd,sm)
prect_trd_access,trdstd = lintrd(Ttrd,prect)


Aname = ['CMIP6 (ssp370)','CESM2-LE (ssp370)','ACCESS-ESM1.5-LE (ssp370)','Observed (TOGA)',\
         'La Niña-like (TOGA)','El Niño-like (TOGA)']
meanAtas = [np.nanmean(tas_trd_cmip6),np.nanmean(tas_trd_cesm2le),\
            np.nanmean(tas_trd_access),np.nanmean(tastrd_O),\
                np.nanmean(tastrd_L),np.nanmean(tastrd_E)]
maxAtas = [np.nanmax(tas_trd_cmip6),np.nanmax(tas_trd_cesm2le),\
           np.nanmax(tas_trd_access),np.nanmax(tastrd_O),\
               np.nanmax(tastrd_L),np.nanmax(tastrd_E)]
minAtas = [np.nanmin(tas_trd_cmip6),np.nanmin(tas_trd_cesm2le),\
           np.nanmin(tas_trd_access),np.nanmin(tastrd_O),\
               np.nanmin(tastrd_L),np.nanmin(tastrd_E)]
meanAprect = [np.nanmean(prect_trd_cmip6),np.nanmean(prect_trd_cesm2le),\
              np.nanmean(prect_trd_access),np.nanmean(precttrd_O),\
                  np.nanmean(precttrd_L),np.nanmean(precttrd_E)]
maxAprect = [np.nanmax(prect_trd_cmip6),np.nanmax(prect_trd_cesm2le),\
             np.nanmax(prect_trd_access),np.nanmax(precttrd_O),\
                 np.nanmax(precttrd_L),np.nanmax(precttrd_E)]
minAprect = [np.nanmin(prect_trd_cmip6),np.nanmin(prect_trd_cesm2le),\
             np.nanmin(prect_trd_access),np.nanmin(precttrd_O),\
                 np.nanmin(precttrd_L),np.nanmin(precttrd_E)]

nyr_trd = 10
slim = [-0.6,0.6]
cbrtick = np.arange(slim[0],slim[1]+0.1,0.2)


colors = [matplotlib.colors.to_rgb('saddlebrown'),matplotlib.colors.to_rgb('white'),matplotlib.colors.to_rgb('forestgreen')];
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=18)   
scolor = 'saddlebrown'   
markerssp = ['^','s','X','d']
sspname = ['ssp126 (1980-2016)','ssp245 (1980-2016)','ssp370 (1980-2016)','ssp585 (1980-2016)']
sspname2 = ['ssp126 (2014-2050)','ssp245 (2014-2050)','ssp370 (2014-2050)','ssp585 (2014-2050)']
fig = plt.figure(facecolor = 'w')
fig.set_size_inches(10,6)
ax = plt.subplot(111)

ax.plot([-10,10],[0,0],color = 'lightgray')
ax.plot([0,0],[-100,100],color = 'lightgray')
ax.plot([scipy.stats.t.ppf(0.05, df = nyr-2, loc=0, scale=1)*tstrdstd_obs*nyr_trd,\
          scipy.stats.t.ppf(0.05, df = nyr-2, loc=0, scale=1)*tstrdstd_obs*nyr_trd],\
         [scipy.stats.t.ppf(0.95, df = nyr-2, loc=0, scale=1)*precttrdstd_obs*nyr_trd,100],'--',color = 'lightgray')
ax.plot([scipy.stats.t.ppf(0.95, df = nyr-2, loc=0, scale=1)*tstrdstd_obs*nyr_trd,\
          scipy.stats.t.ppf(0.95, df = nyr-2, loc=0, scale=1)*tstrdstd_obs*nyr_trd],\
         [-100,scipy.stats.t.ppf(0.05, df = nyr-2, loc=0, scale=1)*precttrdstd_obs*nyr_trd],'--',color = 'lightgray')
ax.plot([scipy.stats.t.ppf(0.95, df = nyr-2, loc=0, scale=1)*tstrdstd_obs*nyr_trd,10],\
         [scipy.stats.t.ppf(0.05, df = nyr-2, loc=0, scale=1)*precttrdstd_obs*nyr_trd,\
          scipy.stats.t.ppf(0.05, df = nyr-2, loc=0, scale=1)*precttrdstd_obs*nyr_trd],\
         '--',color = 'lightgray')
ax.plot([-10,scipy.stats.t.ppf(0.05, df = nyr-2, loc=0, scale=1)*tstrdstd_obs*nyr_trd],\
         [scipy.stats.t.ppf(0.95, df = nyr-2, loc=0, scale=1)*precttrdstd_obs*nyr_trd,\
          scipy.stats.t.ppf(0.95, df = nyr-2, loc=0, scale=1)*precttrdstd_obs*nyr_trd],\
         '--',color = 'lightgray')

### Merging the three PiControl sources (CMIP6, ACCESS-ESM-LE, CESM2-LE)
ellipse_x = np.empty((4256+866+1966,)) * np.nan
ellipse_x[0:4256] = np.reshape(tas_pi,(4256));
ellipse_x[4256:4256+866] = np.reshape(tas_pi_access,(866));
ellipse_x[4256+866:4256+866+1966] = np.reshape(tas_pi_cesm2,(1966));
ellipse_y = np.empty((4256+866+1966,)) * np.nan
ellipse_y[0:4256] = np.reshape(prect_pi,(4256));
ellipse_y[4256:4256+866] = np.reshape(prect_pi_access,(866));
ellipse_y[4256+866:4256+866+1966] = np.reshape(prect_pi_cesm2,(1966));    
ellipse_z = np.empty((4256+866+1966,)) * np.nan
ellipse_z[0:4256] = np.reshape(sm_pi,(4256));
ellipse_z[4256:4256+866] = np.reshape(sm_pi_access,(866));
ellipse_z[4256+866:4256+866+1966] = np.reshape(sm_pi_cesm2,(1966));    

xmin = -0.6; xmax = 0.8; ymin = -10; ymax = 10
nbin = 30
xbin = np.linspace(xmin,xmax,nbin+1); ybin = np.linspace(ymin,ymax,nbin+1)
H, xedges, yedges = np.histogram2d(ellipse_x[~np.isnan(ellipse_x)]*nyr_trd, \
                                   ellipse_y[~np.isnan(ellipse_x)]*nyr_trd, \
                                       bins = [xbin, ybin], weights = ellipse_z[~np.isnan(ellipse_x)]*nyr_trd)
H_counts, xedges, yedges = np.histogram2d(ellipse_x[~np.isnan(ellipse_x)]*nyr_trd, \
                                          ellipse_y[~np.isnan(ellipse_x)]*nyr_trd, bins = [xbin, ybin])
Hplot = np.empty_like(H) * np.nan
Hplot[H_counts>0] = H[H_counts>0]/H_counts[H_counts>0]
tick_font_size = 14
XX = 0.5 * (xedges[0:nbin] + xedges[1:nbin+1])
YY = 0.5 * (yedges[0:nbin] + yedges[1:nbin+1])
XX_m, YY_m = np.meshgrid(XX,YY)
cs = ax.pcolormesh(XX_m, YY_m, Hplot.T,cmap = cmap, vmax = 0.6, vmin = -0.6)


for i in range(6):
    ax.plot([minAtas[i]*nyr_trd,maxAtas[i]*nyr_trd],[meanAprect[i]*nyr_trd,meanAprect[i]*nyr_trd],\
             color = 'saddlebrown',zorder = 750,linewidth = 3)
    ax.plot([meanAtas[i]*nyr_trd,meanAtas[i]*nyr_trd],[minAprect[i]*nyr_trd,maxAprect[i]*nyr_trd],\
             color = 'saddlebrown',zorder = 750,linewidth =  3)
ax.scatter(np.mean(tas_trd_cmip6)*nyr_trd,np.mean(prect_trd_cmip6)*nyr_trd, \
            c = np.mean(sm_trd_cmip6)*nyr_trd,linewidths = 2.5,\
            cmap=cmap,alpha=1,marker = '^',\
            vmax = slim[1],vmin = slim[0], s = 200,zorder = 800,edgecolors = 'saddlebrown')
ax.scatter(np.mean(tas_trd_cesm2le)*nyr_trd,np.mean(prect_trd_cesm2le)*nyr_trd, \
            c = np.mean(sm_trd_cesm2le)*nyr_trd,linewidths = 2.5,\
            cmap=cmap,alpha=1,marker = '>',\
            vmax = slim[1],vmin = slim[0], s = 200,zorder = 800,edgecolors = 'saddlebrown')
ax.scatter(np.mean(tas_trd_access)*nyr_trd,np.mean(prect_trd_access)*nyr_trd, \
            c = np.mean(sm_trd_access)*nyr_trd,linewidths = 2.5,\
            cmap=cmap,alpha=1,marker = '<',\
            vmax = slim[1],vmin = slim[0], s = 200,zorder = 800,edgecolors = 'saddlebrown')
ax.scatter(np.nanmean(tastrd_O)*nyr_trd,np.nanmean(precttrd_O)*nyr_trd, \
            c = np.nanmean(smtrd_O)*nyr_trd,linewidths = 2.5,\
            cmap=cmap,alpha=1,marker = 'X',\
            vmax = slim[1],vmin = slim[0], s = 200,zorder = 800,edgecolors = 'saddlebrown')
ax.scatter(np.mean(tastrd_E)*nyr_trd,np.mean(precttrd_E)*nyr_trd,linewidths = 2.5, \
            c = np.mean(smtrd_E)*nyr_trd,\
            cmap=cmap,alpha=1,marker = 's',\
            vmax = slim[1],vmin = slim[0], s = 200,zorder = 800,edgecolors = 'saddlebrown')
ax.scatter(np.mean(tastrd_L)*nyr_trd,np.mean(precttrd_L)*nyr_trd,linewidths = 2.5, \
            c = np.mean(smtrd_L)*nyr_trd,\
            cmap=cmap,alpha=1,marker = 'd',\
            vmax = slim[1],vmin = slim[0], s = 200,zorder = 800,edgecolors = 'saddlebrown')

cs1 = ax.scatter(tstrd_obs*nyr_trd,precttrd_obs*nyr_trd, \
            c = smtrd_obs*nyr_trd,\
            cmap=cmap,alpha=1,marker = '*',linewidths = 2.5,\
            vmax = slim[1],vmin = slim[0], s = 400,zorder = 1000,edgecolors = 'saddlebrown')
cbr = fig.colorbar(cs1,fraction=0.05,orientation = 'vertical',\
                   ticks = cbrtick)
cbr.set_label(size=14, \
                  label='JJA ${mrsos}$ (Normalized; per decade)')

#### Plotting the markers outside of the canvas to create the legend
ax.scatter([-100],[-100],alpha = 0.7, s = 400, marker = '*', facecolor = scolor,label = 'Observation-based')
ax.scatter([-100],[-100],alpha = 0.7, s = 150, marker = '^', facecolor = scolor,label = Aname[0])
ax.scatter([-100],[-100],alpha = 0.7, s = 150, marker = '>', facecolor = scolor,label = Aname[1])
ax.scatter([-100],[-100],alpha = 0.7, s = 150, marker = '<', facecolor = scolor,label = Aname[2])
ax.scatter([-100],[-100],alpha = 0.7, s = 150, marker = 'X', facecolor = scolor,label = Aname[3])
ax.scatter([-100],[-100],alpha = 0.7, s = 150, marker = 'd', facecolor = scolor,label = Aname[4])
ax.scatter([-100],[-100],alpha = 0.7, s = 150, marker = 's', facecolor = scolor,label = Aname[5])

plt.xticks(fontsize = 14); plt.yticks(fontsize = 14)
plt.ylabel('DJFMAM ${pr}$ (mm/month  per decade)', fontsize = 14)
plt.xlabel('JJA ${tas}$ (K  per decade)', fontsize = 14)
plt.axis([-0.6,0.8,-10,10])
plt.legend(ncol = 1, loc = 'upper right',fontsize = 12)
plt.title('Relationship of Trends: DJFMAM ${pr}$, JJA ${tas}$, and JJA ${mrsos}$', fontsize = 18)
plt.savefig(savepath+'figure4_scatter_plot_cesm2le100_legend.svg',dpi=300)
plt.show()
