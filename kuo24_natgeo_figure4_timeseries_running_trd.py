#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:36:55 2024

@author: yk545
"""
from scipy.io import savemat
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
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import sys
sys.path.append("/Users/yk545/Documents/Research/Codes/Function_YNKuo/")
from function_YNKuo import *

nyr = 34
nyr_trd = 34

sspcolor = ['forestgreen','blue','red','purple']
sspcolorL = ['yellowgreen','skyblue','coral','violet']
ssps = ['ssp126','ssp245','ssp370','ssp585']
fpath = '/Users/yk545/Documents/Research/Data/CMIP6/'

nens_cmip6 = 17  
    
fpath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_natgeo_code/'
fname = 'kuo24_natgeo_figure4_timeseries_running_trd_cmip6_cesm2le_accesmesmle_1950-2050.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
## Reading pr
prect = mat_contents['prect'][:,:,:]
prect_cesm = mat_contents['prect_cesm'][:,:]
prect_access = mat_contents['prect_access'][:,:,:]
prect_runntrd = mat_contents['prect_runntrd'][:,:,:]
prect_cesm_runntrd = mat_contents['prect_cesm_runntrd'][:,:]
prect_access_runntrd = mat_contents['prect_access_runntrd'][:,:,:]
## Reading tas
tas = mat_contents['tas'][:,:,:]
tas_cesm = mat_contents['tas_cesm'][:,:]
tas_access = mat_contents['tas_access'][:,:,:]
tas_runntrd = mat_contents['tas_runntrd'][:,:,:]
tas_cesm_runntrd = mat_contents['tas_cesm_runntrd'][:,:]
tas_access_runntrd = mat_contents['tas_access_runntrd'][:,:,:]
## Reading mrsos
sm = mat_contents['sm'][:,:,:]
sm_cesm = mat_contents['sm_cesm'][:,:]
sm_access = mat_contents['sm_access'][:,:,:]
sm_runntrd = mat_contents['sm_runntrd'][:,:,:]
sm_cesm_runntrd = mat_contents['sm_cesm_runntrd'][:,:]
sm_access_runntrd = mat_contents['sm_access_runntrd'][:,:,:]


savepath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_LIM_SWUS_figure/'
titlef = 20; tickf = 16; labelf = 16.3;
sspname = ['CMIP6-ssp126','CMIP6-ssp245','CMIP6-ssp370','CMIP6-ssp585']
sspcolor = ['green','orange','red','purple']
i=2
nyr_trd = 10; nyr = 34
fig = plt.figure(facecolor = 'w')
fig.set_size_inches(7,10.5)
ax1 = plt.subplot(2,1,1)
ax1.plot([1950-1/12, 2051],[0,0],color = 'lightgray')
ax1.plot([1998+1/6, 1998+1/6],[-200, 200],'--',color = 'gray')
for i in range(4):
    ax1.fill_between(np.arange(1950,1950+101,1),\
                 np.mean(sm[:,:,i],1)+np.std(sm[:,:,i],1),\
                 np.mean(sm[:,:,i],1)-np.std(sm[:,:,i],1),\
                 color = sspcolor[i],alpha = 0.15)
    ax1.plot(np.arange(1950,1950+101,1),np.mean(sm[:,:,i],1),linewidth = 4,\
         color = sspcolor[i],label = sspname[i])
ax1.fill_between(np.arange(1950,1950+101,1),\
                 np.mean(sm_cesm[:,:],1)+np.std(sm_cesm[:,:],1),\
                 np.mean(sm_cesm[:,:],1)-np.std(sm_cesm[:,:],1),\
                 color = 'k',alpha = 0.15)
ax1.plot(np.arange(1950,1950+101,1),np.mean(sm_cesm[:,:],1),linewidth = 4,\
         color = 'k',label = 'CESM2-LE')
ax1.fill_between(np.arange(1950,1950+101,1),\
                 np.mean(sm_access[:,:,2],1)+np.std(sm_access[:,:,2],1),\
                 np.mean(sm_access[:,:,2],1)-np.std(sm_access[:,:,2],1),\
                 color = 'b',alpha = 0.15)
ax1.plot(np.arange(1950,1950+101,1),np.mean(sm_access[:,:,2],1),linewidth = 4,\
         color = 'b',label = 'ACCESS-ESM1.5 LE')
ax1.set_xticks(np.arange(1960,2061,20))
ax1.add_patch(plt.Rectangle((1980, -2.3), width=36, height=4.6,\
                            fill=False, edgecolor='gray', linewidth=3))
ax1.set_title('${mrsos}$ (1950-2050)',fontsize = titlef)
ax1.set_ylabel('$mm^{3}mm^{-3}$',fontsize = labelf)
ax1.set_xticklabels(np.arange(1960,2061,20),fontsize=tickf)
ax1.set_yticklabels(np.arange(-4,5,1),fontsize=tickf)
ax1.axis([1950-1/12, 2051, -2.5, 2.5])
#ax1.axis([1950-1/12, 2051, -0.01, 0.01])

ax2 = plt.subplot(2,1,2)
ax2.plot([1950-1/12, 2051],[0,0],color = 'lightgray')
ax2.plot([1998+1/6, 1998+1/6],[-200, 200],'--',color = 'gray')
for i in range(4):
    ax2.fill_between(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),\
                 (np.mean(sm_runntrd[:,:,i],1)+np.std(sm_runntrd[:,:,i],1))*nyr_trd,\
                 (np.mean(sm_runntrd[:,:,i],1)-np.std(sm_runntrd[:,:,i],1))*nyr_trd,\
                 color = sspcolor[i],alpha = 0.15)
    ax2.plot(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),np.mean(sm_runntrd[:,:,i],1)*nyr_trd,linewidth = 4,\
         color = sspcolor[i],label = sspname[i])

ax2.fill_between(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),\
                 (np.mean(sm_cesm_runntrd[:,:],1)+np.std(sm_cesm_runntrd[:,:],1))*nyr_trd,\
                 (np.mean(sm_cesm_runntrd[:,:],1)-np.std(sm_cesm_runntrd[:,:],1))*nyr_trd,\
                 color = 'k',alpha = 0.15)
ax2.plot(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),np.mean(sm_cesm_runntrd[:,:],1)*nyr_trd,linewidth = 4,\
         color = 'k')

ax2.fill_between(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),\
                 (np.mean(sm_access_runntrd[:,:,2],1)+np.std(sm_access_runntrd[:,:,2],1))*nyr_trd,\
                 (np.mean(sm_access_runntrd[:,:,2],1)-np.std(sm_access_runntrd[:,:,2],1))*nyr_trd,\
                 color = 'b',alpha = 0.15)
ax2.plot(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),np.mean(sm_access_runntrd[:,:,2],1)*nyr_trd,linewidth = 4,\
         color = 'b')
ax2.set_xticks(np.arange(1960,2061,20))
ax2.set_title('running 34-year trend of ${mrsos}$',fontsize = titlef)
ax2.set_ylabel('$mm^{3}mm^{-3}$ per decade',fontsize = labelf)
ax2.set_xticklabels(np.arange(1960,2061,20),fontsize=tickf)
ax2.set_yticks(np.arange(-0.4,0.41,0.2))
ax2.set_yticklabels(np.round(np.arange(-0.4,0.41,0.2),2),fontsize=tickf)
ax2.axis([1950-1/12, 2051, -0.45, 0.45])

ax1.set_rasterized(True)
ax2.set_rasterized(True)
plt.savefig(savepath+'figure4_mrsos_timeseries_runningtrd_cesm100.svg',dpi=300)
plt.show()

fig = plt.figure(facecolor = 'w')
fig.set_size_inches(7,10.5)
ax1 = plt.subplot(2,1,1)
ax1.plot([1950-1/12, 2051],[0,0],color = 'lightgray')
ax1.plot([1998+1/6, 1998+1/6],[-200, 200],'--',color = 'gray')
ax1.add_patch(plt.Rectangle((1980, -4+0.5), width=36, height=8-1,\
                            fill=False, edgecolor='gray', linewidth=3))
for i in range(4):
    ax1.fill_between(np.arange(1950,1950+101,1),\
                 (np.mean(tas[:,:,i],1)+np.std(tas[:,:,i],1)),\
                 (np.mean(tas[:,:,i],1)-np.std(tas[:,:,i],1)),\
                 color = sspcolor[i],alpha = 0.15)
    ax1.plot(np.arange(1950,1950+101,1),np.mean(tas[:,:,i],1),linewidth = 4,\
         color = sspcolor[i],label = sspname[i])
ax1.fill_between(np.arange(1950,1950+101,1),\
                 (np.mean(tas_cesm[:,:],1)+np.std(tas_cesm[:,:],1)),\
                 (np.mean(tas_cesm[:,:],1)-np.std(tas_cesm[:,:],1)),\
                 color = 'k',alpha = 0.15)
ax1.plot(np.arange(1950,1950+101,1),np.mean(tas_cesm[:,:],1),linewidth = 4,\
         color = 'k',label = 'cesm2-le')

ax1.fill_between(np.arange(1950,1950+101,1),\
                 (np.mean(tas_access[:,:,2],1)+np.std(tas_access[:,:,2],1)),\
                 (np.mean(tas_access[:,:,2],1)-np.std(tas_access[:,:,2],1)),\
                 color = 'b',alpha = 0.15)
ax1.plot(np.arange(1950,1950+101,1),np.mean(tas_access[:,:,2],1),linewidth = 4,\
         color = 'b',label = 'access-le')
ax1.set_xticks(np.arange(1960,2061,20))
ax1.set_xticklabels(np.arange(1960,2061,20),fontsize=tickf)
ax1.set_yticklabels(np.arange(-4,5,1),fontsize=tickf)
ax1.axis([1950-1/12, 2051, -4, 4])
ax1.set_title('${tas}$ (1950-2050)',fontsize = titlef)
ax1.set_ylabel('K',fontsize = labelf)

ax2 = plt.subplot(2,1,2)
ax2.plot([1950-1/12, 2051],[0,0],color = 'lightgray')
ax2.plot([1998+1/6, 1998+1/6],[-200, 200],'--',color = 'gray')
for i in range(4):
    ax2.fill_between(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),\
                 (np.mean(tas_runntrd[:,:,i],1)+np.std(tas_runntrd[:,:,i],1))*nyr_trd,\
                 (np.mean(tas_runntrd[:,:,i],1)-np.std(tas_runntrd[:,:,i],1))*nyr_trd,\
                 color = sspcolor[i],alpha = 0.15)
    ax2.plot(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),np.mean(tas_runntrd[:,:,i],1)*nyr_trd,linewidth = 4,\
         color = sspcolor[i],label = sspname[i])
ax2.fill_between(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),\
                 (np.mean(tas_cesm_runntrd[:,:],1)+np.std(tas_cesm_runntrd[:,:],1))*nyr_trd,\
                 (np.mean(tas_cesm_runntrd[:,:],1)-np.std(tas_cesm_runntrd[:,:],1))*nyr_trd,\
                 color = 'k',alpha = 0.15)
ax2.plot(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),np.mean(tas_cesm_runntrd[:,:],1)*nyr_trd,linewidth = 4,\
         color = 'k',label = 'CESM2-LE')

ax2.fill_between(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),\
                 (np.mean(tas_access_runntrd[:,:,2],1)+np.std(tas_access_runntrd[:,:,2],1))*nyr_trd,\
                 (np.mean(tas_access_runntrd[:,:,2],1)-np.std(tas_access_runntrd[:,:,2],1))*nyr_trd,\
                 color = 'b',alpha = 0.15)
ax2.plot(np.arange(1950+nyr/2+1/6,1950+67+nyr/2+1/6,1),np.mean(tas_access_runntrd[:,:,2],1)*nyr_trd,linewidth = 4,\
         color = 'b',label = 'ACCESS-ESM1.5 LE')
ax2.set_xticks(np.arange(1960,2061,20))
ax2.legend()
ax2.axis([1950-1/12, 2051, -1, 1])
ax2.set_title('running 34-year trend of ${tas}$',fontsize = titlef)
ax2.set_ylabel('K per decade',fontsize = labelf)
ax2.set_xticklabels(np.arange(1960,2061,20),fontsize=tickf)
ax2.set_yticks(np.arange(-1,1.1,0.5))
ax2.set_yticklabels(np.arange(-1,1.1,0.5),fontsize=tickf)
ax1.set_rasterized(True)
ax2.set_rasterized(True)
plt.savefig(savepath+'figure4_tas_timeseries_runningtrd_cesm100.svg',dpi=300)
plt.show()

nyr = 10

fig = plt.figure(facecolor = 'w')
fig.set_size_inches(7,10.5)
ax1 = plt.subplot(2,1,1)
ax1.plot([1950-1/12, 2051],[0,0],color = 'lightgray')
ax1.plot([1998+1/6, 1998+1/6],[-200, 200],'--',color = 'gray')
ax1.add_patch(plt.Rectangle((1980, -20+0.5), width=36, height=20*2-1,\
                            fill=False, edgecolor='gray', linewidth=3))
for i in range(4):
    ax1.fill_between(np.arange(1950,1950+100,1),\
                 np.mean(prect[:,:,i],1)+np.std(prect[:,:,i],1),\
                 np.mean(prect[:,:,i],1)-np.std(prect[:,:,i],1),\
                 color = sspcolor[i],alpha = 0.15)
    ax1.plot(np.arange(1950,1950+100,1),np.mean(prect[:,:,i],1),linewidth = 4,\
         color = sspcolor[i],label = sspname[i])
ax1.fill_between(np.arange(1950,1950+100,1),\
                 np.mean(prect_cesm[:,:],1)+np.std(prect_cesm[:,:],1),\
                 np.mean(prect_cesm[:,:],1)-np.std(prect_cesm[:,:],1),\
                 color = 'k',alpha = 0.15)
ax1.plot(np.arange(1950,1950+100,1),np.mean(prect_cesm[:,:],1),linewidth = 4,\
         color = 'k',label = sspname[i])
ax1.fill_between(np.arange(1950,1950+100,1),\
                 np.mean(prect_access[:,:,2],1)+np.std(prect_access[:,:,2],1),\
                 np.mean(prect_access[:,:,2],1)-np.std(prect_access[:,:,2],1),\
                 color = 'b',alpha = 0.15)
ax1.plot(np.arange(1950,1950+100,1),np.mean(prect_access[:,:,2],1),linewidth = 4,\
         color = 'b',label = sspname[i])
ax1.set_xticks(np.arange(1960,2061,20))
ax1.axis([1950-1/12, 2051, -20, 20])
ax1.set_title('$pr$ (1950-2050)',fontsize = titlef)
ax1.set_ylabel('mm/month',fontsize = labelf)
ax1.set_xticklabels(np.arange(1960,2061,20),fontsize=tickf)
ax1.set_yticks(np.round(np.arange(-20,21,10),2))
ax1.set_yticklabels(np.round(np.arange(-20,21,10),2),fontsize=tickf)

ax2 = plt.subplot(2,1,2)
ax2.plot([1950-1/12, 2051],[0,0],color = 'lightgray')
ax2.plot([1998+1/6, 1998+1/6],[-200, 200],'--',color = 'gray')
for i in range(4):
    ax2.fill_between(np.arange(1950+18+1/6,1950+67+18+1/6,1),\
                 (np.mean(prect_runntrd[:,:,i],1)+np.std(prect_runntrd[:,:,i],1))*nyr,\
                 (np.mean(prect_runntrd[:,:,i],1)-np.std(prect_runntrd[:,:,i],1))*nyr,\
                 color = sspcolor[i],alpha = 0.15)
    ax2.plot(np.arange(1950+18+1/6,1950+67+18+1/6,1),np.mean(prect_runntrd[:,:,i],1)*nyr,linewidth = 4,\
         color = sspcolor[i],label = sspname[i])
ax2.fill_between(np.arange(1950+18+1/6,1950+67+18+1/6,1),\
                 (np.mean(prect_cesm_runntrd[:,:],1)+np.std(prect_cesm_runntrd[:,:],1))*nyr,\
                 (np.mean(prect_cesm_runntrd[:,:],1)-np.std(prect_cesm_runntrd[:,:],1))*nyr,\
                 color = 'k',alpha = 0.15)
ax2.plot(np.arange(1950+18+1/6,1950+67+18+1/6,1),np.mean(prect_cesm_runntrd[:,:],1)*nyr,linewidth = 4,\
         color = 'k')
ax2.fill_between(np.arange(1950+18+1/6,1950+67+18+1/6,1),\
                 (np.mean(prect_access_runntrd[:,:,2],1)+np.std(prect_access_runntrd[:,:,2],1))*nyr,\
                 (np.mean(prect_access_runntrd[:,:,2],1)-np.std(prect_access_runntrd[:,:,2],1))*nyr,\
                 color = 'b',alpha = 0.15)
ax2.plot(np.arange(1950+18+1/6,1950+67+18+1/6,1),np.mean(prect_access_runntrd[:,:,2],1)*nyr,linewidth = 4,\
         color = 'b')

ax2.set_xticks(np.arange(1960,2061,20))
ax2.set_title('running 34-year trend of $pr$',fontsize = titlef)
ax2.set_ylabel('mm/month per decade',fontsize = labelf)
ax2.set_xticklabels(np.arange(1960,2061,20),fontsize=tickf)
ax2.set_yticklabels(np.round(np.arange(-1.5,1.51,0.5),2),fontsize=tickf)
ax2.axis([1950-1/12, 2051, -1.5, 1.5])
ax1.set_rasterized(True)
ax2.set_rasterized(True)
plt.savefig(savepath+'figure4_pr_timeseries_runningtrd_cesm100.svg',dpi=300)
plt.show()
