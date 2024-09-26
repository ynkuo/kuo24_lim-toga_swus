#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:51:35 2024

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
import pylab

fpath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_natgeo_code/'
fname = 'kuo24_natgeo_figure2_nino34_swus_prect_cesm2_access-esm1-5_pi_hist_toga.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fpath+fname)
mat_contents = sio.loadmat(mat_fname)
prect_pi = mat_contents['prect_pi'][0,:]
prect_pi_access = mat_contents['prect_pi_access'][0,:]
prect_cesm = mat_contents['prect_cesm'][0,:]
prect_access = mat_contents['prect_access'][0,:]
prect_ersst = mat_contents['prect_ersst'][0,:]
prect_lim_e = mat_contents['prect_lim_e'][0,:]
prect_lim_l = mat_contents['prect_lim_l'][0,:]

nino34_pi = mat_contents['nino34_pi'][0,:]
nino34_pi_access = mat_contents['nino34_pi_access'][0,:]
nino34_cesm = mat_contents['nino34_cesm'][0,:]
nino34_access = mat_contents['nino34_access'][0,:]
nino34_ersst = mat_contents['nino34_ersst'][0,:]
nino34_lim_e = mat_contents['nino34_lim_e'][0,:]
nino34_lim_l = mat_contents['nino34_lim_l'][0,:]

###### Sorting
flabel = 26; ftick = 24
nyr_trd = 10
hist_pi = np.histogram(nino34_pi*nyr_trd,30,range =(np.min(nino34_pi*nyr_trd),np.max(nino34_access*nyr_trd)))
bounding = hist_pi[1]
nino34_bin = np.zeros((30,)) * np.nan
count_pi_nino34 = np.zeros((30,)) * np.nan
binned_pi_prect = np.zeros((30,)) * np.nan
for i in range(len(hist_pi[0])):
    nino34_bin[i] = 1/2*(bounding[i]+bounding[i+1])
    tempdata = nino34_pi*10
    tempdata2 = tempdata[tempdata<=bounding[i+1]]
    if (len(tempdata2)!=0):
        tempdata_plotting = tempdata2[tempdata2>bounding[i]]
        print(len(tempdata_plotting))
        if (len(tempdata_plotting)!=0):
            count_pi_nino34[i] = len(tempdata_plotting)/len(tempdata)*100
            temp_prect = np.zeros((len(tempdata_plotting),))
            for j in range(len(tempdata_plotting)):
                temp_prect[j] = prect_pi[np.where(tempdata==tempdata_plotting[j])[0]]*nyr_trd
            binned_pi_prect[i] = np.mean(temp_prect)
        else:
            continue
    else:
        continue

hist_pi_access = np.histogram(nino34_pi_access*nyr_trd,30,range =(np.min(nino34_pi*nyr_trd),np.max(nino34_access*nyr_trd)))
bounding = hist_pi_access[1]
nino34_bin_access = np.zeros((30,)) * np.nan
count_pi_nino34_access = np.zeros((30,)) * np.nan
binned_pi_prect_access = np.zeros((30,)) * np.nan
for i in range(len(hist_pi_access[0])):
    nino34_bin_access[i] = 1/2*(bounding[i]+bounding[i+1])
    tempdata = nino34_pi_access*10
    tempdata2 = tempdata[tempdata<=bounding[i+1]]
    if (len(tempdata2)!=0):
        tempdata_plotting = tempdata2[tempdata2>bounding[i]]
        print(len(tempdata_plotting))
        if (len(tempdata_plotting)!=0):
            count_pi_nino34_access[i] = len(tempdata_plotting)/len(tempdata)*100
            temp_prect = np.zeros((len(tempdata_plotting),))
            for j in range(len(tempdata_plotting)):
                temp_prect[j] = prect_pi_access[np.where(tempdata==tempdata_plotting[j])[0]]*nyr_trd
            binned_pi_prect_access[i] = np.mean(temp_prect)
        else:
            continue
    else:
        continue

hist_cesm2le = np.histogram(nino34_cesm*10,30,range =(np.min(nino34_pi*10),np.max(nino34_access*10)))
bounding = hist_cesm2le[1]
count_cesm2le_nino34 = np.zeros((30,)) * np.nan
binned_cesm2le_prect = np.zeros((30,)) * np.nan
for i in range(len(hist_cesm2le[0])):
    tempdata = nino34_cesm*10
    tempdata2 = tempdata[tempdata<=bounding[i+1]]
    if (len(tempdata2)!=0):
        tempdata_plotting = tempdata2[tempdata2>bounding[i]]
        print(len(tempdata_plotting))
        if (len(tempdata_plotting)!=0):
            count_cesm2le_nino34[i] = len(tempdata_plotting)/len(tempdata)*100
            temp_prect = np.zeros((len(tempdata_plotting),))
            for j in range(len(tempdata_plotting)):

                temp_prect[j] = prect_cesm[np.where(tempdata==tempdata_plotting[j])[0]]*nyr_trd
            binned_cesm2le_prect[i] = np.mean(temp_prect)
        else:
            continue
    else:
        continue

hist_access = np.histogram(nino34_access*10,30,range =(np.min(nino34_pi*10),np.max(nino34_access*10)))
bounding = hist_access[1]
count_access_nino34 = np.zeros((30,)) * np.nan
binned_access_prect = np.zeros((30,)) * np.nan
for i in range(len(hist_cesm2le[0])):
    tempdata = nino34_access*10
    tempdata2 = tempdata[tempdata<=bounding[i+1]]
    if (len(tempdata2)!=0):
        tempdata_plotting = tempdata2[tempdata2>bounding[i]]
        print(len(tempdata_plotting))
        if (len(tempdata_plotting)!=0):
            count_access_nino34[i] = len(tempdata_plotting)/len(tempdata)*100
            temp_prect = np.zeros((len(tempdata_plotting),))
            for j in range(len(tempdata_plotting)):

                temp_prect[j] = prect_access[np.where(tempdata==tempdata_plotting[j])[0]]*nyr_trd
            binned_access_prect[i] = np.mean(temp_prect)
        else:
            continue
    else:
        continue

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(5,20)
ax = fig.add_axes([0.05, 0.25, 0.7, 0.72])
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.grid(visible=True)
plt.plot([-100,2],[0,0],\
          color = 'k',linestyle='-',linewidth = 1)

### flip the histogram to the left
l1 = ax.stairs(-count_pi_nino34, bounding, orientation='horizontal',\
          color = 'darkgray',fill='darkgray',alpha = 0.5, linewidth = 3,label='PiCtrl (CESM2)')
l2 = ax.stairs(-count_pi_nino34_access, bounding, orientation='horizontal',\
          color = 'darkgray',linewidth = 3,hatch='//',label='PiCtrl (ACCESS-ESM1.5)')
l3 = ax.stairs(-count_cesm2le_nino34, bounding, orientation='horizontal',\
           color = 'red',fill='red',alpha = 0.5,linewidth = 3,label='Hist (CESM2)')
l4 = ax.stairs(-count_access_nino34, bounding, orientation='horizontal',\
           color = 'red',linewidth = 3,hatch='//',label='Hist (ACCESS-ESM1.5)')
l5 = plt.plot([-100,2],[nino34_ersst*10,nino34_ersst*10],\
          color = 'violet',linewidth = 6,label='TOGA: ERSSTv5')
l6 = plt.plot([-100,2],[nino34_lim_e*10,nino34_lim_e*10],\
          color = 'coral',linewidth = 6,label='TOGA: El Niño-like')
l7 = plt.plot([-100,2],[nino34_lim_l*10,nino34_lim_l*10],\
          color = 'skyblue',linewidth = 6,label='TOGA: La Niña-like')
plt.yticks(np.arange(-0.4,0.6,0.1),np.round(np.arange(-0.4,0.6,0.1),2),fontsize = ftick)
plt.xticks([-20,-15,-10,-5,0],['20','15','10','5','0'],fontsize = ftick)
plt.xlabel('%',fontsize = flabel+2)
plt.ylabel('Niño 3.4 trend (K per decade)',fontsize = flabel+2)
plt.title('Niño 3.4 trend',fontsize = flabel+4)
plt.legend(fontsize = 23, bbox_to_anchor=(0.6, -0.05),loc='upper center')
plt.axis([-22,0.3,-0.45,0.55])
ax.set_rasterized(True)
savepath = '/Users/yk545/Documents/Research/Manuscript/kuo23_LIM_TOGAs/kuo24_LIM_SWUS_figure/'
plt.savefig(savepath+'figure2_nino34.svg',dpi=300)
plt.show()


fig = plt.figure(facecolor = 'white')
fig.set_size_inches(5,20)
ax = fig.add_axes([0.05, 0.25, 0.7, 0.72])
plt.yticks(np.arange(-0.4,0.6,0.1),fontsize = ftick)
plt.plot(binned_pi_prect,nino34_bin,\
         color = 'darkgray',linewidth = 5,label = 'PiCtrl (CESM2)')
plt.plot(binned_pi_prect_access,nino34_bin,\
         color = 'darkgray',linestyle = '--', linewidth = 5,label = 'PiCtrl (ACCESS-ESM1.5)')
plt.plot(binned_cesm2le_prect,nino34_bin,\
         color = 'red',linewidth = 5,label = 'Hist (CESM2)')
plt.plot(binned_access_prect,nino34_bin,\
         color = 'red',linestyle = '--',linewidth = 5,label = 'Hist (ACCESS-ESM1.5)')
plt.xticks([-4, -2, 0, 2],fontsize = ftick)
plt.xlabel('mm/mon per decade',fontsize = flabel)
plt.plot([-5,5],[0,0],\
          color = 'k',linestyle='-',linewidth = 1)
plt.grid(visible=True)
plt.plot([0,0],[-1,1],\
          color = 'k',linestyle='-',linewidth = 1)
plt.barh(nino34_ersst*10, np.mean(prect_ersst)*10, \
         color = 'violet',orientation = 'horizontal',\
         height = 0.03,xerr = np.std(prect_ersst*10),label = 'TOGA: ERSSTv5')
plt.barh(nino34_lim_e*10, np.mean(prect_lim_e)*10, \
         color = 'coral',orientation = 'horizontal',\
         height = 0.03,xerr = np.std(prect_lim_e*10),label = 'TOGA: El Niño-like')
plt.barh(nino34_lim_l*10, np.mean(prect_lim_l)*10, \
         color = 'skyblue',orientation = 'horizontal',\
         height = 0.03,xerr = np.std(prect_lim_l*10),label = 'TOGA: La Niña-like')
plt.legend(fontsize = 23, bbox_to_anchor=(0.6, -0.05),loc='upper center')
plt.title('SWUS ${pr}$ trend',fontsize = flabel+4)
plt.axis([-5,3,-0.45,0.55])
ax.spines[['right']].set_visible(False)
plt.xticks(np.arange(-4,2.1,2),fontsize = ftick)
plt.yticks(np.arange(-0.4,0.6,0.1),[],fontsize = ftick)
plt.annotate('', xy=(1.1, 0.8), xycoords='axes fraction', xytext=(1.1, 0.55), 
            arrowprops=dict(fc='red',ec = 'red', lw=3))
plt.annotate('', xy=(1.1, 0.05), xycoords='axes fraction', xytext=(1.1, 0.3), 
            arrowprops=dict(fc='blue',ec = 'blue', lw=3))
plt.savefig(savepath+'figure2_swus_pr_sorted_by_nino34.svg', transparent=True)
plt.show()
