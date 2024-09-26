#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
import numpy as np
import scipy.stats
from netCDF4 import Dataset
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from os import walk
import glob
import re
import sys
from scipy.io import savemat
from os.path import dirname, join as pjoin
import scipy.io as sio
sys.path.append("/glade/work/kuoyan/code_YNKuo/")
from function_script_YNKuo import *

#### Scatter plot
fname = '/glade/u/home/kuoyan/code_analysis/code_YNKuo/kuo24_natgeo_figure3_scatter_plot_vpg_200_npi_nino34.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fname)
mat_contents = sio.loadmat(mat_fname)
vp_trd_pi = mat_contents['vp_trd_pi'][0,:]
vp_trd_le = mat_contents['vp_trd_le'][0,:]
vp_trd_lim_e = mat_contents['vp_trd_lim_e'][0,:]
vp_trd_lim_l = mat_contents['vp_trd_lim_l'][0,:]
vp_trd_rf = mat_contents['vp_trd_rf'][0,:]

npi_trd_pi = mat_contents['npi_trd_pi'][0,:]
npi_trd_le = mat_contents['npi_trd_le'][0,:]
npi_trd_lim_e = mat_contents['npi_trd_lim_e'][0,:]
npi_trd_lim_l = mat_contents['npi_trd_lim_l'][0,:]
npi_trd_rf = mat_contents['npi_trd_rf'][0,:]

nino34_trd_pi = mat_contents['nino34_trd_pi'][0,:]
nino34_trd_le = mat_contents['nino34_trd_le'][0,:]
nino34_trd_lim_e = mat_contents['nino34_trd_lim_e'][0,:]
nino34_trd_lim_l = mat_contents['nino34_trd_lim_l'][0,:]

xmin = -1.5e6; xmax = 1.5e6; ymin = -2; ymax = 2
nbin = 30
xbin = np.linspace(xmin,xmax,nbin+1); ybin = np.linspace(ymin,ymax,nbin+1)
H, xedges, yedges = np.histogram2d(vp_trd_pi*10, npi_trd_pi*10, bins = [xbin, ybin], weights = nino34_trd_pi*10)
H_counts, xedges, yedges = np.histogram2d(vp_trd_pi*10, npi_trd_pi*10, bins = [xbin, ybin])
Hplot = np.empty_like(H) * np.nan
Hplot[H_counts>0] = H[H_counts>0]/H_counts[H_counts>0]
XX = 0.5 * (xedges[0:nbin] + xedges[1:nbin+1])
YY = 0.5 * (yedges[0:nbin] + yedges[1:nbin+1])
el = [0, 9, 15, 32, 51, 79, 81, 83, 86, 96]
la = [8, 26, 31, 36, 41, 67, 75, 82, 90, 94]
fig = plt.figure()
fig.set_size_inches(8.5,6)
plt.ticklabel_format(axis='x', style='sci', scilimits=(5,0))
ax = plt.subplot(111)
cs = ax.pcolormesh(XX, YY, Hplot.T,cmap = plt.cm.RdBu_r,vmax = 0.5, vmin = -0.5)

select_target = nino34_trd_pi[nino34_trd_pi>np.percentile(nino34_trd_pi,97.5)]
temp_pi = np.empty((len(select_target),3)) * np.nan
for i in range(len(select_target)):
    ind = np.where(nino34_trd_pi==select_target[i])[0]
    temp_pi[i,0] = vp_trd_pi[ind]
    temp_pi[i,1] = npi_trd_pi[ind]
    temp_pi[i,2] = nino34_trd_pi[ind]
x1 = np.mean(temp_pi[:,0])*10; y1 = np.mean(temp_pi[:,1])*10
plt.plot([np.min(temp_pi[:,0]*10), np.max(temp_pi[:,0]*10)], [np.mean(temp_pi[:,1]*10), np.mean(temp_pi[:,1]*10)],color = 'k',lw = 3)
plt.plot([np.mean(temp_pi[:,0]*10), np.mean(temp_pi[:,0]*10)], [np.min(temp_pi[:,1]*10), np.max(temp_pi[:,1]*10)],color = 'k',lw = 3)
cs1 = plt.scatter(np.mean(temp_pi[:,0])*10, np.mean(temp_pi[:,1])*10,c = np.mean(temp_pi[:,2])*10,cmap=plt.cm.RdBu_r,marker = 'o',\
            vmax = 0.5, vmin = -0.5, s = 200,label = 'El Niño-like (PiCtrl)',zorder = 101, edgecolor = 'k',linewidths = 2.5)
select_target = nino34_trd_pi[nino34_trd_pi<np.percentile(nino34_trd_pi,2.5)]
temp_pi = np.empty((len(select_target),3)) * np.nan
for i in range(len(select_target)):
    ind = np.where(nino34_trd_pi==select_target[i])[0]
    temp_pi[i,0] = vp_trd_pi[ind]
    temp_pi[i,1] = npi_trd_pi[ind]
    temp_pi[i,2] = nino34_trd_pi[ind]
plt.plot([np.min(temp_pi[:,0]*10), np.max(temp_pi[:,0]*10)], [np.mean(temp_pi[:,1]*10), np.mean(temp_pi[:,1]*10)],color = 'k',lw = 3)
plt.plot([np.mean(temp_pi[:,0]*10), np.mean(temp_pi[:,0]*10)], [np.min(temp_pi[:,1]*10), np.max(temp_pi[:,1]*10)],color = 'k',lw = 3)
cs1 = plt.scatter(np.mean(temp_pi[:,0])*10, np.mean(temp_pi[:,1])*10,c = np.mean(temp_pi[:,2])*10,cmap=plt.cm.RdBu_r,marker = 'o',\
            vmax = 0.5, vmin = -0.5, s = 200,label = 'La Niña-like (PiCtrl)',zorder = 101, edgecolor = 'k',linewidths = 2.5)

#confidence_ellipse(vp_trd_le*10, npi_trd*10, ax, n_std = 2.95, edgecolor='red')
plt.plot([np.min(vp_trd_le[el]*10), np.max(vp_trd_le[el]*10)], [np.mean(npi_trd_le[el]*10), np.mean(npi_trd_le[el]*10)],color = 'k',lw = 3)
plt.plot([np.mean(vp_trd_le[el]*10), np.mean(vp_trd_le[el]*10)], [np.min(npi_trd_le[el]*10), np.max(npi_trd_le[el]*10)],color = 'k',lw = 3)
plt.plot([np.min(vp_trd_le[la]*10), np.max(vp_trd_le[la]*10)], [np.mean(npi_trd_le[la]*10), np.mean(npi_trd_le[la]*10)],color = 'k',lw = 3)
plt.plot([np.mean(vp_trd_le[la]*10), np.mean(vp_trd_le[la]*10)], [np.min(npi_trd_le[la]*10), np.max(npi_trd_le[la]*10)],color = 'k',lw = 3)
cs1 = plt.scatter(np.mean(vp_trd_le[el])*10, np.mean(npi_trd_le[el])*10,c = np.mean(nino34_trd_le[el])*10,cmap=plt.cm.RdBu_r,marker = 's',\
            vmax = 0.5, vmin = -0.5, s = 200,label = 'El Niño-like (Hist)',zorder = 101, edgecolor = 'k',linewidths = 2.5)
cs1 = plt.scatter(np.mean(vp_trd_le[la])*10, np.mean(npi_trd_le[la])*10,c = np.mean(nino34_trd_le[la])*10,cmap=plt.cm.RdBu_r,marker = 's',\
            vmax = 0.5, vmin = -0.5, s = 200, label = 'La Niña-like (Hist)',zorder = 101, edgecolor = 'k',linewidths = 2.5)

plt.plot([np.min(vp_trd_lim_l*10), np.max(vp_trd_lim_l*10)], [np.mean(npi_trd_lim_l*10), np.mean(npi_trd_lim_l*10)],color = 'k',lw = 3)
plt.plot([np.mean(vp_trd_lim_l*10), np.mean(vp_trd_lim_l*10)], [np.min(npi_trd_lim_l*10), np.max(npi_trd_lim_l*10)],color = 'k',lw = 3)
plt.plot([np.min(vp_trd_lim_e*10), np.max(vp_trd_lim_e*10)], [np.mean(npi_trd_lim_e*10), np.mean(npi_trd_lim_e*10)],color = 'k',lw = 3)
plt.plot([np.mean(vp_trd_lim_e)*10, np.mean(vp_trd_lim_e)*10], [np.min(npi_trd_lim_e*10), np.max(npi_trd_lim_e*10)],color = 'k',lw = 3)
cs1 = plt.scatter(np.mean(vp_trd_lim_e)*10, np.mean(npi_trd_lim_e)*10,c = nino34_trd_lim_e*10,cmap=plt.cm.RdBu_r,marker = 'd',\
            vmax = 0.5, vmin = -0.5, s = 250, label = 'El Niño-like (CS-LIM)',zorder = 201, edgecolor = 'k',linewidths = 2.5)
cs1 = plt.scatter(np.mean(vp_trd_lim_l)*10, np.mean(npi_trd_lim_l)*10,c = nino34_trd_lim_l*10,cmap=plt.cm.RdBu_r,marker = 'd',\
            vmax = 0.5, vmin = -0.5, s = 250, label = 'La Niña-like (CS-LIM)',zorder = 201, edgecolor = 'k',linewidths = 2.5)

plt.plot([np.mean(vp_trd_lim_e)*10, np.mean(vp_trd_lim_l)*10],\
         [np.mean(npi_trd_lim_e)*10,np.mean(npi_trd_lim_l)*10],linewidth = 2,color = 'k',linestyle = '--',zorder = 100)

plt.plot([np.mean(vp_trd_le[el])*10, np.mean(vp_trd_le[la])*10],\
         [np.mean(npi_trd_le[el])*10,np.mean(npi_trd_le[la])*10],linewidth = 2,color = 'k',linestyle = '--',zorder = 100)
plt.plot([x1, np.mean(temp_pi[:,0])*10],[y1, np.mean(temp_pi[:,1])*10],linewidth = 2, color = 'k',linestyle = '--',zorder = 100)

plt.xlabel('Equatorial 200hPa VPG Trend (m$^2$s$^{-1}$ per decade)',fontsize = 16)
plt.xticks(np.arange(-1800000,1500001,300000),fontsize = 14);
plt.yticks(np.arange(-2,2.1,0.50),fontsize = 14)
plt.ylabel('NPI trend (hPa per decade)',fontsize = 18)
plt.plot([0,0],[-2,2],color = 'lightgray')
plt.plot([-10000000,10000000],[0,0],color = 'lightgray')
plt.axis([-1500000,1500000,-2,2])
cbr = fig.colorbar(cs1,fraction=0.05,orientation = 'vertical', ticks = np.arange(-0.5,0.5+0.1,0.1))
plt.plot([np.min(vp_trd_rf*10), np.max(vp_trd_rf*10)], [np.mean(npi_trd_rf*10), np.mean(npi_trd_rf*10)],color = 'k',lw = 3)
plt.plot([np.mean(vp_trd_rf)*10, np.mean(vp_trd_rf)*10], [np.min(npi_trd_rf*10), np.max(npi_trd_rf*10)],color = 'k',lw = 3)
plt.scatter(np.mean(vp_trd_rf)*10, np.mean(npi_trd_rf)*10,facecolor = "white",edgecolor = 'k',marker = '^',\
            s = 200,label = 'Radiatively Forced', zorder = 80,linewidths = 2.5)
cbr.set_label(size=16, \
                  label='Niño 3.4 ${SST}$ trend (K per decade)')
cbr.ax.tick_params(labelsize=14)

### plot for legend
legend1 = plt.scatter(0,100,facecolor = "white",marker = 'o',\
            s = 200,zorder = 101, edgecolor = 'k',linewidths = 2.5)
legend2 = plt.scatter(0,100,facecolor = "white",edgecolor = 'k',marker = 's',\
            s = 200, zorder = 80,linewidths = 2.5)
legend3 = plt.scatter(0,100,facecolor = "white",edgecolor = 'k',marker = 'd',\
            s = 250, zorder = 80,linewidths = 2.5)
legend4 = plt.scatter(0,100,facecolor = "white",edgecolor = 'k',marker = '^',\
            s = 200, zorder = 80,linewidths = 2.5)
plt.legend([legend1, legend2, legend3, legend4],['${PiControl}$ ${(PiCtrl)}$','${Historical}$ ${(Hist)}$','TOGA (LIM)','${RF}$ ${only}$'],fontsize = 11, ncol = 4, loc = 4)
plt.savefig('figure3_nino34_200hPavpg_npi_meshgrid_mean_error_edge_k.svg',dpi=300)
plt.show()

fname = '/glade/u/home/kuoyan/code_analysis/code_YNKuo/kuo24_natgeo_figure3_reg_pi_hist_vpg200_npi_boostrapping.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fname)
mat_contents = sio.loadmat(mat_fname)
reg_pi = mat_contents['reg_pi'][0,:]
reg_hist = mat_contents['reg_hist'][0,:]

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(8,4)
plt.hist(reg_pi*10e6, 40,range=(-5,10),histtype='step',color = 'gray',linewidth=2,label = 'PiControl')
plt.hist(reg_hist*10e6, 40,range=(-5,10), histtype='step',color = 'r',linewidth=2,label = 'Historical')
plt.axis([-6, 10, 0, 1500])
plt.title('NPI trends explained by equatorial 200hPa VPG trends',fontsize = 18)
plt.xlabel('Regression Coefficient (hPa per $10^6$m$^2$s$^{-1}$)',fontsize = 18)
plt.ylabel('Count',fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.savefig('figure3_npi_200hPavpg_reg.svg',dpi=300)
plt.show()

fname = '/glade/u/home/kuoyan/code_analysis/code_YNKuo/kuo24_natgeo_figure3_reg_pi_hist_nino34_vpg200_boostrapping.mat'
data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = pjoin(data_dir,\
                  fname)
mat_contents = sio.loadmat(mat_fname)
reg_pi = mat_contents['reg_pi'][0,:]
reg_hist = mat_contents['reg_hist'][0,:]

fig = plt.figure(facecolor = 'white')
fig.set_size_inches(8,4)
plt.hist(reg_pi*10e-6, 40, range=(-22,-4), histtype='step',color = 'gray',linewidth=2,label = 'PiCtrl')
plt.hist(reg_hist*10e-6, 40,range=(-22,-4), histtype='step',color = 'r',linewidth=2,label = 'Hist')
plt.axis([-22, -2.5, 0, 1100])
plt.title('Equatorial 200hPa VPG trends explained by Niño 3.4 trends',fontsize = 18)
plt.xlabel('Regression Coefficient ($10^{-6}$m$^2$s$^{-1}$ per K)',fontsize = 18)
plt.ylabel('count',fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(np.arange(0,1001,250), fontsize = 16)
plt.legend(fontsize = 18)
plt.savefig('figure3_200hPavpg_nino34_reg.svg',dpi=300)
plt.show()
