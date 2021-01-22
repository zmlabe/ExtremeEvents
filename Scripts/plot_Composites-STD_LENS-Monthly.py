"""
Script plots standard deviation composites for large ensemble data (monthly) 
using several variables

Author    : Zachary M. Labe
Date      : 13 August 2020
"""

### Import modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import calc_Utilities as UT
import calc_dataFunctions as df
import cmocean
import calc_Stats as dSS

### Set preliminaries
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v1/Composites/LENS/'
reg_name = 'Globe'
dataset = 'lens'
rm_ensemble_mean = True
variq = ['T2M']
monthlychoice = 'annual'

def read_primary_dataset(variq,dataset,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons

for i in range(len(variq)):
    ### Read in data for selected region                
    lat_bounds,lon_bounds = UT.regions(reg_name)
    dataall,lats,lons = read_primary_dataset(variq[i],dataset,
                                              lat_bounds,lon_bounds)
    
    ### Remove ensemble mean
    if rm_ensemble_mean == True:
        data= dSS.remove_ensemble_mean(dataall)
        print('*Removed ensemble mean*')
    elif rm_ensemble_mean ==  False:
        data = dataall
    
    ### Composite over selected period (x2)
    if monthlychoice == 'DJF':
        years = np.arange(dataall.shape[1]) + 1921
    else:
        years = np.arange(dataall.shape[1]) + 1920
        
    length = years.shape[0]//2
    historicalall = dataall[:,:length,:,:]
    futureall = dataall[:,length:,:,:]
    
    ### Calculate standard deviation
    historical = np.nanstd(historicalall,axis=1)
    future = np.nanstd(futureall,axis=1)
    
    ### Average over composites for plotting
    historicalm = np.nanmean(historical,axis=0)
    futurem = np.nanmean(future,axis=0)
    
    ### Calculate significance
    pruns = UT.calc_FDR_ttest(future[:,:,:],historical[:,:,:],0.05) #FDR
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Begin plots!!!
    fig = plt.figure(figsize=(4,8))
    
    ### Select graphing preliminaries
    if variq[i] == 'T2M':
        label = r'\textbf{$\bf{\sigma}$T2M [$\bf{^{\circ}}$C]}'
        cmap = cm.classic_16.mpl_colormap 
        limit = np.arange(0,8.1,0.25)
        barlim = np.arange(0,9,2)
        limitd = np.arange(-2,2.1,0.1)
        barlimd = np.round(np.arange(-2,3,2),2)
    elif variq[i] == 'SLP':
        label = r'\textbf{$\bf{\sigma}$SLP [hPa]}'
        cmap = cm.classic_16.mpl_colormap 
        limit = np.arange(0,10.25,0.25)
        barlim = np.arange(0,11,5)
        limitd = np.arange(-1,1.001,0.001)
        barlimd = np.round(np.arange(-1,1.5,0.5),2)
    elif variq[i] == 'U700':
        label = r'\textbf{$\bf{\sigma}$U700 [m/s]}'
        cmap = cm.classic_16.mpl_colormap 
        limit = np.arange(0,5.1,0.1)
        barlim = np.arange(0,6,5)
        limitd = np.arange(-1,1.001,0.001)
        barlimd = np.round(np.arange(-1,1.5,0.5),2)
    
    ###########################################################################
    ax = plt.subplot(311)
    m = Basemap(projection='moll',lon_0=0,resolution='l')   
    var, lons_cyclic = addcyclic(historicalm, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.45)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='max')
              
    m.drawcoastlines(color='dimgray',linewidth=0.7) 
      
    cs.set_cmap(cmap) 
    cbar = plt.colorbar(cs,drawedges=False,orientation='vertical',
                    pad = 0.07,fraction=0.035)
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(labelsize=5,pad=5) 
    ticklabs = cbar.ax.get_xticklabels()
    cbar.ax.set_xticklabels(ticklabs,ha='center',color='dimgrey')
    cbar.ax.tick_params(axis='y', size=.001)
    cbar.outline.set_edgecolor('dimgrey')
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(label,labelpad=5,color='dimgrey',
                    fontsize=8)
    
    plt.text(0,0,r'\textbf{1921-2010}',color='dimgrey',fontsize=6.5)
    
    ###########################################################################
    ax = plt.subplot(312)
    m = Basemap(projection='moll',lon_0=0,resolution='l')   
    var, lons_cyclic = addcyclic(futurem, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.45)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='max')
              
    m.drawcoastlines(color='dimgray',linewidth=0.7)   
    
    cs.set_cmap(cmap) 
    cbar = plt.colorbar(cs,drawedges=False,orientation='vertical',
                    pad = 0.07,fraction=0.035)
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(labelsize=5,pad=5) 
    ticklabs = cbar.ax.get_xticklabels()
    cbar.ax.set_xticklabels(ticklabs,ha='center',color='dimgrey')
    cbar.ax.tick_params(axis='y', size=.001)
    cbar.outline.set_edgecolor('dimgrey')
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(label,labelpad=5,color='dimgrey',
                    fontsize=8)
    
    plt.text(0,0,r'\textbf{2011-2100}',color='dimgrey',fontsize=6.5)
    
    ###########################################################################
    ax = plt.subplot(313)
    m = Basemap(projection='moll',lon_0=0,resolution='l')   
    var, lons_cyclic = addcyclic(futurem-historicalm, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.45)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limitd,extend='both')
              
    m.drawcoastlines(color='dimgray',linewidth=0.7)       
    cs.set_cmap(cmocean.cm.balance) 
    
    cbar = plt.colorbar(cs,drawedges=False,orientation='vertical',
                    pad = 0.07,fraction=0.035)
    cbar.set_ticks(barlimd)
    cbar.set_ticklabels(list(map(str,barlimd)))
    cbar.ax.tick_params(labelsize=5,pad=5) 
    ticklabs = cbar.ax.get_xticklabels()
    cbar.ax.set_xticklabels(ticklabs,ha='center',color='dimgrey')
    cbar.ax.tick_params(axis='y', size=.001)
    cbar.outline.set_edgecolor('dimgrey')
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(r'\textbf{Difference}',labelpad=5,color='dimgrey',
                    fontsize=8)
    
    plt.text(0,0,r'\textbf{Future--Historical}',color='dimgrey',fontsize=6.5)
    
    ###########################################################################
    ###########################################################################
    ###########################################################################   
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.16,wspace=0,hspace=0.01)
    
    if rm_ensemble_mean == True:
        plt.savefig(directoryfigure + 'Composites-STD_LENS_%s_%s.png' \
                    % (monthlychoice,variq[i]),dpi=300)
    elif rm_ensemble_mean == False:
        plt.savefig(directoryfigure + 'Composites-STD_LENS_%s_%s_ORIGINAL.png' \
            % (monthlychoice,variq[i]),dpi=300)
