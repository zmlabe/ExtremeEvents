"""
Created on Thu Aug 13 08:20:11 2020

@author: zlabe
"""

"""
Script plots composites for large ensemble data (monthly) using 
several variables

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
import calc_Stats as dSS

### Set preliminaries
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v1/Composites/LENS/'
reg_name = 'Globe'
dataset = 'lens'
rm_ensemble_mean = False
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
        
    ### Calculate ensemble mean
    meandata = np.nanmean(data,axis=0)
    del data #save storage
    
    ### Composite over selected period (x2)
    if monthlychoice == 'DJF':
        years = np.arange(meandata.shape[0]) + 1921
    else:
        years = np.arange(meandata.shape[0]) + 1920
        
    length = years.shape[0]//2
    historical = meandata[:length,:,:]
    future = meandata[length:,:,:]
    
    ### Average over composites for plotting
    historicalm = np.nanmean(historical,axis=0)
    futurem = np.nanmean(future,axis=0)
    
    ### Calculate significance
    pruns = UT.calc_FDR_ttest(future[:,:,:],historical[:,:,:],0.05) #FDR
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Begin plots!!!
    fig = plt.figure()
    
    ### Select graphing preliminaries
    if rm_ensemble_mean == True:
        if variq[i] == 'T2M':
            label = r'\textbf{T2M [$\bf{^{\circ}}$C]}'
            cmap = cm.cubehelix3_16_r.mpl_colormap 
        elif variq[i] == 'SLP':
            label = r'\textbf{SLP [hPa]}'
            cmap = cm.cubehelix3_16_r.mpl_colormap 
        elif variq[i] == 'U700':
            label = r'\textbf{U700 [m/s]}'
            cmap = cm.cubehelix3_16_r.mpl_colormap 
        limit = np.linspace(futurem.min(),futurem.max(),300)
        barlim = np.linspace(futurem.min(),futurem.max(),2)
    elif rm_ensemble_mean == False:
        if variq[i] == 'T2M':
            label = r'\textbf{T2M [$\bf{^{\circ}}$C]}'
            cmap = plt.cm.twilight
            limit = np.arange(-35,35.1,0.5)
            barlim = np.arange(-35,36,35)
        elif variq[i] == 'SLP':
            label = r'\textbf{SLP [hPa]}'
            cmap = plt.cm.cividis
            limit = np.arange(985,1035.1,2)
            barlim = np.arange(985,1036,10)
        elif variq[i] == 'U700':
            label = r'\textbf{U700 [m/s]}'
            cmap = cm.classic_16.mpl_colormap 
            limit = np.arange(-10,20.1,0.5)
            barlim = np.arange(-10,21,5)
    
    ###########################################################################
    ax = plt.subplot(211)
    m = Basemap(projection='moll',lon_0=0,resolution='l')   
    var, lons_cyclic = addcyclic(historicalm, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.45)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='both')
              
    m.drawcoastlines(color='dimgray',linewidth=0.7)       
    cs.set_cmap(cmap) 
    plt.text(0,0,r'\textbf{1921-2010}',color='dimgrey',fontsize=10)
    
    ###########################################################################
    ax = plt.subplot(212)
    m = Basemap(projection='moll',lon_0=0,resolution='l')   
    var, lons_cyclic = addcyclic(futurem, lons)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='white',color='dimgray',
                      linewidth=0.45)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='both')
              
    m.drawcoastlines(color='dimgray',linewidth=0.7)       
    cs.set_cmap(cmap) 
    plt.text(0,0,r'\textbf{2011-2100}',color='dimgrey',fontsize=10)
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    cbar_ax = fig.add_axes([0.293,0.1,0.4,0.03])             
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='both',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(label,fontsize=14,color='k',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=6)
    cbar.outline.set_edgecolor('dimgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16,wspace=0,hspace=0.01)
    
    if rm_ensemble_mean == True:
        plt.savefig(directoryfigure + 'Composites_LENS_%s_%s.png' \
                    % (monthlychoice,variq[i]),dpi=300)
    elif rm_ensemble_mean == False:
        plt.savefig(directoryfigure + 'Composites_LENS_%s_%s_ORIGINAL.png' \
            % (monthlychoice,variq[i]),dpi=300)
