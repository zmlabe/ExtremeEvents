"""
Create composites of the raw data after removing the ensemble mean and then
calculating a rolling standard deviation

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 11 February 2021
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries ########
directorydata = '/Users/zlabe/Documents/Research/ExtremeEvents/Data/Composites-STD/'
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v2_STD-RMENS/Composites-RawData/'

###############################
###############################
land_only = False
rm_ensemble_mean = True
rm_standard_dev = True
###############################
###############################
num_of_class = 2
window = 5
###############################
###############################
modelgcm = [r'LENS',r'MPI',r'MK3.6',r'EC-EARTH']
variq = 'T2M'
###############################
###############################
season = 'annual'
reg_name = 'SMILEGlobe'
years = np.arange(1920,2099+1,1)
###############################
###############################

### Read in data for meanSTD
typemodel = 'meanSTD'
data = Dataset(directorydata + 'Composites_RawData_Maps-STDDEV%syrs_%s_%s_%s_land_only-%s_%s.nc' % (window,typemodel,season,variq,land_only,reg_name))
lat1 = data.variables['lat'][:]
lon1 = data.variables['lon'][:]
meanstd = data.variables['stdev'][:]
data.close()

### Read in data for maxSTD
typemodel = 'maxSTD'
data = Dataset(directorydata + 'Composites_RawData_Maps-STDDEV%syrs_%s_%s_%s_land_only-%s_%s.nc' % (window,typemodel,season,variq,land_only,reg_name))
maxstd = data.variables['stdev'][:]
data.close()

### Read in data for minSTD
typemodel = 'minSTD'
data = Dataset(directorydata + 'Composites_RawData_Maps-STDDEV%syrs_%s_%s_%s_land_only-%s_%s.nc' % (window,typemodel,season,variq,land_only,reg_name))
minstd = data.variables['stdev'][:]
data.close()

### Read in data for spread
typemodel = 'spread'
data = Dataset(directorydata + 'Composites_RawData_Maps-STDDEV%syrs_%s_%s_%s_land_only-%s_%s.nc' % (window,typemodel,season,variq,land_only,reg_name))
spreadstd = data.variables['stdev'][:]
data.close()

### Function for calculating mean composites
def comp(data,yearcomp):
    """
    Calculate composites for first yearcomp and last yearcomp and take
    the difference
    """
    
    ### Take periods
    early = data[:,:yearcomp,:,:]
    late = data[:,-yearcomp:,:,:]
    
    ### Average periods
    earlym = np.nanmean(early,axis=1)
    latem = np.nanmean(late,axis=1)
    
    ### Difference
    diff = latem - earlym
    
    return diff

### Return information
yearq = 10
diff_mean = comp(meanstd,yearq) 
diff_max = comp(maxstd,yearq) 
diff_min = comp(minstd,yearq) 
diff_spread = comp(spreadstd,yearq) 

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(-0.5,0.51,0.01)
barlim = np.round(np.arange(-0.5,0.6,0.25),2)
cmap = cmocean.cm.balance
label = r'\textbf{Difference (%s)}' % yearq

fig = plt.figure()
for r in range(len(modelgcm)):
    var = diff_mean[r]
    
    ax1 = plt.subplot(2,2,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
    
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='both')
            
    cs.set_cmap(cmap) 
    # if any([r==0,r==2,r==4]):
    #     ax1.annotate(r'\textbf{%s}' % datasets[r],xy=(0,0),xytext=(-0.07,0.5),
    #                   textcoords='axes fraction',color='k',fontsize=10,
    #                   rotation=90,ha='center',va='center')
    # if any([r==0,r==1]):
    #     ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.08),
    #                   textcoords='axes fraction',color='dimgrey',fontsize=11,
    #                   rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{%s}' % modelgcm[r],xy=(0,0),xytext=(0.82,0.95),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=340,ha='center',va='center')
    
###############################################################################
cbar_ax = fig.add_axes([0.32,0.09,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'Composites_differenceInRawData_meanSTD_%s.png' % yearq,dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(-1,1.01,0.01)
barlim = np.round(np.arange(-1,2,1),2)
cmap = cmocean.cm.balance
label = r'\textbf{Spread Difference (%s)}' % yearq

fig = plt.figure()
for r in range(len(modelgcm)):
    var = diff_spread[r]
    
    ax1 = plt.subplot(2,2,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
    
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs = m.contourf(x,y,var,limit,extend='both')
            
    cs.set_cmap(cmap) 
    # if any([r==0,r==2,r==4]):
    #     ax1.annotate(r'\textbf{%s}' % datasets[r],xy=(0,0),xytext=(-0.07,0.5),
    #                   textcoords='axes fraction',color='k',fontsize=10,
    #                   rotation=90,ha='center',va='center')
    # if any([r==0,r==1]):
    #     ax1.annotate(r'\textbf{%s}' % timeq[r],xy=(0,0),xytext=(0.5,1.08),
    #                   textcoords='axes fraction',color='dimgrey',fontsize=11,
    #                   rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{%s}' % modelgcm[r],xy=(0,0),xytext=(0.82,0.95),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=340,ha='center',va='center')
    
###############################################################################
cbar_ax = fig.add_axes([0.32,0.09,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'Composites_differenceInRawData_spreadSTD_%s.png' % yearq,dpi=300)
            