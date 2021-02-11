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
import palettable.scientific.sequential as ssss
import palettable.scientific.diverging as dddd
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
directorydata = '/Users/zlabe/Documents/Research/ExtremeEvents/Data/Class-STDDEV/'
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v2_STD-RMENS/CLASS/AllModels/'

###############################
###############################
land_only = True
rm_ensemble_mean = True
rm_standard_dev = True
###############################
###############################
num_of_class = 2
typemodel = 'train'
window = 5
###############################
###############################
modellabels = [r'LENS',r'MPI',r'MK3.6',r'EC-EARTH']
modelGCMs = ['lens','MPI','CSIRO_MK3.6','KNMI_ecearth']
variq = 'T2M'
###############################
###############################
season = 'annual'
reg_name = 'SMILEGlobe'
years = np.arange(1920,2099+1,1)
yearq = 30
###############################
###############################

### Read in data
def readData(window,typemodel,variq,simuqq,land_only,reg_name):
    """
    Read in LRP maps
    """
    
    data = Dataset(directorydata + 'LRP_Maps-STDDEV%syrs_%s_Annual_%s_%s_land_only-%s_%s.nc' % (window,typemodel,variq,simuqq,land_only,reg_name))
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lrp = data.variables['LRP'][:]
    data.close()
    
    return lrp,lat,lon

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

### Call data for each model
modeldata = []
for i in range(len(modelGCMs)):
    simuqq = modelGCMs[i]
    lrpq,lat1,lon1 = readData(window,typemodel,variq,simuqq,land_only,reg_name)
    
    lrpmean = np.nanmean(lrpq,axis=0)
    modeldata.append(lrpmean)
lrpall = np.asarray(modeldata,dtype=object)

### Composite data
diff = comp(lrpall,yearq)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(-0.2,0.201,0.0025)
barlim = np.round(np.arange(-0.2,0.21,0.1),2)
cmap = dddd.Berlin_12.mpl_colormap
label = r'\textbf{LRP Difference (%s)}' % yearq

fig = plt.figure()
for r in range(len(modellabels)):
    var = diff[r]
    
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
    
    if land_only == True:
        m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=True,zorder=5)

    ax1.annotate(r'\textbf{%s}' % modellabels[r],xy=(0,0),xytext=(0.82,0.95),
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

plt.savefig(directoryfigure + 'Composites_differenceInLRP_%s_4models_%s_land-only%s.png' % (yearq,reg_name,land_only),dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(-0.2,0.201,0.0025)
barlim = np.round(np.arange(-0.2,0.21,0.1),2)
cmap = dddd.Berlin_12.mpl_colormap
label = r'\textbf{LRP Difference (%s)}' % yearq

fig = plt.figure(figsize=(6,6))
for r in range(len(modellabels)):
    var = diff[r]
    
    ax1 = plt.subplot(2,2,r+1)
    m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                            area_thresh=10000.)
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
    
    if land_only == True:
        m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',lakes=True,zorder=5)

    ax1.annotate(r'\textbf{%s}' % modellabels[r],xy=(0,0),xytext=(0.82,0.93),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=323,ha='center',va='center')
    
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
plt.subplots_adjust(top=0.85,wspace=-0.35,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'Composites_differenceInLRP_%s_4models_ortho_%s_land-only%s.png' % (yearq,reg_name,land_only),dpi=300)
                        