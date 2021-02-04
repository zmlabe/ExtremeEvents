"""
Script plots LRP maps for correct and incorrect cases

Author    : Zachary M. Labe
Date      : 4 February 2021
"""

### Import modules
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import calc_Utilities as UT
import calc_dataFunctions as df
import palettable.cubehelix as cm
import palettable.scientific.sequential as ssss
import palettable.scientific.diverging as dddd
import calc_Stats as dSS
import cmocean
from sklearn.metrics import accuracy_score
from netCDF4 import Dataset

### Set preliminaries
dataset = 'LENS'
dataset_obs = '20CRv3'
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v2_STD-RMENS/CLASS/%s/' % dataset
directorydata = '/Users/zlabe/Documents/Research/ExtremeEvents/Data/Class-STDDEV/'
reg_name = 'GlobeNoSP'
rm_ensemble_mean = True
variq = 'T2M'
monthlychoice = 'Annual'
land_only = True
ocean_only = False
rm_merid_mean = False
rm_annual_mean = False
rm_ensemble_mean = True
ensnum = 40
num_of_class = 2
iterations = [100]
nomonths = 12
window = 5 
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
labely = [r'1920-2009',r'1920-2009',r'1920-2009',
          r'2009-2099',r'2009-2099',r'2009-2099']
headers = [r'\textbf{GOOD}',r'\textbf{BAD}',r'\textbf{DIFF}']

### Special cases
if reg_name == 'Globe':
    if dataset == 'MPI':
        reg_name = 'MPIGlobe'

### Create sample class labels for 1920-2099
if num_of_class == 3:
    yearlabels = np.arange(1920+window,2099+1,1)
    lengthlabels = yearlabels.shape[0]//num_of_class
    array1 = np.asarray([0]*lengthlabels)
    array2 = np.asarray([1]*lengthlabels)
    array3 = np.asarray([2]*lengthlabels)
    classesl = np.concatenate([array1,array2,array3],axis=None)
elif num_of_class == 2:
    yearlabels = np.arange(1920+window,2099+1,1)
    lengthlabels = yearlabels.shape[0]//num_of_class
    array1 = np.asarray([0]*lengthlabels)
    array2 = np.asarray([1]*(yearlabels.shape[0]-lengthlabels))
    classesl = np.concatenate([array1,array2],axis=None)
    
### Read in prediction data
trainq = np.genfromtxt(directorydata + 'training_STDDEVCentury%syrs_%s_%s_%s_%s_iterations%s_land_only-%s_v1.txt' % (window,variq,monthlychoice,reg_name,dataset,iterations[0],land_only))
testq = np.genfromtxt(directorydata + 'testing_STDDEVCentury%syrs_%s_%s_%s_%s_iterations%s_land_only-%s_v1.txt' % (window,variq,monthlychoice,reg_name,dataset,iterations[0],land_only))
obsq = np.genfromtxt(directorydata + 'obsout_STDDEVCentury%syrs_%s_%s_%s_%s-%s_iterations%s_land_only-%s_v1.txt' % (window,variq,monthlychoice,reg_name,dataset_obs,dataset,iterations[0],land_only))

### Reshape
train = np.reshape(trainq,(trainq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],trainq.shape[1]))
test = np.reshape(testq,(testq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],testq.shape[1]))
obs = obsq

### Read in LRP training data
data1 = Dataset(directorydata + 'LRP_Maps-STDDEV%syrs_train_%s_%s_%s_land_only-%s_%s.nc' % (window,monthlychoice,variq,dataset,land_only,reg_name))
lat1 = data1.variables['lat'][:]
lon1 = data1.variables['lon'][:]
lrptrain = data1.variables['LRP'][:].reshape(train.shape[0],yearlabels.shape[0],lat1.shape[0]*lon1.shape[0])
data1.close()

### Read in LRP testing data
data2 = Dataset(directorydata + 'LRP_Maps-STDDEV%syrs_test_%s_%s_%s_land_only-%s_%s.nc' % (window,monthlychoice,variq,dataset,land_only,reg_name))
lrptest = data2.variables['LRP'][:].reshape(test.shape[0],yearlabels.shape[0],lat1.shape[0]*lon1.shape[0])
lat1 = data1.variables['lat'][:]
lon1 = data1.variables['lon'][:]
data2.close()

### Meshgrid
lon2,lat2 = np.meshgrid(lon1,lat1)

###############################################################################
###############################################################################
###############################################################################
### Calculate accuracy
def truelabel(data):
    """
    Calculate argmax
    """
    maxindexdata = np.empty((data.shape[0],data.shape[1]))
    for i in range(data.shape[0]):
        maxindexdata[i,:] = np.argmax(data[i,:,:],axis=1)    
    meanmaxindexdata= np.nanmean(maxindexdata,axis=0)
    
    return maxindexdata,meanmaxindexdata

def accuracyTotalTime(data_pred,data_true):
    """
    Compute accuracy for the entire time series
    """
    accdata_pred = np.empty((data_pred.shape[0]))
    for i in range(data_pred.shape[0]):
        accdata_pred[i] = accuracy_score(data_true,data_pred[i,:])
        
    return accdata_pred

def accuracyTPeriodTime(data_pred,data_true):
    """
    Compute accuracy for the three periods
    """
    time = data_true.shape[0]
    period = int(time//2)+1
    
    accdata_pred = np.empty((data_pred.shape[0],2))
    for i in range(data_pred.shape[0]):
        for save,j in enumerate(range(0,time,period)):
            accdata_pred[i,save] = accuracy_score(data_true[j:j+period],
                                                data_pred[i,j:j+period])
        
    return accdata_pred

### Calculate statistics
indextrain,meanindextrain = truelabel(train)
indextest,meanindextest = truelabel(test)

acctrain = accuracyTotalTime(indextrain,classesl)
acctest = accuracyTotalTime(indextest,classesl)

periodtrain = accuracyTPeriodTime(indextrain,classesl)
periodtest = accuracyTPeriodTime(indextest,classesl)

### Save good lrp maps
def lrpType1(indexdata,lrpdata,classesl,lat1,lon1):
    lrpdata_good = []
    lrpdata_bad = []
    for i in range(indexdata.shape[0]):
        for j in range(int(indexdata.shape[1]-indexdata.shape[1]/2)):
            if indexdata[i,j] == classesl[j]:
                lrpdata_good.append(lrpdata[i,j,:])
            else:
                lrpdata_bad.append(lrpdata[i,j,:])
    lrpdata_goodmap = np.asarray(lrpdata_good).reshape(len(lrpdata_good),lat1.shape[0],lon1.shape[0])
    lrpdata_badmap = np.asarray(lrpdata_bad).reshape(len(lrpdata_bad),lat1.shape[0],lon1.shape[0])
    meangood_data = np.nanmean(lrpdata_goodmap,axis=0)
    meanbad_data = np.nanmean(lrpdata_badmap,axis=0)
    
    sizegood = len(lrpdata_goodmap)
    sizebad = len(lrpdata_badmap)
    
    return meangood_data,meanbad_data,sizegood,sizebad

def lrpType2(indexdata,lrpdata,classesl,lat1,lon1):
    lrpdata_good = []
    lrpdata_bad = []
    for i in range(indexdata.shape[0]):
        for j in range(int(indexdata.shape[1]-indexdata.shape[1]/2)):
            if indexdata[i,j] == classesl[int(indexdata.shape[1]/2)+j]:
                lrpdata_good.append(lrpdata[i,int(indexdata.shape[1]/2)+j,:])
            else:
                lrpdata_bad.append(lrpdata[i,j,:])
    lrpdata_goodmap = np.asarray(lrpdata_good).reshape(len(lrpdata_good),lat1.shape[0],lon1.shape[0])
    lrpdata_badmap = np.asarray(lrpdata_bad).reshape(len(lrpdata_bad),lat1.shape[0],lon1.shape[0])
    meangood_data = np.nanmean(lrpdata_goodmap,axis=0)
    meanbad_data = np.nanmean(lrpdata_badmap,axis=0)
    
    sizegood = len(lrpdata_goodmap)
    sizebad = len(lrpdata_badmap)
    
    return meangood_data,meanbad_data,sizegood,sizebad

goodtrain1,badtrain1,sizetraingood1,sizetrainbad1 = lrpType1(indextrain,lrptrain,classesl,lat1,lon1)
goodtest1,badtest1,sizetestgood1,sizetestbad1  = lrpType1(indextest,lrptest,classesl,lat1,lon1)
difftrain_1 = goodtrain1 - badtrain1
difftest_1 = goodtest1 - badtest1

goodtrain2,badtrain2,sizetraingood2,sizetrainbad2 = lrpType2(indextrain,lrptrain,classesl,lat1,lon1)
goodtest2,badtest2,sizetestgood2,sizetestbad2  = lrpType2(indextest,lrptest,classesl,lat1,lon1)
difftrain_2 = goodtrain2 - badtrain2
difftest_2 = goodtest2 - badtest2

trainplot = [goodtrain1,badtrain1,difftrain_1,
              goodtrain2,badtrain2,difftrain_2]
testplot = [goodtest1,badtest1,difftest_1,
              goodtest2,badtest2,difftest_2]

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(0,0.5001,0.005)
barlim = np.round(np.arange(0,0.6,0.1),2)
limitdiff = np.arange(-0.3,0.301,0.0025)
barlimdiff = np.round(np.arange(-0.3,0.31,0.15),2)
cmap = [cm.cubehelix2_16.mpl_colormap,cm.cubehelix2_16.mpl_colormap,dddd.Berlin_12.mpl_colormap,
          cm.cubehelix2_16.mpl_colormap,cm.cubehelix2_16.mpl_colormap,dddd.Berlin_12.mpl_colormap]
limits = [limit,limit,limitdiff,limit,limit,limitdiff]
barlimits = [barlim,barlimdiff]
label = r'\textbf{RELEVANCE}'
labeldiff = r'\textbf{LRP DIFFERENCE}'

fig = plt.figure(figsize=(9,4))
for r in range(len(trainplot)):
    var = trainplot[r]
    
    ax1 = plt.subplot(2,3,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    if any([r==0,r==1,r==3,r==4]):
        m.drawcoastlines(color='darkgrey',linewidth=0.3)
    elif any([r==2,r==5]):
        m.drawcoastlines(color='darkgrey',linewidth=0.3)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    if any([r==0,r==1,r==3,r==4]):
        cs1 = m.contourf(x,y,var,limits[r],extend='max')
        cs1.set_cmap(cmap[r]) 
    elif any([r==2,r==5]):
        cs2 = m.contourf(x,y,var,limits[r],extend='both')
        cs2.set_cmap(cmap[r]) 
            
    if any([r==0,r==3]):
        ax1.annotate(r'\textbf{%s}' % labely[r],xy=(0,0),xytext=(-0.07,0.5),
                      textcoords='axes fraction',color='k',fontsize=10,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1,r==2]):
        ax1.annotate(r'\textbf{%s}' % headers[r],xy=(0,0),xytext=(0.5,1.10),
                      textcoords='axes fraction',color='dimgrey',fontsize=20,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.2,0.09,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlimits[0])
cbar1.set_ticklabels(list(map(str,barlimits[0])))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
cbar_ax2 = fig.add_axes([0.675,0.09,0.3,0.03])                
cbar2 = fig.colorbar(cs2,cax=cbar_ax2,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar2.set_label(labeldiff,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar2.set_ticks(barlimits[1])
cbar2.set_ticklabels(list(map(str,barlimits[1])))
cbar2.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar2.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'training_differenceLRP_composites_%s_%s_%s.png' % (variq,dataset,reg_name),dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means testing
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

cmap = [cm.cubehelix2_16.mpl_colormap,cm.cubehelix2_16.mpl_colormap,dddd.Berlin_12.mpl_colormap,
          cm.cubehelix2_16.mpl_colormap,cm.cubehelix2_16.mpl_colormap,dddd.Berlin_12.mpl_colormap]
limits = [limit,limit,limitdiff,limit,limit,limitdiff]
barlimits = [barlim,barlimdiff]
label = r'\textbf{RELEVANCE}'
labeldiff = r'\textbf{LRP DIFFERENCE}'

fig = plt.figure(figsize=(9,4))
for r in range(len(testplot)):
    var = testplot[r]
    
    ax1 = plt.subplot(2,3,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    circle = m.drawmapboundary(fill_color='dimgrey')
    circle.set_clip_on(False) 
    if any([r==0,r==1,r==3,r==4]):
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
    elif any([r==2,r==5]):
        m.drawcoastlines(color='darkgrey',linewidth=0.27)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    if any([r==0,r==1,r==3,r==4]):
        cs1 = m.contourf(x,y,var,limits[r],extend='max')
        cs1.set_cmap(cmap[r]) 
    elif any([r==2,r==5]):
        cs2 = m.contourf(x,y,var,limits[r],extend='both')
        cs2.set_cmap(cmap[r]) 
            
    if any([r==0,r==3]):
        ax1.annotate(r'\textbf{%s}' % labely[r],xy=(0,0),xytext=(-0.07,0.5),
                      textcoords='axes fraction',color='k',fontsize=10,
                      rotation=90,ha='center',va='center')
    if any([r==0,r==1,r==2]):
        ax1.annotate(r'\textbf{%s}' % headers[r],xy=(0,0),xytext=(0.5,1.10),
                      textcoords='axes fraction',color='dimgrey',fontsize=20,
                      rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.2,0.09,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlimits[0])
cbar1.set_ticklabels(list(map(str,barlimits[0])))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

###############################################################################
cbar_ax2 = fig.add_axes([0.675,0.09,0.3,0.03])                
cbar2 = fig.colorbar(cs2,cax=cbar_ax2,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar2.set_label(labeldiff,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar2.set_ticks(barlimits[1])
cbar2.set_ticklabels(list(map(str,barlimits[1])))
cbar2.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar2.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'testing_differenceLRP_composites_%s_%s_%s.png' % (variq,dataset,reg_name),dpi=300)
