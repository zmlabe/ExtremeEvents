"""
Scipt plots histograms of data with mean removed over 4 time periods

Author    : Zachary M. Labe
Date      : 13 January 2021
"""

### Import modules
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import calc_Utilities as UT
import calc_dataFunctions as df
import palettable.wesanderson as ww
import calc_Stats as dSS

### Set preliminaries
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_NewANN_v1/LENS/'
directorydata = '/Users/zlabe/Documents/Research/ExtremeEvents/Data/'
reg_name = 'Globe'
dataset = 'lens'
dataset_obs = '20CRv3'
rm_ensemble_mean = True
variq = ['T2M']
seasons = ['annual']
land_only = False
ocean_only = False
rm_merid_mean = False
rm_annual_mean = False
rm_ensemble_mean = True
ensnum = 40
num_of_class = 2
iterations = 100

### Create sample class labels for 1920-2099
if num_of_class == 3:
    yearlabels = np.arange(1920,2099+1,1)
    lengthlabels = yearlabels.shape[0]//num_of_class
    array1 = np.asarray([0]*lengthlabels)
    array2 = np.asarray([1]*lengthlabels)
    array3 = np.asarray([2]*lengthlabels)
    classesl = np.concatenate([array1,array2,array3],axis=None)
elif num_of_class == 2:
    yearlabels = np.arange(1920,2099+1,1)
    lengthlabels = yearlabels.shape[0]//num_of_class
    array1 = np.asarray([0]*lengthlabels)
    array2 = np.asarray([1]*lengthlabels)
    classesl = np.concatenate([array1,array2],axis=None)
    
### Read in data
trainq = np.genfromtxt(directorydata + 'training_Century_%s_%s_%s_%s_iterations%s_v2-nomean.txt' % (variq[0],seasons[0],reg_name,dataset,iterations))
testq = np.genfromtxt(directorydata + 'testing_Century_%s_%s_%s_%s_iterations%s_v2-nomean.txt' % (variq[0],seasons[0],reg_name,dataset,iterations))
obsq = np.genfromtxt(directorydata + 'obsout_Century_%s_%s_%s_%s-%s_iterations%s_v2-nomean.txt' % (variq[0],seasons[0],reg_name,dataset_obs,dataset,iterations))

### Reshape
train = np.reshape(trainq,(trainq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],trainq.shape[1]))
test = np.reshape(testq,(testq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],testq.shape[1]))
obs = obsq

### Medians
meantrain = np.nanmedian(train,axis=0)
meantest = np.nanmedian(test,axis=0)

### Combination of data
total = np.append(train,test,axis=0)

###############################################################################
###############################################################################
###############################################################################    
### Create graph 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])
        
c2=ww.FantasticFox2_5.mpl_colormap
        
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey',
                labelsize=6)  
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)

for i in range(train.shape[0]):
    plt.scatter(yearlabels[:lengthlabels],train[i,:lengthlabels,0],s=15,edgecolor='w',
                color='darkgrey',alpha=0.4)
plt.plot(yearlabels[:lengthlabels],meantrain[:lengthlabels,0],color=c2(0.01),linewidth=2,
          linestyle='-')
         
for i in range(train.shape[0]):
    plt.scatter(yearlabels[lengthlabels:lengthlabels*2],train[i,lengthlabels:lengthlabels*2,1],s=15,edgecolor='w',
                color='k',alpha=0.4)
    plt.plot(yearlabels[lengthlabels:lengthlabels*2],meantrain[lengthlabels:lengthlabels*2,1],color=c2(0.6),linewidth=2,
              linestyle='-')
    
plt.xticks(np.arange(1920,2100+1,20),map(str,np.arange(1920,2100+1,20)))
plt.yticks(np.arange(0,1.1,0.2),map(str,np.round(np.arange(0,1.1,0.2),2)))
plt.xlim([1920,2100])
plt.ylim([0,1.0])
plt.savefig(directoryfigure + 'training_century_3_nomean.png',dpi=300)

###############################################################################
###############################################################################
############################################################################### 
### Testing figure
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey',
                labelsize=6)  
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)

for i in range(test.shape[0]):
    plt.scatter(yearlabels[:lengthlabels],test[i,:lengthlabels,0],s=15,edgecolor='w',
                color='darkgrey',alpha=0.4)
plt.plot(yearlabels[:lengthlabels],meantest[:lengthlabels,0],color=c2(0.01),linewidth=2,
          linestyle='-')
         
for i in range(test.shape[0]):
    plt.scatter(yearlabels[lengthlabels:lengthlabels*2],test[i,lengthlabels:lengthlabels*2,1],s=15,edgecolor='w',
                color='k',alpha=0.4)
    plt.plot(yearlabels[lengthlabels:lengthlabels*2],meantest[lengthlabels:lengthlabels*2,1],color=c2(0.6),linewidth=2,
              linestyle='-')

plt.xticks(np.arange(1920,2100+1,20),map(str,np.arange(1920,2100+1,20)))
plt.yticks(np.arange(0,1.1,0.2),map(str,np.round(np.arange(0,1.1,0.2),2)))
plt.xlim([1920,2100])
plt.ylim([0,1.0])

plt.savefig(directoryfigure + 'testing_century_3_nomean.png',dpi=300)