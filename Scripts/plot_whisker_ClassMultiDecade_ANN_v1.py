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
num_of_class = 3
iterations = 100
labelx = [r'1920-1979',r'1980-2039',r'2040-2099']

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
trainq = np.genfromtxt(directorydata + 'training_%s_%s_%s_%s_iterations%s_v2.txt' % (variq[0],seasons[0],reg_name,dataset,iterations))
testq = np.genfromtxt(directorydata + 'testing_%s_%s_%s_%s_iterations%s_v2.txt' % (variq[0],seasons[0],reg_name,dataset,iterations))
obsq = np.genfromtxt(directorydata + 'obsout_%s_%s_%s_%s_iterations%s_v2.txt' % (variq[0],seasons[0],reg_name,dataset_obs,iterations))

### Reshape
train = np.reshape(trainq,(trainq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],trainq.shape[1]))
test = np.reshape(testq,(testq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],testq.shape[1]))
obs = obsq

### Medians
meantrain = np.nanmean(train,axis=0)
meantest = np.nanmean(test,axis=0)

### First
meantrain1 = np.nanmean(meantrain[:lengthlabels,0],axis=0)
train5 = np.percentile(meantrain[:lengthlabels,0],5,axis=0)
train95 = np.percentile(meantrain[:lengthlabels,0],95,axis=0)
meantest1 = np.nanmean(meantest[:lengthlabels,0],axis=0)
test5 = np.percentile(meantest[:lengthlabels,0],5,axis=0)
test95 = np.percentile(meantest[:lengthlabels,0],95,axis=0)

### Second
meantrain2 = np.nanmean(meantrain[lengthlabels:lengthlabels*2,1],axis=0)
train25 = np.percentile(meantrain[lengthlabels:lengthlabels*2,1],5,axis=0)
train295 = np.percentile(meantrain[lengthlabels:lengthlabels*2,1],95,axis=0)
meantest2 = np.nanmean(meantest[lengthlabels:lengthlabels*2,1],axis=0)
test25 = np.percentile(meantest[lengthlabels:lengthlabels*2,1],5,axis=0)
test295 = np.percentile(meantest[lengthlabels:lengthlabels*2,1],95,axis=0)

### Third
meantrain3 = np.nanmean(meantrain[-lengthlabels:,2],axis=0)
train35 = np.percentile(meantrain[-lengthlabels:,2],5,axis=0)
train395 = np.percentile(meantrain[-lengthlabels:,2],95,axis=0)
meantest3 = np.nanmean(meantest[-lengthlabels:,2],axis=0)
test35 = np.percentile(meantest[-lengthlabels:,2],5,axis=0)
test395 = np.percentile(meantest[-lengthlabels:,2],95,axis=0)

### Combination of data
total = np.append(train,test,axis=0)

datameantrain = [meantrain1,meantrain2,meantrain3]
datameantest = [meantest1,meantest2,meantest3]
train5 = [train5,train25,train35]
train95 = [train95,train295,train395]
test5 = [test5,test25,test35]
test95 = [test95,test295,test395]

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

###############################################################################
###############################################################################
############################################################################### 
### Training figure
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',
               labelbottom='off',bottom='off')
ax.tick_params(axis = "x", which = "both", bottom = False, top = False)

plt.axhline(0.5,linestyle='--',linewidth=2,color='dimgrey',dashes=(1,0.3))
ccc=ww.Aquatic1_5.mpl_colormap(np.linspace(0,0.8,len(datameantrain)))
for i in range(len(datameantrain)):
    plt.scatter(i,datameantrain[i],s=100,c=ccc[i],edgecolor=ccc[i],zorder=5,clip_on=False)
    plt.errorbar(i,datameantrain[i],
                  yerr=np.array([[datameantrain[i]-train5[i],train95[i]-datameantrain[i]]]).T,
                  color=ccc[i],linewidth=1.5,capthick=3,capsize=10,clip_on=False)

plt.ylabel(r'\textbf{TRAINING DATA}',color='k',fontsize=11)    
plt.xticks(np.arange(0,5,1),labelx)
plt.yticks(np.arange(0,1.1,0.2),map(str,np.round(np.arange(0,1.1,0.2),2)))
plt.xlim([-1,3])
plt.ylim([0,1.0])
plt.savefig(directoryfigure + 'training_3_errorbar.png',dpi=300)

###############################################################################
###############################################################################
############################################################################### 
### Testing figure
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(0)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',
               labelbottom='off',bottom='off')
ax.tick_params(axis = "x", which = "both", bottom = False, top = False)

plt.axhline(0.5,linestyle='--',linewidth=2,color='dimgrey',dashes=(1,0.3))
ccc=ww.Aquatic1_5.mpl_colormap(np.linspace(0,0.8,len(datameantrain)))
for i in range(len(datameantest)):
    plt.scatter(i,datameantest[i],s=100,c=ccc[i],edgecolor=ccc[i],zorder=5,clip_on=False)
    plt.errorbar(i,datameantest[i],
                  yerr=np.array([[datameantest[i]-test5[i],test95[i]-datameantest[i]]]).T,
                  color=ccc[i],linewidth=1.5,capthick=3,capsize=10,clip_on=False)

plt.ylabel(r'\textbf{TESTING DATA}',color='k',fontsize=11)    
plt.xticks(np.arange(0,5,1),labelx)
plt.yticks(np.arange(0,1.1,0.2),map(str,np.round(np.arange(0,1.1,0.2),2)))
plt.xlim([-1,3])
plt.ylim([0,1.0])
plt.savefig(directoryfigure + 'testing_3_errorbar.png',dpi=300)