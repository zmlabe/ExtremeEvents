"""
Script plots boxplots

Author    : Zachary M. Labe
Date      : 27 January 2021
"""

### Import modules
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import calc_Utilities as UT
import calc_dataFunctions as df
import palettable.wesanderson as ww
import calc_Stats as dSS
from sklearn.metrics import accuracy_score

### Set preliminaries
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_NewANN_v1/LENS/'
directorydata = '/Users/zlabe/Documents/Research/ExtremeEvents/Data/'
reg_name = 'Globe'
dataset = 'LENS'
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
iterations = 100

### Read in century data
trainacc = np.genfromtxt(directorydata + 'train_totalaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()
trainper = np.genfromtxt(directorydata + 'train_periodaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()
testacc = np.genfromtxt(directorydata + 'test_totalaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()
testper = np.genfromtxt(directorydata + 'test_periodaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()

### Read in multi-decade data
dectrainacc = np.genfromtxt(directorydata + 'train_totalaccuracy_alldata_ClassMultiDecade_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()
dectrainper = np.genfromtxt(directorydata + 'train_periodaccuracy_alldata_ClassMultiDecade_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()
dectestacc = np.genfromtxt(directorydata + 'test_totalaccuracy_alldata_ClassMultiDecade_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()
dectestper = np.genfromtxt(directorydata + 'test_periodaccuracy_alldata_ClassMultiDecade_ANNv3_%s.txt' % dataset,
              unpack=True).transpose()

###############################################################################
###############################################################################
###############################################################################    
### Plot box plots
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Adjust axes in time series plots 
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
        
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis="x",which="both",bottom = False,top=False,
               labelbottom=False)

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=2)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='w',linewidth=2)

positionstrain = np.array(range(trainper.shape[1]))*2.0-0.3
positionstest = np.array(range(testper.shape[1]))*2.0+0.3
bpl = plt.boxplot(trainper,positions=positionstrain,widths=0.5,
                  patch_artist=True,sym='')
bpr = plt.boxplot(testper,positions=positionstest, widths=0.5,
                  patch_artist=True,sym='')

# Modify boxes
ctrain = 'deepskyblue'
ctest = 'indianred'
set_box_color(bpl,ctrain)
set_box_color(bpr,ctest)
plt.plot([], c=ctrain, label=r'\textbf{Training}')
plt.plot([], c=ctest, label=r'\textbf{Testing}')
l = plt.legend(shadow=False,fontsize=7,loc='upper center',
            fancybox=True,frameon=False,ncol=2,bbox_to_anchor=(0.5,1.1),
            labelspacing=1,columnspacing=1,handletextpad=0.4)

for i in range(trainper.shape[1]):
    y = trainper[:,i]
    x = np.random.normal(positionstrain[i], 0.04, size=len(y))
    plt.plot(x, y,color='darkblue', alpha=1,zorder=10,marker='.',linewidth=0)
for i in range(testper.shape[1]):
    y = testper[:,i]
    x = np.random.normal(positionstest[i], 0.04, size=len(y))
    plt.plot(x, y,color='darkred', alpha=1,zorder=10,marker='.',linewidth=0)

plt.ylabel(r'\textbf{Accuracy}',color='k',fontsize=8)
plt.yticks(np.arange(0,1.1,0.1),list(map(str,np.round(np.arange(0,1.1,0.1),2))),
            fontsize=6) 
plt.ylim([0.5,1])

plt.text(-0.3,0.51,r'\textbf{1920-2009}',fontsize=10,color='dimgrey',
          ha='left',va='center')
plt.text(2.3,0.51,r'\textbf{2010-2099}',fontsize=10,color='dimgrey',
          ha='right',va='center')

plt.savefig(directoryfigure + 'ClassCentury_Accuracy_BOX_ANNv3_%s.png' % dataset,
            dpi=300)

###############################################################################  

fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis="x",which="both",bottom = False,top=False,
                labelbottom=False)

plt.axhline(1/3.,color='dimgrey',dashes=(1,0.3),linewidth=2,
            zorder=1)

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=2)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='w',linewidth=2)

positionstraindec = np.array(range(dectrainper.shape[1]))*2.0-0.3
positionstestdec = np.array(range(dectestper.shape[1]))*2.0+0.3
bpl = plt.boxplot(dectrainper,positions=positionstraindec,widths=0.5,
                  patch_artist=True,sym='')
bpr = plt.boxplot(dectestper,positions=positionstestdec, widths=0.5,
                  patch_artist=True,sym='')

# Modify boxes
ctrain = 'deepskyblue'
ctest = 'indianred'
set_box_color(bpl,ctrain)
set_box_color(bpr,ctest)
plt.plot([], c=ctrain, label=r'\textbf{Training}')
plt.plot([], c=ctest, label=r'\textbf{Testing}')
l = plt.legend(shadow=False,fontsize=7,loc='upper center',
            fancybox=True,frameon=False,ncol=2,bbox_to_anchor=(0.5,1.1),
            labelspacing=1,columnspacing=1,handletextpad=0.4)

for i in range(dectrainper.shape[1]):
    y = dectrainper[:,i]
    x = np.random.normal(positionstraindec[i], 0.04, size=len(y))
    plt.plot(x, y,color='darkblue', alpha=1,zorder=10,marker='.',linewidth=0)
for i in range(dectestper.shape[1]):
    y = dectestper[:,i]
    x = np.random.normal(positionstestdec[i], 0.04, size=len(y))
    plt.plot(x, y,color='darkred', alpha=1,zorder=10,marker='.',linewidth=0)
    
plt.ylabel(r'\textbf{Accuracy}',color='k',fontsize=8)
plt.yticks(np.arange(0,1.1,0.1),list(map(str,np.round(np.arange(0,1.1,0.1),2))),
            fontsize=6) 
plt.ylim([0,1])

plt.text(-0.4,0.01,r'\textbf{1920-1979}',fontsize=10,color='dimgrey',
          ha='left',va='center')
plt.text(2,0.01,r'\textbf{1980-2039}',fontsize=10,color='dimgrey',
          ha='center',va='center')
plt.text(4.4,0.01,r'\textbf{2040-2099}',fontsize=10,color='dimgrey',
          ha='right',va='center')

plt.savefig(directoryfigure + 'ClassMultiDecade_Accuracy_BOX_ANNv3_%s.png' % dataset,
            dpi=300)