"""
Script plots boxplots

Author    : Zachary M. Labe
Date      : 26 January 2021
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
reg_name = 'narrowTropics'
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
trainacc = np.genfromtxt(directorydata + 'train_totalaccuracy_ClassCentury_ANNv3_%s_%s.txt' % (dataset,reg_name),
              unpack=True).transpose()
trainper = np.genfromtxt(directorydata + 'train_periodaccuracy_ClassCentury_ANNv3_%s_%s.txt' % (dataset,reg_name),
              unpack=True).transpose()
testacc = np.genfromtxt(directorydata + 'test_totalaccuracy_ClassCentury_ANNv3_%s_%s.txt' % (dataset,reg_name),
              unpack=True).transpose()
testper = np.genfromtxt(directorydata + 'test_periodaccuracy_ClassCentury_ANNv3_%s_%s.txt' % (dataset,reg_name),
              unpack=True).transpose()

# ### Read in multi-decade data
# dectrainacc = np.genfromtxt(directorydata + 'train_totalaccuracy_ClassMultiDecade_ANNv3_%s.txt' % dataset,
#               unpack=True).transpose()
# dectrainper = np.genfromtxt(directorydata + 'train_periodaccuracy_ClassMultiDecade_ANNv3_%s.txt' % dataset,
#               unpack=True).transpose()
# dectestacc = np.genfromtxt(directorydata + 'test_totalaccuracy_ClassMultiDecade_ANNv3_%s.txt' % dataset,
#               unpack=True).transpose()
# dectestper = np.genfromtxt(directorydata + 'test_periodaccuracy_ClassMultiDecade_ANNv3_%s.txt' % dataset,
#               unpack=True).transpose()

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

plt.axhline(0.5,color='dimgrey',dashes=(1,0.3),linewidth=2,
            zorder=1)

pos = [0,1]
plt.plot(pos,trainper,c='deepskyblue', label=r'\textbf{Training}',
         marker='o',markersize=10)
plt.plot(pos,testper,c='indianred', label=r'\textbf{Testing}',
         marker='o',markersize=10)
l = plt.legend(shadow=False,fontsize=7,loc='upper center',
            fancybox=True,frameon=False,ncol=2,bbox_to_anchor=(0.5,1.1),
            labelspacing=1,columnspacing=1,handletextpad=0.4)

plt.ylabel(r'\textbf{Accuracy}',color='k',fontsize=8)
plt.yticks(np.arange(0,1.1,0.1),list(map(str,np.round(np.arange(0,1.1,0.1),2))),
            fontsize=6) 
plt.ylim([0.,1])

plt.text(0.,0.01,r'\textbf{1920-2009}',fontsize=10,color='dimgrey',
         ha='left',va='center')
plt.text(1.,0.01,r'\textbf{2010-2099}',fontsize=10,color='dimgrey',
         ha='right',va='center')

plt.savefig(directoryfigure + 'ClassCentury_Accuracy_PointPlots_ANNv3_%s_%s.png' % (dataset,reg_name),
            dpi=300)

# ###############################################################################  

# fig = plt.figure()
# ax = plt.subplot(111)
# adjust_spines(ax, ['left', 'bottom'])
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['left'].set_color('dimgrey')
# ax.spines['bottom'].set_color('none')
# ax.spines['left'].set_linewidth(2)
# ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
# ax.tick_params(axis="x",which="both",bottom = False,top=False,
#                 labelbottom=False)

# plt.axhline(1/3.,color='dimgrey',dashes=(1,0.3),linewidth=2,
#             zorder=1)

# pos = [0,1,2]
# plt.plot(pos,dectrainper,c='deepskyblue', label=r'\textbf{Training}',
#           marker='o',markersize=10)
# plt.plot(pos,dectestper,c='indianred', label=r'\textbf{Testing}',
#           marker='o',markersize=10)

# l = plt.legend(shadow=False,fontsize=7,loc='upper center',
#             fancybox=True,frameon=False,ncol=2,bbox_to_anchor=(0.5,1.1),
#             labelspacing=1,columnspacing=1,handletextpad=0.4)

# plt.ylabel(r'\textbf{Accuracy}',color='k',fontsize=8)
# plt.yticks(np.arange(0,1.1,0.1),list(map(str,np.round(np.arange(0,1.1,0.1),2))),
#             fontsize=6) 
# plt.ylim([0.,1])

# plt.text(0.,0.01,r'\textbf{1920-1979}',fontsize=10,color='dimgrey',
#           ha='left',va='center')
# plt.text(1.,0.01,r'\textbf{1980-2039}',fontsize=10,color='dimgrey',
#           ha='center',va='center')
# plt.text(2.,0.01,r'\textbf{2040-2099}',fontsize=10,color='dimgrey',
#           ha='right',va='center')

# plt.savefig(directoryfigure + 'ClassMultiDecade_Accuracy_BPointPlots_ANNv3_%s.png' % dataset,
#             dpi=300)