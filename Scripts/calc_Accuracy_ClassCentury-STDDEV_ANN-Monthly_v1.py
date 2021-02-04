"""
Script calculates accuracy of century STDDEV ANNv1

Author    : Zachary M. Labe
Date      : 1 February 2021
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
dataset = 'LENS'
dataset_obs = '20CRv3'
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v2_STD-RMENS/CLASS-Monthly/%s/' % dataset
directorydata = '/Users/zlabe/Documents/Research/ExtremeEvents/Data/Class-STDDEV-Monthly/'
reg_name = 'Globe'
rm_ensemble_mean = True
variq = 'SLP'
monthlychoice = 'none'
land_only = False
ocean_only = False
rm_merid_mean = False
rm_annual_mean = False
rm_ensemble_mean = True
ensnum = 40
num_of_class = 2
iterations = [100]
nomonths = 12
window = 5 * nomonths

### Special cases
if reg_name == 'Globe':
    if dataset == 'MPI':
        reg_name = 'MPIGlobe'

### Create sample class labels for 1920-2099 monthly data
if num_of_class == 3:
    yearlabels = np.repeat(np.arange(1920+window//nomonths,2099+1,1),12)
    lengthlabels = yearlabels.shape[0]//num_of_class
    array1 = np.asarray([0]*lengthlabels)
    array2 = np.asarray([1]*lengthlabels)
    array3 = np.asarray([2]*lengthlabels)
    classesl = np.concatenate([array1,array2,array3],axis=None)
elif num_of_class == 2:
    yearlabels = np.repeat(np.arange(1920+window//nomonths,2099+1,1),12)
    lengthlabels = yearlabels.shape[0]//num_of_class
    array1 = np.asarray([0]*lengthlabels)
    array2 = np.asarray([1]*(yearlabels.shape[0]-lengthlabels))
    classesl = np.concatenate([array1,array2],axis=None)
    
### Read in data
trainq = np.genfromtxt(directorydata + 'training_STDDEVCentury-Monthly%syrs_%s_%s_%s_%s_iterations%s_land_only-%s_v1.txt' % (window,variq,monthlychoice,reg_name,dataset,iterations[0],land_only))
testq = np.genfromtxt(directorydata + 'testing_STDDEVCentury-Monthly%syrs_%s_%s_%s_%s_iterations%s_land_only-%s_v1.txt' % (window,variq,monthlychoice,reg_name,dataset,iterations[0],land_only))
obsq = np.genfromtxt(directorydata + 'obsout_STDDEVCentury-Monthly%syrs_%s_%s_%s_%s-%s_iterations%s_land_only-%s_v1.txt' % (window,variq,monthlychoice,reg_name,dataset_obs,dataset,iterations[0],land_only))

### Reshape
train = np.reshape(trainq,(trainq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],trainq.shape[1]))
test = np.reshape(testq,(testq.shape[0]//yearlabels.shape[0],yearlabels.shape[0],testq.shape[1]))
obs = obsq

### Combination of data
total = np.append(train,test,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Calculate accuracy

### Argmax
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

    data_predr = np.reshape(data_pred,(data_pred.shape[0]*data_pred.shape[1]))
    data_truer = np.tile(data_true,data_pred.shape[0])
    accdata_pred = accuracy_score(data_truer,data_predr)
        
    return accdata_pred

def accuracyTPeriodTime(data_pred,data_true):
    """
    Compute accuracy for the two periods
    """
    time = data_true.shape[0]
    period = int(time//2)
    
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
print(acctrain,acctest)
print(acctrain-acctest)

periodtrain = accuracyTPeriodTime(indextrain,classesl)
periodtest = accuracyTPeriodTime(indextest,classesl)

### Save statistics
np.savetxt(directorydata + 'train_totalaccuracy_STDDEVClassCentury_ANNv1-Monthly_%s_%s_%s.txt' % (dataset,reg_name,variq),
            np.array([acctrain]))
np.savetxt(directorydata + 'test_totalaccuracy_STDDEVClassCentury_ANNv1-Monthly_%s_%s_%s.txt' % (dataset,reg_name,variq),
            np.array([acctest]))
np.savetxt(directorydata + 'train_periodaccuracy_STDDEVClassCentury_ANNv1-Monthly_%s_%s_%s.txt' % (dataset,reg_name,variq),
            periodtrain)
np.savetxt(directorydata + 'test_periodaccuracy_STDDEVClassCentury_ANNv1-Monthly_%s_%s_%s.txt' % (dataset,reg_name,variq),
            periodtest)
