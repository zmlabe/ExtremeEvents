"""
Script calculates accuracy of multi-decadal ANNv3

Author    : Zachary M. Labe
Date      : 28 January 2021
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
trainq = np.genfromtxt(directorydata + 'training_Century_%s_%s_%s_%s_iterations%s_v3.txt' % (variq[0],seasons[0],reg_name,dataset,iterations))
testq = np.genfromtxt(directorydata + 'testing_Century_%s_%s_%s_%s_iterations%s_v3.txt' % (variq[0],seasons[0],reg_name,dataset,iterations))
obsq = np.genfromtxt(directorydata + 'obsout_Century_%s_%s_%s_%s-%s_iterations%s_v3.txt' % (variq[0],seasons[0],reg_name,dataset_obs,dataset,iterations))

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
    accdata_pred = np.empty((data_pred.shape[0]))
    for i in range(data_pred.shape[0]):
        accdata_pred[i] = accuracy_score(data_true,data_pred[i,:])
        
    return accdata_pred

def accuracyTPeriodTime(data_pred,data_true):
    """
    Compute accuracy for the three periods
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

periodtrain = accuracyTPeriodTime(indextrain,classesl)
periodtest = accuracyTPeriodTime(indextest,classesl)

np.savetxt(directorydata + 'train_totalaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
            np.array([acctrain]))
np.savetxt(directorydata + 'test_totalaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
            np.array([acctest]))
np.savetxt(directorydata + 'train_periodaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
            periodtrain)
np.savetxt(directorydata + 'test_periodaccuracy_alldata_ClassCentury_ANNv3_%s.txt' % dataset,
            periodtest)
