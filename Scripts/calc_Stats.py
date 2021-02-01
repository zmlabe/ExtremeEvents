"""
Functions are useful statistical untilities for data processing in the NN
 
Notes
-----
    Author : Zachary Labe
    Date   : 15 July 2020
    
Usage
-----
    [1] rmse(a,b)
    [2] remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs)
    [3] remove_merid_mean(data,data_obs)
    [4] remove_ensemble_mean(data)
    [5] remove_ocean(data,data_obs)
    [6] remove_land(data,data_obs)
    [7] standardize_data(Xtrain,Xtest)
    [8] rm_ensemble_mean(data,window)
"""

def rmse(a,b):
    """calculates the root mean squared error
    takes two variables, a and b, and returns value
    """
    
    ### Import modules
    import numpy as np
    
    ### Calculate RMSE
    rmse_stat = np.sqrt(np.mean((a - b)**2))
    
    return rmse_stat

###############################################################################

def remove_annual_mean(data,data_obs,lats,lons,lats_obs,lons_obs):
    """
    Removes annual mean from data set
    """
    
    ### Import modulates
    import numpy as np
    import calc_Utilities as UT
    
    ### Create 2d grid
    lons2,lats2 = np.meshgrid(lons,lats)
    lons2_obs,lats2_obs = np.meshgrid(lons_obs,lats_obs)
    
    ### Calculate weighted average and remove mean
    data = data - UT.calc_weightedAve(data,lats2)[:,:,np.newaxis,np.newaxis]
    data_obs = data_obs - UT.calc_weightedAve(data_obs,lats2_obs)[:,np.newaxis,np.newaxis]
    
    return data,data_obs

###############################################################################

def remove_merid_mean(data, data_obs):
    """
    Removes annual mean from data set
    """
    
    ### Import modulates
    import numpy as np
    
    ### Move mean of latitude
    data = data - np.nanmean(data,axis=2)[:,:,np.newaxis,:]
    data_obs = data_obs - np.nanmean(data_obs,axis=1)[:,np.newaxis,:]

    return data,data_obs

###############################################################################

def remove_ensemble_mean(data):
    """
    Removes ensemble mean
    """
    
    ### Import modulates
    import numpy as np
    
    ### Remove ensemble mean
    datameangone = data - np.nanmean(data,axis=0)
    
    return datameangone

###############################################################################

def remove_ocean(data,data_obs,lat_bounds,lon_bounds):
    """
    Masks out the ocean for land_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    
    ### Read in land mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'lsmask_19x25.nc'
    datafile = Dataset(directorydata + filename)
    maskq = datafile.variables['nmask'][:]
    lats = datafile.variables['latitude'][:]
    lons = datafile.variables['longitude'][:]
    datafile.close()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    return datamask, data_obsmask

###############################################################################

def remove_land(data,data_obs,lat_bounds,lon_bounds):
    """
    Masks out the ocean for ocean_only == True
    """
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_dataFunctions as df
    
    ### Read in ocean mask
    directorydata = '/Users/zlabe/Data/masks/'
    filename = 'ocmask_19x25.nc'
    datafile = Dataset(directorydata + filename)
    maskq = datafile.variables['nmask'][:]
    lats = datafile.variables['latitude'][:]
    lons = datafile.variables['longitude'][:]
    datafile.close()
    
    mask,lats,lons = df.getRegion(maskq,lats,lons,lat_bounds,lon_bounds)
    
    ### Mask out model and observations
    datamask = data * mask
    data_obsmask = data_obs * mask
    
    return datamask, data_obsmask

###############################################################################

def standardize_data(Xtrain,Xtest):
    """
    Standardizes training and testing data
    """
    
    ### Import modulates
    import numpy as np

    Xmean = np.nanmean(Xtrain,axis=0)
    Xstd = np.nanstd(Xtrain,axis=0)
    Xtest = (Xtest - Xmean)/Xstd
    Xtrain = (Xtrain - Xmean)/Xstd
    
    stdVals = (Xmean,Xstd)
    stdVals = stdVals[:]
    
    return Xtrain,Xtest,stdVals

###############################################################################

def convert_fuzzyDecade(data,startYear,classChunk):
    ### Import modules
    import numpy as np
    import scipy.stats as stats
    
    years = np.arange(startYear-classChunk*2,2100+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    labels = np.zeros((np.shape(data)[0],len(chunks)))
    
    for iy,y in enumerate(data):
        norm = stats.uniform.pdf(years,loc=y-classChunk/2.,scale=classChunk)
        
        vec = []
        for sy in years[::classChunk]:
            j=np.logical_and(years>sy,years<sy+classChunk)
            vec.append(np.sum(norm[j]))
        vec = np.asarray(vec)
        vec[vec<.0001] = 0. # This should not matter
        
        vec = vec/np.sum(vec)
        
        labels[iy,:] = vec
    return labels, chunks

###############################################################################

def convert_fuzzyDecade_toYear(label,startYear,classChunk):
    ### Import modules
    import numpy as np
    
    print('SELECT END YEAR - HARD CODED IN FUNCTION')
    years = np.arange(startYear-classChunk*2,2099+classChunk*2)
    chunks = years[::int(classChunk)] + classChunk/2
    
    return np.sum(label*chunks,axis=1)

###############################################################################

def invert_year_output(ypred,startYear):
    ### Import modules
    import numpy as np
    import scipy.stats as stats
    
    if(option4):
        inverted_years = convert_fuzzyDecade_toYear(ypred,startYear,classChunk)
    else:
        inverted_years = invert_year_outputChunk(ypred,startYear)
    
    return inverted_years

###############################################################################

def invert_year_outputChunk(ypred,startYear):
    ### Import modules
    import numpy as np
    import scipy.stats as stats
    
    if(len(np.shape(ypred))==1):
        maxIndices = np.where(ypred==np.max(ypred))[0]
        if(len(maxIndices)>classChunkHalf):
            maxIndex = maxIndices[classChunkHalf]
        else:
            maxIndex = maxIndices[0]

        inverted = maxIndex + startYear - classChunkHalf

    else:    
        inverted = np.zeros((np.shape(ypred)[0],))
        for ind in np.arange(0,np.shape(ypred)[0]):
            maxIndices = np.where(ypred[ind]==np.max(ypred[ind]))[0]
            if(len(maxIndices)>classChunkHalf):
                maxIndex = maxIndices[classChunkHalf]
            else:
                maxIndex = maxIndices[0]
            inverted[ind] = maxIndex + startYear - classChunkHalf
    
    return inverted

###############################################################################

def convert_to_class(data,startYear):
    ### Import modules
    import numpy as np
    
    data = np.array(data) - startYear + classChunkHalf
    dataClass = to_categorical(data)
    
    return dataClass

###############################################################################

def create_multiClass(xInput,yOutput):
    ### Import modules
    import numpy as np
    import copy as copy
    
    yMulti = copy.deepcopy(yOutput)
    
    for stepVal in np.arange(-classChunkHalf,classChunkHalf+1,1.):
        if(stepVal==0):
            continue
        y = yOutput + stepVal
        
    return xInput, yMulti

###############################################################################

def create_multiLabel(yClass):
    ### Import modules
    import numpy as np
    
    youtClass = yClass
    
    for i in np.arange(0,np.shape(yClass)[0]):
        v = yClass[i,:]
        j = np.argmax(v)
        youtClass[i,j-classChunkHalf:j+classChunkHalf+1] = 1
    
    return youtClass

###############################################################################
    
def rm_standard_dev(var,window):
    """
    Smoothed standard deviation
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling std!\n\n')
    
    rollingstd = np.empty((var.shape))
    for ens in range(var.shape[0]):
        for i in range(var.shape[2]):
            for j in range(var.shape[3]):
                series = pd.Series(var[ens,:,i,j])
                rollingstd[ens,:,i,j] = series.rolling(window).std().to_numpy()
    
    newdata = rollingstd[:,window:,:,:] 
    print('-----------COMPLETED: Rolling std!\n\n')     
    return newdata 

###############################################################################
    
def rm_variance_dev(var,window):
    """
    Smoothed variance
    """
    import pandas as pd
    import numpy as np
    
    print('\n\n-----------STARTED: Rolling vari!\n\n')
    
    rollingvar = np.empty((var.shape))
    for ens in range(var.shape[0]):
        for i in range(var.shape[2]):
            for j in range(var.shape[3]):
                series = pd.Series(var[ens,:,i,j])
                rollingvar[ens,:,i,j] = series.rolling(window).var().to_numpy()
    
    newdata = rollingvar[:,window:,:,:] 
    print('-----------COMPLETED: Rolling vari!\n\n')     
    return newdata 