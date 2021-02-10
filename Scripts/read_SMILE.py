"""
Function(s) reads in monthly data from the MMLEA for selected
variables
 
Notes
-----
    Author : Zachary Labe
    Date   : 13 August 2020
    
Usage
-----
    [1] read_SMILE(directory,simulation,vari,sliceperiod,
                  slicebase,sliceshape,addclimo,
                  slicenan,takeEnsMean)
"""

def read_SMILE(directory,simulation,vari,sliceperiod,slicebase,sliceshape,addclimo,slicenan,takeEnsMean):
    """
    Function reads monthly data from the MMLEA
    
    Parameters
    ----------
    directory : string
        path for data
    simulation : string
        name of the model
    vari : string
        variable for analysis
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    addclimo : binary
        True or false to add climatology
    slicenan : string or float
        Set missing values
    takeEnsMean : binary
        whether to take ensemble mean
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable
    ENSmean : numpy array
        ensemble mean
        
    Usage
    -----
    read_SMILE(directory,simulation,vari,sliceperiod,
                  slicebase,sliceshape,addclimo,
                  slicenan,takeEnsMean)
    """
    print('\n>>>>>>>>>> STARTING read_SMILE function!')
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Parameters
    if simulation=='CCCma_canesm2':
        time = np.arange(1950,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        allens = np.arange(1,50+1,1)
        ens = allens
    elif simulation == 'CSIRO_MK3.6':
        time = np.arange(1850,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        allens = np.arange(1,30+1,1)
        ens = allens
    elif simulation == 'GFDL_CM3':
        time = np.arange(1920,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        allens = np.arange(1,20+1,1)
        ens = allens
    elif simulation == 'GFDL_ESM2M':
        time = np.arange(1950,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        allens = np.arange(1,30+1,1)
        ens = allens
    elif simulation == 'KNMI_ecearth':
        time = np.arange(1860,2100+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        allens = np.arange(1,16+1,1)
        ens = allens
    elif simulation == 'MPI':
        time = np.arange(1850,2099+1,1)
        timeslice = '%s-%s' % (time.min(),time.max())
        mon = 12
        # allens = np.arange(1,100+1,1)
        allens = np.arange(1,40+1,1)
        ens = allens
    else:
        ValueError('WRONG SMILE SELECTED!')
    
    ###########################################################################
    ### Read in data
    membersvar = []
    for i,ensmember in enumerate(ens):
        filename = directory + '%s/monthly/%s_%s_%s.nc' % (simulation,vari,ensmember,timeslice)                                                          
        data = Dataset(filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        var = data.variables['%s' % vari][:,:,:]
        data.close()
        
        print('Completed: read ensemble --%s for %s for %s--' % (simulation,ensmember,vari))
        membersvar.append(var)
        del var
    membersvar = np.asarray(membersvar)
    ensvar = np.reshape(membersvar,(len(ens),time.shape[0],mon,
                                    lat1.shape[0],lon1.shape[0]))
    del membersvar
    print('Completed: read all members!\n')
    
    ###########################################################################
    ### Check for missing data
    ensvar[np.where(ensvar <= -999)] = np.nan
    
    ###########################################################################
    ### Calculate anomalies or not
    if addclimo == True:
        ensvalue = ensvar
        print('Completed: calculated absolute variable!')
    elif addclimo == False:
        yearsq = np.where((time >= slicebase.min()) & (time <= slicebase.max()))[0]
        yearssel = time[yearsq]
        
        mean = np.nanmean(ensvar[:,yearsq,:,:,:])
        ensvalue = ensvar - mean
        print('Completed: calculated anomalies from',
              slicebase.min(),'to',slicebase.max())
        
    ###########################################################################
    ### Slice over months (currently = [ens,yr,mn,lat,lon])
    ### Shape of output array
    if sliceperiod == 'annual':
        ensvalue = np.nanmean(ensvalue,axis=2)
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape = ensvalue
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: ANNUAL MEAN!')
    elif sliceperiod == 'DJF':
        ensshape = np.empty((ensvalue.shape[0],ensvalue.shape[1]-1,
                             lat1.shape[0],lon1.shape[0]))
        for i in range(ensvalue.shape[0]):
            ensshape[i,:,:,:] = UT.calcDecJanFeb(ensvalue[i,:,:,:,:],
                                                 lat1,lon1,'surface',1)
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: DJF MEAN!')
    elif sliceperiod == 'MAM':
        enstime = np.nanmean(ensvalue[:,:,2:5,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: MAM MEAN!')
    elif sliceperiod == 'JJA':
        enstime = np.nanmean(ensvalue[:,:,5:8,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JJA MEAN!')
    elif sliceperiod == 'SON':
        enstime = np.nanmean(ensvalue[:,:,8:11,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: SON MEAN!')
    elif sliceperiod == 'JFM':
        enstime = np.nanmean(ensvalue[:,:,0:3,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JFM MEAN!')
    elif sliceperiod == 'AMJ':
        enstime = np.nanmean(ensvalue[:,:,3:6,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: AMJ MEAN!')
    elif sliceperiod == 'JAS':
        enstime = np.nanmean(ensvalue[:,:,6:9,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: JAS MEAN!')
    elif sliceperiod == 'OND':
        enstime = np.nanmean(ensvalue[:,:,9:,:,:],axis=2)
        if sliceshape == 1:
            ensshape = enstime.ravel()
        elif sliceshape == 4:
            ensshape = enstime
        print('Shape of output = ', ensshape.shape,[[ensshape.ndim]])
        print('Completed: OND MEAN!')
    elif sliceperiod == 'none':
        if sliceshape == 1:
            ensshape = ensvalue.ravel()
        elif sliceshape == 4:
            ensshape= np.reshape(ensvalue,(ensvalue.shape[0],ensvalue.shape[1]*ensvalue.shape[2],
                                             ensvalue.shape[3],ensvalue.shape[4]))
        elif sliceshape == 5:
            ensshape = ensvalue
        print('Shape of output =', ensshape.shape, [[ensshape.ndim]])
        print('Completed: ALL RAVELED MONTHS!')
        
    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        ensshape[np.where(np.isnan(ensshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    elif slicenan == False:
        ensshape = ensshape
    else:
        ensshape[np.where(np.isnan(ensshape))] = slicenan

    ###########################################################################
    ### Take ensemble mean
    if takeEnsMean == True:
        ENSmean = np.nanmean(ensshape,axis=0)
        print('Ensemble mean AVAILABLE!')
    elif takeEnsMean == False:
        ENSmean = np.nan
        print('Ensemble mean NOT available!')
    else:
        ValueError('WRONG OPTION!')
        
    ###########################################################################
    ### Change units
    if vari == 'SLP':
        ensshape = ensshape/100 # Pa to hPa
        ENSmean = ENSmean/100 # Pa to hPa
        print('Completed: Changed units (Pa to hPa)!')
    elif vari == 'T2M':
        ensshape = ensshape - 273.15 # K to C
        ENSmean = ENSmean - 273.15 # K to C
        print('Completed: Changed units (K to C)!')
        
    print('>>>>>>>>>> ENDING read_SMILE function!')
    
    ###########################################################################
    ### Change years
    if simulation == 'MPI':
        if sliceperiod == 'none':
            ensshaper = ensshape[:,70*12:,:,:] # 1920-2099
        else:
            ensshaper = ensshape[:,70:,:,:] # 1920-2099
        print('[**HARD CODE**: Shape of output =', ensshaper.shape, [[ensshaper.ndim]])
    elif simulation == 'CSIRO_MK3.6':
        if sliceperiod == 'none':
            ensshaper = ensshape[:,70*12:-12,:,:] # 1920-2099
        else:
            ensshaper = ensshape[:,70:-1,:,:] # 1920-2099
        print('[**HARD CODE**: Shape of output =', ensshaper.shape, [[ensshaper.ndim]])
    elif simulation == 'KNMI_ecearth':
        if sliceperiod == 'none':
            ensshaper = ensshape[:,60*12:-12,:,:] # 1920-2099
        else:
            ensshaper = ensshape[:,60:-1,:,:] # 1920-2099
        print('[**HARD CODE**: Shape of output =', ensshaper.shape, [[ensshaper.ndim]])
    else:
        ensshaper = ensshape

    return lat1,lon1,ensshaper,ENSmean
        

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/Users/zlabe/Data/SMILE/'
# simulation = 'KNMI_ecearth'
# vari = 'T2M'
# sliceperiod = 'none'
# slicebase = np.arange(1951,1980+1,1)
# sliceshape = 4
# slicenan = 'nan'
# addclimo = True
# takeEnsMean = False
# lat,lon,var,ENSmean = read_SMILE(directory,simulation,vari,sliceperiod,
#                                         slicebase,sliceshape,addclimo,
#                                         slicenan,takeEnsMean)