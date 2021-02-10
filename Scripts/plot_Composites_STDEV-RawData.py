"""
Create composites of the raw data after removing the ensemble mean and then
calculating a rolling standard deviation

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 10 February 2021
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
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
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directorydataENS = '/Users/zlabe/Data/SMILE/'
directorydataBB = '/Users/zlabe/Data/BEST/'
directorydataEE = '/Users/zlabe/Data/ERA5/'
directoryoutput = '/Users/zlabe/Documents/Research/ExtremeEvents/Data/Composites-STD/'
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v2_STD-RMENS/Composites-RawData/'

###############################
###############################
land_only = False
rm_ensemble_mean = True
rm_standard_dev = True
###############################
###############################
num_of_class = 2
window = 5
###############################
###############################
datasetsingle = ['lens','MPI','CSIRO_MK3.6','KNMI_ecearth']
# datasetsingle = ['KNMI_ecearth']
variq = 'T2M'
###############################
###############################
seasons = ['annual']
reg_name = 'SMILEGlobe'
years = np.arange(1920,2099+1,1)
###############################
###############################

### Functions
def read_primary_dataset(variq,dataset,lat_bounds,lon_bounds,monthlychoice):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons

###############################
### Set model defauls
meanstd = np.empty((len(datasetsingle),years.shape[0]-window,94,144))
maxstd = np.empty((len(datasetsingle),years.shape[0]-window,94,144))
minstd = np.empty((len(datasetsingle),years.shape[0]-window,94,144))
for i in range(len(datasetsingle)):
    if datasetsingle[i] == 'lens':
        simuqq = 'LENS'
        ensnum = 40
        timelens = np.arange(1920+window,2099+1,1)
        yearsall = [timelens]
        directoriesall = [directorydataLLL]
    elif datasetsingle[i] == 'MPI':
        simuqq = datasetsingle[0]
        ensnum = 40
        timempi = np.arange(1920+window,2099+1,1)
        yearsall = [timempi]
        directoriesall = [directorydataENS]
    elif datasetsingle[i] == 'CSIRO_MK3.6':
        simuqq = datasetsingle[0]
        ensnum = 30
        timecsiro= np.arange(1920+window,2099+1,1)
        yearsall = [timecsiro]
        directoriesall = [directorydataENS]
    elif datasetsingle[i] == 'KNMI_ecearth':
        simuqq = datasetsingle[0]
        ensnum = 16
        timeknmi = np.arange(1920+window,2099+1,1)
        yearsall = [timeknmi]
        directoriesall = [directorydataENS]
        
    ### Read in parameters
    monthlychoice = seasons[0]
    if reg_name == 'Globe':
        if any([datasetsingle[0]=='MPI',
                datasetsingle[0]=='CSIRO_MK3.6',
                datasetsingle[0]=='KNMI_ecearth']):
            reg_name = 'SMILEGlobe'
    lat_bounds,lon_bounds = UT.regions(reg_name)
        
    ### Define primary dataset to use
    dataset = datasetsingle[i]
    modelType = dataset
    
    ### Read in data
    data,lats,lons = read_primary_dataset(variq,dataset,lat_bounds,lon_bounds,monthlychoice)
    
    ### Slice data
    if rm_ensemble_mean == True:
        datae = dSS.remove_ensemble_mean(data)
        print('*Removed ensemble mean*')
        
    if rm_standard_dev == True:
        datas = dSS.rm_standard_dev(datae,window)
        print('*Removed standard deviation*')
        
    ### Calculate ensemble mean
    meanstd[i,:,:] = np.nanmean(datas,axis=0)
    maxstd[i,:,:]  = np.nanmean(datas,axis=0)
    minstd[i,:,:]  = np.nanmean(datas,axis=0)
    
    print('%s -- %s' % (i,datasetsingle[i]))
    
##############################################################################
##############################################################################
##############################################################################
def netcdfComp(lats,lons,var,directory,window,typemodel,season,variq,simuqq,land_only,reg_name):
    print('\n>>> Using netcdfComp function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'Composites_RawData_Maps-STDDEV%syrs_%s_%s_%s_%s_land_only-%s_%s.nc' % (window,typemodel,season,variq,simuqq,land_only,reg_name)
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'Composites of standard deviation maps' 
    
    ### Dimensions
    ncfile.createDimension('models',var.shape[0])
    ncfile.createDimension('years',var.shape[1])
    ncfile.createDimension('lat',var.shape[2])
    ncfile.createDimension('lon',var.shape[3])
    
    ### Variables
    models = ncfile.createVariable('models','f4',('models'))
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    varns = ncfile.createVariable('LRP','f4',('models','years','lat','lon'))
    
    ### Units
    varns.units = 'degrees C'
    ncfile.title = 'Standard Deviation - Ensemble mean removed'
    ncfile.instituion = 'Colorado State University'
    ncfile.references = 'Deser et al. [2020]'
    
    ### Data
    models[:] = np.arange(var.shape[0])
    years[:] = np.arange(var.shape[1])
    latitude[:] = lats
    longitude[:] = lons
    varns[:] = var
    
    ncfile.close()
    print('*Completed: Created netCDFComp File!')
    
netcdfComp(lats,lons,meanstd,directoryoutput,window,'meanSTD',seasons[0],variq,simuqq,land_only,reg_name)
netcdfComp(lats,lons,maxstd,directoryoutput,window,'maxSTD',seasons[0],variq,simuqq,land_only,reg_name)
netcdfComp(lats,lons,minstd,directoryoutput,window,'mminSTD',seasons[0],variq,simuqq,land_only,reg_name)
    