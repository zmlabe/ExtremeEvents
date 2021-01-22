"""
Scipt plots histograms of data with mean removed over 4 time periods

Author    : Zachary M. Labe
Date      : 7 January 2021
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
directoryfigure = '/Users/zlabe/Desktop/ExtremeEvents_v1/Composites/MPI/'
reg_name = 'Globe'
dataset = 'MPI'
rm_ensemble_mean = True
variq = ['T2M']
monthlychoice = 'annual'
yeartype = ['1920-1964','1965-2009','2010-2054','2055-2099']

###############################################################################
###############################################################################
###############################################################################
def read_primary_dataset(variq,dataset,lat_bounds,lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons
###############################################################################
###############################################################################
###############################################################################
### Call functions
for i in range(len(variq)):
    ### Read in data for selected region                
    lat_bounds,lon_bounds = UT.regions(reg_name)
    dataall,lats,lons = read_primary_dataset(variq[i],dataset,
                                              lat_bounds,lon_bounds)
    
    ### Remove ensemble mean
    if rm_ensemble_mean == True:
        data= dSS.remove_ensemble_mean(dataall)
        print('*Removed ensemble mean*')
    elif rm_ensemble_mean ==  False:
        data = dataall
    
    ### Composite over selected period (x2)
    if monthlychoice == 'DJF':
        years = np.arange(dataall.shape[1]) + 1921
    else:
        years = np.arange(dataall.shape[1]) + 1920
        
meancomp = np.empty((len(years)//40,data.shape[0],data.shape[2],data.shape[3]))
for count,i in enumerate(range(0,len(years)-44,45)):
    meancomp[count,:,:,:,] = np.nanmean(data[:,i:i+45,:,:],axis=1)
    
### Mesh latxlon
lon2,lat2 = np.meshgrid(lons,lats)

### Set up different regions
# Globe
lat_globe = lats.copy()
lon_globe = lons.copy()
globe = meancomp.copy()
latmin_globe = -88.
latmax_globe = 88.
lonmin_globe = 0.
lonmax_globe = 360.
name_globe = 'Globe'

# Tropics
lat_trop = lats.copy()
lon_trop = lons.copy()
trop = meancomp.copy()
latmin_trop = -30.
latmax_trop = 30.
lonmin_trop = 0.
lonmax_trop = 360.
name_trop = 'Tropics'

# Northern Hemisphere
lat_nh = lats.copy()
lon_nh = lons.copy()
nh = meancomp.copy()
latmin_nh = 0.
latmax_nh = 88.
lonmin_nh = 0.
lonmax_nh = 360.
name_nh = 'Northern Hemisphere'

# Southern Hemisphere
lat_sh = lats.copy()
lon_sh = lons.copy()
sh = meancomp.copy()
latmin_sh = -88.
latmax_sh = 0.
lonmin_sh = 0.
lonmax_sh = 360.
name_sh = 'Southern Hemisphere'

# Indian Ocean
lat_io = lats.copy()
lon_io = lons.copy()
io = meancomp.copy()
latmin_io = -10.
latmax_io = 10.
lonmin_io = 50.
lonmax_io = 110.
name_io = 'Indian Ocean'

# ENSO region
lat_enso = lats.copy()
lon_enso = lons.copy()
enso = meancomp.copy()
latmin_enso = -5.
latmax_enso = 5.
lonmin_enso = 160.
lonmax_enso = 280.
name_enso = 'ENSO'

# North Atlantic
lat_na = lats.copy()
lon_na = lons.copy()
na = meancomp.copy()
latmin_na = 50.
latmax_na = 60.
lonmin_na = 315.
lonmax_na = 340.
name_na = 'North Atlantic'

# Arctic
lat_a = lats.copy()
lon_a = lons.copy()
a = meancomp.copy()
latmin_a = 67.
latmax_a = 88.
lonmin_a= 0.
lonmax_a = 360.
name_a = 'Arctic'

# Central Africa
lat_africa = lats.copy()
lon_africa = lons.copy()
africa = meancomp.copy()
latmin_africa = 0.
latmax_africa = 15.
lonmin_africa = 0.
lonmax_africa = 15.
name_africa = 'Central Africa'

# Southern Ocean
lat_so = lats.copy()
lon_so = lons.copy()
so = meancomp.copy()
latmin_so = -66.
latmax_so = 40.
lonmin_so = 5.
lonmax_so = 70.
name_so = 'Southern Ocean'

# Create lists
names = [name_globe,name_trop,name_nh,name_sh,name_io,
          name_enso,name_na,name_a,name_africa,name_so]

lattall = [lat_globe,lat_trop,lat_nh,lat_sh,lat_io,
            lat_enso,lat_na,lat_a,lat_africa,lat_so]
latallmin = [latmin_globe,latmin_trop,latmin_nh,latmin_sh,latmin_io,
              latmin_enso,latmin_na,latmin_a,latmin_africa,latmin_so]
latallmax = [latmax_globe,latmax_trop,latmax_nh,latmax_sh,latmax_io,
              latmax_enso,latmax_na,latmax_a,latmax_africa,latmax_so]

lonnall = [lon_globe,lon_trop,lon_nh,lon_sh,lon_io,
            lon_enso,lon_na,lon_a,lon_africa,lon_so]
lonallmin = [lonmin_globe,lonmin_trop,lonmin_nh,lonmin_sh,lonmin_io,
              lonmin_enso,lonmin_na,lonmin_a,lonmin_africa,lonmin_so]
lonallmax = [lonmax_globe,lonmax_trop,lonmax_nh,lonmax_sh,lonmax_io,
              lonmax_enso,lonmax_na,lonmax_a,lonmax_africa,lonmax_so]

regionsall = [globe,trop,nh,sh,io,enso,na,a,africa,so]

### Calculate regional averages for histograms
regions_average = []
for i in range(len(regionsall)):
    latq = np.where((lats >= latallmin[i]) & (lats <= latallmax[i]))[0]
    lonq = np.where((lons >= lonallmin[i]) & (lons <= lonallmax[i]))[0]
    latnew = lattall[i][latq]
    lonnew = lonnall[i][lonq]
    lonnew2,latnew2 = np.meshgrid(lonnew,latnew)
    
    regiongrid1 = regionsall[i][:,:,latq,:]
    regiongrid = regiongrid1[:,:,:,lonq]
    
    ave = UT.calc_weightedAve(regiongrid,latnew2)
    regions_average.append(ave)
    
### Calculate PDFs
num_bins = np.arange(-0.4,0.401,0.005)
pdfregions = np.empty((len(regions_average),meancomp.shape[0],len(num_bins)))
for rrr in range(len(regions_average)):
    for hist in range(meancomp.shape[0]):
        m,s = sts.norm.fit(regions_average[rrr][hist].ravel())
        pdfregions[rrr,hist,:] = sts.norm.pdf(num_bins,m,s)

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
    
### Begin each histogram set
color=ww.Chevalier_4.mpl_colormap(np.linspace(0,1,meancomp.shape[0]))
pp = np.empty((pdfregions.shape[0]))
for rrrr in range(pdfregions.shape[0]):
    
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
    
    ### Calculate statistical difference
    t,p = sts.ks_2samp(pdfregions[rrrr][0,:],pdfregions[rrrr][-1,:])
    pp[rrrr] = p
    
    for i,c in zip(range(pdfregions.shape[1]),color): 
        data = pdfregions[rrrr,i,:]
        
        plt.plot(num_bins,data,color=c,linewidth=2,label=r'\textbf{%s}' % yeartype[i],
                  clip_on=False)
        
    plt.xticks(np.arange(-0.4,0.41,0.1),map(str,np.round(np.arange(-0.4,0.41,0.1),2)))
    plt.yticks(np.arange(0,21,2),map(str,np.arange(0,21,2)))
    plt.xlim([-0.4,0.4])
    plt.ylim([0,12])
        
    l = plt.legend(shadow=False,fontsize=7,loc='upper center',
            fancybox=True,frameon=False,ncol=4,bbox_to_anchor=(0.5,1.1),
            labelspacing=0.2,columnspacing=1,handletextpad=0.4)
    for text in l.get_texts():
        text.set_color('k')
        
    plt.text(-0.4,10.9,r'\textbf{%s}' % names[rrrr],fontsize=20,
              color='dimgrey',ha='left',va='center')

    if p < 0.0001:
            plt.text(0.4,10.9,r'\textbf{\textit{p} $\bf{<<}$ 0.01}',fontsize=7,
                      color='k',ha='right',va='center')
    elif p < 0.01:
        plt.text(0.4,10.9,r'\textbf{\textit{p} $\bf{<}$ 0.01}',fontsize=7,
                  color='k',ha='right',va='center')
    elif p < 0.05:
        plt.text(0.4,10.9,r'\textbf{\textit{p} $\bf{<}$ 0.05}',fontsize=7,
                  color='k',ha='right',va='center')
        
    plt.savefig(directoryfigure + 'PDFs_%s_PeriodsInternal.png' % names[rrrr],
                dpi=300)

###############################################################################
###############################################################################
###############################################################################     
### Begin each histogram set
c2=ww.FantasticFox2_5.mpl_colormap
pp = np.empty((pdfregions.shape[0]))
for rrrr in range(pdfregions.shape[0]):
    
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
    
    ### Calculate statistical difference
    datafirst = regions_average[rrrr][0,:]
    datalasts = regions_average[rrrr][-1,:]
    
    n_lensf, bins_lensf, patches_lensf = plt.hist(datafirst,
              bins=np.arange(-0.4,0.41,0.02),density=False,color=c2(0.1),
              label=r'\textbf{1920-1964}',alpha=0.75,clip_on=False)
    for i in range(len(patches_lensf)):
        patches_lensf[i].set_facecolor(c2(0.1))
        patches_lensf[i].set_edgecolor('white')
        patches_lensf[i].set_linewidth(0.5)
        
    n_lensl, bins_lensl, patches_lensl = plt.hist(datalasts,
              bins=np.arange(-0.4,0.41,0.02),density=False,color=c2(0.6),
              label=r'\textbf{2055-2099}',alpha=0.75,clip_on=False)
    for i in range(len(patches_lensl)):
        patches_lensl[i].set_facecolor(c2(0.6))
        patches_lensl[i].set_edgecolor('white')
        patches_lensl[i].set_linewidth(0.5)
        
    plt.xticks(np.arange(-0.4,0.41,0.1),map(str,np.round(np.arange(-0.4,0.41,0.1),2)))
    plt.yticks(np.arange(0,21,2),map(str,np.arange(0,21,2)))
    plt.xlim([-0.4,0.4])
    plt.ylim([0,14])
        
    l = plt.legend(shadow=False,fontsize=7,loc='upper center',
            fancybox=True,frameon=False,ncol=4,bbox_to_anchor=(0.5,1.1),
            labelspacing=0.2,columnspacing=1,handletextpad=0.4)
    for text in l.get_texts():
        text.set_color('k')
        
    plt.text(-0.4,12.8,r'\textbf{%s}' % names[rrrr],fontsize=20,
              color='dimgrey',ha='left',va='center')
        
    plt.savefig(directoryfigure + 'Histogram_%s_PeriodsInternal.png' % names[rrrr],
                dpi=300)
    