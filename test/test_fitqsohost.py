from q3dfit.common import fitqsohost
import numpy
from astropy.io import fits
import lmfit
import matplotlib.pyplot as plt

def load_in_data(infile):
    '''Temporary function to load in a fits cube
        
        Parametrs
        ---------
        infile: string
        name or path to the fits file that needs to be loaded in
        
        Returns
        -------
        cube: numpy array
        
        '''
    
    
    
    hdul = fits.open(infile)
    print(hdul)
    cube = hdul[1].data
    
    return cube


def makeqsotemplate(incube):
    
    '''Function defined to extract the quasar spectrum
        
        
        Parameters
        ----------
        
        
        
        Returns
        -------
        array
        2 by 1-D array of wavelength and quasar flux
        
        
        
        
        
        
        '''
    
    
    
    white_light_image = numpy.median(incube,axis=0) #Right now hard coding which axis is wavelength ra,dec.
    
    loc_max = numpy.where(white_light_image == white_light_image.max())
    print(loc_max)
    qsotemplate = incube[:,loc_max[0][0],loc_max[1][0]]
    
    
    return qsotemplate


data = load_in_data('../../pyfsfit/pg1411rb3.fits')

#k=k/numpy.median(k)
test_spec_to_fit = data[:,15,10] #15,5  #15,3

wave = numpy.arange(0,len(test_spec_to_fit))
qsotemplate=[wave,makeqsotemplate(data)]
x_ranges_to_fit = [[11,190],[500,1404],[1893,1997],[2078,3350],[4180,5485],[5495,5700],[5800,6195]] #[[0,6195]] [4180,5400],[5500,5700]
x_to_fit = numpy.array([])
y_to_fit = numpy.array([])
model_to_fit = numpy.array([])

for i in x_ranges_to_fit:
    print(i)
    x_to_fit=numpy.concatenate([x_to_fit,numpy.arange(i[0],i[1])])



result,comps,y_final=fitqsohost.fitqsohost(wave,test_spec_to_fit,test_spec_to_fit,0,0,x_to_fit,qsotemplate,qsoonly=1,qsoord=1,hostonly=1)#hostonly=1,hostord=1)
print(result.fit_report())
plt.plot(test_spec_to_fit,'blue',label='data')
plt.plot(x_to_fit,result.init_fit, 'k--', label='initial fit')
#plt.plot(x_to_fit,y_final[x_to_fit.astype('int')], 'r-', label='best fit')
plt.plot(y_final, 'r-', label='best fit')
plt.plot(test_spec_to_fit-y_final,label='residuals')
#plt.plot(comps['g1_'],label='g1_')
#plt.plot(comps['g2_'],label='g1_')
#plt.plot(comps['qsotemplate_model_exponential'],label='QSO')

for i in comps.keys():
    plt.plot(wave,comps[i],label=i)
plt.legend(loc='best')
plt.show()


