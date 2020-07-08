from q3dfit.common import fitqsohost,makeqsotemplate,readcube
import numpy
from astropy.io import fits
import lmfit
import matplotlib.pyplot as plt
import os

#def load_in_data(infile):
#    '''Temporary function to load in a fits cube
#
#        Parametrs
#        ---------
#        infile: string
#        name or path to the fits file that needs to be loaded in
#
#        Returns
#        -------
#        cube: numpy array
#
#        '''
#
#
#
#    hdul = fits.open(infile)
#    print(hdul)
#    cube = hdul[1].data
#
#    return cube






#k=k/numpy.median(k)
data = readcube.CUBE(infile='../../pyfsfit/pg1411rb3.fits')
test_spec_to_fit = data.dat[10,15] #15,5  #15,3

wave = numpy.arange(0,len(test_spec_to_fit))

outxdr = ''
infits = '../../pyfsfit/pg1411rb3.fits'

qsotemplate = makeqsotemplate.makeqsotemplate(infits,outxdr,dataext=None,dqext=None,waveext=None)
#os.system('rm nucleartemplate.npy')

qsotemplate=[wave,qsotemplate['flux']]

x_ranges_to_fit = [[11,190],[500,1404],[1893,1997],[2078,3350],[4180,5485],[5495,5700],[5800,6195]] #[[0,6195]] [4180,5400],[5500,5700]
x_to_fit = numpy.array([])
y_to_fit = numpy.array([])
model_to_fit = numpy.array([])

for i in x_ranges_to_fit:
    print(i)
    x_to_fit=numpy.concatenate([x_to_fit,numpy.arange(i[0],i[1])])



result,comps,y_final=fitqsohost.fitqsohost(wave,test_spec_to_fit,test_spec_to_fit,0,0,x_to_fit,qsoxdr='nucleartemplate.npy',qsoonly=1,qsoord=1,hostonly=1,fcn_test=1)#hostonly=1,hostord=1)
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


