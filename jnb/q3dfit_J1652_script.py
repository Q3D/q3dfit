
import numpy as np
import sys
sys.path.append('../')
from q3dfit.q3df import q3dfit
import q3dfit.q3dpro as q3dpro
import traceback
from matplotlib import pyplot as plt


#q3di = '../jnb/XID2028/q3di.npy'
q3di = '../jnb/J1652/q3di.npy'


cols = [19]
rows = [21]


q3di_dict = np.load('../jnb/J1652/q3di.npy', allow_pickle=True)[()]


# q3dfit(q3di,cols=cols,rows=rows, nocrash=True)



q3di = q3di_dict

cols = 19
rows = 21
from q3dfit.q3dout import load_q3dout
q3do = load_q3dout(q3di, cols, rows)


do_plotlin = False
if do_plotlin:
	argsplotline = dict()
	argsplotline['nx'] = 1
	argsplotline['ny'] = 1
	argsplotline['line'] = ['Pab'] # 'H2_00_S5', '[ArII]6.99', '[NeVI]7.65'] #['Hbeta', '[OIII]5007']
	argsplotline['size'] = [0.5] #[0.05, 0.07]

	q3do.plot_line(q3di, plotargs=argsplotline)
	from matplotlib import pyplot as plt
	#plt.plot(q3do.wave, q3do.spec, color='k')
	#plt.plot(q3do.wave, q3do.cont_fit+q3do.line_fit, color='r')
	plt.show()
	plt.close()


do_plotcont_basic = False
if do_plotcont_basic:
	argscontplot = dict()
	argscontplot['xstyle'] = 'lin'
	argscontplot['ystyle'] = 'lin'
	argscontplot['fluxunit_out'] = 'flambda'
	argscontplot['mode'] = 'dark'

	argscontplot['IR'] = True

	q3do.sepcontpars(q3di)
	q3do.plot_cont(q3di, plotargs=argscontplot)
	plt.show()


do_plot_decomp = True
if do_plot_decomp:
	from q3dfit.plot import plotdecomp
	plotdecomp(q3do, q3di, show=True, do_lines=True)


import sys; sys.exit()

cols = [0,35]
rows = [0,35]
from q3dfit.q3dcollect import q3dcollect
# q3dcollect(q3di, cols=cols, rows=rows)


import q3dfit.q3dpro as q3dpro
qpro = q3dpro.Q3Dpro(q3di, PLATESCALE=0.025, NOCONT=True)


do_psf_map = False
if do_psf_map:
	qpro.__init__(q3di)

	white_light = np.sum(qpro.contdat.qso_mod, axis=2)

	from astropy.io import fits
	infile = '../../../DATA_Reduction/MIRI_MRS_reduction/my_jwebbinar_prep/ifu_session/sci/stage3_SkipOutlier/SkipOutlier_chancube_ch4-short_s3d.fits'
	hdul_in = fits.open(infile)
	hdul_new = hdul_in.copy()
	hdul_new[1].data = white_light
	#hdul_new.writeto('XID_2028_q3dfitPSF_whitelight.fits', overwrite=True)
	breakpoint()

	plt.imshow(white_light, vmin=np.percentile(white_light.flatten(), 90), vmax=np.percentile(white_light.flatten(), 99) )
	cb = plt.colorbar(pad=0.02)
	cb.ax.tick_params(labelsize=17)
	cb.set_label('Summed flux', fontsize=18, labelpad=10.3)
	plt.title('Recovered PSF - white light image', fontsize=15)
	plt.tight_layout()
	plt.savefig(qpro.dataDIR + q3di.label + '_psf_white')
	plt.show()







