# -*- coding: utf-8 -*-
"""
  @author: Caroline Bertemes, based on q3da by hadley

  This class defines a q3dout object, which is created by q3df when running on any single spaxel. 
  It collects all the output of q3df/fitspec and contains functions to generate plots for a single spaxel. 

  (For multi-spaxel processing instead, please see q3dpro.)

"""

import numpy as np
from q3dfit.readcube import Cube
from q3dfit.sepfitpars import sepfitpars
import importlib
from ppxf.ppxf_util import log_rebin
from q3dfit.linelist import linelist
from q3dfit.cmpweq import cmpweq
from q3dfit import qsohostfcn
from scipy.special import legendre
from scipy import interpolate
from timeit import default_timer as timer
import os
import copy as copy
from matplotlib import pyplot as plt
import q3dfit
from astropy.table import Table
from q3dfit.cmplin import cmplin



class q3dout:
	def __init__(self, iuse, juse, outstr):
		self.outstr = outstr
		self.struct = outstr
		self.initdat = outstr['q3di_initdat']
		self.fitran = outstr['fitran']
		self.iuse = iuse
		self.juse = juse

		# Continuum fit parameters
		self.ct_method = outstr['ct_method']
		self.ct_coeff = outstr['ct_coeff']
		self.ct_ebv = outstr['ct_ebv']
		self.zstar = outstr['zstar']
		self.zstar_err = outstr['zstar_err']
		self.ct_add_poly_weights = outstr['ct_add_poly_weights']
		self.ct_ppxf_sigma = outstr['ct_ppxf_sigma']
		self.ct_ppxf_sigma_err = outstr['ct_ppxf_sigma_err']
		self.ct_rchisq = outstr['ct_rchisq']

		# Spectrum in various forms
		self.wave = outstr['wave']
		self.spec = outstr['spec']
		self.spec_err = outstr['spec_err']
		self.cont_dat = outstr['cont_dat']
		self.cont_fit = outstr['cont_fit']
		self.cont_fit_pretweak = outstr['cont_fit_pretweak']
		self.emlin_dat = outstr['emlin_dat']
		self.emlin_fit = outstr['emlin_fit']

		# gd_indx is applied, and then ct_indx
		self.gd_indx = outstr['gd_indx']
		self.fitran_indx = outstr['fitran_indx']
		self.ct_indx = outstr['ct_indx']

		# Line fit parameters
		self.noemlinfit = outstr['noemlinfit']
		self.noemlinmask = outstr['noemlinmask']
		self.redchisq = outstr['redchisq']
		self.linelist = outstr['linelist']
		self.linelabel = outstr['linelabel']
		self.parinfo = outstr['parinfo']
		self.maxncomp = outstr['maxncomp']
		self.param = outstr['param']
		self.perror = outstr['perror']
		self.perror_resid = outstr['perror_resid']
		#self.covar = outstr['covar']
		self.siglim = outstr['siglim']

		# switch to track when first continuum processed
		self.firstcontproc = True



		### READ DATA
		if not ('datext' in self.initdat):
			datext = 1
		else:
			datext = self.initdat['datext']
		if not ('varext' in self.initdat):
			varext = 2
		else:
			varext = self.initdat['varext']
		if not ('dqext' in self.initdat):
			dqext = 3
		else:
			dqext = self.initdat['dqext']
		header = bytes(1)

		if 'argsreadcube' in self.initdat:
			self.cube = Cube(infile=self.initdat['infile'], quiet=True,
						datext=datext, varext=varext,
						dqext=dqext, **self.initdat['argsreadcube'])
		else:
			self.cube = Cube(infile=self.initdat['infile'], quiet=True,
						datext=datext, varext=varext,
						dqext=dqext)

		# set this to true if we're using Voronoi binning
		# and the tiling is missing
		novortile = False
		if self.cube.dat.ndim == 1:
			self.flux = self.cube.dat
			self.err = self.cube.err
			self.dq = self.cube.dq
		elif self.cube.dat.ndim == 2:
			self.flux = self.cube.dat[:, i]
			self.err = self.cube.err[:, i]
			self.dq = self.cube.dq[:, i]
		else:
			if 'vormap' in self.initdat:
				if np.isfinite(self.initdat['vormap'][i][j]) and \
						(self.initdat['vormap'][i][j] is not bad):
					iuse = vorcoords[self.initdat['vormap'][i][j] - 1, 0]
					juse = vorcoords[self.initdat['vormap'][i][j] - 1, 1]
				else:
					novortile = True
			else:
				iuse = self.iuse
				juse = self.juse

			if not novortile:
				self.flux = self.cube.dat[iuse, juse, :].flatten()
				self.err = self.cube.err[iuse, juse, :].flatten()
				self.dq = self.cube.dq[iuse, juse, :].flatten()


		### Line lists
		if 'noemlinfit' not in self.initdat or not outstr['noemlinfit']:
			# # get linelist
			# if 'argslinelist' in self.initdat:
			# 	self.listlines = linelist(self.initdat['lines'], **self.initdat['argslinelist'])
			# else:
			# 	self.listlines = linelist(self.initdat['lines'])
			self.listlines = self.linelist

			# table with doublets to combine
			data_path = q3dfit.__path__[0]+'/data/linelists/'
			self.doublets = Table.read(data_path+'doublets.tbl', format='ipac')
			# make a copy of singlet list
			self.lines_with_doublets = copy.deepcopy(self.initdat['lines'])
			# append doublet names to singlet list
			min1doubl = False
			for (name1, name2) in zip(self.doublets['line1'], self.doublets['line2']):
				if name1 in self.listlines['name'] and name2 in self.listlines['name']:
					self.lines_with_doublets.append(name1+'+'+name2)
					min1doubl = True
			if not min1doubl:
				self.lines_with_doublets = None



	def get_cont_props(self, do_save=False):
		struct = self.outstr
		bad = 1.0 * 10**99
		# make and populate output data self.cubes
		if self.firstcontproc is True:
			self.hostcube = \
			   {'dat': np.zeros(self.cube.nwave), #np.zeros((self.cube.ncols, self.cube.nrows, self.cube.nwave)),
				'err': np.zeros(self.cube.nwave),
				'dq':  np.zeros(self.cube.nwave),
				'norm_div': np.zeros(self.cube.nwave),
				'norm_sub': np.zeros(self.cube.nwave)}

			if 'decompose_ppxf_fit' in self.initdat:
				self.contcube = \
					{'wave': struct['wave'],
					 'all_mod': np.zeros(self.cube.nwave), #np.zeros((self.cube.ncols, self.cube.nrows, self.cube.nwave)),
					 'stel_mod': np.zeros(self.cube.nwave),
					 'poly_mod': np.zeros(self.cube.nwave),
					 'stel_mod_tot': 0. #np.zeros((self.cube.ncols, self.cube.nrows))
					 + bad,
					 'poly_mod_tot': 0.
					 + bad,
					 'poly_mod_tot_pct': 0.
					 + bad,
					 'stel_sigma': 0.
					 + bad,
					 'stel_sigma_err': np.array([0., 0.]) #np.zeros((self.cube.ncols, self.cube.nrows, 2))
					 + bad,
					 'stel_z': 0. 
					 + bad,
					 'stel_z_err': np.array([0., 0.])
					 + bad,
					 'stel_rchisq': 0.
					 + bad,
					 'stel_ebv': 0.
					 + bad,
					 'stel_ebv_err': np.array([0., 0.])
					 + bad}

			elif 'decompose_qso_fit' in self.initdat:
				self.contcube = \
					{'wave': struct['wave'],
					 'qso_mod':
						 np.zeros(self.cube.nwave), #np.zeros((self.cube.ncols, self.cube.nrows, self.cube.nwave)),
					 'qso_poly_mod':
						 np.zeros(self.cube.nwave),
					 'host_mod':
						 np.zeros(self.cube.nwave),
					 'poly_mod':
						 np.zeros(self.cube.nwave),
					 'npts':
						 0. + bad,  # np.zeros((self.cube.ncols, self.cube.nrows)) + bad,
					 'stel_sigma':
						 0. + bad,
					 'stel_sigma_err':
						 np.array([0., 0.]) + bad,
					 'stel_z':
						 0. + bad,
					 'stel_z_err':
						 np.array([0., 0.]) + bad,
					 'stel_rchisq':
						 0. + bad,
					 'stel_ebv':
						 0. + bad,
					 'stel_ebv_err':
						 0. + bad}
			else:
				self.contcube = \
					{'all_mod': np.zeros(self.cube.nwave),  #np.zeros((self.cube.ncols, self.cube.nrows, self.cube.nwave)),
					 'stel_z':
						 0. + bad,  # np.zeros((self.cube.ncols, self.cube.nrows)) + bad,
					 'stel_z_err':
						 np.array([0., 0.]) + bad,
					 'stel_rchisq':
						 0. + bad,
					 'stel_ebv':
						 0. + bad,
					 'stel_ebv_err':
						 np.array([0., 0.]) + bad}
			self.firstcontproc = False

		self.hostcube['dat'][struct['fitran_indx']] = struct['cont_dat']
		self.hostcube['err'][struct['fitran_indx']] = \
			self.err[struct['fitran_indx']]
		self.hostcube['dq'][struct['fitran_indx']] = \
			self.dq[struct['fitran_indx']]
		self.hostcube['norm_div'][struct['fitran_indx']] \
			= np.divide(struct['cont_dat'], struct['cont_fit'])
		self.hostcube['norm_sub'][struct['fitran_indx']] \
			= np.subtract(struct['cont_dat'], struct['cont_fit'])


		if 'decompose_ppxf_fit' in self.initdat:
			add_poly_degree = 4  # should match fitspec
			if 'argscontfit' in self.initdat:
				if 'add_poly_degree' in self.initdat['argscontfit']:
					add_poly_degree = \
						self.initdat['argscontfit']['add_poly_degree']
			# Compute polynomial
			dumy_log, wave_log,_ = \
				log_rebin([struct['wave'][0],
						   struct['wave'][len(struct['wave'])-1]],
						  struct['spec'])
			xnorm = np.linspace(-1., 1., len(wave_log))
			cont_fit_poly_log = 0.0
			for k in range(0, add_poly_degree):
				cfpllegfun = legendre(k)
				cont_fit_poly_log += \
					cfpllegfun(xnorm) * struct['ct_add_poly_weights'][k]
			interpfunction = \
				interpolate.interp1d(cont_fit_poly_log, wave_log,
									 kind='linear', fill_value="extrapolate")
			cont_fit_poly = interpfunction(np.log(struct['wave']))
			# Compute stellar continuum
			cont_fit_stel = np.subtract(struct['cont_fit'], cont_fit_poly)
			# Total flux fromd ifferent components
			cont_fit_tot = np.sum(struct['cont_fit'])
			self.contcube['all_mod'][struct['fitran_indx']] = \
				struct['cont_fit']
			self.contcube['stel_mod'][struct['fitran_indx']] = \
				cont_fit_stel
			self.contcube['poly_mod'][struct['fitran_indx']] = \
				cont_fit_poly
			self.contcube['stel_mod_tot'] = np.sum(cont_fit_stel)
			self.contcube['poly_mod_tot'] = np.sum(cont_fit_poly)
			self.contcube['poly_mod_tot_pct'] \
				= np.divide(self.contcube['poly_mod_tot'], cont_fit_tot)
			self.contcube['stel_sigma'] = struct['ct_ppxf_sigma']
			self.contcube['stel_z'] = struct['zstar']

			if 'ct_errors' in struct:
				self.contcube['stel_sigma_err'][:] \
					= struct['ct_errors']['ct_ppxf_sigma']
			# assuming that ct_errors is a dictionary
			else:  # makes an array with two arrays
				self.contcube['stel_sigma_err'][:] \
					= [struct['ct_ppxf_sigma_err'],
					   struct['ct_ppxf_sigma_err']]

			if 'ct_errors' in struct:
				self.contcube['stel_z_err'][:] = \
					struct['ct_errors']['zstar']
			else:
				self.contcube['stel_z_err'][:] \
					= [struct['zstar_err'], struct['zstar_err']]

		elif 'decompose_qso_fit' in self.initdat:
			if self.initdat['fcncontfit'] == 'fitqsohost':
				if 'qsoord' in self.initdat['argscontfit']:
					qsoord = self.initdat['argscontfit']['qsoord']
				else:
					qsoord = None  # ?

				if 'hostord' in self.initdat['argscontfit']:
					hostord = self.initdat['argscontfit']['hostord']
				else:
					hostord = None  # ?

				if 'blrpar' in self.initdat['argscontfit']:
					self.blrpar = self.initdat['argscontfit']['blrpar']
				else:
					self.blrpar = None
				# default here must be same as in IFSF_FITQSOHOST
				if 'add_poly_degree' in self.initdat['argscontfit']:
					self.add_poly_degree = \
						self.initdat['argscontfit']['add_poly_degree']
				else:
					self.add_poly_degree = 30

				# These lines mirror ones in IFSF_FITQSOHOST
				struct_tmp = struct

				# Get and renormalize template
				self.qsotemplate = \
					np.load(self.initdat['argscontfit']['qsoxdr'],
							allow_pickle='TRUE').item()
				try:
					self.qsowave = self.qsotemplate['wave']
					qsoflux_full = self.qsotemplate['flux']
				except:
					self.qsotemplate = \
						np.load(self.initdat['argscontfit']['qsoxdr'],
							allow_pickle='TRUE')
					self.qsowave = self.qsotemplate['wave'][0]
					qsoflux_full = self.qsotemplate['flux'][0]


				iqsoflux = \
					np.where((self.qsowave >= struct_tmp['fitran'][0]) &
							 (self.qsowave <= struct_tmp['fitran'][1]))
				self.qsoflux = qsoflux_full[iqsoflux]
				struct = struct_tmp
				#If polynomial residual is re-fit with PPXF, separate out best-fit
				#parameter structure created in IFSF_FITQSOHOST and compute polynomial
				#and stellar components
				if 'refit' in self.initdat['argscontfit'] and 'args_questfit' not in self.initdat['argscontfit']:
					self.par_qsohost = struct['ct_coeff']['qso_host']
					self.par_stel = struct['ct_coeff']['stel']
					dumy_log, wave_rebin,_ = log_rebin([struct['wave'][0],
						struct['wave'][len(struct['wave'])-1]],
						struct['spec'])
					xnorm = np.linspace(-1., 1., len(wave_rebin))
					if self.add_poly_degree > 0:
						par_poly = struct['ct_coeff']['poly']
						polymod_log = \
							legendre.legval(xnorm, par_poly)
						interpfunct = \
							interpolate.interp1d(wave_rebin,
												 polymod_log,
												 kind='cubic',
												 fill_value="extrapolate")
						self.polymod_refit = interpfunct(np.log(struct['wave']))
					else:
						self.polymod_refit = np.zeros(len(struct['wave']), dtype=float)
					self.contcube['stel_sigma'] = struct['ct_coeff']['ppxf_sigma']
					self.contcube['stel_z'] = struct['zstar']

					#Don't know ct_error's type
					if 'ct_errors' in struct:
						self.contcube['stel_sigma_err'][:] \
							= struct['ct_errors']['ct_ppxf_sigma']
					else:
						self.contcube['stel_sigma_err'][:] \
							= [struct['ct_ppxf_sigma_err'], struct['ct_ppxf_sigma_err']]
					if 'ct_errors' in struct:
						self.contcube['stel_z_err'][:] \
							= struct['ct_errors']['zstar']
					else:
						self.contcube['stel_z_err'][:] \
							= [struct['zstar_err'], struct['zstar_err']]
					#again why aren't those two if statements combined
				elif 'refit' in self.initdat['argscontfit'] and self.initdat['argscontfit']['refit']=='questfit': # Refitting with questfit in the MIR
					self.par_qsohost = struct['ct_coeff']['qso_host']
					dumy_log, wave_rebin,_ = log_rebin([struct['wave'][0],
						struct['wave'][len(struct['wave'])-1]],
						struct['spec'])
					xnorm = cap_range(-1.0, 1.0, len(wave_rebin)) #1D?
					self.polymod_refit = np.zeros(len(struct['wave']), dtype=float)  # Double-check

				else:
					self.par_qsohost = struct['ct_coeff']
					self.polymod_refit = 0.0

				#produce fit with template only and with template + host. Also
				#output QSO multiplicative polynomial
				self.qsomod_polynorm = 0.
				self.qsomod = \
					qsohostfcn(struct['wave'], params_fit=self.par_qsohost,
							   qsoflux=self.qsoflux, qsoonly=True,
							   blrpar=self.blrpar, qsoord=qsoord,
							   hostord=hostord)
				self.hostmod = struct['cont_fit_pretweak'] - self.qsomod

				#if continuum is tweaked in any region, subide resulting residual
				#proportionality @ each wavelength btwn qso and host components
				self.qsomod_notweak = self.qsomod
				if 'tweakcntfit' in self.initdat:
					modresid = struct['cont_fit'] - struct['cont_fit_pretweak']
					inz = np.where((self.qsomod != 0) & (self.hostmod != 0))[0]
					qsofrac = np.zeros(len(self.qsomod))
					for ind in inz:
						qsofrac[ind] = self.qsomod[ind] / (self.qsomod[ind] + self.hostmod[ind])
					self.qsomod += modresid * qsofrac
					self.hostmod += modresid * (1.0 - qsofrac)
				#components of qso fit for plotting
				self.qsomod_normonly = self.qsoflux
				if self.blrpar is not None:
					self.qsomod_blronly = \
						qsohostfcn(struct['wave'],
								   params_fit=self.par_qsohost,
								   qsoflux=self.qsoflux, blronly=True,
								   blrpar=self.blrpar, qsoord=qsoord,
								   hostord=hostord)
				else:
					self.qsomod_blronly = 0.
			elif  self.initdat['fcncontfit'] == 'questfit':      # CB: adding option to plot decomposed QSO fit if questfit is used
				from q3dfit.questfit import quest_extract_QSO_contrib
				self.qsomod, self.hostmod, self.qsomod_intr, self.hostmod_intr = quest_extract_QSO_contrib(struct['ct_coeff'], self.initdat)
				self.qsomod_polynorm = 1.
				self.qsomod_notweak = self.qsomod
				self.qsoflux = self.qsomod.copy()/np.median(self.qsomod)
				self.qsomod_normonly = self.qsoflux
				self.polymod_refit = 0.
				self.blrpar = None
				self.qsomod_blronly = 0.

		elif self.initdat['fcncontfit'] == 'ppxf' and 'qsotempfile' in self.initdat:
			qsotempfile = np.load(self.initdat['qsotempfile'], allow_pickle='TRUE').item()
			struct_qso = qsotempfile
			self.qsomod = struct_qso['cont_fit'] * struct['ct_coeff'][len(struct['ct_coeff']) - 1]
			self.hostmod = struct['cont_fit'] - self.qsomod
		elif self.initdat['fcncontfit'] == 'questfit':
		#else:
			self.contcube['all_mod'][struct['fitran_indx']] = struct['cont_fit']
			self.contcube['stel_z'] = struct['zstar']
			if 'ct_errors' in struct:
				self.contcube['stel_z_err'][:] = struct['ct_errors']['zstar']
			else:
				self.contcube['stel_z_err'][:] = [0, 0]

		self.contcube['stel_ebv'] = struct['ct_ebv']
		if 'ct_errors' in struct:
			self.contcube['stel_ebv_err'][:]=struct['ct_errors']['ct_ebv']
		else:
			self.contcube['stel_rchisq']=0.0

		# Print ppxf results to stdout
		if ('decompose_ppxf_fit' in self.initdat) or \
				('decompose_qso_fit' in self.initdat):
			if 'argscontfit' in self.initdat:
				if 'print_output' in self.initdat['argscontfit']:
					print("PPXF results: ")
					if 'decompose_ppxf_fit' in self.initdat:
						ct_coeff_tmp = struct['ct_coeff']
						poly_tmp_pct = self.contcube['poly_mod_tot_pct']
					else:
						ct_coeff_tmp = struct['ct_coeff']['stel']
						poly_tmp_pct = \
							np.sum(self.polymod_refit) / np.sum(self.hostmod)
					inz = np.where(ct_coeff_tmp != 0.0)
					ctnz = len(inz)
					if ctnz > 0:
						coeffgd = ct_coeff_tmp[inz]
						# normalize coefficients to % of total stellar coeffs.
						totcoeffgd = np.sum(coeffgd)
						coeffgd /= totcoeffgd
						# re-normalize to % of total flux
						coeffgd *= (1.0 - poly_tmp_pct)
						# TODO: xdr file
						startempfile = \
							np.load(self.initdat['startempfile']+".npy",
									allow_pickle='TRUE').item()
						agesgd = startempfile['ages'][inz]  # check
						# sum coefficients over age ranges
						iyoung = np.where(agesgd < 1e7)
						ctyoung = len(iyoung)
						iinter1 = np.where(agesgd > 1e7 and agesgd < 1e8)
						ctinter1 = len(iinter1)
						iinter2 = np.where(agesgd > 1e8 and agesgd < 1e9)
						ctinter2 = len(iinter2)
						iold = np.where(agesgd > 1e9)
						ctold = len(iold)
						if ctyoung > 0:
							coeffyoung = np.sum(coeffgd[iyoung]) * 100.0
						else:
							coeffyoung = 0.0
						if ctinter1 > 0:
							coeffinter1 = np.sum(coeffgd[iinter1]) * 100.0
						else:
							coeffinter1 = 0.0
						if ctinter2 > 0:
							coeffinter2 = np.sum(coeffgd[iinter2]) * 100.0
						else:
							coeffinter2 = 0.0
						if ctold > 0:
							coeffold = np.sum(coeffgd[iold]) * 100.0
						else:
							coeffold = 0.0
						print(str(round(coeffyoung)) +
							  ' contribution from ages <= 10 Myr.')
						print(str(round(coeffinter1)) +
							  ' contribution from 10 Myr < age <= 100 Myr.')
						print(str(round(coeffinter2)) +
							  ' contribution from 100 Myr < age <= 1 Gyr.')
						print(str(round(coeffold)) +
							  ' contribution from ages > 1 Gyr.')
						print(' Stellar template convolved with sigma = ' +
							  str(struct['ct_ppxf_sigma']) + 'km/s')

		if do_save:
			np.save('{[outdir]}{[label]}'.format(self.initdat, self.initdat)+'.cont.npy', self.contcube)




	def plot_cont(self, decompose_qso_fit=False, decompose_ppxf_fit=False, argspltcont=None, quiet=True, show=False):
		'''
		Continuum plotting function
		'''
		if 'fcnpltcont' in self.initdat:
			fcnpltcont = self.initdat['fcnpltcont']
		else:
			fcnpltcont = 'plot_cont'
		module = importlib.import_module('q3dfit.' + fcnpltcont)
		pltcontfcn = getattr(module, fcnpltcont)

		flux = np.array(self.cube.dat)[self.iuse, self.juse, :].flatten()
		err = np.array(np.sqrt(abs(self.cube.var[self.iuse, self.juse, :]))).flatten()
		dq = np.array(self.cube.dq)[self.iuse, self.juse, :].flatten()
		labin = '{[outdir]}{[label]}_{:04d}_{:04d}'.\
			format(self.initdat, self.initdat, self.iuse+1, self.juse+1)
		labout = '{[outdir]}{[label]}_{:04d}_{:04d}'.\
			format(self.initdat, self.initdat, self.iuse+1, self.juse+1)
		outfile = labout

		# if oned:
		# 	err = []
		# 	for a in self.cube.var[:, i]:
		# 		err.append(np.sqrt(abs(a)))


		# set this to true if we're using Voronoi binning
		# and the tiling is missing
		# novortile = False
		# # if oned:  # i think?
		# # 	flux = np.array(self.cube.dat)[:, i]
		# # 	err = []
		# # 	for a in self.cube.var[:, i]:
		# # 		err.append(np.sqrt(abs(a)))
		# # 	dq = self.cube.dq[:, i]
		# # 	labin = '{[outdir]}{[label]}_{:04d}'.\
		# # 		format(initdat, initdat, i+1)
		# # 	labout = labin
		# # else:
		# if True:
		# 	if not quiet:
		# 		print(f'    Row {self.juse+1} of {self.cube.nrows}')

		# 	if 'vormap' in self.initdat:
		# 		if np.isfinite(self.initdat['vormap'][self.iuse][self.juse]) and \
		# 				(self.initdat['vormap'][self.iuse][self.juse] is not bad):
		# 			self.iuse = vorcoords[self.initdat['vormap'][self.iuse][self.juse] - 1, 0]
		# 			self.juse = vorcoords[self.initdat['vormap'][self.iuse][self.juse] - 1, 1]
		# 		else:
		# 			novortile = True

		# 	if not novortile:
		# 		flux = np.array(self.cube.dat)[self.iuse, self.juse, :].flatten()
		# 		err = np.array(np.sqrt(abs(self.cube.var[self.iuse, self.juse, :]))).flatten()
		# 		dq = np.array(self.cube.dq)[self.iuse, self.juse, :].flatten()
		# 		labin = '{[outdir]}{[label]}_{:04d}_{:04d}'.\
		# 			format(self.initdat, self.initdat, self.iuse+1, self.juse+1)
		# 		labout = '{[outdir]}{[label]}_{:04d}_{:04d}'.\
		# 			format(self.initdat, self.initdat, self.iuse+1, self.juse+1)

		# # Restore fit after a couple of sanity checks
		# # these sanity checks are wearing down my sanity
		# if not novortile:
		# 	infile = labin + '.npy'
		# 	outfile = labout
		# 	nodata = flux.nonzero()
		# 	ct = len(nodata[0])
		# 	filepresent = os.path.isfile(infile)  # check file
		# else:
		# 	# missing Voronoi bin for this spaxel
		# 	filepresent = False
		# 	ct = 0

		# if not filepresent or ct == 0:

		# 	badmessage = f'        No data for [{i+1}, {j+1}]'
		# 	print(badmessage)

		# else:
		# 	# Restore fit
		# 	#infile = self.initdat['infile']
		# 	struct = (np.load(infile, allow_pickle='TRUE')).item()
		# 	for el in struct:
		# 		if el not in self.outstr:
		# 			print(el, ' not in outstr')
		# 	breakpoint()

		if True:

			self.get_cont_props()

			struct = self.outstr

			# Restore original error.
			struct['spec_err'] = err[struct['fitran_indx']]

			if sum(struct['cont_fit']) != 0.0:

				if decompose_qso_fit:

					# if self.initdat['fcncontfit'] == 'fitqsohost':
					# 	if 'qsoord' in self.initdat['argscontfit']:
					# 		qsoord = self.initdat['argscontfit']['qsoord']
					# 	else:
					# 		qsoord = None  # ?

					# 	if 'hostord' in self.initdat['argscontfit']:
					# 		hostord = self.initdat['argscontfit']['hostord']
					# 	else:
					# 		hostord = None  # ?

					# 	if 'blrpar' in self.initdat['argscontfit']:
					# 		blrpar = self.initdat['argscontfit']['blrpar']
					# 	else:
					# 		blrpar = None
					# 	# default here must be same as in IFSF_FITQSOHOST
					# 	if 'add_poly_degree' in self.initdat['argscontfit']:
					# 		add_poly_degree = \
					# 			initdat['argscontfit']['add_poly_degree']
					# 	else:
					# 		add_poly_degree = 30

					# 	# These lines mirror ones in IFSF_FITQSOHOST
					# 	struct_tmp = struct

					# 	# Get and renormalize template
					# 	qsotemplate = \
					# 		np.load(self.initdat['argscontfit']['qsoxdr'],
					# 				allow_pickle='TRUE').item()
					# 	try:
					# 		qsowave = qsotemplate['wave']
					# 		qsoflux_full = qsotemplate['flux']
					# 	except:
					# 		qsotemplate = \
					# 			np.load(self.initdat['argscontfit']['qsoxdr'],
					# 				allow_pickle='TRUE')
					# 		qsowave = qsotemplate['wave'][0]
					# 		qsoflux_full = qsotemplate['flux'][0]

					# 	iqsoflux = \
					# 		np.where((qsowave >= struct_tmp['fitran'][0]) &
					# 				 (qsowave <= struct_tmp['fitran'][1]))
					# 	qsoflux = qsoflux_full[iqsoflux]
					# 	struct = struct_tmp

					# 	#If polynomial residual is re-fit with PPXF, separate out best-fit
					# 	#parameter structure created in IFSF_FITQSOHOST and compute polynomial
					# 	#and stellar components
					# 	if 'refit' in self.initdat['argscontfit'] and 'args_questfit' not in self.initdat['argscontfit']:
					# 		par_qsohost = struct['ct_coeff']['qso_host']
					# 		par_stel = struct['ct_coeff']['stel']
					# 		dumy_log, wave_rebin,_ = log_rebin([struct['wave'][0],
					# 			struct['wave'][len(struct['wave'])-1]],
					# 			struct['spec'])
					# 		xnorm = np.linspace(-1., 1., len(wave_rebin))
					# 		if add_poly_degree > 0:
					# 			par_poly = struct['ct_coeff']['poly']
					# 			polymod_log = \
					# 				legendre.legval(xnorm, par_poly)
					# 			interpfunct = \
					# 				interpolate.interp1d(wave_rebin,
					# 									 polymod_log,
					# 									 kind='cubic',
					# 									 fill_value="extrapolate")
					# 			polymod_refit = interpfunct(np.log(struct['wave']))
					# 		else:
					# 			polymod_refit = np.zeros(len(struct['wave']), dtype=float)

					# 	# elif 'refit' in self.initdat['argscontfit'] and self.initdat['argscontfit']['refit']=='questfit': # Refitting with questfit in the MIR
					# 	# 	par_qsohost = struct['ct_coeff']['qso_host']
					# 	# 	dumy_log, wave_rebin,_ = log_rebin([struct['wave'][0],
					# 	# 		struct['wave'][len(struct['wave'])-1]],
					# 	# 		struct['spec'])
					# 	# 	xnorm = cap_range(-1.0, 1.0, len(wave_rebin)) #1D?
					# 	# 	polymod_refit = np.zeros(len(struct['wave']), dtype=float)  # Double-check

					# 	else:
					# 		par_qsohost = struct['ct_coeff']
					# 		polymod_refit = 0.0


					# 	#produce fit with template only and with template + host. Also
					# 	#output QSO multiplicative polynomial
					# 	qsomod_polynorm = 0.
					# 	qsomod = \
					# 		qsohostfcn(struct['wave'], params_fit=par_qsohost,
					# 				   qsoflux=qsoflux, qsoonly=True,
					# 				   blrpar=blrpar, qsoord=qsoord,
					# 				   hostord=hostord)
					# 	hostmod = struct['cont_fit_pretweak'] - qsomod

					# 	#if continuum is tweaked in any region, subide resulting residual
					# 	#proportionality @ each wavelength btwn qso and host components
					# 	qsomod_notweak = qsomod
					# 	if 'tweakcntfit' in self.initdat:
					# 		modresid = struct['cont_fit'] - struct['cont_fit_pretweak']
					# 		inz = np.where((qsomod != 0) & (hostmod != 0))[0]
					# 		qsofrac = np.zeros(len(qsomod))
					# 		for ind in inz:
					# 			qsofrac[ind] = qsomod[ind] / (qsomod[ind] + hostmod[ind])
					# 		qsomod += modresid * qsofrac
					# 		hostmod += modresid * (1.0 - qsofrac)
					# 	#components of qso fit for plotting
					# 	qsomod_normonly = qsoflux
					# 	if blrpar is not None:
					# 		qsomod_blronly = \
					# 			qsohostfcn(struct['wave'],
					# 					   params_fit=par_qsohost,
					# 					   qsoflux=qsoflux, blronly=True,
					# 					   blrpar=blrpar, qsoord=qsoord,
					# 					   hostord=hostord)
					# 	else:
					# 		qsomod_blronly = 0.
					# elif self.initdat['fcncontfit'] == 'questfit':      # CB: adding option to plot decomposed QSO fit if questfit is used
					# 	from q3dfit.questfit import quest_extract_QSO_contrib
					# 	qsomod, hostmod, qsomod_intr, hostmod_intr = quest_extract_QSO_contrib(struct['ct_coeff'], self.initdat)
					# 	qsomod_polynorm = 1.
					# 	qsomod_notweak = qsomod
					# 	qsoflux = qsomod.copy()/np.median(qsomod)
					# 	qsomod_normonly = qsoflux
					# 	polymod_refit = 0.
					# 	blrpar = None
					# 	qsomod_blronly = 0.

					struct_host = copy.deepcopy(struct)
					struct_qso = copy.deepcopy(struct_host)

					struct_host['spec'] -= self.qsomod
					struct_host['cont_dat'] -= self.qsomod
					struct_host['cont_fit'] -= self.qsomod

					struct_qso['spec'] -= self.hostmod
					struct_qso['cont_dat'] -= self.hostmod
					struct_qso['cont_fit'] -= self.hostmod


					if np.sum(struct_host['cont_fit']) != 0.0:
						if 'refit' in self.initdat['argscontfit']:
							compspec = np.array([self.polymod_refit,
												 self.hostmod-self.polymod_refit])
							comptitles = ['ord. ' + str(self.add_poly_degree) +
									   ' Leg. poly.', 'stel. temp.']
						else:
							compspec = [self.hostmod]
							# if initdat['fcncontfit'] == 'questfit':     ##  CB: Work-around - think about this more later
							#     compspec = [hostmod]
							comptitles = ['exponential terms']


						if argspltcont is not None or 'argspltcont' in self.initdat:
							if argspltcont is None:		argspltcont=self.initdat['argspltcont']
							pltcontfcn(struct_host, outfile + '_cnt_host',
								   compspec=compspec, comptitles=comptitles,
								   title='Host', fitran=self.initdat['fitran'],
								   initdat=self.initdat, **argspltcont)
						else:
							pltcontfcn(struct_host, outfile + '_cnt_host',
								   compspec=compspec, comptitles=comptitles,
								   title='Host', fitran=self.initdat['fitran'],
								   initdat=self.initdat)

						if self.blrpar is not None and max(self.qsomod_blronly) != 0.:
							qsomod_blrnorm = np.median(qsomod) / \
								max(self.qsomod_blronly)
							compspec = np.array([self.qsomod_normonly,
												 self.qsomod_blronly *
												 qsomod_blrnorm])
							comptitles = ['raw template', 'scattered*' +
									   str(qsomod_blrnorm)]
						else:
							compspec = [self.qsomod_normonly]
							comptitles = ['raw template']
							


						if self.initdat['fcncontfit'] != 'questfit':
							if argspltcont is not None or 'argspltcont' in self.initdat:
								if argspltcont is None:		argspltcont=self.initdat['argspltcont']
								pltcontfcn(struct_qso, str(outfile) + '_cnt_qso',
									   compspec=compspec, comptitles=comptitles,
									   title='QSO', fitran=self.initdat['fitran'],
									   initdat=self.initdat, **argspltcont)
							else:
								pltcontfcn(struct_qso, str(outfile) + '_cnt_qso',
									   compspec=compspec, comptitles=comptitles,
									   title='QSO', fitran=self.initdat['fitran'],
									   initdat=self.initdat)
						else:
							if argspltcont is not None or 'argspltcont' in self.initdat:
								if argspltcont is None:		argspltcont=self.initdat['argspltcont']
								pltcontfcn(struct_qso, str(outfile) + '_cnt_qso',
									   compspec=[struct_qso['cont_fit']],
									   title='QSO', fitran=self.initdat['fitran'],
									   comptitles=['QSO'], initdat=self.initdat,
									   **argspltcont)
							else:
								pltcontfcn(struct_qso, str(outfile) + '_cnt_qso',
									   compspec=[struct_qso['cont_fit']],
									   title='QSO', fitran=self.initdat['fitran'],
									   comptitles=['QSO'], initdat=self.initdat)								


						if 'argspltcont' in self.initdat:
							pltcontfcn(struct, outfile + '_cnt',
								   compspec=np.array([self.qsomod, self.hostmod]),
								   title='Total', comptitles=['QSO', 'host'],
								   fitran=self.initdat['fitran'], initdat=self.initdat,
								   **self.initdat['argspltcont'])
						else:
							pltcontfcn(struct, outfile + '_cnt',
								   compspec=np.array([self.qsomod, self.hostmod]),
								   title='Total', comptitles=['QSO', 'host'],
								   fitran=self.initdat['fitran'], initdat=self.initdat)


				elif self.initdat['fcncontfit'] == 'ppxf' and 'qsotempfile' in self.initdat:
					qsotempfile = np.load(self.initdat['qsotempfile'], allow_pickle='TRUE').item()
					struct_qso = qsotempfile
					self.qsomod = struct_qso['cont_fit'] * struct['ct_coeff'][len(struct['ct_coeff']) - 1]
					self.hostmod = struct['cont_fit'] - self.qsomod

					if 'argspltcont' in self.initdat:
						pltcontfcn(struct, outfile + '_cnt',
								   compspec=np.array([self.qsomod, self.hostmod]),
								   title='Total', comptitles=['QSO', 'host'],
								   fitran=self.fitran, initdat=self.initdat,
								   **self.initdat['argspltcont'])
					else:
						pltcontfcn(struct, outfile + '_cnt',
								   compspec=np.array([self.qsomod, self.hostmod]),
								   title='Total', comptitles=['QSO', 'host'],
								   fitran=self.fitran, initdat=self.initdat)

				elif decompose_ppxf_fit:

					self.add_poly_degree = 4  # should match fitspec
					if 'argscontfit' in self.initdat:
						if 'add_poly_degree' in self.initdat['argscontfit']:
							self.add_poly_degree = \
								self.initdat['argscontfit']['add_poly_degree']
					# Compute polynomial
					dumy_log, wave_log,_ = \
						log_rebin([struct['wave'][0],
								   struct['wave'][len(struct['wave'])-1]],
								  struct['spec'])
					xnorm = np.linspace(-1., 1., len(wave_log))
					cont_fit_poly_log = 0.0
					for k in range(0, self.add_poly_degree):
						cfpllegfun = legendre(k)
						cont_fit_poly_log += \
							cfpllegfun(xnorm) * struct['ct_add_poly_weights'][k]
					interpfunction = \
						interpolate.interp1d(cont_fit_poly_log, wave_log,
											 kind='linear', fill_value="extrapolate")
					cont_fit_poly = interpfunction(np.log(struct['wave']))
					# Compute stellar continuum
					cont_fit_stel = np.subtract(struct['cont_fit'], cont_fit_poly)
					# Total flux fromd ifferent components
					cont_fit_tot = np.sum(struct['cont_fit'])


					if 'argspltcont' in self.initdat:
						pltcontfcn(struct, outfile + '_cnt',
								   compspec=np.array([cont_fit_stel, cont_fit_poly]),
								   title='Total',
								   comptitless=['stel. temp.',
											'ord. ' + str(self.add_poly_degree) +
											'Leg.poly'],
								   fitran=self.fitran, initdat=self.initdat,
								   **self.initdat['argspltcont'])
					else:
						pltcontfcn(struct, outfile + '_cnt',
								   compspec=np.array([cont_fit_stel, cont_fit_poly]),
								   title='Total', initdat=self.initdat,
								   comptitles=['stel. temp.', 'ord. ' +
											str(self.add_poly_degree) +
											' Leg. poly'],
								   fitran=self.fitran)


				else:
					if 'argspltcont' in self.initdat:
						pltcontfcn(struct, outfile + '_cnt',
								   fitran=self.fitran, initdat=self.initdat,
								   **self.initdat['argspltcont'])
					else:
						pltcontfcn(struct, outfile + '_cnt',
								   fitran=self.fitran, initdat=self.initdat)


				if 'argscontfit' in self.initdat.keys():
					if 'plot_decomp' in self.initdat['argscontfit'].keys():
						if self.initdat['argscontfit']['plot_decomp']:
							from q3dfit.plot_quest import plot_quest
							lam_lines = struct['linelist']['lines'].tolist()
							plot_quest(struct['wave'],
									   struct['cont_dat']+struct['emlin_dat'],
									   struct['cont_fit']+struct['emlin_fit'],
									   struct['ct_coeff'], self.initdat,
									   lines=lam_lines,
									   linespec=struct['emlin_fit'])

			if show:
				plt.show()

				# if decompose_qso_fit:
				# 	if argspltcont is not None or 'argspltcont' in self.initdat:
				# 		if argspltcont is None:		argspltcont=self.initdat['argspltcont']
				# 		pltcontfcn(struct_host, outfile + '_cnt_host',
				# 		   compspec=compspec, compfit=compfit,
				# 		   title='Host', fitran=self.fitran,
				# 		   **argspltcont)
				# 	else:
				# 		pltcontfcn(struct_host, outfile + '_cnt_host',
				# 		   compspec=compspec, title='Host', fitran=self.fitran)

				# 	if 'blrpar' in self.initdat['argscontfit']:
				# 					qsomod_blrnorm = np.median(qsomod) / \
				# 						max(qsomod_blronly)
				# 					compspec = np.array([qsomod_normonly,
				# 								qsomod_blronly * qsomod_blrnorm])
				# 					compfit = ['raw template', 'scattered\times' +
				# 							   str(qsomod_blrnorm)]
				# 	else:
				# 					compspec = [[qsomod_normonly]]
				# 					compfit = ['raw template']
				# 	if 'argspltcont' in self.initdat:
				# 		pltcontfcn(struct_qso, str(outfile) + '_cnt_qso',
				# 		   compspec=compspec, compfit=compfit,
				# 		   title='QSO', fitran=self.fitran,
				# 		   **self.initdat['argspltcont'])
				# 	else:
				# 		pltcontfcn(struct_qso, outfile + '_cnt_qso',
				# 		   compspec=compspec, comptitles=compfit,
				# 		   title='QSO', fitran=self.fitran)



	def plot_lin(self, show=False):
		# plot emission lines
		struct = self.outstr
		if not hasattr(self, 'listlines') or not hasattr(self, 'self.linepars') or not hasattr(self, 'self.tflux'):
			self.get_lin_props(do_save=False)

		labout = '{[outdir]}{[label]}_{:04d}_{:04d}_lin1'.\
			format(self.initdat, self.initdat, self.iuse+1, self.juse+1)
		outfile = labout

		if struct['noemlinfit'] == b'0' or struct['noemlinfit'] == False:
			# get line fit params
			self.linepars, self.tflux = \
				sepfitpars(self.listlines, struct['param'], struct['perror'],
				   # struct['parinfo'], self.tflux=True,
				   self.maxncomp, tflux=True,
				   doublets=self.lines_with_doublets)

			if 'nolines' not in self.linepars:
				if 'fcnpltlin' in self.initdat:
					fcnpltlin = self.initdat['fcnpltlin']
				else:
					fcnpltlin = 'pltlin'
				module = \
					importlib.import_module('q3dfit.' + 
											fcnpltlin)
				pltlinfcn = getattr(module, fcnpltlin)
				if 'argspltlin1' in self.initdat:
					pltlinfcn(struct, self.initdat['argspltlin1'],
							  outfile + '_lin1')
				if 'argspltlin2' in self.initdat:
					pltlinfcn(struct, self.initdat['argspltlin2'],
							  outfile + '_lin2')
		if show:	plt.show()


	def get_lin_props(self, do_save=True):

		bad = 1.0 * 10**99

		labout = '{[outdir]}{[label]}_{:04d}_{:04d}'.\
					format(self.initdat, self.initdat, self.iuse+1, self.juse+1)

		# Get emission line properties
		if 'noemlinfit' not in self.initdat or not self.initdat['noemlinfit']:
			# get linelist
			if not self.listlines:
				if 'argslinelist' in self.initdat:
					self.listlines = linelist(self.initdat['lines'], **self.initdat['argslinelist'])
				else:
					self.listlines = linelist(self.initdat['lines'])

			# if self.lines_with_doublets.shape is not None:
			# 	ndoublets = self.lines_with_doublets.shape[0]
			# else: 
			# 	ndoublets = 0

			# lines_with_doublets = self.initdat['lines']

			# for i in range(0, ndoublets):
			# 	if (self.lines_with_doublets[i][0] in self.listlines['name']) and \
			# 			(self.lines_with_doublets[i][1] in self.listlines['name']):
			# 		dkey = self.lines_with_doublets[i][0]+'+'+self.lines_with_doublets[i][1]
			# 		lines_with_doublets.append(dkey)

			# if 'argslinelist' in self.initdat:
			# 	self.listlines_with_doublets = linelist(lines_with_doublets,
			# 									   **self.initdat['argslinelist'])
			# else:
			# 	self.listlines_with_doublets = linelist(lines_with_doublets)


			# INITIALIZE LINE HASH
			if True:
				emlwav = dict()
				emlwaverr = dict()
				emlsig = dict()
				emlsigerr = dict()
				emlweq = dict()
				emlflx = dict()
				emlflxerr = dict()
				emlweq['ftot'] = dict()
				emlflx['ftot'] = dict()
				emlflxerr['ftot'] = dict()
				for k in range(0, self.initdat['maxncomp']):
					cstr = 'c' + str(k + 1)
					emlwav[cstr] = dict()
					emlwaverr[cstr] = dict()
					emlsig[cstr] = dict()
					emlsigerr[cstr] = dict()
					emlweq['f' + cstr] = dict()
					emlflx['f' + cstr] = dict()
					emlflxerr['f' + cstr] = dict()
					emlflx['f' + cstr + 'pk'] = dict()
					emlflxerr['f' + cstr + 'pk'] = dict()
				if self.lines_with_doublets is not None:
					for line in self.lines_with_doublets:
						emlweq['ftot'][line] = 0. + bad 
						emlflx['ftot'][line] = 0. + bad #np.zeros((self.cube.ncols, self.cube.nrows), dtype=float) + bad
						emlflxerr['ftot'][line] = 0. + bad
						for k in range(0, self.initdat['maxncomp']):
							cstr = 'c' + str(k + 1)
							emlwav[cstr][line] = 0. + bad  #np.zeros((self.cube.ncols, self.cube.nrows),
													#	  dtype=float) + bad
							emlwaverr[cstr][line] = 0. + bad
							emlsig[cstr][line] = 0. + bad
							emlsigerr[cstr][line] = 0. + bad
							emlweq['f'+cstr][line] = 0. + bad
							emlflx['f'+cstr][line] = 0. + bad
							emlflxerr['f'+cstr][line] = 0. + bad
							emlflx['f'+cstr+'pk'][line] = 0. + bad
							emlflxerr['f'+cstr+'pk'][line] = 0. + bad

			struct = self.outstr
			i = self.iuse
			j = self.juse

			if struct['noemlinfit'] == b'0' or struct['noemlinfit'] == False:
				# get line fit params
				self.linepars, self.tflux = \
					sepfitpars(self.listlines, struct['param'], struct['perror'],
					   self.maxncomp, tflux=True,
					   doublets=self.lines_with_doublets)

				# get correct number of components in this spaxel
				thisncomp = 0
				thisncompline = ''

# 				for line in self.lines_with_doublets:
				for line in self.linepars['flux'].columns:
					sigtmp = self.linepars['sigma'][line]
					fluxtmp = self.linepars['flux'][line]

					igd = np.where(sigtmp != 0 and sigtmp != bad and
								   fluxtmp != 0 and fluxtmp != bad)
					ctgd = len(igd)

					if ctgd > 0:
						thisncomp = ctgd
						thisncompline = line

					if ctgd > 0:
# 						emlflx['ftot'][line][i, j] = self.tflux['tflux'][line]
# 						emlflxerr['ftot'][line][i, j] = self.tflux['tfluxerr'][line]
						emlflx['ftot'][line] = self.linepars['flux'][line][0]
						emlflxerr['ftot'][line] = self.linepars['fluxerr'][line][0]


				if thisncomp == 1:
					isort = [0]
					if 'flipsort' in self.initdat:
						# if flipsort[i, j]:
						if self.initdat['flipsort']:
							print('Flipsort set, but ' +
								  'only 1 component. Setting to 2 components' +
								  ' and flipping anyway.')
							isort = [0, 1]  # flipped
				elif thisncomp >= 2:
					# sort components
					# igd = np.arange(thisncomp)
					igd = np.arange(self.initdat['maxncomp'])
					# indices = np.arange(initdat['maxncomp'])
					sigtmp = self.linepars['sigma'][:, thisncompline]
					fluxtmp = self.linepars['flux'][:, thisncompline]
					if 'sorttype' not in self.initdat:
						isort = sigtmp[igd].sort()
					elif self.initdat['sorttype'] == 'wave':
						isort = self.linepars['wave'][igd, line].sort()  # reversed?
					elif self.initdat['sorttype'] == 'reversewave':
						isort = self.linepars['wave'][igd, line].sort(reverse=True)

					if 'flipsort' in self.initdat:
						# if flipsort[i,j ] is not None:
						if self.initdat['flipsort']:
							isort = isort.sort(reverse=True)

				if thisncomp > 0:
					# for line in lines_with_doublets:
					for line in self.linepars['flux'].columns:
						kcomp = 1
						for sindex in isort:
							cstr = 'c' + str(kcomp)
							emlwav[cstr][line] \
								= self.linepars['wave'][line].data[sindex]
							emlwaverr[cstr][line] \
								= self.linepars['waveerr'][line].data[sindex]
							emlsig[cstr][line] \
								= self.linepars['sigma'][line].data[sindex]
							emlsigerr[cstr][line] \
								= self.linepars['sigmaerr'][line].data[sindex]
		#                            emlweq['f' + cstr][line][i, j] \
		#                                = lineweqs['comp'][line].data[sindex]
							emlflx['f' + cstr][line] \
								= self.linepars['flux'][line].data[sindex]
							emlflxerr['f' + cstr][line] \
								= self.linepars['fluxerr'][line].data[sindex]
							emlflx['f' + cstr + 'pk'][line] \
								= self.linepars['fluxpk'][line].data[sindex]
							emlflxerr['f' + cstr + 'pk'][line] \
								= self.linepars['fluxpkerr'][line].data[sindex]
							kcomp += 1		


			self.emlwav = emlwav
			self.emlwaverr=emlwaverr
			self.emlsig=emlsig
			self.emlsigerr=emlsigerr
			self.emlflx=emlflx 
			self.emlflxerr=emlflxerr
			self.emlweq=emlweq

			if do_save:
				# Save emission line dictionary
				np.savez(labout+'.lin.npz',
					 emlwav=emlwav, emlwaverr=emlwaverr,
					 emlsig=emlsig, emlsigerr=emlsigerr,
					 emlflx=emlflx, emlflxerr=emlflxerr,
					 emlweq=emlweq)







