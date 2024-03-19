
import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import copy
from scipy import constants
from CBPlot import do_plot

def gaussian(x, amplitude, mean, stddev):
	return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
	return (
		A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
		A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)) )

def triple_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
	return (
		A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
		A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)) +
		A3 * np.exp(-(x - mu3)**2 / (2 * sigma3**2)) )

def triple_gaussian_B(x, A1, sigma1, A2, sigma2, A3, sigma3):
	mu1 = 0
	mu2 = 0
	mu3 = 0	
	return ( A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)) + A3 * np.exp(-(x - mu3)**2 / (2 * sigma3**2)) )


def get_avpoint_weighted(lam_in, F_in, huse_in):
	del_lam = np.append(lam_in[huse_in][1:]-lam_in[huse_in][:-1], lam_in[huse_in][-1]-lam_in[huse_in][-2])
	F_av = np.average(F_in[huse_in], weights=del_lam)
	lam_av = 0.5*(lam_in[huse_in][0]+lam_in[huse_in][-1])

	return lam_av, F_av


def Read_in_L_FWHM_dbl(file_Ftot_map, file_Ftot_map2, file_line_npz, line, line2, fluxnorm, dist_cm, ncol=19, nrow=19, ncomp_broad=-1, do_print=False, do_total_profile=False, lam0_micron=-1, return_velcen=False, return_profile=False):
	hdul_fit = fits.open(file_Ftot_map)
	hdul_fit2 = fits.open(file_Ftot_map2)
	Ftot = (hdul_fit[0].data[ncol, nrow] + hdul_fit2[0].data[ncol, nrow]) * fluxnorm    # [erg/s/cm^2]
	data = np.load(file_line_npz, allow_pickle=True)

	if do_total_profile or (isinstance(ncomp_broad, list) or isinstance(ncomp_broad, np.ndarray)) or ncomp_broad>0:
		F_out = Ftot
		if lam0_micron<0.:
			print('Need to specify the rest-frame wavelength of the line for getting the FWHM of the summed line profile... Halting.')
			import sys; sys.exit()
		ncomp = data['emlncomp'][()][line][ncol, nrow]
		xx = np.linspace(-12000., 12000., 500)
		profile = np.zeros(len(xx))
		profile_broadest = np.zeros(len(xx))
		profile_narrow = np.zeros(len(xx))
		if do_total_profile:
			ncomp_arr = np.arange(ncomp)+1
		elif (isinstance(ncomp_broad, list) or isinstance(ncomp_broad, np.ndarray)):
			ncomp_arr = ncomp_broad
		else:
			ncomp_arr = [ncomp_broad]
		lines_arr = np.array([line, line2])


		Fsum = 0.
		eFsum = 0.

		F_avg_arr = np.zeros(ncomp)
		eF_avg_arr = np.zeros(ncomp)
		cwv_arr = np.zeros(ncomp)
		ecwv_arr = np.zeros(ncomp)
		profile_oneline = np.zeros(len(xx))
		for line_i in lines_arr:
			Fc2_nonorm = data['emlflx'][()]['fc2'][line_i][ncol, nrow]
			Fc1_nonorm = data['emlflx'][()]['fc1'][line_i][ncol, nrow]
			eFsum_i = 0.
			for i in ncomp_arr: # range(1, ncomp+1):
				comp_i = 'c'+str(i)
				cenwave_i = data['emlwav'][()][comp_i][line_i][ncol, nrow]
				velcen_i, velcen_i = wave_to_vel(np.array([cenwave_i]), np.array([cenwave_i]), lam0_micron*(1.+zred), zred)
				sig_i = data['emlsig'][()][comp_i][line_i][ncol, nrow]
				Fpk_nonorm_i = data['emlflx'][()]['f'+comp_i+'pk'][line_i][ncol, nrow] # Peak flux - don't care about the norm, we just want to get the width of the final profile
				F_nonorm_i = data['emlflx'][()]['f'+comp_i][line_i][ncol, nrow]# Flux - don't care about the norm, we just want to get the width of the final profile
				profile += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i)
				profile_broadest += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i+data['emlsigerr'][()][comp_i][line_i][ncol, nrow])
				profile_narrow += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i-data['emlsigerr'][()][comp_i][line_i][ncol, nrow])
				if line_i=='MgII2803':
					profile_oneline += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i)
				F_comp = Ftot * F_nonorm_i/(Fc1_nonorm+Fc2_nonorm)
				eF_comp = F_comp * data['emlflxerr'][()]['f'+comp_i][line_i][ncol, nrow]/F_nonorm_i

				cwv_arr[i-1] += data['emlwav'][()][comp_i][line_i][ncol, nrow]
				ecwv_arr[i-1] += data['emlwaverr'][()][comp_i][line_i][ncol, nrow]

				F_avg_arr[i-1] += F_comp
				eF_avg_arr[i-1] += eF_comp
				if do_total_profile:
					Fsum += F_comp
					eFsum_i += eF_comp**2
			eFsum_i = np.sqrt(eFsum_i)
			eFsum += eFsum_i
		Fsum = Fsum/2.
		eFsum = eFsum/2.
		F_avg_arr = F_avg_arr/2.
		eF_avg_arr = eF_avg_arr/2.
		cwv_arr /= 2.
		ecwv_arr /= 2.

		# F_comp_avg = np.mean(F_arr)
		# eF_comp_avg = np.mean(eF_arr)

		# if 'MgII' in file_Ftot_map:
		#     breakpoint()

		half_max_flux = np.max(profile) / 2.0
		left_idx = np.argmin(np.abs(profile[:np.argmax(profile)] - half_max_flux))
		right_idx = np.argmin(np.abs(profile[np.argmax(profile):] - half_max_flux)) + np.argmax(profile)
		FWHM_out = xx[right_idx] - xx[left_idx]

		half_max_flux = np.max(profile_broadest) / 2.0
		left_idx = np.argmin(np.abs(profile_broadest[:np.argmax(profile_broadest)] - half_max_flux))
		right_idx = np.argmin(np.abs(profile_broadest[np.argmax(profile_broadest):] - half_max_flux)) + np.argmax(profile_broadest)
		FWHM_broadest_out = xx[right_idx] - xx[left_idx]

		half_max_flux = np.max(profile_narrow) / 2.0
		left_idx = np.argmin(np.abs(profile_narrow[:np.argmax(profile_narrow)] - half_max_flux))
		right_idx = np.argmin(np.abs(profile_narrow[np.argmax(profile_narrow):] - half_max_flux)) + np.argmax(profile_narrow)
		FWHM_narrow_out = xx[right_idx] - xx[left_idx]
		if True: #ncomp>1:
			plt.figure()
			plt.plot(xx, profile)
			plt.axhline(np.max(profile)/2., color='grey', zorder=0, linestyle='--', alpha=0.7)
			plt.xlabel('Velocity')
			plt.title('Line profile -- '+file_line_npz.split('/')[-2]+'_'+file_line_npz.split('/')[-1])
			# plt.annotate('{}'.format(FWHM_out))
			plt.savefig('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652/plots_CB/' + file_line_npz.split('/')[-2]+'_'+file_line_npz.split('/')[-1].replace('.line.npz', ''))
			plt.close()

	L_out = F_out * (4*np.pi*dist_cm**2)
	if return_profile:
		return L_out, FWHM_out, FWHM_broadest_out, FWHM_narrow_out, F_avg_arr, eF_avg_arr, Fsum, eFsum, [xx, profile_oneline]
	elif not return_velcen:
		return L_out, FWHM_out, FWHM_broadest_out, FWHM_narrow_out, F_avg_arr, eF_avg_arr, Fsum, eFsum
	else:
		return L_out, FWHM_out, FWHM_broadest_out, FWHM_narrow_out, F_avg_arr, eF_avg_arr, Fsum, eFsum, cwv_arr, ecwv_arr

def Read_in_L_FWHM_B(file_Ftot_map, file_line_npz, line, fluxnorm, dist_cm, ncol=19, nrow=19, ncomp_broad=-1, do_print=False, do_total_profile=False, lam0_micron=-1, return_Fsum=False):
    hdul_fit = fits.open(file_Ftot_map)
    Ftot = hdul_fit[0].data[ncol, nrow] * fluxnorm    # [erg/s/cm^2]
    data = np.load(file_line_npz, allow_pickle=True)
    Fc1 = data['emlflx'][()]['fc1'][line][ncol, nrow] * fluxnorm
    eFc1 = data['emlflxerr'][()]['fc1'][line][ncol, nrow] * fluxnorm

    if do_total_profile:
        if (isinstance(ncomp_broad, list) or isinstance(ncomp_broad, np.ndarray)) or ncomp_broad>0:
            print('do_total_profile is switched on. Using the total summed profile. Breakpoint...')
            breakpoint()
    if do_total_profile or (isinstance(ncomp_broad, list) or isinstance(ncomp_broad, np.ndarray)):
        F_out = Ftot
        if lam0_micron<0.:
            print('Need to specify the rest-frame wavelength of the line for getting the FWHM of the summed line profile... Halting.')
            import sys; sys.exit()
        ncomp = data['emlncomp'][()][line][ncol, nrow]
        xx = np.linspace(-12000., 12000., 500)
        profile = np.zeros(len(xx))
        profile_broadest = np.zeros(len(xx))
        profile_narrow = np.zeros(len(xx))
        if do_total_profile:
            ncomp_arr = np.arange(ncomp)+1
        else:
            ncomp_arr = ncomp_broad

        Fsum = 0.
        eFsum = 0.
        F_nonorm_sum = 0.
        for i in ncomp_arr:
            F_nonorm_sum += data['emlflx'][()]['fc'+str(i)][line][ncol, nrow]
        for i in ncomp_arr: # range(1, ncomp+1):
            comp_i = 'c'+str(i)
            cenwave_i = data['emlwav'][()][comp_i][line][ncol, nrow]
            velcen_i, velcen_i = wave_to_vel(np.array([cenwave_i]), np.array([cenwave_i]), lam0_micron*(1.+zred), zred)
            sig_i = data['emlsig'][()][comp_i][line][ncol, nrow]
            Fpk_nonorm_i = data['emlflx'][()]['f'+comp_i+'pk'][line][ncol, nrow] # Peak flux - don't care about the norm, we just want to get the width of the final profile
            F_nonorm_i = data['emlflx'][()]['f'+comp_i][line][ncol, nrow]# Flux - don't care about the norm, we just want to get the width of the final profile
            profile += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i)
            profile_broadest += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i+data['emlsigerr'][()][comp_i][line][ncol, nrow])
            profile_narrow += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i-data['emlsigerr'][()][comp_i][line][ncol, nrow])

            if do_total_profile:
                F_comp = Ftot * F_nonorm_i/F_nonorm_sum
                eF_comp = F_comp * data['emlflxerr'][()]['f'+comp_i][line][ncol, nrow]/F_nonorm_i
                Fsum += F_comp
                eFsum += eF_comp**2
        eFsum = np.sqrt(eFsum)
    
        half_max_flux = np.max(profile) / 2.0
        left_idx = np.argmin(np.abs(profile[:np.argmax(profile)] - half_max_flux))
        right_idx = np.argmin(np.abs(profile[np.argmax(profile):] - half_max_flux)) + np.argmax(profile)
        FWHM_out = xx[right_idx] - xx[left_idx]

        half_max_flux = np.max(profile_broadest) / 2.0
        left_idx = np.argmin(np.abs(profile_broadest[:np.argmax(profile_broadest)] - half_max_flux))
        right_idx = np.argmin(np.abs(profile_broadest[np.argmax(profile_broadest):] - half_max_flux)) + np.argmax(profile_broadest)
        FWHM_broadest_out = xx[right_idx] - xx[left_idx]

        half_max_flux = np.max(profile_narrow) / 2.0
        left_idx = np.argmin(np.abs(profile_narrow[:np.argmax(profile_narrow)] - half_max_flux))
        right_idx = np.argmin(np.abs(profile_narrow[np.argmax(profile_narrow):] - half_max_flux)) + np.argmax(profile_narrow)
        FWHM_narrow_out = xx[right_idx] - xx[left_idx]
        if True: #ncomp>1:
            plt.figure()
            plt.plot(xx, profile)
            plt.axhline(np.max(profile)/2., color='grey', zorder=0, linestyle='--', alpha=0.7)
            plt.xlabel('Velocity')
            plt.title('Line profile -- '+file_line_npz.split('/')[-2]+'_'+file_line_npz.split('/')[-1])
            # plt.annotate('{}'.format(FWHM_out))
            plt.savefig('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652/plots_CB/' + file_line_npz.split('/')[-2]+'_'+file_line_npz.split('/')[-1].replace('.line.npz', ''))
            plt.close()
    elif ncomp_broad<0.:
        try:
            Fc2 = data['emlflx'][()]['fc2'][line][ncol, nrow] * fluxnorm
            FWHM_broad = data['emlsig'][()]['c2'][line][ncol, nrow] * 2.355 #[km/s]
            F_broad = Ftot * (Fc2/(Fc1+Fc2))
            print('FWHM values: {:.2f}'.format(FWHM_broad, data['emlsig'][()]['c2'][line][ncol, nrow] * 2.355))
            FWHM_broadest_out = FWHM_broad + data['emlsigerr'][()]['c2'][line][ncol, nrow] * 2.355
            FWHM_narrow_out = FWHM_broad - data['emlsigerr'][()]['c2'][line][ncol, nrow] * 2.355
        except:
            FWHM_broad = data['emlsig'][()]['c1'][line][ncol, nrow] * 2.355 #[km/s]
            FWHM_broadest_out = FWHM_broad + data['emlsigerr'][()]['c1'][line][ncol, nrow] * 2.355
            FWHM_narrow_out = FWHM_broad - data['emlsigerr'][()]['c1'][line][ncol, nrow] * 2.355
            F_broad = Ftot 
        F_out = F_broad
        FWHM_out = FWHM_broad


    else:
        FWHM_broad = data['emlsig'][()]['c'+str(ncomp_broad)][line][ncol, nrow] * 2.355 #[km/s]
        FWHM_broadest_out = FWHM_broad + data['emlsigerr'][()]['c'+str(ncomp_broad)][line][ncol, nrow] * 2.355
        FWHM_narrow_out = FWHM_broad - data['emlsigerr'][()]['c'+str(ncomp_broad)][line][ncol, nrow] * 2.355
        comps_arr = [el for el in data['emlflx'][()] if 'fc' in el and 'pk' not in el]
        denom = 0.
        for comp_i in comps_arr:
            denom += data['emlflx'][()][comp_i][line][ncol, nrow] * fluxnorm
        F_broad = Ftot * data['emlflx'][()]['fc'+str(ncomp_broad)][line][ncol, nrow] * fluxnorm / denom
        F_out = F_broad
        FWHM_out = FWHM_broad

    L_out = F_out * (4*np.pi*dist_cm**2)

    if not return_Fsum:
        return L_out, FWHM_out, FWHM_broadest_out, FWHM_narrow_out
    else:
        return L_out, FWHM_out, FWHM_broadest_out, FWHM_narrow_out, Fsum, eFsum



def Read_in_L_FWHM(file_Ftot_map, file_line_npz, line, fluxnorm, dist_cm, ncol=19, nrow=19, ncomp_broad=-1, do_print=False, do_total_profile=False, lam0_micron=-1, return_tot_profile=False):
	hdul_fit = fits.open(file_Ftot_map)
	Ftot = hdul_fit[0].data[ncol, nrow] * fluxnorm    # [erg/s/cm^2]
	data = np.load(file_line_npz, allow_pickle=True)
	Fc1 = data['emlflx'][()]['fc1'][line][ncol, nrow]
	xx = np.linspace(-12000., 12000., 500)
	profile = np.zeros(len(xx))
	if ncomp_broad>0 and do_total_profile:
		print('do_total_profile is switched on. Using the total summed profile. Breakpoint...')
		breakpoint()
	if do_total_profile:
		F_out = Ftot
		if lam0_micron<0.:
			print('Need to specify the rest-frame wavelength of the line for getting the FWHM of the total line profile... Halting.')
			import sys; sys.exit()
		ncomp = data['emlncomp'][()][line][ncol, nrow]
		for i in range(1, ncomp+1):
			comp_i = 'c'+str(i)
			cenwave_i = data['emlwav'][()][comp_i][line][ncol, nrow]
			velcen_i, velcen_i = wave_to_vel(np.array([cenwave_i]), np.array([cenwave_i]), lam0_micron*(1.+zred), zred)
			sig_i = data['emlsig'][()][comp_i][line][ncol, nrow]
			Fpk_nonorm_i = data['emlflx'][()]['f'+comp_i+'pk'][line][ncol, nrow] # Peak flux - don't care about the norm, we just want to get the width of the final profile
			F_nonorm_i = data['emlflx'][()]['f'+comp_i][line][ncol, nrow] # Peak flux - don't care about the norm, we just want to get the width of the final profile
			profile += gaussian(xx, Fpk_nonorm_i, velcen_i, sig_i)
		half_max_flux = np.max(profile) / 2.0
		left_idx = np.argmin(np.abs(profile[:np.argmax(profile)] - half_max_flux))
		right_idx = np.argmin(np.abs(profile[np.argmax(profile):] - half_max_flux)) + np.argmax(profile)
		FWHM_out = xx[right_idx] - xx[left_idx]
		if True: #ncomp>1:
			plt.figure()
			plt.plot(xx, profile)
			plt.axhline(np.max(profile)/2., color='grey', zorder=0, linestyle='--', alpha=0.7)
			plt.xlabel('Velocity')
			plt.title('Line profile -- '+file_line_npz.split('/')[-2]+'_'+file_line_npz.split('/')[-1])
			# plt.annotate('{}'.format(FWHM_out))
			plt.savefig('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652/plots_CB/' + file_line_npz.split('/')[-2]+'_'+file_line_npz.split('/')[-1].replace('.line.npz', ''))
			plt.close()
	elif ncomp_broad<0.:
		try:
			Fc2 = data['emlflx'][()]['fc2'][line][ncol, nrow]
			FWHM_broad = data['emlsig'][()]['c2'][line][ncol, nrow] * 2.355 #[km/s]
			F_broad = Ftot * (Fc2/(Fc1+Fc2))
			print('FWHM values: {:.2f}'.format(FWHM_broad, data['emlsig'][()]['c2'][line][ncol, nrow] * 2.355))
		except:
			FWHM_broad = data['emlsig'][()]['c1'][line][ncol, nrow] * 2.355 #[km/s]
			F_broad = Ftot 
		F_out = F_broad
		FWHM_out = FWHM_broad
	else:
		FWHM_broad = data['emlsig'][()]['c'+str(ncomp_broad)][line][ncol, nrow] * 2.355 #[km/s]
		comps_arr = [el for el in data['emlflx'][()] if 'fc' in el and 'pk' not in el]
		denom = 0.
		for comp_i in comps_arr:
			denom += data['emlflx'][()][comp_i][line][ncol, nrow]
		F_broad = Ftot * data['emlflx'][()]['fc'+str(ncomp_broad)][line][ncol, nrow] / denom
		F_out = F_broad
		FWHM_out = FWHM_broad

	L_out = F_out * (4*np.pi*dist_cm**2)
	if not return_tot_profile:
		return L_out, FWHM_out
	else:
		return L_out, FWHM_out, [xx, profile]


def subtract_cont(lam_in, F_in, huse_ll, huse_rr, huse_in=[], weighted=False, do_median=False):
	'''
	This function is used to subtract the continuum below a line profile
	huse_ll, huse_rr: left and right window for estimating the continuum to the left/right of the line
	'''
	if len(huse_in)==0:
		huse_in = np.ones(len(lam_in)).astype(bool)

	if weighted:
		lam_ll, F_ll = get_avpoint_weighted(lam_in, F_in, huse_ll)
		lam_rr, F_rr = get_avpoint_weighted(lam_in, F_in, huse_rr)
	elif do_median:
		F_ll = np.median(F_in[huse_ll])
		lam_ll = np.median(lam_in[huse_ll])
		F_rr = np.median(F_in[huse_rr])
		lam_rr = np.median(lam_in[huse_rr])
	else:
		F_ll = np.average(F_in[huse_ll])
		lam_ll = np.average(lam_in[huse_ll])
		F_rr = np.average(F_in[huse_rr])
		lam_rr = np.average(lam_in[huse_rr])

	slope_cont = (F_rr-F_ll)/(lam_rr-lam_ll)
	intercept_cont = 0.5 * (F_ll+F_rr - slope_cont*(lam_rr+lam_ll) )


	return F_in[huse_in] - (slope_cont*lam_in[huse_in] + intercept_cont)
	


def wave_to_vel(lam_obs_left, lam_obs_right, lam_obs_cen, zred):
	# Get linewidth in velocity units [km/s], starting from wavelengths
	#
	# In: de-redshifted wavelengths (left, right, centre)  [micron],  redshift
	# Out: velocity left and right [km/s]
	# 

	lam0_start = lam_obs_left / (1.+zred)
	lam0_end = lam_obs_right / (1.+zred)
	lam0_cen = lam_obs_cen / (1.+zred)

	freq_start = (constants.c * 1e-3) / (lam0_start * 1e-6)  # meters
	freq_end = (constants.c * 1e-3) / (lam0_end * 1e-6)      # meters
	freq_cen =  (constants.c * 1e-3) / (lam0_cen * 1e-6)

	vel_at_freq = lambda nu, nu_cen: (nu_cen**2-nu**2)/(nu_cen**2+nu**2) * (constants.c * 1e-3)

	vel_left = vel_at_freq(freq_start, freq_cen)
	vel_right = vel_at_freq(freq_end, freq_cen)

	return vel_left, vel_right



# Open data - nuclear spectrum of NIRSpec NRS data, and full NIRSpec cube
path_NRS_AV = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/DATA/J1652/NIRSpec_AVayner_lines/'
spec_NRS_PSF = 'J1652_PSF_spec.fits'
hdul_NRS_PSF = fits.open(path_NRS_AV+spec_NRS_PSF)
F_NRS = hdul_NRS_PSF[0].data
cube_NIRSpec = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/DATA/J1652/NIRSpec_v2022dec8_reprojectcomb_no_leakcal_emsm/v2022dec8_reprojectcomb_no_leakcal_emsm/NRS1NRS2_s3d_cgs_emsm.fits'
hdul_NRScube = fits.open(cube_NIRSpec)
lam_NRScube = ((np.arange(hdul_NRScube[0].data.shape[0]) + 1.0) - hdul_NRScube[0].header['CRPIX3']) * hdul_NRScube[0].header['CDELT3'] + hdul_NRScube[0].header['CRVAL3']
Ha_cut = np.logical_and(lam_NRScube > 2.475, lam_NRScube < 2.7)
huse_interp = np.logical_or( np.logical_and(lam_NRScube > 2.475, lam_NRScube < 2.5),  np.logical_and(lam_NRScube > 2.675, lam_NRScube < 2.7))

Hb_OIII_cut = np.logical_and(lam_NRScube>1.87, lam_NRScube<2.05)



file_SDSS = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/DATA/J1652/SDSS_spectrum/spec-4176-55682-0370.fits'
hdul_SDSS = fits.open(file_SDSS)
lam_SDSS = 10**hdul_SDSS[1].data['LOGLAM'] * 1e-4  # micron
F_SDSS = hdul_SDSS[1].data['FLUX']	# [1e-17 erg/s/cm^2]
lam_CIV = 1549 * 1e-4
huse_CIV = np.logical_and(lam_SDSS>0.6, lam_SDSS<0.62)
huse_contfit = np.logical_and(np.logical_and(lam_SDSS>0.57, lam_SDSS<0.64), np.invert(huse_CIV))
popt = np.polyfit(lam_SDSS[huse_contfit], F_SDSS[huse_contfit], 3)
F_SDSS = F_SDSS - (popt[0]*lam_SDSS**3 + popt[1]*lam_SDSS**2 + popt[2]*lam_SDSS + popt[3])


do_save_NRS_as_cube = False
if do_save_NRS_as_cube:
	hdul_NRS_new = copy.copy(hdul_NRScube)
	hdul_NRS_new[0].data = np.broadcast_to(hdul_NRS_PSF[0].data[:, np.newaxis, np.newaxis], hdul_NRScube[0].data.shape)
	hdul_NRS_new[1].data = np.broadcast_to(hdul_NRS_PSF[1].data[:, np.newaxis, np.newaxis],  hdul_NRScube[1].data.shape)
	hdul_NRS_new[2].data = np.zeros(hdul_NRScube[2].data.shape) # np.broadcast_to(hdul_NRS_PSF[2].data[:, np.newaxis, np.newaxis], hdul_NRScube[2].data.shape)
	hdul_list = fits.HDUList([hdul_NRS_new[0], hdul_NRS_new[1], hdul_NRS_new[2]])
	hdul_list.writeto(path_NRS_AV + 'J1652_NIRSpec_PSF_spec_CUBE.fits', overwrite=True)




# Open MRS data
path_MRScube = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/DATA_Reduction/MIRI_MRS_reduction/J1652_reduction/Spilker_DR/DRP_v1_10_2/poststage3_BadPixOff_masterbkg_IFUalign_OutlierOnStg3/'
hdul_MRScube = fits.open(path_MRScube + 'Level3_ch1-short_s3d_bkgsubtracted.fits')
lam_MRScube = ((np.arange(hdul_MRScube[1].data.shape[0]) + 1.0) - hdul_MRScube[1].header['CRPIX3']) * hdul_MRScube[1].header['CDELT3'] + hdul_MRScube[1].header['CRVAL3']

plotdir = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652/plots_CB/'

# spec_conv = np.load('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/q3dfit/data/questfit_templates/J1652_qso.npy', allow_pickle=True)[()]['flux']
# spec_conv = np.load('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/q3dfit/data/questfit_templates/J1652_qso.npy', allow_pickle=True)[()]['flux']
hdul_conv = fits.open('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_PSFfit/Pab_BLR_summed_1D.fits')
spec_conv = hdul_conv[1].data
zred = 2.94


wave_Ha_micron = 6562.8 * 1e-4
wave_Hb_micron = 4861. * 1e-4
wave_Pab_micron = 12822.16 * 1e-4




### Get Halpha profile from NIRSpec NRS
# Subtract continuum via interpolation
huse_ll = np.logical_and(lam_NRScube>2.475, lam_NRScube<2.5)
huse_rr = np.logical_and(lam_NRScube>2.675, lam_NRScube<2.7)
F_Ha_sub = subtract_cont(lam_NRScube, F_NRS, huse_ll, huse_rr)

initial_guess = [3.5, 2.59, 0.05]  # Initial guess for amplitude, mean, and stddev
popt, pcov = curve_fit(gaussian, lam_NRScube[Ha_cut], F_NRS[Ha_cut], p0=initial_guess)

plt.figure()
plt.plot(lam_NRScube[Ha_cut], F_NRS[Ha_cut])
plt.plot(lam_NRScube[huse_interp], F_NRS[huse_interp])
plt.plot(lam_NRScube[Ha_cut], F_Ha_sub[Ha_cut])
# plt.show()
plt.close()

# Place on wavelength axis
lam_Ha_peak = lam_NRScube[Ha_cut][np.argmax(F_Ha_sub[Ha_cut])]
# vel_Ha, vel_Ha_right = wave_to_vel(lam_NRScube[Ha_cut], lam_NRScube[Ha_cut], lam_Ha_peak, zred)
vel_Ha, vel_Ha_right = wave_to_vel(lam_NRScube[Ha_cut], lam_NRScube[Ha_cut], wave_Ha_micron*(1.+zred), zred)
plt.figure()
plt.plot(vel_Ha, F_Ha_sub[Ha_cut])
# plt.show()
plt.close()



### Get CIV profile from SDSS
initial_guess = [13, 0.61, 0.1]  # Initial guess for amplitude, mean, and stddev
popt, pcov = curve_fit(gaussian, lam_SDSS[huse_CIV], F_SDSS[huse_CIV], p0=initial_guess)
lam_CIV_peak = lam_SDSS[huse_CIV][np.argmax(F_SDSS[huse_CIV])]
F_SDSS_norm = F_SDSS/popt[0]

# Place on wavelength axis
lam_CIV_peak = lam_SDSS[huse_CIV][np.argmax(F_SDSS[huse_CIV])]
vel_CIV, vel_CIV_right = wave_to_vel(lam_SDSS[huse_CIV], lam_SDSS[huse_CIV], lam_CIV_peak, zred)
plt.figure()
plt.plot(vel_CIV, F_SDSS[huse_CIV])
# plt.show()
plt.close()


### Get Paschen beta profile from MIRI MRS
# Get brightest spaxel
med_img = np.median(hdul_MRScube[1].data, axis=0)
med_img_B = med_img.copy()
med_img_B[np.isnan(med_img_B)] = -np.inf
idx_brightest = np.unravel_index(np.argmax(med_img_B), med_img_B.shape)
spec_central = hdul_MRScube[1].data[:, idx_brightest[0], idx_brightest[1]]

save_PSFonly = False
if save_PSFonly:
		hdul_new = hdul_MRScube.copy()
		hdul_new[1].data = np.broadcast_to(spec_central[:, np.newaxis, np.newaxis], hdul_MRScube[1].data.shape)
		hdul_new[2].data = np.broadcast_to(hdul_MRScube[2].data[:, idx_brightest[0], idx_brightest[1]][:, np.newaxis, np.newaxis],  hdul_MRScube[2].data.shape)
		hdul_new[3].data = np.broadcast_to(hdul_MRScube[3].data[:, idx_brightest[0], idx_brightest[1]][:, np.newaxis, np.newaxis],  hdul_MRScube[3].data.shape)
		hdul_new[4].data = np.broadcast_to(hdul_MRScube[4].data[:, idx_brightest[0], idx_brightest[1]][:, np.newaxis, np.newaxis],  hdul_MRScube[4].data.shape)
		hdul_list = fits.HDUList([hdul_new[0], hdul_new[1], hdul_new[2], hdul_new[4], hdul_new[5]])
		hdul_list.writeto(path_MRScube + 'Ch1_PSF_only.fits', overwrite=True)


huse_ll = (lam_MRScube<4.95)
huse_rr = np.logical_and(lam_MRScube>5.165, lam_MRScube<5.2)
F_Pab_sub = subtract_cont(lam_MRScube, spec_central, huse_ll, huse_rr, do_median=True)
F_Pab_conv_sub = subtract_cont(lam_MRScube, spec_conv, huse_ll, huse_rr, do_median=True)
Pab_cut = np.logical_and(lam_MRScube>4.9, lam_MRScube<5.18)

plt.plot(lam_MRScube, spec_central)
plt.plot(lam_MRScube[huse_ll], spec_central[huse_ll])
plt.plot(lam_MRScube[huse_rr], spec_central[huse_rr])
# plt.show()
plt.close()


# Get peak flux position
initial_guess = [180, 5.07, 0.1]  # Initial guess for amplitude, mean, and stddev
popt_Pab, pcov = curve_fit(gaussian, lam_MRScube[Pab_cut], F_Pab_sub[Pab_cut], p0=initial_guess)
initial_guess = [1.17, 5.07, 0.1]  # Initial guess for amplitude, mean, and stddev
popt_Pab_conv, pcov = curve_fit(gaussian, lam_MRScube[Pab_cut], F_Pab_conv_sub[Pab_cut], p0=initial_guess)


plt.plot(lam_MRScube[Pab_cut], gaussian(lam_MRScube[Pab_cut], *popt_Pab))
plt.plot(lam_MRScube[Pab_cut], F_Pab_sub[Pab_cut])
# plt.show()
plt.close()


# Place on wavelength axis
lam_Pab_peak = popt_Pab[1]
vel_Pab, vel_Pab_right = wave_to_vel(lam_MRScube[Pab_cut], lam_MRScube[Pab_cut], lam_Pab_peak, zred)
plt.plot(vel_Pab, F_Pab_sub[Pab_cut])
# plt.show()
plt.close()
lam_Pab_conv_peak = popt_Pab_conv[1]
vel_Pab_conv, vel_Pab_conv_right = wave_to_vel(lam_MRScube[Pab_cut], lam_MRScube[Pab_cut], lam_Pab_conv_peak, zred)
vel_Pab_conv, vel_Pab_conv_right = wave_to_vel(lam_MRScube[Pab_cut], lam_MRScube[Pab_cut], wave_Pab_micron*(1.+zred), zred)

# # Fit double Gauss to Pab profile in velocity space   via double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2)
# initial_guess = [1., 0., 1200., 0.25, 0., 3000.]  # Initial guess for amplitude, mean, and stddev
# popt_Pab_vel, pcov = curve_fit(double_gaussian, vel_Pab_conv, F_Pab_conv_sub[Pab_cut], p0=initial_guess)

# initial_guess = [0.5, 0., 1200., 0.5, 0., 1000., 0.25, 0., 3000. ]  # Initial guess for amplitude, mean, and stddev
# initial_guess = [0.7, 1200., 0.5, 1000., 0.25, 3600. ]  # Initial guess for amplitude, mean, and stddev
# popt_Pab_vel, pcov = curve_fit(triple_gaussian_B, vel_Pab_conv, F_Pab_conv_sub[Pab_cut], p0=initial_guess, maxfev=5000)

# # Spectrally smooth Pab profile using a moving average
# # Function to perform spectral smoothing using a moving average
# def spectral_smooth(data, window_size):
#     smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
#     return smoothed_data


### Overplot Ha and Pab
plt.figure()
plt.axhline(0., color='grey', linestyle='--')
plt.plot(vel_Ha, F_Ha_sub[Ha_cut]/F_Ha_sub[Ha_cut].max(), label=r'$\mathrm{H}_\alpha$', color='k')
#plt.plot(vel_Pab, F_Pab_sub[Pab_cut] / popt_Pab[0], label='Paschen beta')
plt.plot(vel_Pab_conv, F_Pab_conv_sub[Pab_cut] / popt_Pab_conv[0], color='tab:red', label=r'$\mathrm{Pa}_\beta$')
# plt.plot(vel_CIV, F_SDSS_norm[huse_CIV], color='dodgerblue', label=r'$\mathrm{C_{IV}}$')
# plt.plot(vel_Pab_conv, triple_gaussian_B(vel_Pab_conv, *popt_Pab_vel) )
plt.axvline(0., color='grey', alpha=0.7, linestyle='--', zorder=0)
plt.xlabel('Velocity [km/s]')
plt.xlim(-10000, 10000)
plt.ylim(-0.3, 1.8)
plt.ylabel('Flux (normalised)')
plt.legend(fontsize=16)
do_plot.adjust_plot(plt, C_xlabelsize_figsize=1.)
plt.savefig(plotdir + 'spec/' + 'Ha_vs_Pab_line_profile')
# plt.show()
plt.close()


### Overplot Ha and Pab broad comp only
plt.figure()
xx =  np.linspace(-9500, 9500, 500)
data_Hb = np.load('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit_Hb_OIII/J1652_wFeII.line.npz', allow_pickle=True)
lam_Hb_broad = data_Hb['emlwav'][()]['c3']['Hbeta'][19,19]
sig_Hb_broad = data_Hb['emlsig'][()]['c3']['Hbeta'][19,19]
Fc1_Hb = data_Hb['emlflx'][()]['fc1']['Hbeta'][19,19]
Fc2_Hb = data_Hb['emlflx'][()]['fc2']['Hbeta'][19,19]
Fc3_Hb = data_Hb['emlflx'][()]['fc3']['Hbeta'][19,19]
hdul_Hb_fit = fits.open('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit_Hb_OIII/J1652_wFeII_Hbeta_Ftot_map.fits')
Ftot_Hb = hdul_Hb_fit[0].data[19,19]	# (1e4*10^-17) erg/s/cm^2  ## erg/cm2/s/A
F_Hb = Ftot_Hb * 1e-13 
F_Hb_broad = F_Hb * (Fc3_Hb/(Fc1_Hb+Fc2_Hb+Fc3_Hb))
Hb_cut = np.logical_and(lam_NRScube>1.88, lam_NRScube<1.96)
#plt.plot(lam_NRScube[Hb_cut], hdul_NRS_PSF[1].data[Hb_cut][:])
velcen_Hb_broad, velcen_Hb_broad = wave_to_vel(np.array([lam_Hb_broad]), np.array([lam_Hb_broad]), wave_Hb_micron*(1.+zred), zred)
fit_Hb_broad = gaussian(xx, amplitude=F_Hb_broad, mean=velcen_Hb_broad, stddev=sig_Hb_broad)
plt.plot(xx, fit_Hb_broad/np.max(fit_Hb_broad), label=r'$\mathrm{H}_\beta$', color='dodgerblue')

data_Ha = np.load('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_2comp.line.npz', allow_pickle=True)
sig_Ha_broad = data_Ha['emlsig'][()]['c2']['Halpha'][19,19]
lam_Ha_broad = data_Ha['emlwav'][()]['c2']['Halpha'][19,19]
Fc1_Ha = data_Ha['emlflx'][()]['fc1']['Halpha'][19,19]
Fc2_Ha = data_Ha['emlflx'][()]['fc2']['Halpha'][19,19]
hdul_Ha_fit = fits.open('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_2comp_Halpha_Ftot_map.fits')
Ftot_Ha = hdul_Ha_fit[0].data[19,19]    # (1e4*10^-17) erg/s/cm^2  ## erg/cm2/s/A
F_Ha = Ftot_Ha * 1e-13 
F_Ha_broad = F_Ha * (Fc2_Ha/(Fc1_Ha+Fc2_Ha))
velcen_Ha_broad, velcen_Ha_broad = wave_to_vel(np.array([lam_Ha_broad]), np.array([lam_Ha_broad]), wave_Ha_micron*(1.+zred), zred)
fit_Ha_broad = gaussian(xx, amplitude=F_Ha_broad, mean=velcen_Ha_broad, stddev=sig_Ha_broad)
plt.plot(xx, fit_Ha_broad/np.max(fit_Ha_broad), label=r'$\mathrm{H}_\alpha$', color='k')

data_Pab = np.load('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_PSFfit/J1652_2comp_symm.line.npz', allow_pickle=True)
Fc1_Pab = data_Pab['emlflx'][()]['fc1']['Pab'][19,19]
Fc2_Pab = data_Pab['emlflx'][()]['fc2']['Pab'][19,19]
sig_Pab_broad = data_Pab['emlsig'][()]['c2']['Pab'][19,19]
lam_Pab_broad = data_Pab['emlwav'][()]['c2']['Pab'][19,19]
Fc1_Pab = data_Pab['emlflx'][()]['fc1']['Pab'][19,19]
Fc2_Pab = data_Pab['emlflx'][()]['fc2']['Pab'][19,19]
hdul_Pab_fit = fits.open('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_PSFfit/J1652_2comp_symm_Pab_Ftot_map.fits')
Ftot_Pab = hdul_Pab_fit[0].data[19,19]
F_Pab = Ftot_Pab * 1e-13  # [erg/s/cm^2]
F_Pab_broad = F_Pab * (Fc2_Pab/(Fc1_Pab+Fc2_Pab))
#vel_Pab_broad, vel_Pab_broad_right = wave_to_vel(lam_MRScube[Pab_cut], lam_MRScube[Pab_cut], lam_Pab_broad*(1.+zred), zred)
#fit_Pab_broad = gaussian(vel_Pab_broad, amplitude=F_Pab_broad, mean=lam_Pab_broad, stddev=sig_Pab_broad)
velcen_Pab, velcen_Pab = wave_to_vel(np.array([lam_Pab_broad]), np.array([lam_Pab_broad]), wave_Pab_micron*(1.+zred), zred)
fit_Pab_broad = gaussian(xx, amplitude=F_Pab_broad, mean=velcen_Pab, stddev=sig_Pab_broad)
plt.plot(xx, fit_Pab_broad/np.max(fit_Pab_broad), label=r'$\mathrm{Pa}_\beta$', color='tab:red')
plt.axvline(0., color='grey', alpha=0.7, linestyle='--', zorder=0)
plt.ylabel('Flux (normalised)')
plt.xlabel('Velocity [km/s]')
plt.legend(fontsize=17)
do_plot.adjust_plot(plt, C_xlabelsize_figsize=0.3)
plt.savefig(plotdir + 'spec/' + 'Pab_vs_Balmer_broad_comps')
# plt.show()
plt.close()

cosmo = FlatLambdaCDM(Om0=0.3, H0=70.)
dist_Mpc = cosmo.luminosity_distance(z=zred).value
dist_cm = dist_Mpc*1e6*3.08567758*1e16*100


Ha_Ftot_map = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_3comp_Halpha_Ftot_map.fits'
Ha_line_3comp_npz = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_3comp.line.npz'
Hb_Ftot_map = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit_Hb_OIII/J1652_wFeII_Hbeta_Ftot_map.fits'
Hb_line_npz = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit_Hb_OIII/J1652_wFeII.line.npz'
Pab_Ftot_map = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_PSFfit/J1652_2comp_symm_Pab_Ftot_map.fits'
Pab_line_npz = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_PSFfit/J1652_2comp_symm.line.npz'
dir_GNIRS = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_GNIRS_MgII/'
MgII_Ftot_map = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_GNIRS_MgII/J1652_GNIRS_wFeII_MgII2796_Ftot_map.fits'
Mg_line_npz = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_GNIRS_MgII/J1652_GNIRS_wFeII.line.npz'
wave_MgII = 2800.
L_Pab_broad, FWHM_Pab_broad, [xx, profile_Pab] = Read_in_L_FWHM(file_Ftot_map=Pab_Ftot_map, file_line_npz=Pab_line_npz, line='Pab', fluxnorm=1e-13, dist_cm=dist_cm, ncol=19, nrow=19, do_total_profile=True, lam0_micron=wave_Pab_micron, return_tot_profile=True)
L_Ha_3comp_tot, FWHM_Ha_3comp_tot, [xx, profile_Ha] = Read_in_L_FWHM(file_Ftot_map=Ha_Ftot_map, file_line_npz=Ha_line_3comp_npz, line='Halpha', fluxnorm=1e-13, dist_cm=dist_cm, ncol=19, nrow=19, do_total_profile=True, lam0_micron=wave_Ha_micron, return_tot_profile=True)
L_Hb_3comp_tot, FWHM_Hb_3comp_tot, [xx, profile_Hb] = Read_in_L_FWHM(file_Ftot_map=Hb_Ftot_map, file_line_npz=Hb_line_npz, line='Hbeta', fluxnorm=1e-13, dist_cm=dist_cm, ncol=19, nrow=19, do_total_profile=True, lam0_micron=wave_Hb_micron, return_tot_profile=True)
L_MgII_tot, FWHM_tot_MgII, FWHM_tot_broadest, FWHM_tot_narrow, F_MgII_avg_arr, eF_MgII_avg_arr, Fsum_MgII, eFsum_MgII, [xx, profile_Mg] = Read_in_L_FWHM_dbl(file_Ftot_map=dir_GNIRS+'J1652_GNIRS_wFeII_MgII2796_Ftot_map.fits', file_Ftot_map2=dir_GNIRS+'J1652_GNIRS_wFeII_MgII2803_Ftot_map.fits', 
	file_line_npz=Mg_line_npz, line='MgII2803', line2='MgII2796', fluxnorm=1e-16, dist_cm=dist_cm, ncol=0, nrow=0, do_total_profile=True, lam0_micron=2803/1e4, return_velcen=False, return_profile=True)
plt.plot(xx, profile_Pab/np.max(profile_Pab), label=r'$\mathrm{Pa}_\beta$', color='tab:red')
plt.plot(xx, profile_Ha/np.max(profile_Ha), label=r'$\mathrm{H}_\alpha$', color='k')
plt.plot(xx, profile_Hb/np.max(profile_Hb), label=r'$\mathrm{H}_\beta$', color='grey')
plt.plot(xx, profile_Mg/np.max(profile_Mg), label=r'$\mathrm{Mg_{II}}$', color='dodgerblue')
plt.axvline(0., color='grey', alpha=0.75, linestyle='--', zorder=0)
plt.axhline(0.5, color='grey', alpha=0.6, linestyle=':', zorder=0)
plt.xlabel('Velocity [km/s]')
plt.ylabel('Flux (normalised)')
plt.legend(fontsize=17)
do_plot.adjust_plot(plt, C_xlabelsize_figsize=0.3)
plt.savefig(plotdir + 'spec/' + 'Pab_vs_Balmer_tot')
plt.show()
plt.close()


do_get_BHmass = False
if do_get_BHmass:
	### Get virial BH mass
	M_BH = lambda L_Pab, FWHM_Pab: 10**(7.83) * (L_Pab**0.436) * (FWHM_Pab**1.74) # L_Pab in [1e40 erg/s], FWHM_Pab in [1e4 km/s]. From La Franca+15 / Lamperti+17

	FWHM_Pab_nonorm = 4000. 	# [km/s]
	FWHM_Pab_norm = FWHM_Pab_nonorm * 1e-4	#[1e4 km/s]


	F_Pab = 0.5*1e-18  # [erg/s/cm^2/micron]     ((# OLD: [erg/s/cm^2/arsec^2] within a radius of 2.5 spaxels))


	spax_len_arcec = hdul_MRScube[1].header['CDELT1'] * 3600. # arcsec  # Note: Also, the area of a spaxel in arcsec^2 is in hdul_MRScube[1].header['PIXAR_A2']
	area_conv_arcsec2 = np.pi * (2.5 * spax_len_arcec)**2

	lam_Pab = 1.2822 * (1.+zred) # micron

	L_Pab_nonorm = F_Pab * (4*np.pi*dist_cm**2) * lam_Pab	# [erg/s]
	L_Pab = L_Pab_nonorm * 1e-40	# [1e40 erg/s]

	print('\nlog M_BH: {:.3f}\n'.format(np.log10(M_BH(L_Pab, FWHM_Pab_norm))))

	do_plot_with_extremeQSOs = True 
	if do_plot_with_extremeQSOs:
		### BH masses from Ferris+21 - extremely luminous radio-WISE selected galaxies ###
		MBH_Mstar_OIII = 1e-3 * np.array([3.19, 3.81 , 47.73, 12.37, 13.45, 0.43 , 5.83 , 0.64 ,  7.02,  0.38,  7.73,  3.80,  11.9,  12.1,  4.98])
		IDs_WISE_OIII = np.array(["J081131.61-222522", "J082311.24-062408", "J130817.00-344754", "J134331.37-113609", "J140050.13-291924", "J141243.15-202011", 
		  "J143419.59-023543", "J150048.73-064939", "J151003.71-220311", "J151424.12-341100", "J154141.64-114409", "J163426.87-172139", 
		  "J170204.65-081108", "J195141.22-042024", "J200048.58-280251"])
		IDs_OIII  = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17])

		IDs_WISE_Balmer = np.array(["J081131.61-222522", "J082311.24-062408", "J130817.00-344754", "J140050.13-291924", "J141243.15-202011", "J143419.59-023543", 
		  "J143931.76-372523", "J150048.73-064939", "J151003.71-220311", "J151310.42-221004", "J154141.64-114409", "J163426.87-172139", "J195141.22-042024", 
		  "J204049.51-390400"])
		IDs_Balmer = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18])
		MBH_Mstar_Ha = 1e-3 * np.array([0.24, 1.68, 1.88, 2.60, 0.17, 0.54, 3.54, 1.89, 0.53, 0.91, 1.99, 1.84, 0.32, -99.])
		eMBH_Mstar_Ha = np.array([0.10, 1.02, 0.66, 0.61, 0.15, 0.29, 2.76, 0.50, 0.43, 0.45, 1.68, 0.94, 0.37, -99.])

		MBH_Mstar_Hb = 1e-3 * np.array([-99., 0.44, 0.29, 0.18, -99., -99., -99., 0.08, -99., -99., -99., -99., 0.07, 0.03])
		eMBH_Mstar_Hb = np.array([-99., 0.18, 0.16, 0.09, -99., -99., -99., 0.12, -99., -99., -99., -99., 0.05, 0.02])

		lMstar = np.array([9.97, 10.83, 10.90, 11.07, 10.66, 11.24, 11.02, 11.57, 11.21, 10.24, 11.12, 10.54, 11.74, 10.53, 11.64, 10.89, 11.08, 11.29, 10.92, 10.86, 11.23, 10.76, 11.99, 10.95, 11.42, 11.01, 10.81, 11.52, 12.14, 11.04])
		IDs_WISE = np.array(["J071433.54-363552", "J071912.58-334944", "J081131.61-222522", "J082311.24-062408", "J130817.00-344754", "J134331.37-113609", 
		  "J140050.13-291924", "J141243.15-202011", "J143419.59-023543", "J143931.76-372523", "J150048.73-064939", "J151003.71-220311", "J151310.42-221004", 
		  "J151424.12-341100", "J152116.59+001755", "J154141.64-114409", "J163426.87-172139", "J164107.22-054827", "J165305.40-010230", "J165742.88-174049", 
		  "J170204.65-081108", "J170325.05-051742", "J170746.08-093916", "J193622.58-335420", "J195141.22-042024", "J195801.72-074609", "J200048.58-280251", 
		  "J202148.06-261159", "J204049.51-390400", "J205946.93-354134"])



from dust_attenuation.averages import C00
k_Ha_C00 = 3.32476
k_Hb_C00 = 4.5965

def E_BmV(Ha_Hb_obs, k_Ha=k_Ha_C00, k_Hb=k_Hb_C00, Ha_Hb_t = 3.1, log_scale=False):
	#Ha_Hb_t is the intrinsic Ha/Hb ratio
	
	f = 1/(-0.4*(k_Ha-k_Hb))
	if(log_scale):
		E_BV = f*(Ha_Hb_obs - np.log10(Ha_Hb_t))
	else:
		E_BV = f*np.log10(Ha_Hb_obs/Ha_Hb_t)
	
	return E_BV


def line_ratio_att_correct(f1_f2_obs, E_BV, l1, l2):
	#l1, l2 in angstrom
	# ratio in log
	Av = E_BV*4.05
	l1 = l1/1e4 # um
	l2 = l2/1e4 #
	
	A1_list = []
	A2_list = []
	for Av_i in Av:
		if(Av_i>0):
			att_model = C00(Av=Av_i)
			A1_list.append(att_model(l1))
			A2_list.append(att_model(l2))
		else:
			A1_list.append(0)
			A2_list.append(0)
#     f1_f2_int = f1_f2_obs*pow(10, 0.4*(A1-A2))
	f1_f2_int = f1_f2_obs + 0.4*(np.array(A1_list)-np.array(A2_list))
	return f1_f2_int


def att_CCM(lam_micron, R_V, A_V):
	### Returning A(lambda) for Cardelli, Clayton, & Mathis (1989) extinction
	xx = 1./lam_micron
	if xx > 0.3 and xx < 1.1:  # IR   
		a_CCM =  0.574 * xx**(1.61)
		b_CCM = -0.527 * xx**(1.61)
	elif xx > 1.1 and xx < 3.3: # optical/NIR
		yy = xx - 1.82
		c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085, 0.01979, -0.77530,  0.32999 ] [::-1]
		c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434, -0.62251,  5.30260, -2.09002 ] [::-1]
		a_CCM = np.polyval(c1, yy)
		b_CCM = np.polyval(c2, yy)
	elif xx > 3.3 and xx < 8.:
		y1 = xx - 5.9
		F_a = -0.04473 * y1**2 - 0.009779 * y1**3
		F_b =   0.2130 * y1**2  +  0.1207 * y1**3
		a_CCM =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
		b_CCM = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b

	return  A_V * (a_CCM + b_CCM/R_V)

def k_lambda(lam_micron, R_V):
	### Cardelli, Clayton, & Mathis (1989) extinction. Following https://github.com/moustakas/impro/blob/master/pro/dust/k_lambda.pro
	xx = 1./lam_micron
	if xx > 0.3 and xx < 1.1:  # IR   
		a_CCM =  0.574 * xx**(1.61)
		b_CCM = -0.527 * xx**(1.61)
	elif xx > 1.1 and xx < 3.3: # optical/NIR
		yy = xx - 1.82
		c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085, 0.01979, -0.77530,  0.32999 ] [::-1]
		c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434, -0.62251,  5.30260, -2.09002 ] [::-1]
		a_CCM = np.polyval(c1, yy)
		b_CCM = np.polyval(c2, yy)
	elif xx > 3.3 and xx < 8.:
		y1 = xx - 5.9
		F_a = -0.04473 * y1**2 - 0.009779 * y1**3
		F_b =   0.2130 * y1**2  +  0.1207 * y1**3
		a_CCM =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
		b_CCM = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b

	k_lambda = R_V * (a_CCM + b_CCM/R_V)
	return k_lambda


### Infer extinction from decrement
do_get_ext = True
if do_get_ext:
	k_lambda_C00 = lambda lam_micron, R_V: 2.659*(-2.156 + 1.509/lam_micron - 0.198/lam_micron**2 + 0.011/lam_micron**3) + R_V		# from Calzetti 2000; k_lam = A_lam/E(B-V)

	wave_Ha = 6562.8
	wave_Hb = 4861.
	wave_Pab = 12822.16
	R_V = 4.05 ## Ratio of total to selective extinction; Calzetti et al. (2000) estimate R_V = 4.05 +/- 0.80   ### 3.1
	R_int = 17.6		## intrinsic ratio

	hdul_Ha_fit = fits.open('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_Halpha_Ftot_map.fits')
	Ftot_Ha = hdul_Ha_fit[0].data[19,19]	# (1e4*10^-17) erg/s/cm^2  ## erg/cm2/s/A
	F_Ha = Ftot_Ha * 1e-13 

	# data_Ha_2comp = np.load('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_2comp.line.npz', allow_pickle=True)
	# Fc1_Ha = data_Ha_2comp['emlflx'][()]['fc1']['Halpha'][19,19]
	# Fc2_Ha = data_Ha_2comp['emlflx'][()]['fc2']['Halpha'][19,19]
	# FWHM_Ha_broad = data_Ha_2comp['emlsig'][()]['c2']['Halpha'][19,19] * 2.355
	# F_Ha_broad = F_Ha * (Fc2_Ha/(Fc1_Ha+Fc2_Ha))
	# F_Ha = F_Ha_broad

	data_Ha_3comp = np.load(Ha_line_3comp_npz, allow_pickle=True)
	Fc1_Ha_3comp = data_Ha_3comp['emlflx'][()]['fc1']['Halpha'][19,19]
	Fc2_Ha_3comp = data_Ha_3comp['emlflx'][()]['fc2']['Halpha'][19,19]
	Fc3_Ha_3comp = data_Ha_3comp['emlflx'][()]['fc3']['Halpha'][19,19]
	FWHM_Ha_broad_3comp = data_Ha_3comp['emlsig'][()]['c3']['Halpha'][19,19] * 2.355
	F_Ha_broad_3comp = F_Ha * ((Fc3_Ha_3comp+Fc2_Ha_3comp)/(Fc1_Ha_3comp+Fc2_Ha_3comp+Fc3_Ha_3comp))
	F_Ha_narrow_3comp = F_Ha * (Fc1_Ha_3comp/(Fc1_Ha_3comp+Fc2_Ha_3comp+Fc3_Ha_3comp))
	L_Ha_3comp_tot, FWHM_Ha_3comp_tot = Read_in_L_FWHM(file_Ftot_map=Ha_Ftot_map, file_line_npz=Ha_line_3comp_npz, line='Halpha', fluxnorm=1e-13, dist_cm=dist_cm, ncol=19, nrow=19, do_total_profile=True, lam0_micron=wave_Ha*1e-4)


	Ha_Ftot_map = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_wNIISII_linevary_Halpha_Ftot_map.fits'
	Ha_line_npz = '/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_NIRSpec_PSFfit/J1652_wNIISII_linevary.line.npz'
	L_Ha_wNIISII_tot, FWHM_Ha_wNIISII_tot, FWHM_Ha_wNIISII_tot_broadest_out, FWHM_Ha_wNIISII_tot_narrow_out = Read_in_L_FWHM_B(file_Ftot_map=Ha_Ftot_map, file_line_npz=Ha_line_npz, line='Halpha', fluxnorm=1e-13, dist_cm=dist_cm, ncol=19, nrow=19, do_total_profile=True, lam0_micron=wave_Ha/1e4)
	
	
	FWHM_Ha_broad_3comp_mid = data_Ha_3comp['emlsig'][()]['c2']['Halpha'][19,19] * 2.355
	F_Ha_broad_3comp_mid = F_Ha * (Fc2_Ha_3comp/(Fc1_Ha_3comp+Fc2_Ha_3comp+Fc3_Ha_3comp))
	
	hdul_Hb_fit = fits.open(Hb_Ftot_map)
	Ftot_Hb = hdul_Hb_fit[0].data[19,19]	# (1e4*10^-17) erg/s/cm^2  ## erg/cm2/s/A
	F_Hb = Ftot_Hb * 1e-13 

	data_Hb = np.load(Hb_line_npz, allow_pickle=True)
	Fc1_Hb = data_Hb['emlflx'][()]['fc1']['Hbeta'][19,19]
	Fc2_Hb = data_Hb['emlflx'][()]['fc2']['Hbeta'][19,19]
	Fc3_Hb = data_Hb['emlflx'][()]['fc3']['Hbeta'][19,19]
	FWHM_Hb_broad = data_Hb['emlsig'][()]['c3']['Hbeta'][19,19] * 2.355
	F_Hb_broad = F_Hb * (Fc3_Hb/(Fc1_Hb+Fc2_Hb+Fc3_Hb))
	F_Hb_narrow = F_Hb * (Fc1_Hb/(Fc1_Hb+Fc2_Hb+Fc3_Hb))
	L_Hb_3comp_tot, FWHM_Hb_3comp_tot = Read_in_L_FWHM(file_Ftot_map=Hb_Ftot_map, file_line_npz=Hb_line_npz, line='Hbeta', fluxnorm=1e-13, dist_cm=dist_cm, ncol=19, nrow=19, do_total_profile=True, lam0_micron=wave_Hb*1e-4)


	# F_Hb = F_Hb_broad


	E_BmV_0 = E_BmV(F_Ha/F_Hb,log_scale=False)
	AV_0 = R_V * E_BmV_0


	## Following https://github.com/moustakas/impro/blob/master/pro/dust/get_ebv.pro , dust_correct.pro
	# R_obs = F_Ha_broad/F_Hb_broad
	# rcurve = - k_lambda(wave_Ha/1.e4, R_V) + k_lambda(wave_Hb/1.e4, R_V)
	# R_int_HaHb = 3.16  # from Lu+18: https://arxiv.org/abs/1811.11063  # use 3.1 for AGN instead of 2.86  # 3.5 in van den Berk 2001 composite spec
	# color = -2.5 * np.log10(R_int_HaHb/R_obs)
	# ebv = color / rcurve
	# kl = k_lambda(wave_Ha/1.e4, R_V)
	# Ha_newflux = F_Ha * 10**(0.4*ebv*kl)
	# # A_Ha = ebv*kl
	# AV_0 = ebv*k_lambda(0.551, R_V)


	R_obs_tot = L_Ha_3comp_tot/L_Hb_3comp_tot
	rcurve = - k_lambda(wave_Ha/1.e4, R_V) + k_lambda(wave_Hb/1.e4, R_V)
	R_int_HaHb = 3.1 # use 3.1 for AGN instead of 2.86  # 3.5 in van den Berk 2001 composite spec
	color = -2.5 * np.log10(R_int_HaHb/R_obs_tot)
	ebv = color / rcurve
	kl = k_lambda(wave_Ha/1.e4, R_V)
	AV_0_tot = ebv*k_lambda(0.551, R_V)
	breakpoint()



	att_model = C00(Av=AV_0_tot)
	att_model_2 = C00(Av=1.0)
	att_Ha = att_model(wave_Ha/1e4)	# A_lam / A_V
	att_Hb = att_model(wave_Hb/1e4)	# A_lam / A_V
	att_Pab = att_model(wave_Pab/1e4)	# A_lam / A_V


	R_obs_2 = F_Ha_broad_3comp/F_Hb_broad
	rcurve = - k_lambda(wave_Ha/1.e4, R_V) + k_lambda(wave_Hb/1.e4, R_V)
	R_int_HaHb = 3.1 # use 3.1 for AGN instead of 2.86  # 3.5 in van den Berk 2001 composite spec
	color = -2.5 * np.log10(R_int_HaHb/R_obs_2)
	ebv = color / rcurve
	kl = k_lambda(wave_Ha/1.e4, R_V)
	AV_0_broad = ebv*k_lambda(0.551, R_V)


	breakpoint()

	R_obs_3 = F_Ha_broad_3comp_mid/F_Hb_broad
	rcurve = - k_lambda(wave_Ha/1.e4, R_V) + k_lambda(wave_Hb/1.e4, R_V)
	R_int_HaHb = 3.1 # use 3.1 for AGN instead of 2.86  # 3.5 in van den Berk 2001 composite spec
	color = -2.5 * np.log10(R_int_HaHb/R_obs_3)
	ebv = color / rcurve
	kl = k_lambda(wave_Ha/1.e4, R_V)
	AV_0_C = ebv*k_lambda(0.551, R_V)




	R_obs_narrow = F_Ha_narrow_3comp/F_Hb_narrow
	rcurve = - k_lambda(wave_Ha/1.e4, R_V) + k_lambda(wave_Hb/1.e4, R_V)
	R_int_HaHb = 3.1 # use 3.1 for AGN instead of 2.86  # 3.5 in van den Berk 2001 composite spec
	R_int_HaHb_narrow = 4.37
	color = -2.5 * np.log10(R_int_HaHb_narrow/R_obs_narrow)
	ebv = color / rcurve
	kl = k_lambda(wave_Ha/1.e4, R_V)
	AV_0_narrow = ebv*k_lambda(0.551, R_V)


	breakpoint()



	hdul_Pab_fit = fits.open('/Users/caroline/Documents/ARI-Heidelberg/Q3D/QUESTFIT/q3dfit-dev/jnb/J1652_PSFfit/J1652_Pab_Ftot_map.fits')
	Ftot_Pab = hdul_Pab_fit[0].data[19,19]


	F_Pab = Ftot_Pab * 1e-13  # [erg/s/cm^2]

	R_obs = F_Ha/F_Pab


	rcurve = k_lambda(wave_Ha/1.e4, R_V)-k_lambda(wave_Pab/1.e4, R_V)

	color = -2.5 * np.log10(R_int/R_obs)

	ebv = color / rcurve

	kl = k_lambda(wave_Ha/1.e4, R_V)
	Ha_newflux = F_Ha * 10**(0.4*ebv*kl)

	kl_V = k_lambda(0.551, R_V)


	breakpoint()








