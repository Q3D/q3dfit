import numpy as np
import pdb
from astropy.table import Table
from scipy import constants
from q3dfit.gaussflux import gaussflux
from q3dfit.lmlabel import lmlabel


def sepfitpars(linelist, param, perror, maxncomp, waveran=None,
               tflux=False, doublets=None):

    """
    Convert output of MPFIT, with best-fit line parameters in a single
    array, into a structure with separate arrays for different line
    parameters. Compute total line fluxes from the best-fit line
    parameters.

    Parameters
    ----------
    linelist : dict
        List of emission line rest-frame wavelengths.
    param : ndarray, shape (N,)
        Best-fit parameter array output by MPFIT.
    perror : ndarray, shape (N,)
        Errors in best fit parameters, output by MPFIT.
    parinfo : dict (the old version's type is structure)
        Dictionary input into MPFIT. Each tag has N values, one per parameter.
        Used to sort param and perror arrays.
    waveran: ndarray, shape (2,), optional
        Set to upper and lower limits to return line parameters only
        for lines within the given wavelength range. Lines outside this
        range have fluxes set to 0.
    tflux: dict (the old version's type is structure), optional
        A structure with separate hashes for total flux and error. The hashes
        are indexed by line. Tags: tflux, tfluxerr

    Returns
    -------
    dict
        A structure with separate hashes for different line parameters. The dictionaries
        are indexed by line, and each value is an array over components.
        Tags: flux, fluxerr, fluxpk, fluxpkerr, nolines, wave, and sigma.
    """
    # DW: param, perror and parinfo are lists of dictionaries which are not easy to handle
    # I therefore re-structure them into a single dictionary each

    #parinfo_new = {}
    #for k, v in [(key, d[key]) for d in parinfo for key in d]:
    #    if k not in parinfo_new:
    #        parinfo_new[k] = [v]
    #    else:
    #        parinfo_new[k].append(v)

    #parinfo_new = Table(parinfo_new)

    #param = np.array(param)
    #perror = np.array(perror)


    # Return 0 if no lines were fit
    if len(param) == 1:  # DW: this needs to be double checked!!!!###
        outstr = {'nolines': b'0'}
        if tflux is not False:
            tfdict = dict()
            return outstr, tfdict
        else:
            return outstr

    else:

        basearr = np.full(linelist['name'].T.data.shape, np.nan)
        basearr_1comp = basearr
        if maxncomp > 1:
            basearr = np.tile(basearr, (maxncomp, 1))
        flux = Table(basearr, names=linelist['name'])
        fluxerr = Table(basearr, names=linelist['name'])
        fluxpk = Table(basearr, names=linelist['name'])
        fluxpkerr = Table(basearr, names=linelist['name'])
        fluxpk_obs = Table(basearr, names=linelist['name'])
        fluxpkerr_obs = Table(basearr, names=linelist['name'])
        sigma = Table(basearr, names=linelist['name'])
        sigmaerr = Table(basearr, names=linelist['name'])
        sigma_obs = Table(basearr, names=linelist['name'])
        sigmaerr_obs = Table(basearr, names=linelist['name'])
        wave = Table(basearr, names=linelist['name'])
        waveerr = Table(basearr, names=linelist['name'])

        if tflux is not False:
            tf = Table(basearr_1comp, names=linelist['name'])
            tfe = Table(basearr_1comp, names=linelist['name'])

        #in2ha = np.where(parinfo_new['parname'] == '[NII]/Halpha line ratio')
        #ctn2ha = np.count_nonzero(parinfo_new['parname'] == '[NII]/Halpha line ratio')

        #in1rat = np.where(parinfo_new['parname'] == '[NI]5200/5198 line ratio')
        #ctn1rat = np.count_nonzero(parinfo_new['parname'] == '[NI]5200/5198 line ratio')

        #is2rat = np.where(parinfo_new['parname'] == '[SII]6716/6731 line ratio')
        #cts2rat = np.count_nonzero(parinfo_new['parname'] == '[SII]6716/6731 line ratio')

        #   ihahb = np.where(parinfo_new['parname'] == 'Halpha/Hbeta line ratio')
        #   cthahb = np.count_nonzero(parinfo_new['parname'] == 'Halpha/Hbeta line ratio')

        #io2rat = np.where(parinfo_new['parname'] == '[OII]3729/3726 line ratio')
        #cto2rat = np.count_nonzero(parinfo_new['parname'] == '[OII]3729/3726 line ratio')

        #   Populate Tables

        for line in linelist['name']:

            for i in range(0, maxncomp):

                # indices
                lmline = lmlabel(line)
                ifluxpk = f'{lmline.lmlabel}_{i}_flx'
                isigma = f'{lmline.lmlabel}_{i}_sig'
                iwave = f'{lmline.lmlabel}_{i}_cwv'
                #ispecres = f'{lmline.lmlabel}_{i}_srsigslam'

                # make sure the line was fit -- necessary if, e.g., #
                # components reset to 0 by checkcomp
                if iwave in param.keys():
                    wave[line][i] = param[iwave]
                    sigma[line][i] = param[isigma]
                    fluxpk[line][i] = param[ifluxpk]
                    sigma_obs[line][i] = param[isigma]
                    fluxpk_obs[line][i] = param[ifluxpk]

                    waveerr[line][i] = perror[iwave]
                    sigmaerr[line][i] = perror[isigma]
                    sigmaerr_obs[line][i] = perror[isigma]
                    fluxpkerr[line][i] = perror[ifluxpk]
                    fluxpkerr_obs[line][i] = perror[ifluxpk]

            # Because of the way these lines are tied to others (with a division!) they
            # can yield NaNs in components that aren't fit. Correct this.
            # if line eq '[SII]6731' OR line eq 'Hbeta' OR line eq '[NI]5189' then begin

            #if (line == '[SII]6731') or (line == '[NI]5189'):
            #    inan = np.where(np.isfinite(fluxpk[line]) == False)
            #    ctnan = np.count_nonzero(np.isfinite(fluxpk[line]) == False)
            #    if ctnan > 0:
            #        fluxpk[line][inan] = 0.
            #        fluxpkerr[line,inan] = 0.
            #        fluxpk_obs[line,inan] = 0.
            #        fluxpkerr_obs[line,inan] = 0.

        #for line in linelist['name']:

        # Fix flux errors associated with line ratios. E.g., [NII]/Halpha is a fitted
        # parameter and [NII]6583 is tied to it, so the formal error in [NII]6583
        # flux is 0. Add errors in Halpha and [NII]/Halpha in quadrature to get
        #  error in [NII]6583.

            # if (line == "[NII]6583") and (ctn2ha > 0):
            #     fluxpkerr_obs[line][0:ctn2ha] = fluxpk_obs[line][0:ctn2ha] * \
            #     np.sqrt((perror[in2ha]/param[in2ha])**2. + \
            #     (fluxpkerr_obs['Halpha'][0:ctn2ha]/ fluxpk_obs['Halpha'][0:ctn2ha])**2.)
            #     # In pegged case, set errors equal to each other
            #     ipegged = np.where((perror[in2ha] == 0.) and (param[in2ha] != 0.))
            #     ctpegged = np.count_nonzero((perror[in2ha] == 0.) and (param[in2ha] != 0.))
            #     if ctpegged > 0:
            #         fluxpkerr_obs['[NII]6583'][ipegged] = \
            #         fluxpkerr_obs['Halpha'][ipegged]
            #     fluxpkerr[line] = fluxpkerr_obs[line]

            # if (line == '[SII]6731') and (cts2rat > 0):
            #     fluxpkerr_obs[line][0:cts2rat] = \
            #     fluxpk_obs[line][0:cts2rat] * \
            #     np.sqrt((perror[is2rat]/param[is2rat])**2. + \
            #     (fluxpkerr_obs['[SII]6716'][0:cts2rat] / \
            #     fluxpk_obs['[SII]6716'][0:cts2rat])**2.)
            #     # In pegged case, set errors equal to each other
            #     ipegged = np.where((perror[is2rat] == 0.) and (param[is2rat] != 0.))
            #     ctpegged = np.count_nonzero((perror[in2ha] == 0.) and (param[in2ha] != 0.))
            #     if ctpegged > 0:
            #         fluxpkerr_obs['[SII]6731'][ipegged] = \
            #         fluxpkerr_obs['[SII]6716'][ipegged]
            #     fluxpkerr[line] = fluxpkerr_obs[line]

            # if (line == '[NI]5198') and (ctn1rat > 0):
            #     fluxpkerr_obs[line][0:ctn1rat] = \
            #     fluxpk_obs[line][0:ctn1rat] * \
            #     np.sqrt((perror[in1rat]/param[in1rat])**2. + \
            #     (fluxpkerr_obs['[NI]5200'][0:ctn1rat]/ \
            #     fluxpk_obs['[NI]5200'][0:ctn1rat])**2.)

            #     fluxpkerr[line] = fluxpkerr_obs[line]

            #     # In pegged case, set errors equal to each other
            #     ipegged = np.where((perror[in1rat] == 0.) and (param[in1rat] != 0.))
            #     ctpegged = np.count_nonzero((perror[in1rat] == 0.) and (param[in1rat] != 0.))
            #     if ctpegged > 0:
            #         fluxpkerr_obs['[NI]5198'][ipegged] = fluxpkerr_obs['[NI]5200'][ipegged]

            #     fluxpkerr[line] = fluxpkerr_obs[line]

## DW: the following is a commented section in the IDL code:
#      if line eq 'Hbeta' AND cthahb gt 0 then begin
#         fluxpkerr_obs[line,0:cthahb-1] = $
#            fluxpk_obs[line,0:cthahb-1]*sqrt($
#            (perror[ihahb]/param[ihahb])^2d + $
#            (fluxpkerr_obs['Halpha',0:cthahb-1]/$
#            fluxpk_obs['Halpha',0:cthahb-1])^2d)
#         fluxpkerr[line] = fluxpkerr_obs[line]
##        If Halpha/Hbeta gets too high, MPFIT sees it as an "upper limit" and
##        sets perror = 0d. Then the errors seem too low, and it registers as a
##        detection. This bit of code corrects that.
#         ipeggedupper = $
#            where(param[ihahb] gt 1d1 AND perror[ihahb] eq 0d,ctpegged)
#         if ctpegged gt 0 then begin
#            fluxpk[line,ipeggedupper] = 0d
#            fluxpk_obs[line,ipeggedupper] = 0d
#         endif

            # if (line == 'Hbeta') and (np.count_nonzero(linelist['name'] == 'Halpha') == 1):
            #     # If Halpha/Hbeta goes belowlower limit, then we re-calculate the errors
            #     # add discrepancy in quadrature to currently calculated error. Assume
            #     # error in fitting is in Hbeta and adjust accordingly.
            #     fha = fluxpk['Halpha']
            #     ihahb = np.where((fluxpk['Halpha'] > 0.) and (fluxpk['Hbeta'] > 0.))
            #     cthahb = np.count_nonzero((fluxpk['Halpha'] > 0.) and (fluxpk['Hbeta'] > 0.))
            #     if cthahb > 0.:
            #         itoolow = np.where(fluxpk['Halpha'][ihahb]/fluxpk['Hbeta'] < 2.86)
            #         cttoolow = np.count_nonzero(fluxpk['Halpha'][ihahb]/fluxpk['Hbeta'] < 2.86)
            #         if cttoolow > 0:
            #             fluxpkdiff = fluxpk[line][itoolow] - fluxpk['Halpha'][itoolow]/2.86
            #             fluxpk[line][itoolow] -= fluxpkdiff
            #             fluxpk_obs[line][itoolow] -= fluxpkdiff
            #             fluxpkerr[line][itoolow] = np.sqrt(fluxpkerr[line][itoolow]**2. + fluxpkdiff**2.)
            #             fluxpkerr_obs[line][itoolow] = np.sqrt(fluxpkerr_obs[line][itoolow]**2. + fluxpkdiff**2.)

            # if (line == '[OII]3729') and (cto2rat > 0.):
            #     fluxpkerr_obs[line][0:cto2rat] = \
            #     fluxpk_obs[line][0:cto2rat]*np.sqrt( \
            #     (perror[io2rat]/param[io2rat])**2. + \
            #     (fluxpkerr_obs['[OII]3726'][0:cto2rat]/ \
            #     fluxpk_obs['[OII]3726'][0:cto2rat])**2.)
            #     # In pegged case, set errors equal to each other
            #     ipegged = np.where((perror[io2rat] == 0.) and (param[io2rat] != 0.))
            #     ctpegged = np.count_nonzero((perror[io2rat] == 0.) and (param[io2rat] != 0.))
            #     if ctpegged > 0:
            #         fluxpkerr_obs['[OII]3729'][ipegged] = fluxpkerr_obs['[OII]3726'][ipegged]
            #     fluxpkerr[line] = fluxpkerr_obs[line]

                    # Add back in spectral resolution

                    # Make sure we're not adding something to 0 --
                    # i.e. the component wasn't fit.
                    # Can't use sigma = 0 as criterion since the line could be
                    # fitted but unresolved.
                    if fluxpk[line][i] > 0:
                        sigmatmp = \
                            sigma[line][i]/(constants.c/1.e3)*wave[line][i]
                        # sigmatmp = np.sqrt(sigmatmp**2. + param[ispecres]**2.)
                        # in km/s
                        sigma_obs[line][i] = \
                            sigmatmp/wave[line][i]*(constants.c/1.e3)
                        # error propagation for adding in quadrature
                        sigmaerr_obs[line][i] *= \
                            sigma[line][i]/(constants.c/1.e3)*wave[line][i] /\
                            sigmatmp
                        # Correct peak flux and error for deconvolution
                        fluxpk[line][i] *= sigma_obs[line][i]/sigma[line][i]
                        fluxpkerr[line][i] *= sigma_obs[line][i]/sigma[line][i]

                        # Compute total Gaussian flux
                        # sigma and error need to be in wavelength space
                        gflux = gaussflux(fluxpk_obs[line][i], sigmatmp,
                                          normerr=fluxpkerr_obs[line][i],
                                          sigerr=sigmaerr_obs[line][i] /
                                          (constants.c / 1.e3) * wave[line][i])
                    else:
                        gflux = {'flux': 0., 'flux_err': 0.}
                    flux[line][i] = gflux['flux']
                    fluxerr[line][i] = gflux['flux_err']

                    # Set fluxes to 0 outside of wavelength range,
                    # or if NaNs or infinite errors

                    inoflux_wr = False
                    if waveran:
                        inoflux_wr = \
                            ((waveran[0] > wave[line][i] *
                              (1. - 3.*sigma[line][i] /
                               (constants.c/1.e3))) or
                             (waveran[1] < wave[line][i] *
                              (1. + 3.*sigma[line][i] /
                               (constants.c/1.e3))))

                    inoflux = \
                        ((np.isfinite(fluxerr[line][i]) is False) or
                         (np.isfinite(fluxpkerr[line][i]) is False))

                    if inoflux_wr or inoflux:
                        flux[line][i] = 0.
                        fluxerr[line][i] = 0.
                        fluxpk[line][i] = 0.
                        fluxpkerr[line][i] = 0.
                        fluxpk_obs[line][i] = 0.
                        fluxpkerr_obs[line][i] = 0.

            # Compute total fluxes summed over components
            igd = np.where(flux[line] > 0.)
            ctgd = np.count_nonzero(flux[line] > 0.)
            if tflux is not False:
                if ctgd > 0:
                    tf[line] = np.sum(flux[line][igd])
                    tfe[line] = np.sqrt(np.sum(fluxerr[line][igd]**2.))
                else:
                    tf[line] = 0.
                    tfe[line] = 0.

        # Special doublet cases: combine fluxes from each line
        if doublets is not None:

            for (name1, name2) in zip(doublets['line1'], doublets['line2']):
                if name1 in linelist['name'] and name2 in linelist['name']:
                    # new line label
                    dkey = name1+'+'+name2
                    # add fluxes
                    tf[dkey] = tf[name1]+tf[name2]
                    flux[dkey] = flux[name1]+flux[name2]
                    fluxpk[dkey] = fluxpk[name1]+fluxpk[name2]
                    fluxpk_obs[dkey] = fluxpk_obs[name1]+fluxpk_obs[name2]
                    # add flux errors in quadrature
                    tfe[dkey] = np.sqrt(tfe[name1]**2. + tfe[name2]**2.)
                    fluxerr[dkey] = np.sqrt(fluxerr[name1]**2. +
                                            fluxerr[name2]**2.)
                    fluxpkerr[dkey] = np.sqrt(fluxpkerr[name1]**2. +
                                              fluxpkerr[name2]**2.)
                    fluxpkerr_obs[dkey] = np.sqrt(fluxpkerr_obs[name1]**2. +
                                                  fluxpkerr_obs[name2]**2.)
                    # average waves and sigmas and errors
                    wave[dkey] = (wave[name1]+wave[name2])/2.
                    waveerr[dkey] = (waveerr[name1]+waveerr[name2])/2.
                    sigma[dkey] = (sigma[name1]+sigma[name2])/2.
                    sigmaerr[dkey] = (sigmaerr[name1]+sigmaerr[name2])/2.
                    sigma_obs[dkey] = (sigma_obs[name1]+sigma_obs[name2])/2.
                    sigmaerr_obs[dkey] = (sigmaerr_obs[name1] +
                                          sigmaerr_obs[name2])/2.

        outstr = {'flux': flux, 'fluxerr': fluxerr,
                  'fluxpk': fluxpk, 'fluxpkerr': fluxpkerr,
                  'wave': wave, 'waveerr': waveerr,
                  'sigma': sigma, 'sigmaerr': sigmaerr,
                  'sigma_obs': sigma_obs, 'sigmaerr_obs': sigmaerr_obs,
                  'fluxpk_obs': fluxpk_obs, 'fluxpkerr_obs': fluxpkerr_obs}

        if tflux is not False:
            tfdict = {'tflux': tf, 'tfluxerr': tfe}

        if tflux is not False:
            return outstr, tfdict
        else:
            return outstr
