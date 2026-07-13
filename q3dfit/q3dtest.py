#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from q3dfit.q3din import q3din
from q3dfit.plot import plotpopheatmap
'''
Functions to vary a polynomial parameter in q3dfit and plot the results.
'''


def plot_test_continum(polytestdat,
                       cmap : Optional[str]='spring',
                       figsize : Optional[tuple]=(20, 30),
                       savefig : Optional[bool]=False,
                       argssavefig : Optional[dict]={'bbox_inches': 'tight', 'dpi': 300},
                       outfile : Optional[str]=None):
    '''
        describe this function

        Parameters
        ----------
        polytestdat
            Dictionary containing the test data for plotting.
        cmap
            String name of the matplotlib colormap to use for plotting the different components. Default is 'spring'.
        figsize
            Tuple of integers specifying the width and height of the figure in inches.
        savefig
            Boolean indicating whether to save the figure.
        argssavefig
            Dictionary of arguments to pass to the savefig function.
        outfile
            String specifying the path and filename for the saved figure.
    '''
    rcParamsOrig = rcParams.copy()

    plt.style.use('dark_background')
    
    plts = 3
    fig, ax = plt.subplots(plts, 1, figsize=figsize)

    plotwidth = len(polytestdat['wave']) // plts

    # Generating colors for each component
    n_lines = len(polytestdat['cont_fit'])
    cmap = plt.get_cmap(cmap)
    color = cmap(np.linspace(0, 1, n_lines))

    for i in range(len(ax)):
        slicerange = slice(i*plotwidth, (i + 1) * plotwidth)
        x = polytestdat['wave'][slicerange]
        ax[i].plot(x, polytestdat['cont_dat'][slicerange])
        for j in range(len(polytestdat['cont_fit'])):
            ax[i].plot(x, polytestdat['cont_fit'][j][slicerange], color=color[j])
    #ax[0].legend()

    sm = plt.cm.ScalarMappable(cmap = cmap, norm=plt.Normalize(min(polytestdat[polytestdat['tstparam']]), max(polytestdat[polytestdat['tstparam']])))
    fig.colorbar(sm, ax=[ax[0], ax[1], ax[2]], orientation = 'vertical', fraction = 0.02, aspect = 80, label = polytestdat['tstparam'])

    if savefig:
        fig.savefig(outfile, **argssavefig)

    plt.show()

    rcParams.update(rcParamsOrig)

def plot_test_components(polytestdat: dict,
                         cmap: str = 'magma',
                         flux_fraction: bool = True,
                         mass_fraction: bool = False,
                         savefig: Optional[bool] = False,
                         argssavefig: Optional[dict] = {'bbox_inches': 'tight', 'dpi': 300},
                         outfile: Optional[str] = None,
                         age_step: Optional[int] = 2,
                         param_step: Optional[int] = 2
                         ): 
    '''
        Heatmap plotting function that plots the marginalized weights on a polynomial vs age and a polynomial vs metallicty grid

        Parameters
        ----------
        polytestdat
            Dictionary containing the test data for plotting.
        cmap
            String name of the matplotlib colormap to use for plotting the different components. Default is 'magma'.
        flux_fraction
            Boolean indicating whether to plot the flux fraction. Default is True.
        mass_fraction
            Boolean indicating whether to plot the mass fraction. Default is False.
        savefig
            Boolean indicating whether to save the figure.
        argssavefig
            Dictionary of arguments to pass to the savefig function.
        outfile
            String specifying the path and filename for the saved figure.
        age_step
            Integer specifying the step size for the age axis ticks. Default is 2.
        param_step
            Integer specifying the step size for the parameter axis ticks. Default is 2.
            
        
    '''
    templates = np.load(polytestdat['templatefile'], allow_pickle=True)[()]
    tstparam = polytestdat['tstparam']

    rcParamsOrig = rcParams.copy()

    # getting data shapes
    unique_ages = np.unique(np.log10(templates['ages']))
    unique_zs = np.unique(templates['zs'])
    param = polytestdat[tstparam]

    if flux_fraction:
        norm_weights = polytestdat['flux_fraction']
        label = 'Portion of total flux'
    elif mass_fraction:
        norm_weights = polynomial_test['mass_fraction']
        label = 'Fration of total fitted population mass'
    else:
        norm_weights = []
        for weights in polytestdat['stelweights']:
            norm_weights.append(weights / np.sum(weights))
        label = 'Normalized Weight'

    norm_weights = np.reshape(norm_weights, (-1, len(unique_zs), len(unique_ages)))
    norm_weights = norm_weights.transpose(0, 2, 1)

    # Marginalize
    age_heatmap_matrix = np.sum(norm_weights, axis=2) 

    zs_heatmap_matrix = np.sum(norm_weights, axis=1)

    # set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Age Distribution Heatmap
    im1 = ax1.imshow(age_heatmap_matrix, aspect='auto', cmap=cmap, origin='lower')
    ax1.set_title(f"{label} vs. {tstparam}", fontsize=13, fontweight="bold")
    ax1.set_ylabel(f"{tstparam}", fontsize=12)
    ax1.set_xlabel(r"$\log_{10}(\mathrm{Age / yr})$", fontsize=12)

    # Set precise tick positions and labels for Age
    ax1.set_yticks(range(len(param))[::param_step])
    ax1.set_yticklabels(param[::param_step])
    ax1.set_xticks(range(len(unique_ages))[::age_step])
    ax1.set_xticklabels([f"{age:.2f}" for age in unique_ages][::age_step], rotation=45)
    fig.colorbar(im1, ax=ax1, label=label)

    # Metallicity Distribution Heatmap
    im2 = ax2.imshow(zs_heatmap_matrix, aspect='auto', cmap=cmap, origin='lower')
    ax2.set_title(f"{label} vs. {tstparam}", fontsize=13, fontweight="bold")
    ax2.set_ylabel(f"{tstparam}", fontsize=12)
    ax2.set_xlabel("Metallicity", fontsize=12)

    # Set precise tick positions and labels for Metallicity
    ax2.set_yticks(range(len(param))[::param_step])
    ax2.set_yticklabels(param[::param_step])
    ax2.set_xticks(range(len(unique_zs)))
    ax2.set_xticklabels([f"{z:.3f}" for z in unique_zs], rotation=45)
    fig.colorbar(im2, ax=ax2, label=label)

    if savefig:
        fig.savefig(outfile, **argssavefig)
    
    plt.tight_layout()
    plt.show()

    rcParams.update(rcParamsOrig)

def polynomial_test(q3di: q3din.q3din,
                    tstparam: Literal['add_poly_degree', 'mult_poly_degree'], 
                    tstmin: Optional[int]=0,
                    tstmax: Optional[int]=5,
                    tststep: Optional[int]=1,
                    starmassfile: Optional[str]=None,
                    q3dfitargs: Optional[dict]={},
                    componentplots: Optional[bool]=False,
                    componentplotargs: Optional[dict]={},
                    quiet: Optional[bool]=True):
    '''
    Function to test running q3dfit multiple times while varying one parameter.

    Parameters
    ----------
    q3di
        :py:class:`~q3dfit.q3din.q3din` object with input parameters.
    tstparam
        Fitting parameter to be varied. Currently supports 'add_poly_degree' and 'mult_poly_degree'.
    tstmin
        Minmum value to be tested. Defaults to 0.
    tstmax
        Maximum value to be tested. Defailts to 5.
    tstsstep
        Step size between test values. defults to 1.
    q3dfitargs
        Dictionary arguments to pass to q3fit().
    starmassfile
        File containing the population mass data at the same ages and metallicities as the file
        in q3di.startempfile. Required to generate mass fractions in the output dictionary.
    componentplots
        If True, runs q3do.plotcontcomponents() after every fit. Defaults to False.
    componentplotargs
        Dictionary of arguments to pass to q3do.plotcontcomponents().
    quiet
        Sets internal q3dfit() and q3do method calls to quiet. Defaults to true.

    Returns
    -------
    polytestdat
        Dictionary containing the fits, weights, and parameters for all tests.
    '''
    from q3dfit.q3df import q3dfit
    from q3dfit.q3dout import load_q3dout

    tests = np.arange(tstmin, tstmax + 1, tststep)

    polytestdat = {'add_poly_degree' : [],
                    'mult_poly_degree' : [],
                    'av' : [],
                    'stelweights' : [],
                    'flux_fraction' : [],
                    'cont_fit' : [],
                    'chisq' : [],
                    'templatefile' : q3di.startempfile,
                    'tstparam' : '',
                    'wave' : [],
                    'cont_dat' : []}

    templates = np.load(q3di.startempfile, allow_pickle = True)[()]
    norms = templates['norm']

    initargscontfit = q3di.argscontfit

    parammap = {'add_poly_degree' : {'testparam' : 'apoly', 'fixedparam' : 'mult_poly_degree'},
                'mult_poly_degree' : {'testparam' : 'mpoly', 'fixedparam' : 'add_poly_degree'}}
    
    if tstparam in parammap:
        polytestdat['tstparam'] = tstparam#parammap[tstparam]['testparam']
        fixedparam = parammap[tstparam]['fixedparam']
    else:
        raise ValueError('Invalid tstparam')

    if starmassfile is not None:
        polytestdat['mass_fraction'] = []
        massfile = np.load(starmassfile, allow_pickle=True)[()]

    for tstval in tests: 
        q3di.argscontfit[tstparam] = tstval
        q3dfit(q3di, quiet=quiet, **q3dfitargs)

        q3d0 = load_q3dout(q3di)

        q3d0.ct_coeff['flux_fraction'] = q3d0.ct_coeff['stelweights'] / norms
        q3d0.ct_coeff['flux_fraction'] /= np.sum(q3d0.ct_coeff['flux_fraction'])

        if componentplots:
            q3d0.plot_cont_components(q3di, quiet=quiet, **componentplotargs)

        polytestdat[fixedparam].append(q3di.argscontfit[fixedparam])
        polytestdat[tstparam].append(q3di.argscontfit[tstparam])
        polytestdat['av'].append(q3d0.ct_coeff['av'])
        polytestdat['stelweights'].append(q3d0.ct_coeff['stelweights'])
        polytestdat['flux_fraction'].append(q3d0.ct_coeff['flux_fraction'])
        polytestdat['cont_fit'].append(q3d0.cont_fit)
        polytestdat['chisq'].append(q3d0.ct_rchisq)

        if 'mass_fraction' in polytestdat:
            polytestdat['mass_fraction'] = polytestdat['flux_fraction'] * massfile

    polytestdat['wave'] = q3d0.wave
    polytestdat['cont_dat'] = q3d0.cont_dat

    q3di.argscontfit = initargscontfit

    return polytestdat
