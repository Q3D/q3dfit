#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def checkcomp(linepars, linetie, ncomp, siglim, sigcut=3., subone=False, ignore=[]):
    """
    Automatically search for "good" components.

    Parameters
    ----------
    linepars: dicts
        line_fitpars attribute of a q3dout object
    linetie: dict
        linetie attribute of a q3din object
    ncomp: dict
        # of components for each fitted line
    siglim: float
        Sigma limits for emission lines.
    sigcut: float, optional, default=3.
        Sigma threshold in flux for rejection of a component
    subone: bool, optional, default=False
        Remove only one component if multiple lines are found to be
        insignificant. Useful when degeneracy between components
        lowers significance for all components, but some components
        still exist.
    ignore: array, optional
        Array of lines to ignore in looking for good copmonents,
        array of string-typed elements.

    Returns
    -------
    newncomp: dict
        dictionary of # of components for each unique linetie anchor.
        The input dictionary ncomp is also updated to reflect correct new # of
        components.

    Notes
    -----

    """

    # Output dictionary
    newncomp = dict()

    # Find lines associated with each unique anchor.
    # NEWLINETIE is a dict of lists,
    # with the keys being the lines that are tied TO and
    # the list corresponding to
    # each key consisting of the tied lines.
    newlinetie = dict()
    # Find unique anchors
    uanchors = np.unique(sorted(linetie.values()))
    for key in uanchors:
        newlinetie[key] = list()
    for key, val in linetie.items():
        newlinetie[val].append(key)

    # Loop through anch
    for key, tiedlist in newlinetie.items():
        if ncomp[key] > 0:
            goodcomp = np.zeros(ncomp[key], dtype=int)
            # badcomp = np.zeros(ncomp[key], dtype=int) + 1
            # Loop through lines tied to each anchor,
            # looking for good components
            for line in tiedlist:
                if line not in ignore:
                    # check for zeroed out lines
                    # for testing numerical issues with least_squares
                    # import pdb; pdb.set_trace()
                    # izero = ((linepars['fluxpk'][line][:ncomp[line]] < 1e-10) &
                    #          (np.isnan(linepars['fluxpkerr'][line][:ncomp[line]])))
                    # if izero.any():
                    #     goodcomp += 1
                    #     goodcomp[np.where(izero)] = 0
                    # else:
                    # Peak flux is insensitive to issues with sigma
                    #    (linepars['flux'][line][:ncomp[line]] >
                    #     sigcut*linepars['fluxerr'][line][:ncomp[line]]) \
                    igdflx = \
                        (linepars['fluxpk_obs'][line][:ncomp[line]] >
                         sigcut*linepars['fluxpkerr_obs'][line][:ncomp[line]])
                    igdsiglo = \
                        (linepars['sigma'][line][:ncomp[line]] > siglim[0])
                    igdsighi = \
                        (linepars['sigma'][line][:ncomp[line]] < siglim[1])
                    igd = igdflx & igdsiglo & igdsighi
                    # Removing this criterion fixes issues where
                    # fitter can't return parameter errors; focusing on
                    # fluxpkerr means can use empirical estimate rather
                    # than estimate from covariance matrix
                    #& (linepars['fluxerr'][line][:ncomp[line]] > 0.) \
                    if igd.any():
                        goodcomp[np.where(igd)] = 1
            tmpncomp = goodcomp.sum()
            if tmpncomp < ncomp[key]:
                if subone:
                    newncomp[key] = ncomp[key] - 1
                else:
                    newncomp[key] = tmpncomp
                #  Loop through lines tied to each anchor and
                # set proper number of components
                for line in tiedlist:
                    ncomp[line] = newncomp[key]

    return(newncomp)
