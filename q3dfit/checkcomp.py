#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['checkcomp']

from typing import Any
import numpy as np


def checkcomp(linepars: dict[str, Any], 
              linetie: dict[str, str], 
              ncomp: dict[str, int], 
              siglim: dict[str, np.ndarray],
              sigcut: float=3., 
              subone: bool=False, 
              ignore: list[str]=[]) \
                -> dict[str, int]:
    """
    Automatically search for "good" components.

    Parameters
    ----------
    linepars
        :py:attr:`~q3dfit.q3dout.q3dout.line_fitpars` attribute of 
        :py:class:`~q3dfit.q3dout.q3dout`
    linetie
        :py:attr:`~qrdfit.q3din.q3din.linetie` attribute of
        :py:class:`~q3dfit.q3din.q3din`
    ncomp
        # of components for each fitted line.
    siglim
        Sigma limits for emission lines.
    sigcut
        Optional. Sigma threshold in flux for rejection of a component.
        Default is 3.
    subone
        Optional. If True, remove only one component if multiple lines are found to be
        insignificant. Useful when degeneracy between components
        lowers significance for all components, but some components
        still exist. Default is False.
    ignore
        Optional. Array of lines to ignore in looking for good copmonents,
        array of string-typed elements. Default is empty list.

    Returns
    -------
    dict[str, int]
        # of good components for each unique linetie anchor. The input ncomp is also 
        updated to reflect correct new # of components.

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
    uanchors = np.unique(sorted(linetie.values())) # type: ignore
    for key in uanchors:
        newlinetie[key] = list()
    for key, val in linetie.items():
        newlinetie[val].append(key)

    # Loop through anch
    for key, tiedlist in newlinetie.items():
        if ncomp[key] > 0:
            goodcomp = np.zeros(ncomp[key], dtype=np.int8)
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
                        (linepars['sigma'][line][:ncomp[line]] > 
                         siglim[line][:ncomp[line],0])
                    igdsighi = \
                        (linepars['sigma'][line][:ncomp[line]] < 
                         siglim[line][:ncomp[line],1])
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

    return newncomp
