#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def checkcomp(linepars, linetie, ncomp, siglim, sigcut=None, ignore=[]):
    """
    Automatically search for "good" components.

    Parameters
    ----------
    linepars: dict
        Output from IFSF_SEPFITPARS
    linetie: dict
        Lines to which each fitted line is tied to
    ncomp: dict
        # of components for each fitted line
    siglim: float
        Sigma limits for emission lines.
    sigcut: float, optional
        Sigma threshold in flux for rejection of a component,
        default is 3.0
    ignore: array, optional
        Array of lines to ignore in looking for good copmonents,
        array of string-typed elements.

    Returns
    -------
    dict
        NEWNCOMP, dictionary of # of components for each unique linetie anchor.
        The input dictionary NCOMP is also updated to reflect correct new # of
        components.

    Notes
    -----

    """

    if sigcut is None:
        sigcut = 3.

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
            # Loop through lines tied to each anchor,
            # looking for good components
            for line in tiedlist:
                if line not in ignore:
                    igd = \
                        (linepars['flux'][line][:ncomp[line]] >
                         sigcut*linepars['fluxerr'][line][:ncomp[line]]) \
                        & (linepars['fluxerr'][line][:ncomp[line]] > 0.) \
                        & (linepars['sigma'][line][:ncomp[line]] > siglim[0]) \
                        & (linepars['sigma'][line][:ncomp[line]] < siglim[1])
                    if igd.any():
                        goodcomp[np.where(igd)] = 1
            tmpncomp = goodcomp.sum()
            if tmpncomp < ncomp[key]:
                newncomp[key] = tmpncomp
                #  Loop through lines tied to each anchor and
                # set proper number of components
                for line in tiedlist:
                    ncomp[line] = tmpncomp

    return(newncomp)
