#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:38:38 2021

@author: drupke

;
; Automatically search for "good" components.
;
; :Categories:
;    IFSFIT
;
; :Returns:
;
;     NEWNCOMP, hash of # of components for each unique linetie anchor.
;     The input hash NCOMP is also updated to reflect correct new # of components.
;
; :Params:
;
;     linepars: in, required, type=hash
;        Output from IFSF_SEPFITPARS
;     linetie: in, required, type=hash
;        Lines to which each fitted line is tied to
;     ncomp: in, required, type=hash
;        # of components for each fitted line
;     siglim: in, required, type=double(2)
;        Sigma limits for emission lines.
;
; :Keywords:
;     sigcut: in, optional, type=double, default=3d
;        Sigma threshold in flux for rejection of a component.
;     ignore: in, optional, type=strarr
;        Array of lines to ignore in looking for good copmonents.
;
; :Author:
;    David S. N. Rupke::
;      Rhodes College
;      Department of Physics
;      2000 N. Parkway
;      Memphis, TN 38104
;      drupke@gmail.com
;
; :History:
;    ChangeHistory::
;      2014apr30, DSNR, copied from GMOS_CHECKCOMP; rewrote, documented,
;                       added copyright and license
;      2015sep04, DSNR, added some logic to keep BLR components;
;                       not heavily tested
;      2015dec14, DSNR, added test for pegging on upper limit of sigma
;      2016oct08, DSNR, changed flux peak sigma cutoff to total flux cutoff, now
;                       that total flux errors are correctly computed
;      2017aug10, DSNR, will now accept components that hit lower limit in sigma
;                       (previously had to be above lower limit)
;      2021jan20, DSNR, updated documentation, input/output
;      2021jan20, DSNR, translated to Python
;
; :Copyright:
;    Copyright (C) 2014--2021 David S. N. Rupke
;
;    This program is free software: you can redistribute it and/or
;    modify it under the terms of the GNU General Public License as
;    published by the Free Software Foundation, either version 3 of
;    the License or any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;    General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with this program.  If not, see
;    http://www.gnu.org/licenses/.
;-

"""

import numpy as np
import pdb


def checkcomp(linepars, linetie, ncomp, siglim,
              sigcut=None, ignore=[]):

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
                    goodcomp[np.where(igd == True)] = 1
            tmpncomp = goodcomp.sum()
            if tmpncomp < ncomp[key]:
                newncomp[key] = tmpncomp
                #  Loop through lines tied to each anchor and
                # set proper number of components
                for line in tiedlist:
                    ncomp[line] = tmpncomp

    return(newncomp)
