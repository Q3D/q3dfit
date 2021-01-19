#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:38:38 2021

@author: drupke

; docformat = 'rst'
;
;+
;
;
; :Categories:
;    IFSFIT
;
; :Returns:
;
; :Params:
;
; :Keywords:
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
;
; :Copyright:
;    Copyright (C) 2014--2016 David S. N. Rupke
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


def ifsf_checkcomp(linepars, linetie, ncomp, newncomp, siglim,
                   sigcut=None, blrlines=None, blrcomp=None, ignore=None):

    if sigcut is None:
        sigcut = 3.

    newncomp = dict()

    # Find unique anchors
    anchors = (linetie.values()).toarray()
    sanchors = anchors[sort(anchors)]
    uanchors = sanchors[uniq(sanchors)]

;  Find lines associated with each unique anchor. NEWLINETIE is a hash of lists,
;  with the keys being the lines that are tied TO and the list corresponding to
;  each key consisting of the tied lines.
   newlinetie = hash(uanchors)
   foreach val,newlinetie,key do newlinetie[key] = list()
   foreach val,linetie,key do newlinetie[val].add,key

;  Loop through anchors
   foreach tiedlist,newlinetie,key do begin
      if ncomp[key] gt 0 then begin
         goodcomp = intarr(ncomp[key])
;        Loop through lines tied to each anchor, looking for good components
         foreach line,tiedlist do begin
            ctgd = 0
            doignore=0b
            if keyword_set(ignore) then $
               foreach ignoreline,ignore do $
                  if ignoreline eq line then doignore=1b
            if ~doignore then begin
;               igd = where((linepars.fluxpk)[line,0:ncomp[line]-1] ge $
;                           sigcut*(linepars.fluxpkerr)[line,0:ncomp[line]-1] AND $
;                           (linepars.fluxpkerr)[line,0:ncomp[line]-1] gt 0 AND $
               igd = where((linepars.flux)[line,0:ncomp[line]-1] ge $
                           sigcut*(linepars.fluxerr)[line,0:ncomp[line]-1] AND $
                           (linepars.fluxerr)[line,0:ncomp[line]-1] gt 0 AND $
                           (linepars.sigma)[line,0:ncomp[line]-1] ge siglim[0] AND $
                           (linepars.sigma)[line,0:ncomp[line]-1] lt siglim[1],$
                           ctgd)
               if keyword_set(blrcomp) AND keyword_set(blrlines) then begin
                  foreach blr,blrlines do begin
                     if line eq blr then begin
                        foreach ind,blrcomp do begin
                           if ctgd gt 0 then begin
                              goodblr = where(ind-1 eq igd,ct)
                              if ct lt 0 then begin
                                 igd = [igd,ind-1]
                                 ctgd++
                              endif
                           endif else begin
                              igd = ind-1
                              ctgd = 1
                           endelse
                        endforeach
                     endif
                  endforeach
               endif
            endif
            if ctgd gt 0 then goodcomp[igd]++
         endforeach
;        Find number of good components
         tmpncomp = 0
         for i=0,ncomp[key]-1 do if goodcomp[i] gt 0 then tmpncomp++
         if tmpncomp ne ncomp[key] then begin
            newncomp[key]=tmpncomp
;           Loop through lines tied to each anchor and set proper number of
;           components
            foreach line,tiedlist do ncomp[line]=tmpncomp
         endif
      endif
   endforeach

end
