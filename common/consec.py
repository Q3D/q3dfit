#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:15:16 2020

@author: hadley
"""

import numpy as np

def consec(a, l = None, h = None, n = None, same = None, distribution = None):
    nel = len(a)
    """ """
    if nel == 1:
        l = 0
        h = 0
        n = 1
    elif nel == 2:
        if same != None:
            if a[1] - a[0] == 0:
                l = 0
                h = 1
                n = 1
            else:
                l = -1
                h = -1
                n = 0
        else:
            if abs(a[1] - a[0]) == 1:
                l = 0
                h = 1
                n = 1
            else:
                l = -1
                h = -1
                n = 0
    else:
        if same == None:
            #adding padding
            temp = np.concatenate(([a[0]], a))
            arr = np.concatenate((temp, [a[-1]]))

            shiftedright = np.roll(arr, 1)
            shiftedleft = np.roll(arr, -1)


            cond1 = np.absolute(np.subtract(arr, shiftedright)) == 1
            cond2 = np.absolute(np.subtract(arr, shiftedleft)) == 1

        else:
            #adding padding
            temp = np.concatenate(([a[0] + 1], a))
            arr = np.concatenate((temp, [a[-1] - 1]))

            shiftedright = np.roll(arr, 1)
            shiftedleft = np.roll(arr, -1)

            cond1 = np.absolute(np.subtract(arr, shiftedright)) == 0
            cond2 = np.absolute(np.subtract(arr, shiftedleft)) == 0

        #getting rid of padding
        cond1 = cond1[1: -1]
        cond2 = cond2[1: -1]

        #making l
        l = [0]
        l.pop(0)
        for i in range (0, nel):
            if cond2[i] and cond1[i] == False:
                l.append(i)
        nl = len(l)

        #making h
        h = [0]
        h.pop(0)
        for i in range (0, nel):
            if cond1[i] and cond2[i] == False:
                h.append(i)
        nh = len(h)

        if nh * nl == 0:
            l = -1
            h = -1
            n = 0

        else: n = min(nh, nl)

    if l[0] != h[0]: dist = np.subtract(h, l) + 1
    else: dist = 0

    return l, h, n