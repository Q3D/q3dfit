# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:29:06 2020

@author: lily
"""

import numpy as np

def gdtemp (infile, outfile):
    
    file= open(infile, 'r')
    
    #create ages
    for i in range(43):
        file.readline()
        
    agest=file.readline()
    agest= agest.split()
    ages=np.zeros(46, dtype=float)
    for i in range(len(agest)):
        if i>2:
            j=i-3
            agest[i]=float(agest[i])
            ages[j]=agest[i]
     
    #create full data
    header=file.readline()
    
    datarr= np.zeros((13321, 139), dtype=float)
    count = 0
    for i in file.readlines():
        for j in i.split():
            if count < 1851619:
                row = count % 13321
                column= count//13321
                datarr[row][column]= j
                count+=1
    data=np.transpose(datarr)
    data=np.reshape(data, (13321, 139))
    
    #create lambda
    Lambda=np.zeros(13321, dtype=float)
    for i in range (13321):
        Lambda[i]=data[i][0]
        
    #create flux
    ilum=np.zeros(46, dtype=int)
    c=0
    for i in range(139):
        if i%3==1:
                ilum[c]=i
                c=c+1
                
    flux=np.zeros((13321, 46), dtype=float)
    for i in range(13321):
        for j in range (46):
            k=ilum[j]
            flux[i][j]=data[i][k]
    maxf=maxium(flux, 13321, 46)
    flux=(flux/maxf)
    
    templatedict= {'lambda':Lambda, 'flux':flux, 'ages':ages}
    np.save(outfile, templatedict)
    
def maxium (arr, r, c): 
  
    mx = arr[0][0] 
    for i in range(0, r):
        for j in range(0, c):
            if arr[i][j] > mx: 
                mx = arr[i][j]
    return mx

gdtemp('SSPGeneva.z020', 'gdtempoutfile')