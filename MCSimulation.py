#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:59:50 2021

@author: robertoking
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import random
from scipy.stats import norm
import DataPlotting2 as DP


N = 500 #No of pulses


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

def Plot(Angle,Mean,Sigma,c,No):
    rand = []
    noise = []
    energies = []
    
    for i in range(No):
        rand.append(random())
        noise.append(norm.ppf(rand[i]))
        energies.append(noise[i]*Sigma+Mean)
        
    n, edges, patches = plt.hist(energies, bins = 30 , histtype = 'step' , lw = 0)
    plt.plot(edges[0:-1], n, 'o',color=c)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Count')
    plt.grid(True)
    plt.title('Monte Carlo Simulation')

    
    popt,pcov = curve_fit(gaussian,edges[0:-1],n,p0 = [N/10,Mean,Sigma])
    xvals = np.arange(350,800,step = 1)
    plt.plot(xvals,gaussian(xvals,popt[0],popt[1],popt[2]),color=c,label = '%s degrees Mean = %5.3f'%(Angle ,popt[1]))
    plt.legend()
    return(popt[1])

#%%
Plot(Angle=0,Mean=662, Sigma=7.27,c = 'gray',No=N)
Plot(Angle=20,Mean=DP.Popt_20[1],Sigma=np.sqrt(DP.Popt_20[2]/2),No=N//2,c='blue')
Plot(Angle=30,Mean=DP.Popt_30[1],Sigma=np.sqrt(DP.Popt_30[2]/2),No=N//2,c='orange')
Plot(Angle=45,Mean=DP.Popt_45[1],Sigma=np.sqrt(DP.Popt_45[2]/2),No=N,c='green')

#%%
Means = []
for i in range(500):
    Means.append(Plot(Angle=45,Mean=DP.Popt_45[1],Sigma=np.sqrt(DP.Popt_45[2]/2),No=N,c='green'))
    
print('Mean =',np.mean(Means),'Sigma =',np.std(Means))    
    
    
    
#%%

N = [100,200,400,800,1600,3200]
sigmas = [4.65,3.39,2.27,1.67,1.29,0.92] 

def rec(x,h):
    return h/(x**(1/2))

popt1, pcov1 = curve_fit(rec,N,sigmas,p0 = 542 )
xrange = np.arange(100,3300,5)
    
plt.plot(N,sigmas,'o',label = 'Simulation Points') 
plt.plot(xrange,rec(xrange,popt1),label = 'σ = 47/√N')
plt.xlabel('Number of Counts, N') 
plt.ylabel('Standard Deviation in Mean Energy [kev]')
plt.legend(fontsize = 'xx-large')
plt.title('Fluctuation in Mean Energy against Counts in the experiment',weight = 'bold',fontsize = 'large')
    
    
    
    
    