#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:51:43 2021

@author: jakubpazio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

# Fitting function
def func(x, a):
    # return a*np.exp(b*x) + c
    return a * x 
    #return a*x+b
 
#Experimental x and y data points    
xData = np.array([150, 356, 238, 192,10.56,21.72])
yData = np.array([511, 1200, 834, 662, 26.3, 59.6])
'[Na_22,Na_22,Mn_54,Cs_137,Am_241,Am_241]'

No_of_Bins = [55,80,45,55,20,15]
Gaussian_Wids = [97.4,194.8,140.0,105.9,3.94,4.90]          #Wid parameter of each Gaussian given by Calibration code
sigma = np.sqrt(Gaussian_Wids)/np.sqrt(2)                   #Since we designed the gaussian function, we know how its parameters are related to standdev
errors = sigma / np.sqrt(No_of_Bins)                        #Error on each point given by standard deviation of the mean

#Plot experimental data points

 
# Initial guess for the parameters
initialGuess = [10.0]    
 
#Perform the curve-fit
popt, pcov = curve_fit(func, xData, yData, initialGuess, sigma = 3.4152*errors)
Gradient = [popt,pcov]
print('Gradient =',popt[0])
print('Error in Gradient =',pcov[0][0])
 
#x values for the fitted function
xFit = np.arange(0.0, 400.0, 0.01)
 
#Plot the fitted function
def PlotCalibration():
    plt.plot(xData, yData, 'bo', label='experimental-data')
    plt.plot(xFit, func(xFit, *popt), 'r', label='Fit Parameters: Gradient = %5.4f ±%5.4f' % (popt,pcov))
    plt.xlabel('Bin Number')
    plt.ylabel(' Energy [keV]')
    plt.errorbar(xData, yData, yerr=None, xerr=errors,fmt='o')
    plt.legend()
    plt.show()