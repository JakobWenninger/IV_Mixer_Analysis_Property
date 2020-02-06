#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:48:32 2019

@author: wenninger
"""
#TODO/Note: The way the program is written, the program works only under Spyder IPython console, which is not closing a plot after savefig.

import os
from matplotlib.colors import LogNorm
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import numpy as np

from Mixer import Mixer,kwargs_Mixer_John,normalise_2d_array
from plotxy import plot, pltsettings,lbl

plt.close() # in case a plot is still open

headDirectory = 'Impedance_Recovery/Simulated_Curves_Normalised_1_2019_12_10/'

#initialise a Mixer object
M = Mixer('DummyData/John/Unpumped.csv','DummyData/John/Pumped.csv',**kwargs_Mixer_John)
    
M.Unpumped.convolution_most_parameters_Fit_Calc()
M.Unpumped.convolution_without_excessCurrent_Fit_Calc()

vGap = M.Unpumped.gapVoltage
rN = M.Unpumped.rN
cC = vGap*1e3/rN

fits = [M.Unpumped.chalmers_Fit,M.Unpumped.convolution_Fit,M.Unpumped.convolution_without_excessCurrent_Fit,M.Unpumped.convolution_perfect_IV_curve_Fit]
directories = ['Chalmers','Convolution','Convolution_without_excessCurrent','Convolution_perfect_IV']
for i in range(len(fits)):
    print('Process '+directories[i])
    
    # set directory for images
    directory =headDirectory+directories[i]+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #Define the used fit in the mixer calculations
    M.Unpumped.set_simulatedIV(fits[i])
    
    res = M.pumping_Levels_Volt
    res = np.divide(res,vGap)
    plot(res, label='Pumping Level')
    description='Pumping_Level_Volt_over_gap_voltage'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['gap'],title=description.replace('_',' '),close=True)
    
    plot(M.pumping_Levels, label='Pumping Level')
    description='Pumping_Level'
    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel='alhpa',title=description.replace('_',' '),close=True)
    
    plot(normalise_2d_array(M.Unpumped.rawIVDataOffsetCorrected,vGap,cC),'Measurement')
    plot(normalise_2d_array(M.Unpumped.simulatedIV,vGap,cC),'Simulation')
    description='Fitting_to_Measurement'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['cc'],xlim=[-2,2],ylim=[-2,2],title=description.replace('_',' '),close=False)
    description='Fitting_to_Measurement_Zoom_Normal_Resistance'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['cc'],xlim=[1.1,2],ylim=[.9,2],title=description.replace('_',' '),close=False)
    description='Fitting_to_Measurement_Zoom_Subgap_Resistance'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['cc'],xlim=[-.8,.8],ylim=[-.1,.1],title=description.replace('_',' '),close=False)
    description='Fitting_to_Measurement_Zoom_Transission'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['cc'],xlim=[.75,.95],ylim=[0,1.1],title=description.replace('_',' '),close=True)
    
#    M.plot_simulated_and_measured_Unpumped_Pumped_IV_curves()
#    description='Pumped_from_Unpumped_pos'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[0,6.],ylim=[0,400],title=description.replace('_',' '),close=False)
#    description='Pumped_from_Unpumped_neg'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[-6,0],ylim=[-400,0],title=description.replace('_',' '),close=False)
#    description='Pumped_from_Unpumped_Subgap'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[-2.7,2.7],ylim=[-50,50],title=description.replace('_',' '),close=False)
#    description='Pumped_from_Unpumped_Subgap_pos'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[-.1,2.7],ylim=[-2,63],title=description.replace('_',' '),close=False)
#    description='Pumped_from_Unpumped_Subgap_neg'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[-2.7,.1],ylim=[-63,2],title=description.replace('_',' '),close=False)
#    description='Pumped_from_Unpumped_Normal_Resistance_pos'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[2.7,4.7],ylim=[160,310],title=description.replace('_',' '),close=False)
#    description='Pumped_from_Unpumped_Normal_Resistance_neg'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[-4.7,-2.65],ylim=[-310,-140],title=description.replace('_',' '),close=True)
    
    M.plot_simulated_and_measured_AC_currents_normalised()
    description='SIS_Current_pos'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['cc'],xlim=[-.03,2],ylim=[-1,1],title=description.replace('_',' '),close=False)
    description='SIS_Current_neg'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['cc'],xlim=[-2,.03],ylim=[-1,1],title=description.replace('_',' '),close=True)
    
    M.plot_simulated_and_measured_ySIS_normalised()
    description='SIS_Admittance_pos'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['YR'],xlim=[0,2],ylim=[-4,4],title=description.replace('_',' '),legendColumns=2,close=False)
    description='SIS_Admittance_neg'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['YR'],xlim=[-2,0],ylim=[-4,4],title=description.replace('_',' '),legendColumns=2,close=True)
    
#    plt.plot(M.ySIS[0],np.abs(M.ySIS[1]),label='Measurement')
#    plt.plot(M.simulated_ySIS[0],np.abs(M.simulated_ySIS[1]),label='Simulation')
#    description='SIS_Admittance_Absolute_pos'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['Y'],xlim=[0,6],ylim=[0,.2],title=description.replace('_',' '),legendColumns=2,close=False)
#    description='SIS_Admittance_Absolute_neg'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['Y'],xlim=[-6,0],ylim=[0,.2],title=description.replace('_',' '),legendColumns=2,close=True)

    M.plot_simulated_and_measured_zSIS_normalised()
    description='SIS_Impedance_pos'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['ZR'],xlim=[0,2],ylim=[-2.5,2.5],title=description.replace('_',' '),close=False)
    description='SIS_Impedance_neg'
    pltsettings(save=directory+description,xlabel=lbl['gap'],ylabel=lbl['ZR'],xlim=[-2,0],ylim=[-2.5,2.5],title=description.replace('_',' '),close=True)
    
#    plt.plot(M.zSIS[0],np.abs(M.zSIS[1]),label='Measurement')
#    plt.plot(M.simulated_zSIS[0],np.abs(M.simulated_zSIS[1]),label='Simulation')
#    description='SIS_Impedance_Absolute_pos'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['Ohm'],xlim=[0,6],ylim=[0,30],title=description.replace('_',' '),legendColumns=2,close=False)
#    description='SIS_Impedance_Absolute_neg'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['Ohm'],xlim=[-6,0],ylim=[0,40],title=description.replace('_',' '),legendColumns=2,close=True)

#    M.plot_mask_steps()
#    description='Masked_Regions'
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['uA'],title=description.replace('_',' '),close=True)
    
#    description='Admittance_Complete'
#    yEmb = M.simulated_yEmb #TODO does not work
#    zEmb = M.simulated_zEmb
#    yPatch = mpatches.Patch(color='none', label='Y = %.2f %.2fj $\Omega^{-1}$'%(yEmb[0],yEmb[1])) 
#    zPatch = mpatches.Patch(color='none', label='Z = %.2f %.2fj $\Omega$'%(zEmb[0],zEmb[1])) 
#    
#    handles, labels = plt.gca().get_legend_handles_labels() # get existing handles and labels
#    handles.append(yPatch)  # add new patches and labels to list
#    handles.append(zPatch)  # add new patches and labels to list
#    labels.append('Y = %.2f %.2fj $\mho$'%(yEmb[0],yEmb[1]))
#    labels.append('Z = %.2f %.2fj $\Omega$'%(zEmb[0],zEmb[1]))
#    
#    plt.legend(handles, labels,loc='best', shadow=False,ncol=2)
#    leg = plt.gca().get_legend()
#    ltext  = leg.get_texts()  # all the text.Text instance in the legend
#    llines = leg.get_lines()  # all the lines.Line2D instance in the legend
#    plt.setp(ltext, fontsize='small')
#    plt.setp(llines, linewidth=1.5)      # the legend linewidth\z\
#            
#    pltsettings(save=directory+description,xlabel=lbl['mV'],ylabel=lbl['Y'],xlim=[-6,6],ylim=[-.2,.053],title=description.replace('_',' '),close=True,skip_legend=True)
    
    
