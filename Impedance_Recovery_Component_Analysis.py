#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:04:19 2019

@author: wenninger
"""
from matplotlib.colors import LogNorm 
import matplotlib.pylab as plt
import numpy as np
import os

from Mixer import Mixer,kwargs_Mixer_John

directory = 'Impedance_Recovery/Errorsurfaces_2019_12_08/'
if not os.path.exists(directory):
        os.makedirs(directory)

M = Mixer('DummyData/John/Unpumped.csv','DummyData/John/Pumped.csv',**kwargs_Mixer_John)

#vrangeEvaluated = M.mask_photon_steps
#vrangeEvaluated = vrangeEvaluated[np.logical_and(vrangeEvaluated>0,vrangeEvaluated<M.Unpumped.gapVoltage)] 
#The limit of the voltage lower than the gap voltage implies that only photon steps below the transission are token, not the bump beyond the transission.

#ySISreduced = M.ySIS[:,np.isin(M.ySIS[0],M.mask_photon_steps[M.mask_photon_steps>0])]
ySISreduced = M.ySIS_masked_positive
pumping_Levels = M.pumping_Levels_Volt_masked_positive

real=np.arange(0.001,.4,0.0005)
imag = np.arange(-.05,.2,0.0005)
#real=np.arange(0.001,.1,.001)
#imag = np.arange(-.1,.1,0.001)



description = 'Embedding_Admittance_Error_Surface'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,M.yEmb_Errorsurface(ySISreduced,pumping_Levels,real=real,imag=imag).T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.rcParams.update({'font.size': 12})
plt.grid(True)
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Embedding_Admittance_Error_Surface_iLO'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag, M.yEmb_Errorsurface_iLO(ySISreduced,pumping_Levels,real=real,imag=imag).T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'First_Error_Term'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,M.evaluate_first_yEmb_Error_Term(ySISreduced,pumping_Levels,real=real,imag=imag).real.T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Second_Error_Term'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,-M.evaluate_second_yEmb_Error_Term(ySISreduced,pumping_Levels,real=real,imag=imag).T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')  
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Current_LO_real'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,M.evaluate_current_LO_from_Embedding_Circuit(ySISreduced,pumping_Levels,real=real,imag=imag).real.T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Current_LO_abs'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,abs(M.evaluate_current_LO_from_Embedding_Circuit(ySISreduced,pumping_Levels,real=real,imag=imag)).T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Current_LO_Square'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,np.square(M.evaluate_current_LO_from_Embedding_Circuit(ySISreduced,pumping_Levels,real=real,imag=imag)).T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Voltage_over_sqrt_Total_Admittance'
print('Process ' + description)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,np.sum(np.divide(ySISreduced[0],np.sqrt(M.evaluate_total_admittance(ySISreduced,real=real,imag=imag)[:,:,1])),axis=-1).real.T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Sqrt_Total_Admittance'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,np.sum(np.sqrt(M.evaluate_total_admittance(ySISreduced,real=real,imag=imag)[:,:,1]),axis=-1).real.T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Sum_Reciprocal_Total_Admittance'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,np.sum(np.reciprocal(M.evaluate_total_admittance(ySISreduced,real=real,imag=imag)[:,:,1]),axis=-1).real.T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

description = 'Total_Admittance_at_certain_Voltage'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag,M.evaluate_total_admittance(ySISreduced,real=real,imag=imag)[:,:,1,40].real.T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Y$ $\mho$')
ax.set_ylabel('Imaginary $Y$ $\mho$')
plt.savefig(directory+description+'.pdf')
plt.close()

v2='The sum of all evaluated voltages square: %f $V^2$'%(np.sum(np.square(pumping_Levels[1]*1e-3)))
print(v2)
Log = open(directory+'Sum_Voltage_Square.txt','w')
Log.write(v2)
Log.close()

real=np.arange(0.1,15,.1)
imag=np.arange(-15,15,.1)

description = 'Embedding_Impedance_Error_Surface'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.pcolor(real,imag, M.zEmb_Errorsurface(ySISreduced,pumping_Levels,real=real,imag=imag).T,norm=LogNorm())
plt.colorbar()
plt.title(description.replace('_',' '))
ax.set_xlabel('Real $Z$ $\Omega$')
ax.set_ylabel('Imaginary $Z$ $\Omega$')
plt.savefig(directory+description+'.pdf')
plt.close()

     



#ySS = M.evaluate_total_admittance(ySISreduced,real=np.arange(0.0001,.1,.0001),imag = np.arange(-.05,.05,0.0001))[:,:,1]
#
#ySSr = np.reciprocal(ySS)
#ySr = np.sqrt(ySSr)
#
#sySSr = np.sum(ySSr,axis=-1)
#sySr = np.sum(ySr,axis=-1)
#
#first = M.evaluate_first_yEmb_Error_Term(ySISreduced)
#second = M.evaluate_second_yEmb_Error_Term(ySISreduced)
#
#vSum = np.sum(ySISreduced[0]*1e-3)
#
#svySSr = np.sum(np.multiply(ySISreduced[0],ySSr),axis=-1)
#
#svySr = np.sum(np.multiply(ySISreduced[0],ySr),axis=-1)
#
#iLO = M.evaluate_current_LO_from_Embedding_Circuit(ySISreduced,real=np.arange(0.0001,.1,.0001),imag = np.arange(-.05,.05,0.0001))
#
#iLOS = np.square(iLO)
#
#yEmb = M.yEmb_Errorsurface(ySISreduced,real=np.arange(0.0001,.1,.0001),imag = np.arange(-.05,.05,0.0001))