#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:10:28 2019

@author: wenninger
"""
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import fmin,fmin_slsqp,minimize,differential_evolution

from Mixer import Mixer,kwargs_Mixer_John
from plotxy import plot
from IV_Curve_Simulations import iV_Curve_Gaussian_Convolution_with_Excess_Critical_Current,iV_Curve_Gaussian_Convolution_Perfect,iV_Curve_Perfect_with_SubgapResistance,iV_Curve_Gaussian_Convolution_with_SubgapResistance,iV_Chalmers

kwargs_Mixer_John['vmin']=-6
kwargs_Mixer_John['vmax']=6


plt.close()

M = Mixer('DummyData/John/Unpumped.csv','DummyData/John/Pumped.csv',**kwargs_Mixer_John)

plot(M.Unpumped.offsetCorrectedBinedIVData)
subgapCurrent=8 # from plot
sigmaGaussian=.06
#plot(iV_Curve_Gaussian_Convolution_with_Excess_Critical_Current(vGap=M.Unpumped.gapVoltage,criticalCurrent=190,rN=M.Unpumped.rN,sigmaGaussian =sigmaGaussian,subgapLeakage=subgapCurrent))

## with Excess Current; Not Tested
#def cost(params,iVMeasured,rN):
#    critcalCurrent= params[0]
#    sigmaGaussian= params[1]
#    subgapLeakage= params[2]
#    vGap=params[3]
#    excessCriticalCurrent=params[4]
#    evaluated = iVMeasured.binedDataExpansion(iVMeasured.offsetCorrectedBinedIVData[0,-1]+20*sigmaGaussian)
#    sim = iV_Curve_Gaussian_Convolution_with_Excess_Critical_Current(evaluated[0],vGap,excessCriticalCurrent,critcalCurrent,sigmaGaussian,rN,subgapLeakage)
#    iVMeasured=iVMeasured.offsetCorrectedBinedIVData
#    return np.sum(np.abs(np.subtract(sim[1,np.isin(sim[0],iVMeasured[0])==True],iVMeasured[1])))
#
#guess =[190,.08,10,M.Unpumped.gapVoltage,4000]
##fit = fmin(cost,guess,args=(M.Unpumped.offsetCorrectedBinedIVData[:,np.where(M.Unpumped.offsetCorrectedBinedIVData[0]>=0)],M.Unpumped.gapVoltage,M.Unpumped.rN))
##fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.gapVoltage,M.Unpumped.rN))
#fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.rN))
#plot(iV_Curve_Gaussian_Convolution_with_Excess_Critical_Current(M.Unpumped.offsetCorrectedBinedIVData[0],fit[3],fit[4],fit[0],fit[1],
#                                   M.Unpumped.rN,fit[2]))

#Set subgap current to 0
#def cost(params,iVMeasured,rN):
#    vGap=params[0]
#    critcalCurrent= params[1]
#    sigmaGaussian= params[2]
#    #subgapLeakage= params[3]
#    #Extend Voltage, since the simulatedd curve needs to chop frequenceis at the boundaries 
#    evaluated = iVMeasured.binedDataExpansion(iVMeasured.offsetCorrectedBinedIVData[0,-1]+7*sigmaGaussian)
#    sim = iV_Curve_Gaussian_Convolution_Perfect(evaluated[0],vGap,critcalCurrent,sigmaGaussian,rN,0)
#    iVMeasured=iVMeasured.offsetCorrectedBinedIVData
#    return np.sum(np.abs(np.subtract(sim[1,np.isin(sim[0],iVMeasured[0])==True],iVMeasured[1])))
#
#guess =[M.Unpumped.gapVoltage,190,.08]
##fit = fmin(cost,guess,args=(M.Unpumped.offsetCorrectedBinedIVData[:,np.where(M.Unpumped.offsetCorrectedBinedIVData[0]>=0)],M.Unpumped.gapVoltage,M.Unpumped.rN))
##fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.gapVoltage,M.Unpumped.rN))
#fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.rN))
#
#plot(iV_Curve_Gaussian_Convolution_Perfect(np.arange(-5,5,1e-3),fit[0],fit[1],fit[2],M.Unpumped.rN,0))

#Constant Subgap resistance
#def cost_Subgap(params,iVMeasured,rN):
#    vGap=params[0]
#    critcalCurrent= params[1]
#    sigmaGaussian= params[2]
#    subgapLeakage= params[3]
#    #Extend Voltage, since the simulatedd curve needs to chop frequenceis at the boundaries 
#    evaluated = iVMeasured.binedDataExpansion(iVMeasured.offsetCorrectedBinedIVData[0,-1]+7*sigmaGaussian)
#    sim = iV_Curve_Gaussian_Convolution_Perfect(evaluated[0],vGap,critcalCurrent,sigmaGaussian,rN,subgapLeakage)
#    iVMeasured=iVMeasured.offsetCorrectedBinedIVData
#    return np.sum(np.abs(np.subtract(sim[1,np.isin(sim[0],iVMeasured[0])==True],iVMeasured[1])))
#
#guess =[M.Unpumped.gapVoltage,190,.08,10]
##fit = fmin(cost,guess,args=(M.Unpumped.offsetCorrectedBinedIVData[:,np.where(M.Unpumped.offsetCorrectedBinedIVData[0]>=0)],M.Unpumped.gapVoltage,M.Unpumped.rN))
##fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.gapVoltage,M.Unpumped.rN))
#fit = fmin(cost_Subgap,guess,args=(M.Unpumped,M.Unpumped.rN))

#plot(iV_Curve_Gaussian_Convolution_Perfect(np.arange(-5,5,1e-3),fit[0],fit[1],fit[2],M.Unpumped.rN,fit[3]))


#def cost_Excess(params,iVMeasured,rN):
#    vGap=params[0]
#    critcalCurrent= params[1]
#    sigmaGaussian= params[2]
#    subgapLeakage= params[3]
#    excessCriticalCurrent=[4]
#    #Extend Voltage, since the simulatedd curve needs to chop frequenceis at the boundaries 
#    evaluated = iVMeasured.binedDataExpansion(iVMeasured.offsetCorrectedBinedIVData[0,-1]+7*sigmaGaussian)
#    sim = iV_Curve_Gaussian_Convolution_with_Excess_Critical_Current(evaluated[0],vGap,excessCriticalCurrent,critcalCurrent,sigmaGaussian,rN,subgapLeakage)
#    iVMeasured=iVMeasured.offsetCorrectedBinedIVData
#    return np.sum(np.abs(np.subtract(sim[1,np.isin(sim[0],iVMeasured[0])==True],iVMeasured[1])))
#
#guess =[M.Unpumped.gapVoltage,190,.08,10,4000]
##fit = fmin(cost,guess,args=(M.Unpumped.offsetCorrectedBinedIVData[:,np.where(M.Unpumped.offsetCorrectedBinedIVData[0]>=0)],M.Unpumped.gapVoltage,M.Unpumped.rN))
##fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.gapVoltage,M.Unpumped.rN))
#fit = fmin(cost_Excess,guess,args=(M.Unpumped,M.Unpumped.rN))
#
#plot(iV_Curve_Gaussian_Convolution_with_Excess_Critical_Current(M.Unpumped.offsetCorrectedBinedIVData[0],fit[0],fit[4],fit[1],fit[2],M.Unpumped.rN,fit[3]))

#Include subgap resistance term with full parammeters 
def cost_Subgap(params,iVevaluated,iVcompared,rN):
    vGap=params[0]
    critcalCurrent= params[1]
    sigmaGaussian= params[2]
    subgapLeakage= params[3]
    subgapLeakageOffset = params[4]
    excessCriticalCurrent = params[5]
    #Extend Voltage, since the simulatedd curve needs to chop frequenceis at the boundaries 
    evaluated = iVevaluated# iVMeasured.binedDataExpansion(iVMeasured.offsetCorrectedBinedIVData[0,-1]+7*sigmaGaussian)
    sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(evaluated[0],vGap,excessCriticalCurrent=excessCriticalCurrent,criticalCurrent=critcalCurrent,sigmaGaussian =sigmaGaussian,rN=rN,subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
    return np.sum(np.abs(np.subtract(sim[1,np.isin(sim[0],iVcompared[0])==True],iVcompared[1])))

#guess =[M.Unpumped.gapVoltage,190,.08,M.Unpumped.rSG,10,1000]
bounds =[(M.Unpumped.vGapSearchRange[0],M.Unpumped.vGapSearchRange[1]),
         (M.Unpumped.criticalCurrent-100,M.Unpumped.criticalCurrent+100),
         (0,.2),
         (M.Unpumped.rSG-100,M.Unpumped.rSG+100),
         (0,30),
         (M.Unpumped.criticalCurrent-100,10000)
         ]
#fit = fmin(cost,guess,args=(M.Unpumped.offsetCorrectedBinedIVData[:,np.where(M.Unpumped.offsetCorrectedBinedIVData[0]>=0)],M.Unpumped.gapVoltage,M.Unpumped.rN))
#fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.gapVoltage,M.Unpumped.rN))
#This one yields good results
#fit = fmin(cost_Subgap,guess,args=(M.Unpumped,M.Unpumped.rN))
#bounds=np.full([len(guess),2],None)
#bounds[2,0]=0
#bounds[2,1]=.15 #This limit might be extended for a very wide IV response
#bounds[4,0]=0
fit = differential_evolution(cost_Subgap,bounds=bounds,args=(M.Unpumped.binedDataExpansion(M.Unpumped.offsetCorrectedBinedIVData[0,-1]+7*.15),M.Unpumped.offsetCorrectedBinedIVData,M.Unpumped.rN))

#plot(iV_Curve_Gaussian_Convolution_with_SubgapResistance(np.arange(-5,5,1e-3),fit[0],fit[5],fit[1],fit[2],M.Unpumped.rN,fit[3],fit[4]))
plot(iV_Curve_Gaussian_Convolution_with_SubgapResistance(np.arange(-5,5,1e-3),fit.x[0],fit.x[5],fit.x[1],fit.x[2],M.Unpumped.rN,fit.x[3],fit.x[4]))
#
#
##Include subgap resistance term with less parameters 
#def cost_Subgap(params,iVMeasured,rN,rSG,vGap):
#    critcalCurrent= params[0]
#    sigmaGaussian= params[1]
#    subgapLeakage= rSG
#    subgapLeakageOffset = params[2]
#    excessCriticalCurrent = params[3]
#    #Extend Voltage, since the simulatedd curve needs to chop frequenceis at the boundaries 
#    evaluated = iVMeasured.binedDataExpansion(iVMeasured.offsetCorrectedBinedIVData[0,-1]+7*sigmaGaussian)
#    sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(evaluated[0],vGap,excessCriticalCurrent=excessCriticalCurrent,criticalCurrent=critcalCurrent,sigmaGaussian =sigmaGaussian,rN=rN,subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
#    iVMeasured=iVMeasured.offsetCorrectedBinedIVData
#    return np.sum(np.abs(np.subtract(sim[1,np.isin(sim[0],iVMeasured[0])==True],iVMeasured[1])))
#
#guess =[190,.08,10,1000]
##fit = fmin(cost,guess,args=(M.Unpumped.offsetCorrectedBinedIVData[:,np.where(M.Unpumped.offsetCorrectedBinedIVData[0]>=0)],M.Unpumped.gapVoltage,M.Unpumped.rN))
##fit = fmin(cost,guess,args=(M.Unpumped,M.Unpumped.gapVoltage,M.Unpumped.rN))
##This one yields good results
##fit = fmin(cost_Subgap,guess,args=(M.Unpumped,M.Unpumped.rN))
#bounds=np.full([len(guess),2],None)
#bounds[2,0]=0
#fit = minimize(cost_Subgap,guess,bounds=bounds,args=(M.Unpumped,M.Unpumped.rN,M.Unpumped.rSG,M.Unpumped.gapVoltage))
#
#plot(iV_Curve_Gaussian_Convolution_with_SubgapResistance(np.arange(-5,5,1e-3),M.Unpumped.gapVoltage,fit.x[3],fit.x[0],fit.x[1],
 #                                                        M.Unpumped.rN,M.Unpumped.rSG,fit.x[2]))



def cost_Chalmers(params,iVMeasured,Dummy):
    sim = iV_Chalmers(iVMeasured[0],params[0],params[1],params[2],params[3])
    return np.sum(np.abs(np.subtract(sim,iVMeasured[1])))

guess =[30,M.Unpumped.gapVoltage,M.Unpumped.rN,M.Unpumped.rSG]
fit = fmin(cost_Chalmers,guess,args=(M.Unpumped.rawIVDataOffsetCorrected,1))
plot(iV_Chalmers(M.Unpumped.offsetCorrectedBinedIVData[0],fit[0],fit[1],fit[2],fit[3]))



