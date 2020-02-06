#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:40:08 2019

@author: wenninger
"""
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm 
import matplotlib
import numpy as np
import scipy.constants as const
from scipy.optimize import fmin, minimize
from scipy.special import jv

from plotxy import plot
from IV_Class import IV_Response, kwargs_IV_Response_rawData,kwargs_IV_Response_John

import seaborn as sns
sns.set_style("whitegrid",
              {'axes.edgecolor': '.2',
               'axes.facecolor': 'white',
               'axes.grid': True,
               'axes.linewidth': 0.5,
               'figure.facecolor': 'white',
               'grid.color': '.8',
               'grid.linestyle': u'-',
               'legend.frameon': True,
               'xtick.color': '.15',
               'xtick.direction': u'in',
               'xtick.major.size': 3.0,
               'xtick.minor.size': 1.0,
               'ytick.color': '.15',
               'ytick.direction': u'in',
               'ytick.major.size': 3.0,
               'ytick.minor.size': 1.0,
               })
sns.set_context("poster")
matplotlib.rcParams.update({'font.size': 8})


kwargs_Mixer = {
                'fLO':230.2e9 ,
                'tuckerSummationIndex':15, #Same as John used
                'steps_ImpedanceRecovery':1,
                'tHot':300.,
                'tCold':77.,
                'descriptionMixer':'' ,
                'skip_pumping_simulation':True
        }

kwargs_Mixer_rawData = {**kwargs_Mixer,**kwargs_IV_Response_rawData}
kwargs_Mixer_John = {**kwargs_Mixer,**kwargs_IV_Response_John}


class Mixer():
    '''This class represents a mixer including all its IV curves
    '''
    def __init__(self,Unpumped=None,Pumped=None,IFHot=None, IFCold=None,**kwargs):
        '''The initialising of the class.
        Note that it is the IV data is assumed to have the same binning properties.
        
        params
        ------
        Unpumped: object of :class: IV_Response or string
            The unpumped IV response of the mixer
        Pumped: object of :class: IV_Response or string
            The pumped IV response of the mixer
        IFHot: object of :class: IV_Response or string
            The IV response corresponding to a hot load
        IFCold: object of :class: IV_Response or string
            The IV response corresponding to a cold load
            
        kwargs
        ------
        fLO: float
            The frequency of the LO.
        tuckerSummationIndex: int
            The summation index n in the Tucker equations.
        steps_ImpedanceRecovery: int
            The number of photon steps considered for the impedance recovery.
        tHot: float
            The temperature of IF hot.
        tCold: float
            The temperature of IF cold.
        descriptionMixer: str
            A description of the Mixer, eg its temperature.
        skip_pumping_simulation: bool
            Decides if a simulated pumping levels are set during the initialisation of the class.
        + kwargs from IV_Response class.
        
        parameters (not complete)
        ----------
        vPh: float
            The voltage associated with a single photon of the LO.
        '''
        def handle_string_or_object(obj):
            '''This function is used to destinguish if a string is given as parameter or a object. 
            In case a string is given as parameter, a IV_Response object is initiated.
            
            inputs
            ------
            obj: object of :class: IV_Response or string
                The parameter which requires distinction if it is a object or a string
                
            return
            ------
            object of :class: IV_Response
                The initiated object of the IV response
            '''
#            if isinstance(obj,str): #obj is a single string
#                return IV_Response(obj,**self.__dict__)
#            el
            if not obj is None:#Handle the case that not all IV curves are defined
                if isinstance(obj,IV_Response): #obj is an object of class IV_Response
                    return obj  
                else: #obj is an array of strings
                    return IV_Response(obj,**self.__dict__)
                
            
        #preserve parameters
        self.__dict__.update(kwargs)
        self.vPh = const.h*self.fLO/const.e *1e3 # mV

        #Test if the IV binning properties are similar.
        self.Unpumped = handle_string_or_object(Unpumped)
        self.Pumped = handle_string_or_object(Pumped)
        self.IFHot = handle_string_or_object(IFHot)
        self.IFCold = handle_string_or_object(IFCold)
        #try: # to calculate the pumping level. A @property is impractical since the computation is extensive
        #Measured Data
        #compute the whole voltage range, relevant for the Kramers Kronig Transformation
        voltageLimit = np.abs([self.Unpumped.offsetCorrectedBinedIVData[0,-1]+self.tuckerSummationIndex*self.vPh,
                               self.Unpumped.offsetCorrectedBinedIVData[0,0]-self.tuckerSummationIndex*self.vPh]).max()
        self.pumping_Levels = self.pumping_Level_Calc(self.Unpumped.binedDataExpansion(voltageLimit),self.Pumped.offsetCorrectedBinedIVData)
        #TODO put this in a function set_pumping_Levels_Volt 
        self.pumping_Levels_Volt  = np.copy(self.pumping_Levels)
        self.pumping_Levels_Volt[1] = np.divide(const.h * self.fLO*self.pumping_Levels_Volt[1],const.e)*1e3#mV
      
        if not self.skip_pumping_simulation:
            #Simulated Data
            self.simulated_pumping_Levels = self.pumping_Level_Calc(self.Unpumped.simulatedIV,self.simulated_pumped_from_unpumped)
            self.simulated_pumping_Levels_Volt  = np.copy(self.simulated_pumping_Levels)
            self.simulated_pumping_Levels_Volt[1] = np.divide(const.h * self.fLO*self.simulated_pumping_Levels_Volt[1],const.e)*1e3#mV
                
    #        except:
    #            print('Computation of pumping levels failed.')

        
    def physical_Temperature_To_CW_Temperature(physicalTemperature,freq):
        '''This function converts the physical temperature to the effective temperature of an CW signal.
        
        inputs
        ------
        physicalTemperature: float
            The physical temperature of the source.
        freq: float
            The frequency of the CW source signal.
        '''
        return np.multiply(const.h*freq/(2*const.k),np.tanh(const.h*freq/(2*const.k*physicalTemperature)))
    
    def plot_IV_Un_Pumped(self):
        '''This function plots the unpumped and pumped IV curve.
        '''
        plot(self.Unpumped)
        plot(self.Pumped)
    
    def plot_IF_Hot_Cold(self):
        '''This function plots the hot and cold IF curve.
        '''
        plot(self.IFHot)
        plot(self.IFCold)
        
    def overlapping_Voltages_Indexes(self,v0,v1):
        '''This function computes the indexes of arrays where the values are in both arrays.
        This can be used to compute the location of the same gap voltages.
        
        inputs
        ------
        v0: 1d array
            The first array which is partially in v1.
        v1: 1d array
            The second array which is partially in v0.
            
        returns
        -------
        indexes0: 1d array of Booleans
            The indexes in the first array v0 which are found in v1.
        indexes1: 1d array of Booleans
            The indexes in the second array v1 which are found in v0.        
        '''
        indexes0 = np.isin(v0,v1)
        indexes1 = np.isin(v1,v0)
        return indexes0, indexes1
        
    @property
    def y_Factor(self):
        '''Compute the y factor from the hot and cold IF curve.
        
        returns
        -------
        3d array:
            [0] bias voltage
            [1] Y Factor
            [2] Sigma of Y factor
        '''
        hotIndexes,coldIndexes=self.overlapping_Voltages_Indexes(self.IFHot.binedIVData[0],self.IFCold.binedIVData[0])
        yfactor = np.divide(self.IFHot.binedIVData[1,hotIndexes],self.IFCold.binedIVData[1,coldIndexes])
        yfactor_sigma = np.multiply(yfactor,
                                    np.sqrt(np.add(np.square(np.divide(self.IFHot.binedIVData[2,hotIndexes],self.IFHot.binedIVData[1,hotIndexes])),
                                                   np.square(np.divide(self.IFCold.binedIVData[2,coldIndexes],self.IFCold.binedIVData[1,coldIndexes])))))
        return np.vstack([self.IFHot.binedIVData[0,hotIndexes],yfactor,yfactor_sigma])
    
    @property
    def noise_Temperature(self):
        '''Compute the noise temperature for each voltage bias from the y Factor and the temperatures at which the IF has been recorded
        
        returns
        -------
        3d array
            [0] bias voltage
            [1] noise temperature
            [2] Sigma of noise temperature
        '''
        noise_Temperature = np.divide(np.subtract(self.tHot,np.multiply(self.y_Factor[1],self.tCold)),np.subtract(self.y_Factor[1],1))
        sigma_Noise_Temperature = np.multiply(np.divide(np.subtract(self.tCold,self.tHot),np.square(np.subtract(self.y_Factor[1],1))),self.y_Factor[2])
        return np.vstack([self.y_Factor[0],noise_Temperature,sigma_Noise_Temperature])
    
    @property
    def simulated_pumped_from_unpumped(self):#,alphas, unpumpedExpanded, vPh,n):
        '''This function computes the pumped IV curve from the unpumped IV curve and the alpha value.
        #TODO update comments
        inputs
        ------
        alphas: 2d array
            [0] The bias voltage.
            [1] The alpha value at the given bias voltage.
        unpumpedExpanded: 2d array
            The IV data of the unpumped IV curve.
            The voltage data should be enough to include larege summation indexes n*vPh
        vPh: float
            Voltage equivalent of photons arriving in mV
        n: int
            The summation index of the function
    
        returns
        -------
        pumped: 2d array
            The IV data of the pumped IV curve
        '''
        alphas = self.pumping_Levels
        unpumpedExpanded = self.Unpumped.simulatedIV
        vPh=self.vPh
        n = self.tuckerSummationIndex

        #TODO updata with pumped_from_unpumped
        #.reshape((5,4))
        nas= np.arange(-n,n+1)
        na = np.hstack([nas]*len(alphas[0]))
        alphasy = np.vstack([alphas[1]]*(2*n+1)).T
        bessel = np.square(jv(na,alphasy.flatten())) # only positive values due to square
        bessel=bessel.reshape((-1,2*n+1))
        pumped = []
        for bias in range(len(alphas[0])):
            unpumpedOffseted = []
            for nx in nas:
                unpumpedOffseted.append(unpumpedExpanded[1,(np.abs(unpumpedExpanded[0]-(alphas[0,bias]+nx*vPh))).argmin()])
            pumped.append([unpumpedExpanded[0,(np.abs(unpumpedExpanded[0]-(alphas[0,bias]))).argmin()],np.nansum(bessel[bias]*unpumpedOffseted)])#[bias:(bias+2*n+1)]
        return np.array(pumped).T
        
    def plot_simulated_and_measured_Unpumped_Pumped_IV_curves(self):
        '''This function plots the simulated and measured IV curves for comparison.
        For the measured IV curves the offset corrected binned IV curves are displayed, since those are used to compute the pumping level alpha.
        '''
        plot(self.Unpumped.offsetCorrectedBinedIVData,label='Measured Unpumped')
        plot(self.Pumped.offsetCorrectedBinedIVData,label='Measured Pumped')
        plot(self.Unpumped.simulatedIV,label='Fit Unpumped')
        plot(self.simulated_pumped_from_unpumped,label='Fit Pumped')
        
    def pumped_from_unpumped(self,alpha,unpumped):
        '''TODO
        alpha is a single value
        '''
        n = self.tuckerSummationIndex
        n= np.arange(-n,n+1)
         #single alpha value only, n is an array. Otherwise a flattened array needs to be generated and sized after jv computation.
        bessel = np.square(jv(n,alpha))
        pumped = []
        for bias in unpumped[0]:
            voltagesOfInterst = (bias+n*self.vPh)
            #creata a matrix
            voltagesOfInterst = np.expand_dims(voltagesOfInterst, axis=-1) 
            unpumpedOffseted = unpumped[1,np.abs(unpumped[0] - voltagesOfInterst).argmin(axis=-1)]
            pumped.append([bias,np.sum(bessel*unpumpedOffseted)])
        return np.array(pumped).T
        
    
    def pumping_Level_Calc(self,unpumped,pumped):
        '''This function computes alpha and therefore the pumping level from a pumped and unpumped IV curve.
        The unpumped IV data needs to be extended in the normal resistance regime to allow computation at bias voltages with an offset of multiples of the photon step size.
                    
        returns
        -------
        2D array
            [0] The bias voltage.
            [1] The value of alpha.
        
        '''
        def pumpedFunction(alpha,bias, unpumped,pumped, vPh,n):
            '''The function to be minimised.
        
            inputs
            ------
            alpha: float
                Guess values of alpha
            bias: float
                The bias voltage processed
            unpumped: array
                The IV data of the unpumped IV curve   
            pumped: array
                The IV data of the pumped IV curve
            vPh: float
                Voltage equivalent of photons arriving in mV
            n: int
                The summation index of the function
        
            returns
            -------
            float
                The absolute difference of pumped and extract of unpumped IV curve
            ''' #TODO update with pumped_from_unpumped
            n= np.arange(-n,n+1)
            bessel = np.square(jv(n,alpha)) # only positive values due to square
            unpumpedOffseted = []
            voltagesOfInterst = (bias+n*self.vPh)
            #creata a matrix
            voltagesOfInterst = np.expand_dims(voltagesOfInterst, axis=-1) 
            unpumpedOffseted = unpumped[1,np.abs(unpumped[0] - voltagesOfInterst).argmin(axis=-1)]
            return np.abs(np.nansum(bessel*unpumpedOffseted)-pumped[1,(np.abs(pumped[0]-(bias))).argmin()])
    
        alphaArray = []
        #print('Finding alpha for each bias point')
        print('Process pumping level for each voltage bias point.')
        for i in pumped[0]: # evaluate every bias voltage of the pumped IV curve
            #print('Processing ', i) ' TODO process all bias voltages at the same time with pumped_from_unpumped
            alphaArray.append([i,fmin(pumpedFunction,.8,args=(i,unpumped,pumped,self.vPh,self.tuckerSummationIndex),disp=False,ftol=1e-12,xtol=1e-10)[0]])
        print('Computation of pumping levels is finished.')
        return np.array(alphaArray).T
    
    
      
    def iACSISRe_Calc(self,unpumped,pumping_Levels,simulation=False):
        '''The real AC current computed from the unpumped DC IV curve and the pumping level at each bias voltage alpha.
            
        inputs
        ------
        unpumped: 2d array
            The IV data of the unpumped IV curve extended to allow computation of V0+n*Vph.
        pumping_Levels: TODO
        simulation: bool
            In case the function is called with simulated data, it is necessary to use the bias voltages from the unpumped IV curve, rather than the labelling from the pumping level.
            The pumping level bias voltage labelling bases on the pumped IV curve, so that the labelling of the unpumped IV curve can not be utilised (raises errors in later stages).
        returns:
        --------
        array of shape V0
            The real part of the AC current
        '''
        n= np.arange(-self.tuckerSummationIndex-1,self.tuckerSummationIndex+2) # accont for one extra bessel function in each direction
        ifreqRe = []
        for i in range(len(pumping_Levels[0])):
            bessel = jv(n,pumping_Levels[1,i])
            unpumpedOffseted = []
            for nx in n[1:-1]:
                unpumpedOffseted.append(unpumped[1,(np.abs(unpumped[0]-(pumping_Levels[0,i]+nx*self.vPh))).argmin()])
#            ifreqRe.append([unpumped[0,(np.abs(unpumped[0]-(pumping_Levels[0,i]))).argmin()],
#                                              np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
            if simulation: #necessary to fix the bias voltage labeling
                ifreqRe.append([unpumped[0,(np.abs(unpumped[0]-(pumping_Levels[0,i]))).argmin()],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
            else:
                ifreqRe.append([pumping_Levels[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
        return np.array(ifreqRe).T
    
    @property
    def iACSISRe(self):
        '''The real AC current computed from the unpumped bined DC IV curve and the pumping level at each bias voltage alpha.
            The IV data beyond the IV data is token from the normal resistance fit.
        
        returns:
        --------
        array of shape V0
            The real part of the AC current
        '''
        voltageLimit = np.abs([self.pumping_Levels[0,-1]+self.tuckerSummationIndex*self.vPh,
                               self.pumping_Levels[0,0]-self.tuckerSummationIndex*self.vPh]).max()
        unpumped = self.Unpumped.binedDataExpansion(voltageLimit)
        pumping_Levels = self.pumping_Levels
        return self.iACSISRe_Calc(unpumped=unpumped,pumping_Levels=pumping_Levels)
    
    @property
    def simulated_iACSISRe(self):
        '''The real AC current computed from the simulated IV curve.
        
        returns:
        --------
        array of shape V0
            The real part of the AC current
        '''     
        return self.iACSISRe_Calc(self.Unpumped.simulatedIV,self.pumping_Levels,simulation=True)
    
    @property
    def iACSISIm(self):
        '''The imaginary AC current computed from the Kramers Kronig transformed unpumped bined DC IV curve and the pumping level at each bias voltage alpha.
            The IV data beyond the IV data is token from the normal resistance fit.
        
        returns:
        --------
        array of shape V0
            The imaginary part of the AC current
        '''
        voltageLimit = np.abs([self.pumping_Levels[0,-1]+self.tuckerSummationIndex*self.vPh,
                               self.pumping_Levels[0,0]-self.tuckerSummationIndex*self.vPh]).max()
        iKK = self.Unpumped.iKKExpansion(voltageLimit) # 1000 -> no impact
        pumping_Levels = self.pumping_Levels
        return self.iACSISIm_Calc(iKK=iKK,pumping_Levels=pumping_Levels)
    
    @property
    def iACSIS(self):
        '''TODO'''
        return np.vstack([self.iACSISRe[0],self.iACSISRe[1]+1j*self.iACSISIm[1]])
        
    def iACSISIm_Calc(self,iKK,pumping_Levels):
        '''The imaginary AC current computed from the Kramers Kronig transformed IV curve and the pumping level at each bias voltage alpha.
           
        inputs
        ------
        iKK: 2d array
            The Kramers Kronig Transform IV curve.
        
        returns:
        --------
        array of shape V0
            The imaginary part of the AC current
        '''
        n= np.arange(-self.tuckerSummationIndex-1,self.tuckerSummationIndex+2) # accont for one extra bessel function in each direction
        ifreqIm = []
        for i in range(len(pumping_Levels[0])):
            bessel = jv(n,pumping_Levels[1,i])
            iKKOffseted = []
            for nx in n[1:-1]:
                iKKOffseted.append(iKK[1,(np.abs(iKK[0]-(pumping_Levels[0,i]+nx*self.vPh))).argmin()])
            ifreqIm.append([pumping_Levels[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.subtract(bessel[:-2],bessel[2:])),iKKOffseted))])
        return np.array(ifreqIm).T
    
    @property
    def simulated_iACSISIm(self):
        '''The imaginary AC current computed from the Kramers Kronig transformed simulated unpumped DC IV curve and the pumping level at each bias voltage alpha.
            The IV data beyond the IV data is token from the normal resistance fit.
        
        returns:
        --------
        array of shape V0
            The imaginary part of the AC current
        '''
        return self.iACSISIm_Calc(self.Unpumped.simulated_iKK,pumping_Levels = self.pumping_Levels)
    
        
    
    def plot_simulated_and_measured_AC_currents(self):
        '''This function plots the real and imaginary AC currents through the SIS junction for both, measured and the simulated IV curve.
        '''
        plot(self.iACSISRe,'Measured Real')
        plot(self.iACSISIm,'Measured Imaginary')
        plot(self.simulated_iACSISRe,'Simulation Real')
        plot(self.simulated_iACSISIm,'Simulation Imaginary')
        
    def plot_simulated_and_measured_AC_currents_normalised(self):
        '''This function plots the normalised real and imaginary AC currents through the SIS junction for both, measured and the simulated IV curve.
        '''
        vGap = self.Unpumped.gapVoltage*1e3 #uV to compare with uA
        cC = np.divide(vGap,self.Unpumped.rN)
        plot(normalise_2d_array(self.iACSISRe,vGap,cC),'Measured Real')
        plot(normalise_2d_array(self.iACSISIm,vGap,cC),'Measured Imaginary')
        plot(normalise_2d_array(self.simulated_iACSISRe,vGap,cC),'Simulation Real')
        plot(normalise_2d_array(self.simulated_iACSISIm,vGap,cC),'Simulation Imaginary')
     
        
        
    def ySIS_Calc(self,iSISRe,iSISIm,pumping_Levels_Volt):
        '''Compute the admittance of the SIS junction.
        
        inputs
        ------
        iSISRe: 2d array
            [0] The bias voltage.
            [1] The real AC current through the SIS junction.
        iSISIm: 2d array
            [0] The bias voltage.
            [1] The imaginary AC current through the SIS junction.   
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping level alpha in units of volts.
        returns
        -------
        2d array
            The admittance per bias voltage        
        '''        
        #search relevant indexes by bias voltage
        #indexes = np.searchsorted(iSISRe[0],iSISIm[0])
        indexes = np.searchsorted(iSISIm[0],iSISRe[0]) # Here we are 2019/12/09
        # conversion of mV into uV necessary to get admittance as 1/Ohm
        y = np.divide((iSISRe[1] + 1j* iSISIm[1,indexes]),pumping_Levels_Volt[1]*1e3)
        y = np.vstack([iSISRe[0],y])
        return y[:,np.logical_not(np.isinf(y[1]))]  
    
    @property
    def ySIS(self):
        '''Compute the admittance of the SIS junction of the bined IV data.
        
        returns
        -------
        2d array
            The admittance per bias voltage        
        '''
        return self.ySIS_Calc(self.iACSISRe,self.iACSISIm,self.pumping_Levels_Volt)
    
    @property
    def zSIS(self):
        '''The impedance of the SIS junction. 
        '''
        ySIS = self.ySIS
        ySIS[1] = np.reciprocal(ySIS[1])
        return ySIS
    
    
    @property
    def simulated_ySIS(self):
        '''Compute the admittance of the SIS junction of the simulated IV data.
        
        returns
        -------
        2d array
            The admittance per bias voltage        
        '''
        return self.ySIS_Calc(self.simulated_iACSISRe,self.simulated_iACSISIm,self.pumping_Levels_Volt)#TODO change to simulated pumpingLevel simulated_pumping_Levels_Volt_masked_positive
    
    @property
    def simulated_zSIS(self):
        '''The impedance of the simulated SIS junction. 
        '''
        ySIS = self.simulated_ySIS
        ySIS[1] = np.reciprocal(ySIS[1])
        return ySIS
    
    def plot_simulated_and_measured_ySIS(self):
        '''This function plots the admittance basing on a measured IV curve and a simulated IV curve.
        '''
        ySIS = self.ySIS
        simulated_ySIS = self.simulated_ySIS
        plt.plot(ySIS[0],ySIS[1].real,label='Measured Real')
        plt.plot(ySIS[0],ySIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].real,label='Simulated Real')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].imag,label='Simulated Imaginary')
        
    def plot_simulated_and_measured_ySIS_normalised(self):
        '''This function plots the normalised admittance basing on a measured IV curve and a simulated IV curve.
        '''
        ySIS = self.ySIS
        simulated_ySIS = self.simulated_ySIS
        vGap = self.Unpumped.gapVoltage
        rN = np.reciprocal(self.Unpumped.rN)
        ySIS = normalise_2d_array(ySIS,vGap,rN)
        simulated_ySIS = normalise_2d_array(simulated_ySIS,vGap,rN)
        plt.plot(ySIS[0],ySIS[1].real,label='Measured Real')
        plt.plot(ySIS[0],ySIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].real,label='Simulated Real')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].imag,label='Simulated Imaginary')
    
    def plot_simulated_and_measured_zSIS(self):
        '''This function plots the impedance basing on a measured IV curve and a simulated IV curve.
        '''
        zSIS = self.zSIS
        simulated_zSIS = self.simulated_zSIS
        plt.plot(zSIS[0],zSIS[1].real,label='Measured Real')
        plt.plot(zSIS[0],zSIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].real,label='Simulated Real')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].imag,label='Simulated Imaginary')
    
    def plot_simulated_and_measured_zSIS_normalised(self):
        '''This function plots the impedance basing on a measured IV curve and a simulated IV curve.
        '''
        zSIS = self.zSIS
        simulated_zSIS = self.simulated_zSIS
        vGap = self.Unpumped.gapVoltage
        rN = self.Unpumped.rN
        zSIS = normalise_2d_array(zSIS,vGap,rN)
        simulated_zSIS = normalise_2d_array(simulated_zSIS,vGap,rN)
        plt.plot(zSIS[0],zSIS[1].real,label='Measured Real')
        plt.plot(zSIS[0],zSIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].real,label='Simulated Real')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].imag,label='Simulated Imaginary')
    
    def mask_photon_steps_Calc(self,gaussianBinSlopeFit,unpumped,vGap,pumpedIVdata):
        '''This functions computes  masks the steps of the IV curve relative to the positive gap voltage. The returned values are the masked voltages.
        This is necessary to avoid discontinuities between the step in further processing, eg to compute the embedding impedance.
        
        The masked steps are below the gap voltage / the transission.
        
        inputs
        ------
        TODO
        vGap: float 
            The gap voltage of the IV curve.
        pumpedIVdata: 2d np array
            The pumped IV data to recover the voltage.
            In the :class: Mixer, the impedance recovery uses the bias voltages of the pumped IV curve to label the bias voltage of the involved metrices
        returns
        -------
        1d array
            Voltages which are masked for further usage.
        '''
        #actually the masked width is 2*maskWidth
        maskWidth = np.average(gaussianBinSlopeFit[:,1]*8) #use 4 sigma per side to be sure
        n = np.arange(-self.steps_ImpedanceRecovery,self.steps_ImpedanceRecovery)
        lowerBoundaries = n*self.vPh+maskWidth + vGap
        upperBoundaries = np.negative(n)*self.vPh-maskWidth + vGap
        upperBoundaries = upperBoundaries[::-1] # sort it after increasing voltages
        #Test John's settings #TODO
        lowerBoundaries = [-0.75*self.vPh + vGap]
        upperBoundaries = [-0.2*self.vPh + vGap] # actually 0.2
        
        ret = []
        for i in np.arange(self.steps_ImpedanceRecovery):
            # voltage from pumped IV curve since this matches with the ySIS voltages
            ret.append(pumpedIVdata[0,np.logical_and(np.abs(pumpedIVdata[0])>lowerBoundaries[i],
                                                     np.abs(pumpedIVdata[0])<upperBoundaries[i])])
        return np.hstack(ret)  
    
    @property
    def mask_photon_steps(self):
        '''This functions masks the steps of the IV curve relative to the positive gap voltage. The returned values are the masked voltages.
        This is necessary to avoid discontinuities between the step in further processing, eg to compute the embedding impedance.
        
        The masked steps are below the gap voltage / the transission.
            
        returns
        -------
        1d array
            Voltages which are masked for further usage.
        '''
        #Remove after testing 2019/12/06
#        #actually the masked width is 2*maskWidth
#        maskWidth = np.average(self.Unpumped.gaussianBinSlopeFit[:,1]*8) #use 4 sigma per side to be sure
#        n = np.arange(-self.steps_ImpedanceRecovery,self.steps_ImpedanceRecovery)
#        lowerBoundaries = n*self.vPh+maskWidth +self.Unpumped.gapVoltage
#        upperBoundaries = np.negative(n)*self.vPh-maskWidth + self.Unpumped.gapVoltage
#        upperBoundaries = upperBoundaries[::-1] # sort it after increasing voltages
#        ret = []
#        for i in np.arange(self.steps_ImpedanceRecovery):
#            # voltage from pumped IV curve since this matches with the ySIS voltages
#            ret.append(self.Pumped.offsetCorrectedBinedIVData[0,np.logical_and(np.abs(self.Pumped.offsetCorrectedBinedIVData[0])>lowerBoundaries[i],
#                                                                        np.abs(self.Pumped.offsetCorrectedBinedIVData[0])<upperBoundaries[i])])
#        return np.hstack(ret)  
        gaussianBinSlopeFit =  self.Unpumped.gaussianBinSlopeFit
        unpumped = self.Unpumped
        vGap = self.Unpumped.gapVoltage
        pumpedIVdata = self.Pumped.offsetCorrectedBinedIVData
        return self.mask_photon_steps_Calc(gaussianBinSlopeFit,unpumped,vGap,pumpedIVdata)
    
    @property
    def simulated_mask_photon_steps(self):
        '''This functions masks the steps of the simulated IV curve relative to the positive gap voltage. The returned values are the masked voltages.
        This is necessary to avoid discontinuities between the step in further processing, eg to compute the embedding impedance.
        
        The masked steps are below the gap voltage / the transission.
            
        returns
        -------
        1d array
            Voltages which are masked for further usage.
        '''
        gaussianBinSlopeFit = self.Unpumped.simulated_gaussianBinSlopeFit
        unpumped = self.Unpumped
        vGap = self.Unpumped.simulated_gapVoltage
        pumpedIVdata = self.simulated_pumped_from_unpumped
        return self.mask_photon_steps_Calc(gaussianBinSlopeFit,unpumped,vGap,pumpedIVdata)
    
    @property 
    def ySIS_masked(self):
        '''This function returns the admittance of the SIS junction at a bias voltage at the masked regions.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included. This holds for positive and negative bias voltages.
        '''
        return self.ySIS[:,np.isin(self.ySIS[0],self.mask_photon_steps)]

    @property 
    def ySIS_masked_positive(self):
        '''This function returns the admittance of the SIS junction at the masked regions at positive bias voltages.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included at positive bias voltages are included.
        '''
        return self.ySIS[:,np.isin(self.ySIS[0],self.mask_photon_steps[self.mask_photon_steps>0])]

    @property 
    def simulated_ySIS_masked(self):
        '''This function returns the admittance of the simulated SIS junction at a bias voltage at the masked regions.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included. This holds for positive and negative bias voltages.
        '''
        return self.simulated_ySIS[:,np.isin(self.simulated_ySIS[0],self.simulated_mask_photon_steps)]

    @property 
    def simulated_ySIS_masked_positive(self):
        '''This function returns the admittance of the simulated SIS junction at the masked regions at positive bias voltages.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included at positive bias voltages are included.
        '''
        simulated_ySIS =self.simulated_ySIS
        simulated_mask_photon_steps = self.simulated_mask_photon_steps
        return simulated_ySIS[:,np.isin(simulated_ySIS[0],simulated_mask_photon_steps[simulated_mask_photon_steps>0])]

    @property
    def iACSIS_masked(self):
        '''This function returns the masked AC current through the SIS junction.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included. This holds for positive and negative bias voltages.
        '''
        return self.iACSIS[:,np.isin(self.iACSIS[0],self.mask_photon_steps)]

    @property 
    def iACSIS_masked_positive(self):
        '''This function returns the masked AC current through the SIS junction at positive bias voltages.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included at positive bias voltages are included.
        '''
        return self.iACSIS[:,np.isin(self.iACSIS[0],self.mask_photon_steps[self.mask_photon_steps>0])]
    #pumping_Levels_Volt
    @property
    def pumping_Levels_Volt_masked(self):
        '''This function returns the masked pumping level voltages through the SIS junction.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included. This holds for positive and negative bias voltages.
        '''
        return self.pumping_Levels_Volt[:,np.isin(self.pumping_Levels_Volt[0],self.mask_photon_steps)]

    @property 
    def pumping_Levels_Volt_masked_positive(self):
        '''This function returns the masked pumping level voltages through the SIS junction at positive bias voltagese.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included at positive bias voltages are included.
        '''
        masked = self.mask_photon_steps
        return self.pumping_Levels_Volt[:,np.isin(self.pumping_Levels_Volt[0],masked[masked>0])]
    
    @property 
    def simulated_pumping_Levels_Volt_masked_positive(self):
        '''This function returns the masked pumping level voltages through the simulated SIS junction at positive bias voltages.
        The masked regions are defined by the width of a gaussian fitted on the transistion and the number of photon steps involved.
        Only photon steps below the transission are included at positive bias voltages are included.
        '''
        masked = self.simulated_mask_photon_steps
        return self.simulated_pumping_Levels_Volt[:,np.isin(self.simulated_pumping_Levels_Volt[0],masked[masked>0])]
    @property
    def mask_photon_steps_symmetric_to_transission(self):
        '''This functions masks the steps of the IV curve relative to the positive gap voltage. The returned values are the masked voltages.
        This is necessary to avoid discontinuities between the step in further processing, eg to compute the embedding impedance.
        
        The masked steps are below and above the gap voltage / the transission.
            
        returns
        -------
        1d array
            Voltages which are masked for further usage.
        '''
        #actually the masked width is 2*maskWidth
        maskWidth = np.average(self.Unpumped.gaussianBinSlopeFit[:,1]*8) #use 4 sigma per side to be sure
        n = np.arange(-self.steps_ImpedanceRecovery,self.steps_ImpedanceRecovery)
        lowerBoundaries = n*self.vPh+maskWidth +self.Unpumped.gapVoltage
        upperBoundaries = np.negative(n)*self.vPh-maskWidth + self.Unpumped.gapVoltage
        upperBoundaries = upperBoundaries[::-1] # sort it after increasing voltages
        ret = []
        for i in np.arange(2*self.steps_ImpedanceRecovery):
            # voltage from pumped IV curve since this matches with the ySIS voltages
            ret.append(self.Pumped.offsetCorrectedBinedIVData[0,np.logical_and(np.abs(self.Pumped.offsetCorrectedBinedIVData[0])>lowerBoundaries[i],
                                                                        np.abs(self.Pumped.offsetCorrectedBinedIVData[0])<upperBoundaries[i])])
        return np.hstack(ret)  

#    @property
#    def simulated_mask_photon_steps(self):
#        '''This functions masks the steps of the simulated IV curve relative to the positive gap voltage. The returned values are the masked voltages.
#        This is necessary to avoid discontinuities between the step in further processing, eg to compute the embedding impedance.
#            
#        Note that the masked regions are determined from the bined IV data.
#        
#        returns
#        -------
#        1d array
#            Voltages which are masked for further usage.
#        '''
#        #actually the masked width is 2*maskWidth
#        maskWidth = np.average(self.Unpumped.gaussianBinSlopeFit[:,1]*8) #use 4 sigma per side to be sure
#        n = np.arange(-self.steps_ImpedanceRecovery,0)
#        lowerBoundaries = n*self.vPh+maskWidth +self.Unpumped.gapVoltage
#        upperBoundaries = np.negative(n)*self.vPh-maskWidth + self.Unpumped.gapVoltage
#        upperBoundaries = upperBoundaries[::-1] # sort it after increasing voltages
#        ret = []
#        for i in np.arange(self.steps_ImpedanceRecovery):#range does not work in this cased
#              ret.append(self.Unpumped.simulatedIV[0,np.logical_and(np.abs(self.Unpumped.simulatedIV[0])>lowerBoundaries[i],
#                                                                        np.abs(self.Unpumped.simulatedIV[0])<upperBoundaries[i])])
#        return np.hstack(ret)          
    
    
    def plot_mask_steps(self):
        '''This function plots the masked steps along with the offset corrected binned IV data.
        
        inputs
        ------
        Unpumped: object
            The class of the unpumped IV curve.
        vPh: float
            Voltage equivalent of photons arriving.
        nSteps: int
            The number of steps which should be masked out.
        '''
        plot(self.Unpumped.offsetCorrectedBinedIVData, label = 'Unpumped')
        plot(self.Pumped.offsetCorrectedBinedIVData, label = 'Pumped')
        ymin = self.Unpumped.offsetCorrectedBinedIVData[1].min()
        ymax = self.Unpumped.offsetCorrectedBinedIVData[1].max()
        vspan = self.mask_photon_steps
        plt.vlines(vspan,ymin,ymax,alpha=.2,label='Masked')  
    
    def yEmb_cost_Function_iLO(self,yLO,ySIS,pumping_Levels_Volt):
        '''
        '''
        ySS = self.total_admittance(yLO,pumping_Levels_Volt)
        return np.sum(np.square(pumping_Levels_Volt[1]*1e-3))-np.divide(np.square(np.sum(np.divide(pumping_Levels_Volt[1]*1e-3,np.sqrt(ySS[1])))),
                                                         np.sum(np.reciprocal(ySS[1])))
        
    def yEmb_cost_Function(self,params,ySIS,pumping_Levels_Volt):
        '''The error function given by Skalare equation 7. TODO update
        
        inputs
        ------ 
        yLO: 2 element array, real and complex value
            The parameter which requires fit to the admittance of the circuit
            It is necessary to pass real and imaginary split from each other, since scipy.optimize can't deal with complex numbers.
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0]
        dummy: int 
            Necessary to pass args into fmin()
        returns
        -------
        float:
            The error of the function
        '''
        iLO = params[2]
        yLO = [params[0], params[1]]
        #own test
        yLO = params[0]+1j*params[1]
        return np.sum(np.square(np.subtract(pumping_Levels_Volt[1]*1e-3,np.abs(np.divide(iLO,np.add(yLO,ySIS[1]))))))
        
    def yEmb_cost_Function_Skalare(self,params,ySIS,pumping_Levels_Volt):
        '''The error function given by Skalare equation 7.
        
        inputs
        ------ 
        yLO: 2 element array, real and complex value
            The parameter which requires fit to the admittance of the circuit
            It is necessary to pass real and imaginary split from each other, since scipy.optimize can't deal with complex numbers.
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0]
        dummy: int 
            Necessary to pass args into fmin()
        returns
        -------
        float:
            The error of the function
        '''
        iLO = params[2]
        yLO = [params[0], params[1]]
        #iLO = np.multiply(ySIS[0],np.sqrt(ySS)) 
        #iLO = yLO[2]
        #SS -> Sum Squared
        ySS = self.total_admittance(yLO,ySIS)
        #iLO = self.current_LO_from_Embedding_Circuit(ySS,pumping_Levels_Volt)
        #iLO = np.sum(np.multiply(ySIS[0]*1e-3,np.sqrtn(ySS)))
        errorfunc = []
        errorfunc.append( np.sum(np.square(pumping_Levels_Volt[1]*1e-3))) # single value
        #Following Skalare 
        errorfunc.append(self.first_yEmb_Error_Term(iLO,ySS))
        errorfunc.append(self.second_yEmb_Error_Term(iLO,ySS,pumping_Levels_Volt))
        return np.abs(np.sum(errorfunc).real) # the imaginary part is always 0
        #A home made synthesis of John's thesis and Withington
#            errorfunc[-1]=abs(errorfunc[-1])
#            errorfunc.append((np.abs(iLO))*np.sum(np.reciprocal(np.sqrt(ySS))))
        #print(iLO)
        #print(errorfunc)
        #return np.abs(np.diff(errorfunc).real) # the imaginary part is always 0/
        
    def yEmb_cost_Function_Skalare_fixed_iLO(self,params,ySIS,pumping_Levels_Volt):
        '''The error function given by Skalare equation 7.
        
        inputs
        ------ 
        yLO: 2 element array, real and complex value
            The parameter which requires fit to the admittance of the circuit
            It is necessary to pass real and imaginary split from each other, since scipy.optimize can't deal with complex numbers.
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0]
        dummy: int 
            Necessary to pass args into fmin()
        returns
        -------
        float:
            The error of the function
        '''
        yLO = [params[0], params[1]]
        #iLO = np.multiply(ySIS[0],np.sqrt(ySS)) 
        #iLO = yLO[2]
        #SS -> Sum Squared
        ySS = self.total_admittance(yLO,ySIS)
        iLO = self.current_LO_from_Embedding_Circuit(ySS,pumping_Levels_Volt)
        #iLO = np.sum(np.multiply(ySIS[0]*1e-3,np.sqrtn(ySS)))
        errorfunc = []
        errorfunc.append( np.sum(np.square(pumping_Levels_Volt[1]*1e-3))) # single value
        #Following Skalare 
        errorfunc.append(self.first_yEmb_Error_Term(iLO,ySS))
        errorfunc.append(self.second_yEmb_Error_Term(iLO,ySS,pumping_Levels_Volt))
        return np.abs(np.sum(errorfunc).real) # the imaginary part is always 0
        #A home made synthesis of John's thesis and Withington
#            errorfunc[-1]=abs(errorfunc[-1])
#            errorfunc.append((np.abs(iLO))*np.sum(np.reciprocal(np.sqrt(ySS))))
        #print(iLO)
        #print(errorfunc)
        #return np.abs(np.diff(errorfunc).real) # the imaginary part is always 0/
            
    def first_yEmb_Error_Term(self,iLO,ySS):
        '''This function returns the first error term of the Skalare cost function.
        
        inputs
        ------
        iLO: complex float?
            The current through the local osscilator. This is the total current through the system.
        ySS: 2d array
            The square of the total admittance of the circuit.
            [0] The bias voltage.
            [1] The values of the square of the total admittances
        '''
        return  np.multiply(np.square(np.abs(iLO)),np.nansum(np.reciprocal(ySS[1]))).real
    
    def evaluate_first_yEmb_Error_Term(self,ySIS,pumping_Levels_Volt,real=np.arange(0.0001,.1,.0001),imag = np.arange(-.05,.05,0.0001)):
        '''
        '''
        ySS = self.evaluate_total_admittance(ySIS,real,imag)
        iLO = self.evaluate_current_LO_from_Embedding_Circuit(ySIS,pumping_Levels_Volt,real,imag)
        ret = np.zeros(iLO.shape,dtype=np.complex_)
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i,j] = self.first_yEmb_Error_Term(iLO[i,j],ySS[i,j])
        return ret
    
    def second_yEmb_Error_Term(self,iLO,ySS,pumping_Levels_Volt):
        '''
        '''
        return -2*np.multiply(np.abs(iLO),np.nansum(np.divide(pumping_Levels_Volt[1]*1e-3,np.sqrt(ySS[1]))))
    
    def evaluate_second_yEmb_Error_Term(self,ySIS,pumping_Levels_Volt,real=np.arange(0.0001,.1,.0001),imag = np.arange(-.05,.05,0.0001)):
        '''
        '''
        ySS = self.evaluate_total_admittance(ySIS,real,imag)
        iLO = self.evaluate_current_LO_from_Embedding_Circuit(ySIS,pumping_Levels_Volt,real,imag)
#        ySS = self.total_admittance(yLO,ySIS)
#        iLO = self.current_LO_from_Embedding_Circuit(ySS)
        ret = np.zeros(iLO.shape)
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i,j] = self.second_yEmb_Error_Term(iLO[i,j],ySS[i,j],pumping_Levels_Volt)
        return ret
            
    def total_admittance(self,yLO,ySIS):
        '''This method computes the sum of the square of the embedding admittance and the admittance of the SIS junction.
        
        inputs
        ------
        yLO: 2d array
            [0] The real part of the LO admittance.
            [1] The imaginary part of the LO admittance.
        ySIS: 2d array
            [0] The bias voltage for which the admittance of the SIS junction is given.
            [1] The complex admittance of the SIS junction.
        returns
        -------
        2d array:
            [0] The associated applied bias voltage.
            [1] The square of the sum of the real and imaginary admittances of the SIS junction and the LO.
        '''
        #.imag returns real value
        return np.vstack([ySIS[0].real,np.add(np.square(yLO[0]+ySIS[1].real) , np.square(yLO[1]+ySIS[1].imag))])
    
    def evaluate_total_admittance(self,ySIS,real=np.reciprocal(np.arange(0.1,15,.1)),imag=np.reciprocal(np.arange(-15,15,.1))):
        '''This function evaluates the total admittance overa range of LO admittances.
        
        inputs
        ------
        ySIS: 2d array
            [0] The bias voltage for which the admittance of the SIS junction is given.
            [1] The complex admittance of the SIS junction.
        real: 1d array
            The real LO admittances evaluated.
        imag: 1d array
            The imaginary LO admittances evaluated.
        returns
        -------
        4d array: 
            index indicates:
            [0] The axis of the real LO admittance values.
            [1] The axis of the imaginary LO admittance values.
            [2] The bias voltages [0] or the value of the total admittance [1].
            [3] The total admittances associated with a certain bias voltage.        
        '''
        realarray ,imagarray =self.generate_flat_square_array(real,imag)
        totad= []
        for i in range(len(realarray)):
            totad.append(self.total_admittance([realarray[i],imagarray[i]],ySIS))
        totad = np.array(totad)
        totad = totad.reshape((len(real),len(imag),len(ySIS),len(ySIS[0])))
        return totad
            
    def current_LO_from_Embedding_Circuit(self,ySS,pumping_Levels_Volt):
        '''This function computes the LO current of the embedding circuit from given LO and SIS admittance.
        
        inputs
        ------
        ySS: float
            The square of the sum of the real and imaginary admittances of the SIS junction and the LO.
                The Unit is A
        '''
        return  np.divide(np.nansum(np.divide(pumping_Levels_Volt[1]*1e-3,np.sqrt(ySS[1]))),np.nansum(np.reciprocal(ySS[1])))
            
    def evaluate_current_LO_from_Embedding_Circuit(self,ySIS,pumping_Levels_Volt,real=np.reciprocal(np.arange(0.1,15,.1)),imag=np.reciprocal(np.arange(-15,15,.1))):
        '''This function evaluates the LO current at different LO admittances.
        Note: oes not work!
        
        inputs
        ------
        ySIS: 2d array
            [0] The bias voltage for which the admittance of the SIS junction is given.
            [1] The complex admittance of the SIS junction.
        real: 1d array
            The real LO admittances evaluated.
        imag: 1d array
            The imaginary LO admittances evaluated.
        returns
        -------
        2d array:
            The LO at different LO admittances.
        '''        
        realarray ,imagarray =self.generate_flat_square_array(real,imag)
        ySS = self.evaluate_total_admittance(ySIS,real,imag)[:,:,1]
        ySS = ySS.reshape((len(real)*len(imag),len(ySIS[0])))
        iLO= []
        for i in range(len(realarray)):
            iLO.append(self.current_LO_from_Embedding_Circuit(ySS[i],pumping_Levels_Volt))
        iLO = np.array(iLO)
        iLO = np.reshape(iLO,(len(real),len(imag)))
        return iLO.real
    
    def yEmb_Errorsurface(self,ySIS,pumping_Levels_Volt,real=np.reciprocal(np.arange(0.1,15,.01)),imag=np.reciprocal(np.arange(-20,10,.01))):
        '''This function computes the errorsurface between the measured embedding admittance and the embedding admittance given by the parameters.
        
        inputs
        ------
        ySIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage
        real: 1d array
            The real part of the embedding admittances evauluated.
        imag: 1d array
            The imaginary part of the embedding admittances evauluated.
        TODO
        '''
        realarray ,imagarray = self.generate_flat_square_array(real,imag)
        #y = np.add(realarray,1j*imagarray.T)
        errorsurface = []
        for i in range(len(realarray)):
            errorsurface.append(self.yEmb_cost_Function([realarray[i],imagarray[i]],ySIS,pumping_Levels_Volt))
        errorsurface = np.array(errorsurface)
        #x axis is real and y axis is imaginary component
        errorsurface = np.reshape(errorsurface,(len(real),len(imag)))#(len(imag),len(real)))
        #plt.pcolor(errorsurface,norm=LogNorm(vmin=a.min(), vmax=np.unique(a)[-2]))
        return errorsurface
    
    def yEmb_Errorsurface_iLO(self,ySIS,pumping_Levels_Volt,real=np.reciprocal(np.arange(0.1,15,.01)),imag=np.reciprocal(np.arange(-20,10,.01))):
        '''This function computes the errorsurface between the measured embedding admittance and the embedding admittance given by the parameters.
        TODO
        inputs
        ------
        ySIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage
        real: 1d array
            The real part of the embedding admittances evauluated.
        imag: 1d array
            The imaginary part of the embedding admittances evauluated.
        TODO
        '''
        realarray ,imagarray = self.generate_flat_square_array(real,imag)
        #y = np.add(realarray,1j*imagarray.T)
        errorsurface = []
        for i in range(len(realarray)):
            errorsurface.append(self.yEmb_cost_Function_iLO([realarray[i],imagarray[i]],ySIS,pumping_Levels_Volt))
        errorsurface = np.array(errorsurface)
        #x axis is real and y axis is imaginary component
        errorsurface = np.reshape(errorsurface,(len(real),len(imag)))#(len(imag),len(real)))
        #plt.pcolor(errorsurface,norm=LogNorm(vmin=a.min(), vmax=np.unique(a)[-2]))
        return errorsurface
    
    def generate_flat_square_array(self,x,y):
        '''This function computes a linear array which contains only unique combinations of x and the y array.
        
        inputs
        ------
        x: 1d array
            The unique values along the x axis.
        y: 1d array
            The unique values along the y axis.
        returns
        -------
        2 1d arrays:
            The values of the x and y array expanded to match unique combinations of the values.        
        '''
        xarray = np.vstack([x]*len(y)).flatten('F') # 0 0 0 0 0 ... 1 1 1 1... 2 2 2 2 ...
        yarray = np.vstack([y]*len(x)).flatten('C') # 0 1 2 3 4... 0 1 2 ...
        return xarray,yarray
        
    def zEmb_cost_Function(self,zLO,zSIS,pumping_Levels_Volt):
        '''This function computes the error between the measured voltage and the voltage computed from the embedding impedance.
        
        inputs
        ------
        zLO: 1d array
            [0] The real value of the impedance of the LO.
            [1] The imaginary value of the impedance of the LO.
        zSIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage
        dummy: int
            A dummy to pass the args in fmin.
        returns
        -------
        float:
            The remaining error between the bias voltage simulated and measured. 
        '''
        zLO = zLO[0]+1j*zLO[1]
        z = np.divide(zSIS[1],np.add(zLO,zSIS[1]))
        error = []
        error.append(np.sum(np.square(np.abs(pumping_Levels_Volt[1]*1e-3))))
        error.append(-np.divide(np.square(np.sum(np.abs(np.multiply(z,pumping_Levels_Volt[1])*1e-3))),np.sum(np.square(np.abs(z)))))
        return np.abs(np.sum(error))
    
    def findZemb(self,zSIS,pumping_Levels_Volt):
        '''TODO
        '''
        guess = [[6.2,-4]]# Does not take complex value as input
        return fmin(self.zEmb_cost_Function,guess,args=(zSIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
    
    @property
    def zEmb_John(self):
        '''TODO
        '''
        zSIS_masked= self.ySIS_masked_positive
        zSIS_masked[1] = np.reciprocal(zSIS_masked[1])
        pumping_Levels_Volt = self.pumping_Levels_Volt_masked_positive
        return self.findZemb(zSIS_masked,pumping_Levels_Volt)
    
    def zEmb_Errorsurface(self,zSIS,pumping_Levels_Volt,real=np.arange(0.1,15,.1),imag=np.arange(-15,15,.1)):
        '''This function computes the errorsurface between the measured embedding admittance and the embedding admittance given by the parameters.
        
        inputs
        ------
        zSIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage
        currentLO: float
            The current through the local oscillator.
        real: 1d array
            The real impedance of the LO evaluated.
        imag: 1d array
            The imaginary impedance of the LO evaluated.
        '''
        realarray ,imagarray = self.generate_flat_square_array(real,imag)
        #y = np.add(realarray,1j*imagarray.T)
        errorsurface = []
        for i in range(len(realarray)):
            errorsurface.append(self.zEmb_cost_Function([realarray[i],imagarray[i]],zSIS,pumping_Levels_Volt))
        errorsurface = np.array(errorsurface)
        #x axis is real and y axis is imaginary component
        errorsurface = np.reshape(errorsurface,(len(real),len(imag)))#(len(imag),len(real)))
        #plt.pcolor(errorsurface,norm=LogNorm(vmin=a.min(), vmax=np.unique(a)[-2]))
        return errorsurface
    
    def findYemb(self,ySIS,pumping_Levels_Volt):
        '''This function finds the embedding admittance from the voltage dependent admittance of the SIS junction
        
        inputs
        -------
        ySIS: 2d array|`¸
            The admittance of the SIS junction [1] over the applied bias voltage [0].
        TODO
        returns
        -------
        list:
            fmin result including the embedding admittance.
        '''
        ##Skalare 2nd method
        #I_LO is only included as absolute value
        guess = [[.1,.1,1e-3]]# Does not take complex value as input
        #bounds = [(None,None),(None,None),(0.0,None)]
        return fmin(self.yEmb_cost_Function,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)#,bounds=bounds)
        #array([0.08804042, 0.06238886, 0.00017428])
        #return minimize(self.yEmb_cost_Function,guess,args=(ySIS,pumping_Levels_Volt),method='Nelder-Mead',options={'maxiter':3000, 'maxfev':3000},tol=1e-12)#,bounds=bounds)
        #x: array([0.08804042, 0.06238886, 0.00017428])
        #return minimize(self.yEmb_cost_Function,guess,args=(ySIS,pumping_Levels_Volt),method='Nelder-Mead',options={'maxiter':3000, 'maxfev':3000})#,bounds=bounds)
        #x: array([0.14682084, 0.12037847, 0.00024144])
        
    def findYemb_Eyeball(self,iACSIS,pumping_Levels_Volt):
        '''This function finds the embedding admittance from the voltage dependent admittance of the SIS junction
        
        inputs
        -------
        ySIS: 2d array|`¸
            The admittance of the SIS junction [1] over the applied bias voltage [0].
        TODO
        returns
        -------
        list:
            fmin result including the embedding admittance.
        '''
        ##Skalare 2nd method
        #I_LO is only included as absolute value
        guess = [[.1,.1,1e-3,1e-3]]# Does not take complex value as input
        #bounds = [(None,None),(None,None),(0.0,None),(None,None)]
        return fmin(self.eyeBallMethod,guess,args=(iACSIS,pumping_Levels_Volt))#,bounds=bounds)
        #return minimize(self.eyeBallMethod,guess,args=(iACSIS,pumping_Levels_Volt),bounds=bounds)
        '''Some Test results:
            fmin(M.yEmb_cost_Function,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 231
         Function evaluations: 433
Out[217]: array([0.08805099, 0.06239267, 0.00017429])

fmin(M.yEmb_cost_Function_Skalare,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 215
         Function evaluations: 424
Out[218]: array([0.08805074, 0.06239262, 0.00017429])

fmin(M.yEmb_cost_Function_Skalare_fixed_iLO,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 110
         Function evaluations: 232
Out[219]: array([0.08805076, 0.06239259, 0.00094878])

yEmb = .08805076 +1j*.06239259

iLO = .00094878

M.vLO_from_circuit
Out[222]: <bound method Mixer.vLO_from_circuit of <__main__.Mixer object at 0x1c31a379b0>>

M.vLO_from_circuit(iLO,yEmb)

Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 54
         Function evaluations: 105
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 59
         Function evaluations: 114
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 54
         Function evaluations: 105
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 59
         Function evaluations: 114
Optimization terminated successfully.
         Current function value: 0.000001
         Iterations: 21
         Function evaluations: 42
Out[223]: array([0.00584434])

---> Fixed LO results in bad result for the computation of the pumping level!

    
'''
    
    def eyeBallMethod(self,params,iACSIS,pumping_Levels_Volt):
        '''TODO
        '''
        yLO = params[0]+1j*params[1]
        iLO = params[2]+1j*params[3]
        residuals = np.subtract(iLO,np.add(iACSIS[1]*1e-6,np.multiply(yLO,pumping_Levels_Volt[1]*1e-3)))
        return np.sum(np.abs(residuals))
    
    
    @property
    def yEmb(self):
        '''This function determines the embedding admittance from voltages at the photon steps of the IV curve. Other data is ignored.
        
        returns
        -------
        1d np array:
            fmin result including the embedding admittance.    
            [0] real embedding admittance
            [1] imaginary embedding admittance
            [2] LO current
        '''
        ySIS_masked= self.ySIS_masked_positive
        pumping_Levels_Volt = self.pumping_Levels_Volt_masked_positive
        return self.findYemb(ySIS_masked,pumping_Levels_Volt)    
#        iACSISmasked = self.iACSIS_masked_positive
#        return self.findYemb(iACSISmasked,pumping_Levels_Volt)    

    @property
    def zEmb(self):
        '''This function converts the embedding admittance to the embedding admittance.
        
        returns
        -------
        1d np array:
            [0] real embedding impedance
            [1] imaginary embedding impedance
            [2] LO current
        '''
        yEmb = self.yEmb
        return np.hstack([np.reciprocal(yEmb[:2])])
    
    @property
    def simulated_yEmb(self):
        '''This function determines the embedding admittance from voltages at the photon steps of the simulated IV curve. Other data is ignored.
        Note that the masked areas are determined from the bined IV data
        
        returns
        -------
        1d np array:
            fmin result including the embedding admittance.    
            [0] real embedding admittance
            [1] imaginary embedding admittance
            [2] LO current
        '''
        ySIS_masked = self.simulated_ySIS_masked_positive
        pumping_Levels_Volt = self.simulated_pumping_Levels_Volt_masked_positive
        return self.findYemb(ySIS_masked,pumping_Levels_Volt)    
    
    @property
    def simulated_zEmb(self):
        '''This function converts the simulated embedding admittance to the simulated embedding admittance.
        
        returns
        -------
        1d np array:
            [0] real embedding impedance
            [1] imaginary embedding impedance
            [2] LO current
        '''
        yEmb = self.simulated_yEmb
        return np.hstack([np.reciprocal(yEmb[:2])])
    
    @property
    def iLO_from_circuit(self):
        '''TODO
        '''
        iSIS = self.iACSIS
        yEmb = self.yEmb
        yEmb = yEmb[0] + 1j*yEmb[1]
        vLO = self.pumping_Levels_Volt
        return np.vstack([vLO[0],np.add(iSIS[1,np.isin(iSIS[0],vLO[0])],np.multiply(yEmb,vLO[1]*1e3))])
    
    def vLO_from_circuit(self,iLO,yEmb,*args):
        '''Following Boon's thesis equation 3.23
        returns single vLO for a given circuit
        
        inputs
        ------
        iLO: float
            The current of the local oscillator source in Ampere.
        
        '''
        if len(args)==3:
            unpumped = args[0]
            iKK = args[1]
            mask = args[2]
        else:
            voltageLimit = np.abs([self.pumping_Levels[0,-1]+self.tuckerSummationIndex*self.vPh,
                                   self.pumping_Levels[0,0]-self.tuckerSummationIndex*self.vPh]).max()
            unpumped = self.Unpumped.binedDataExpansion(voltageLimit)
            iKK = self.Unpumped.iKKExpansion(voltageLimit)
            mask = self.mask_photon_steps[self.mask_photon_steps>0]
            
        def cost_vLO_from_circuit(vLO,unpumped,iKK,iLO,yEmb,mask):
            '''TODO
            
            inputs
            ------
            vLO: float
                The voltage through the circuit in Volt.
                
                
            '''
            #constant pumping level throughout the whole masked bias voltage range
            pumping_Levels = np.vstack([mask,np.full(mask.shape,np.divide(vLO,self.vPh*1e-3))])
            iACre = self.iACSISRe_Calc(unpumped=unpumped,pumping_Levels=pumping_Levels)
            iACim = self.iACSISIm_Calc(iKK=iKK,pumping_Levels=pumping_Levels)
            iACSIS = np.add(iACre[1],1j*iACim[1])
            return np.sum(np.abs(np.subtract(np.square(np.abs(iLO)),
                        np.square(np.abs(np.add(np.multiply(yEmb,vLO),iACSIS*1e-6))))))
            
        guess = [1e-3]# Does not take complex value as input
        return fmin(cost_vLO_from_circuit,guess,args=(unpumped,iKK,iLO,yEmb,mask),ftol=1e-9,xtol=1e-7)
        
    def yEmb_from_circuit(self):
        '''TODO
        '''
        unpumped = self.Unpumped.offsetCorrectedBinedIVData
        pumped = self.Pumped.offsetCorrectedBinedIVData
        voltageLimit = np.abs([self.pumping_Levels[0,-1]+self.tuckerSummationIndex*self.vPh,
                               self.pumping_Levels[0,0]-self.tuckerSummationIndex*self.vPh]).max()    
        #avoid multiple calls of properties in the vLO_from_circuit function.
        unpumpedexpanded = self.Unpumped.binedDataExpansion(voltageLimit)
        iKK = self.Unpumped.iKKExpansion(voltageLimit)
        mask = self.mask_photon_steps[self.mask_photon_steps>0]
        def cost_function(params,unpumped,pumped,unpumpedexpanded,iKK,mask):
            '''TODO
            '''
            yEmb = params[0] + 1j*params[1] 
            iLO=params[2]
            vLO = self.vLO_from_circuit(iLO,yEmb,unpumpedexpanded,iKK,mask)
            alpha = np.divide(vLO,self.vPh*1e-3)
            pumpsim = self.pumped_from_unpumped(alpha,unpumped)
            voltagesOfInterst = np.expand_dims(mask, axis=-1) 
            return np.abs(np.sum(np.subtract(pumpsim[1,np.abs(pumpsim[0] - voltagesOfInterst).argmin(axis=-1)],pumped[1,np.isin(pumped[0],mask)])))
        guess = [.1,.1,2e-4]
        return fmin(cost_function,guess,args=(unpumped,pumped,unpumpedexpanded,iKK,mask),ftol=1e-9,xtol=1e-7)
    
    def yEmb_from_circuit_abs_sum_revers(self):
        '''TODO
        '''
        unpumped = self.Unpumped.offsetCorrectedBinedIVData
        pumped = self.Pumped.offsetCorrectedBinedIVData
        voltageLimit = np.abs([self.pumping_Levels[0,-1]+self.tuckerSummationIndex*self.vPh,
                               self.pumping_Levels[0,0]-self.tuckerSummationIndex*self.vPh]).max()    
        #avoid multiple calls of properties in the vLO_from_circuit function.
        unpumpedexpanded = self.Unpumped.binedDataExpansion(voltageLimit)
        iKK = self.Unpumped.iKKExpansion(voltageLimit)
        mask = self.mask_photon_steps[self.mask_photon_steps>0]
        def cost_function(params,unpumped,pumped,unpumpedexpanded,iKK,mask):
            '''TODO
            '''
            yEmb = params[0] + 1j*params[1] 
            iLO=params[2]
            vLO = self.vLO_from_circuit(iLO,yEmb,unpumpedexpanded,iKK,mask)
            alpha = np.divide(vLO,self.vPh*1e-3)
            pumpsim = self.pumped_from_unpumped(alpha,unpumped)
            voltagesOfInterst = np.expand_dims(mask, axis=-1) 
            return np.sum(np.abs(np.subtract(pumpsim[1,np.abs(pumpsim[0] - voltagesOfInterst).argmin(axis=-1)],pumped[1,np.isin(pumped[0],mask)])))
        guess = [.1,.1,2e-4]
        return fmin(cost_function,guess,args=(unpumped,pumped,unpumpedexpanded,iKK,mask),ftol=1e-9,xtol=1e-7)
    
def normalise_2d_array(array,xnormalisation,ynormalisation):
    '''TODO
    '''
    array[0] = np.divide(array[0],xnormalisation)
    array[1] = np.divide(array[1],ynormalisation)
    return array
        
        
'''
ToTest: 
    yEmb_from_circuit: takes some time
    yEmb_cost_Function_Skalare_fixed_iLO
    yEmb_cost_Function_Skalare_fixed_iLO
    yEmb_cost_Function
    zEmb_John
'''
        
        
        
        
        
        
        
        
        
        
        
    
    