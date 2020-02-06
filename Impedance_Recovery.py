#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv
from scipy.optimize import fmin
import scipy.constants as const
from scipy.signal import hilbert
from IV_Class import *
from ExpandingFunction import expandFuncWhile

class impedanceRecovery(object):
    def __init__(self,V0,currentUnpumped,currentPumped,rN,vLO=None,freq=None,alpha=None):
        '''This class recovers the impedance of an SIS junction from the pumed IV curve
        
        params
        ------
        V0
            The voltages which are evaluated
        currentUnpumped: same shape as V0
            Data of the unpumped IV curve.
        currentPumped: same shape as V0
            Data of the pumped IV curve
        rN: float
            normal resistance Rn of the junction
        vLO: float
            The voltage of the LO pump.
        freq: float
            The frequency of the LO pump.
        alpha: float
            The Pumping level.
            Two of vLO, freq and alpha must be defined.
        '''
        self.V0 = V0
        self.currentUnpumped = currentUnpumped
        self.currentPumped = currentPumped
        self.rN = rN
        self.freq, self.vLO, self.alpha = alphaFunction(vLO, freq, alpha)

    @property
    def iDCn1(self):
        '''Computes the DC current for the first photon step.
        
        returns array
            I_DC(V_0+hf/e)
        '''
        besselsquare = np.square(jv([0,1],self.alpha))
        return np.divide(np.subtract(self.currentPumped,besselsquare[0]*self.currentUnpumped),besselsquare[1])
    
    @property
    def iDCn1KK(self):
        '''Computes the Kramers Kronig Transformation of the DC current for the first photon step.
        TODO Cauchy Principle Value is assumed to be 1
        
        returns array of size of V0
            Kramers Kronig Transformed Currents
        '''
        dividend = np.subtract(self.iDCn1,np.divide(self.V0,self.rN)) #vector
        divisor = np.subtract(self.V0,np.matrix(self.V0).T) #array
        quotient = np.divide(dividend,divisor)
        quotient[np.isinf(quotient)]=0 # replace infinity by 0 (division through 0)
        integrated = np.sum(quotient,1) #integrate over V' (not V)
        #TODO Cauchy Principle Value is assumed to be 1
        return 1/np.pi * np.array(integrated)[:,0] # np array necessary to get rid of np.matrix to return vector
        
        
    @property
    def ifreq(self):
        '''The AC current caused by the first photon step (n=1)
        '''
        pass
        ifreq = self.ifreqRe + np.multiply( 1j, self.ifreqIm)
        
    @property
    def ifreqRe(self):
        '''The real AC current caused by the first photon step (n=1)
        
        returns:
        --------
        array of shape V0
            The real part of the AC current
        '''
        bessel = jv(np.arange(3),self.alpha)
        return bessel[1]*(bessel[0]+bessel[2])*self.iDCn1
    
    @property
    def ifreqIm(self):
        '''The imaginary AC current caused by the first photon step (n=1)
        
        returns:
        --------
        array of shape V0ß
            The imaginary part of the AC current
        '''
        bessel = jv(np.arange(3),self.alpha)
        return bessel[1]*(bessel[0]-bessel[2])*self.iDCn1KK

    @property
    def admittanceY(self):
        '''The admittance computed from the bias voltage and the complex current through the junction at every voltage bias point
        '''
        pass
        #return np.divide(V0,ifreq)
    
    @property
    def conductanceG(self):
        '''Real part of admittance Y.
        '''
        return np.real(self.admittanceY)
     
    @property
    def subsceptanceB(self):
        '''Imaginary part of admittance Y.
        '''
        return np.imag(self.admittanceY)


    

def i0 (iDC,nMin,nMax,vLO=None,freq=None,alpha=None):
    ''' This function computes the currents of a pumped IV curve from I_DC(V0+n hf/e)
    TODO need information of iDC
    params
    ------
    iDC: array
        The dc current array
    nMin, nMax: integer
        The minimum and maximum of the sum
    vLO: float
        The voltage of the LO pump.
    freq: float
        The frequency of the LO pump.
    alpha: float
        The Pumping level.
        Two of vLO, freq and alpha must be defined.
    '''
    def singleSumElement(n):
        '''Computes a single sum element
    
        params
        ------
        n: integer
            the n evaluated
        
        returns
        -------
        float
        '''
        pass
    nArray = np.arange(nMin,nMax)
    pass 
    

def alphaFunction(vLO=None,freq=None,alpha=None):
    '''
    Computes the missing parameter
     Two of vLO, freq and alpha must be defined.
    
    params
    ------
    vLO: float
            The voltage of the LO pump.
    freq: float
            The frequency of the LO pump.
    alpha: float
            The LO Pumping level

    returns
    -------
    vLO, freq, alpha
    '''
    try:
        if vLO == None:
            vLO = (alpha * const.h * freq/ const.e)
        elif freq == None  :  
            freq = (const.e * vLO/ const.h * alpha)
        elif alpha == None:
            alpha = (const.e * vLO/ const.h/ freq)     
        else:
            print('All parameters were defined, so that a computation of alphaFunction is obsolet')
    except TypeError:
        print('alphaFunction could not be computed, since more than one parameter is not defined')
    return vLO, freq, alpha
    
def alphaDetermination(unpumped,pumped,fLO):
    '''This function computes alpha and therefore the pumping level from a pumped and unpumped IV curve
    
    inputs
    ------
    unpumped: array
        The IV data of the unpumped IV curve   
    pumped: array
        The IV data of the pumped IV curve
    fLO: float
        frequency of the LO pump
    
    returns
    -------
    float
        The value of alpha
    
    '''
    def pumpedFunction(alpha,unpumped,pumped,fLO):
        '''The function to be minimised.
    
        inputs
        ------
        alpha: float
            Guess value of alpha
        unpumped: array
            The IV data of the unpumped IV curve   
        pumped: array
            The IV data of the pumped IV curve
        fLO: float
            frequency of the LO pump
    
        returns
        -------
        float
            The difference between the functions
        '''
        besselsq = jv(np.arange(2),alpha)**2    
        vPh = const.h*fLO/const.e *1000 # mV
        currentV0 = unpumped[1,np.where(unpumped[0]>unpumped[0,0] + vPh)[0]] # chop off lower currents (to add it with the offseted data)
        currentVoffseted = unpumped[1,np.where(unpumped[0]<unpumped[0,-1] - vPh)[0]] # chop off upper currents (to add it with the offseted data) 
      
        #plt.plot(unpumped[0,np.where(unpumped[0]>unpumped[0,0] + vPh)[0]],currentV0)
        #plt.plot(unpumped[0,np.where(unpumped[0]>unpumped[0,0] + vPh)[0]],currentVoffseted)
        #plt.show()
        bessel0 = besselsq[0]*currentVoffseted
        bessel1 =besselsq[1]*currentV0
        pumpCurrent = pumped[1,np.where(pumped[0]<pumped[0,-1] - vPh)[0]]
        #valueMinimised = bessel0 + bessel1 - pumped[1,np.where(pumped[0]>pumped[0,0] + vPh)[0]]
        valueMinimised = bessel0 + bessel1 - pumpCurrent
        #        plt.plot(currentV0)
        #        plt.plot(currentVoffseted)
        #plt.plot(bessel0)
        #plt.plot(bessel1)
        #plt.plot(pumpCurrent)
        #plt.plot(valueMinimised)
        #plt.show()
        return np.nansum(np.abs(valueMinimised))
    return fmin(pumpedFunction,1,args=(unpumped,pumped,fLO))

def alphaDeterminationAtEachBiasWithoutNormalResistanceCorrection(unpumped,pumped,fLO,n):
    '''This function computes alpha and therefore the pumping level from a pumped and unpumped IV curve.
    This is done without fitting the normal resistance above the actual data set. Instead the data at the limits is token for calculation.
    
    inputs
    ------
    unpumped: array
        The IV data of the unpumped IV curve   
    pumped: array
        The IV data of the pumped IV curve
    fLO: float
        frequency of the LO pump
    n: int
            The summation index of the function
            
    returns
    -------
    2D array
        The value of alpha [1] and the associated voltage [0]
    
    '''
    def pumpedFunction(alpha,bias, unpumped,pumped,vPh,n):
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
            Voltage equivalent of photons arriving
        n: int
            The summation index of the function
    
        returns
        -------
        float
            The absolute difference of pumped and extract of unpumped IV curve
        '''
        n= np.arange(-n,n+1)
        bessel = jv(n,alpha)**2
        unpumpedOffseted = []
        for nx in n:
            unpumpedOffseted.append(unpumped[1,(np.abs(unpumped[0]-(bias+nx*vPh))).argmin()])
        #print(unpumpedOffseted)
        return np.abs(np.nansum(bessel*unpumpedOffseted-pumped[1,(np.abs(pumped[0]-(bias))).argmin()]))  
        
        #valueMinimised = bessel0 + bessel1 - pumpCurrent
        #plt.plot(bessel0)
        #plt.plot(bessel1)
        #plt.plot(pumpCurrent)
        #plt.plot(valueMinimised)
        #plt.show()
        #return np.nansum(np.abs(valueMinimised))
        #return np.abs(valueMinimised)
    vPh = const.h*fLO/const.e *1000 # mV
    alphaArray = []
    #print('Finding alpha for each bias point')
    print('Process pumping level for each voltage bias point.')
    for i in pumped[0]: # evaluate every bias voltage of the pumped IV curve
        #print('Processing ', i)
        alphaArray.append([i,fmin(pumpedFunction,.8,args=(i,unpumped,pumped,vPh,n),disp=False)[0]])
    print('Finished.')
#    plt.plot(alphaArray)
#    plt.plot(currentV0)
#    plt.plot(currentVoffseted)
#    plt.plot(pumpCurrent)
#    plt.show()
    return np.array(alphaArray).T

def alphaDeterminationAtEachBias(Unpumped,Pumped,fLO,n):
    '''This function computes alpha and therefore the pumping level from a pumped and unpumped IV curve.
    The IV data beyond the IV data is token from the normal resistance fit. However, the code takes longer than alphaDeterminationAtEachBiasWithoutNormalResistanceCorrection
    
    inputs
    ------
    Unpumped: object
        The IV class of the unpumped IV curve   
    Pumped: object
        The IV class of the pumped IV curve
    fLO: float
        frequency of the LO pump
    n: int
            The summation index of the function
            
    returns
    -------
    2D array
        The value of alpha [1] and the associated voltage [0]
    
    '''
    def pumpedFunction(alpha,bias, unpumped,pumped,rN_LinReg, vPh,n):
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
        rN_LinReg: array
            The fit for negative [0] and positive [1] regression of the normal resistance
        vPh: float
            Voltage equivalent of photons arriving in mV
        n: int
            The summation index of the function
    
        returns
        -------
        float
            The absolute difference of pumped and extract of unpumped IV curve
        ''' 
        n= np.arange(-n,n+1)
        bessel = jv(n,alpha)**2
        unpumpedOffseted = []
        for nx in n:
            unpumpedOffseted.append(unpumped[1,(np.abs(unpumped[0]-(bias+nx*vPh))).argmin()])
        return np.abs(np.nansum(bessel*unpumpedOffseted-pumped[1,(np.abs(pumped[0]-(bias))).argmin()]))  
        
        #valueMinimised = bessel0 + bessel1 - pumpCurrent
        #plt.plot(bessel0)
        #plt.plot(bessel1)
        #plt.plot(pumpCurrent)
        #plt.plot(valueMinimised)
        #plt.show()
        #return np.nansum(np.abs(valueMinimised))
        #return np.abs(valueMinimised)
    pumped = Pumped.binedIVData   
    rN_LinReg=Unpumped.rN_LinReg
    vPh = const.h*fLO/const.e *1000 # mV
    
    #compute the whole voltage range, relevant for the Kramers Kronig Transformation
    voltageLimit = np.abs([Unpumped.binedIVData[0,-1]+n*vPh,Unpumped.binedIVData[0,0]-n*vPh]).max()
    unpumped = Unpumped.binedDataExpansion(voltageLimit)
    plt.plot(unpumped[0],unpumped[1])
    alphaArray = []
    #print('Finding alpha for each bias point')
    print('Process pumping level for each voltage bias point.')
    for i in pumped[0]: # evaluate every bias voltage of the pumped IV curve
        #print('Processing ', i)
        alphaArray.append([i,fmin(pumpedFunction,.8,args=(i,unpumped,pumped,rN_LinReg,vPh,n),disp=False)[0]])
    print('Finished.')
#    plt.plot(alphaArray)
#    plt.plot(currentV0)
#    plt.plot(currentVoffseted)
#    plt.plot(pumpCurrent)
#    plt.show()
    return np.array(alphaArray).T

Unpumped = IV_Response('DummyData/John/Unpumped.csv',columnOffset=0,headerLines=1,currentFactorToMicroampere=1000,rNThresholds = [3.2,6])
Pumped = IV_Response('DummyData/John/Pumped.csv',columnOffset=0,headerLines=1,currentFactorToMicroampere=1000,rNThresholds = [3.2,6])
fLO = 230.2e9 #Hz guessed LO frequency for this data 

#Debugging:
vPh = const.h*fLO/const.e *1000 # mVf
 
unpumped = Unpumped.binedIVData
pumped = Pumped.binedIVData

#alphasWithoutCorrection = alphaDeterminationAtEachBiasWithoutNormalResistanceCorrection(unpumped,pumped,fLO,15)
alphas = alphaDeterminationAtEachBias(Unpumped,Pumped,fLO,15)


#plt.plot(alphas[0],alphas[1])
#plt.plot(alphasWithoutCorrection[0],alphasWithoutCorrection[1])
#plt.show()

#plt.plot(alphas[0],np.multiply(alphas[1],100))
#plt.plot(unpumped[0],unpumped[1])
#plt.plot(pumped[0],pumped[1])
#plt.show()


def ifreqReWithoutCorrection(alphas, unpumped,vPh,n):
    '''The real AC current caused by the first photon step (n=1).
    This is done on behalf of the unpumped DC IV curve and the pumping level at each bias voltage alpha.
    This is done without fitting the normal resistance above the actual data set. Instead the data at the limits is token for calculation.

    inputs
    ------
    alphas: 2D array
        The alphas at each bias voltage
        [0] bias voltage
        [1] pumping level
    unpumped: array
        The IV data of the unpumped IV curve  
    vPh: float
        Voltage equivalent of photons arriving
    n: int
        The summation index of the function
    returns:
    --------
    array of shape V0
        The real part of the AC current
    '''
    n= np.arange(-n-1,n+2) # accont for one extra bessel function in each direction
    ifreqRe = []
    for i in range(len(alphas[0])):
        bessel = jv(n,alphas[1,i])
        unpumpedOffseted = []
        for nx in n[1:-1]:
            unpumpedOffseted.append(unpumped[1,(np.abs(unpumped[0]-(alphas[0,i]+nx*vPh))).argmin()])
        ifreqRe.append([alphas[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
    return np.array(ifreqRe).T


def ifreqRe(alphas, Unpumped,vPh,n):
    '''The real AC current caused by the first photon step (n=1).
    This is done on behalf of the unpumped DC IV curve and the pumping level at each bias voltage alpha
        The IV data beyond the IV data is token from the normal resistance fit.
    
    inputs
    ------
    alphas: 2D array
        The alphas at each bias voltage
        [0] bias voltage
        [1] pumping level
    Unpumped: object
        The IV class of the unpumped IV curve   
    vPh: float
        Voltage equivalent of photons arriving
    n: int
        The summation index of the function
    returns:
    --------
    array of shape V0
        The real part of the AC current
    '''
    voltageLimit = np.abs([alphas[0,-1]+n*vPh,alphas[0,0]-n*vPh]).max()
    unpumped = Unpumped.binedDataExpansion(voltageLimit)
    n= np.arange(-n-1,n+2) # accont for one extra bessel function in each direction
    ifreqRe = []
    for i in range(len(alphas[0])):
        bessel = jv(n,alphas[1,i])
        unpumpedOffseted = []
        for nx in n[1:-1]:
            unpumpedOffseted.append(unpumped[1,(np.abs(unpumped[0]-(alphas[0,i]+nx*vPh))).argmin()])
        ifreqRe.append([alphas[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
    return np.array(ifreqRe).T
 
   
#iACReWO=ifreqReWithoutCorrection(alphas,unpumped,vPh,6)
iACRe=ifreqRe(alphas,Unpumped,vPh,6)
#plt.plot(iACReWO[0],iACReWO[1])
#plt.plot(iACRe[0],iACRe[1])
#plt.show()

#
#plt.plot(alphas[0],(alphas[1]))
#plt.plot(alphas[0],iACRe[1])
#plt.plot(unpumped[0],unpumped[1])
#plt.plot(pumped[0],pumped[1])
#plt.show()

def iKKcalc(ivData):
    '''This function computes the Kramers Kronig Transformation current using scipy.signal.hilbert
    
    inputs
    ------
    ivData: array
        The IV data on which the Kramers Kronig Transformation is applied.
    
    returns
    -------
    array of size of V0
        Kramers Kronig Transformed Currents
    '''
    
    return np.array([ivData[0],-hilbert(ivData[1]-ivData[0]*1e3/15).imag]) #TODO minus sign? subtraction of V/Rn?
 
def ifreqIm(alphas, Unpumped,vPh,n):
    '''The real AC current caused by the first photon step (n=1).
    This is done on behalf of the unpumped DC IV curve and the pumping level at each bias voltage alpha
    
    inputs
    ------
    alphas: 2D array
        The alphas at each bias voltage
        [0] bias voltage
        [1] pumping level
    Unpumped: object
        The class of the unpumped IV curve  
    vPh: float
        Voltage equivalent of photons arriving
    n: int
        The summation index of the function
        
    returns:
    --------
    array of shape V0
        The real part of the AC current
    '''
    voltageLimit = np.abs([alphas[0,-1]+n*vPh,alphas[0,0]-n*vPh]).max()
    iKK = iKKcalc(Unpumped.binedDataExpansion(voltageLimit))
    plt.plot(iKK[0],iKK[1])
    plt.plot(Unpumped.binedDataExpansion(voltageLimit)[0],Unpumped.binedDataExpansion(voltageLimit)[1])
    plt.show()
    
#    plt.plot(voltageRange,currents)
#    plt.plot(iKK[0],iKK[1])
#    plt.show()
    n= np.arange(-n-1,n+2) # accont for one extra bessel function in each direction
    ifreqIm = []
    for i in range(len(alphas[0])):
        bessel = jv(n,alphas[1,i])
        iKKOffseted = []
        for nx in n[1:-1]:
            iKKOffseted.append(iKK[1,(np.abs(iKK[0]-(alphas[0,i]+nx*vPh))).argmin()])
        ifreqIm.append([alphas[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.subtract(bessel[:-2],bessel[2:])),iKKOffseted))])
    return np.array(ifreqIm).T
 
iACIm=ifreqIm(alphas,Unpumped,vPh,6)

def ySIS(iSISRe,iSISIm):
    '''Compute the admittance of the SIS junction.
    
    inputs
    ------
    iSISRe: 2d array
        The real AC current through the SIS junction.
    iSISIm: 2d array
        The imaginary AC current through the SIS junction.
    
    returns
    -------
    2d array
        The admittance per bias voltage        
    '''
    #search relevant indexes by bias voltage
    indexes = np.searchsorted(iSISRe[0],iSISIm[0]) 
    y = np.divide((iSISRe[1] + 1j* iSISIm[1,indexes]),iSISRe[0])
    y = np.vstack([iSISRe[0],y])
    return y[:,np.logical_not(np.isinf(y[1]))]

ySISarray = ySIS(iACRe,iACIm)

def findYemb(ySIS):
    '''This function finds the embedding admittance from the voltage dependent admittance of the SIS junction
    
    innputs
    -------
    ySIS: 2d array
        The admittance of the SIS junction [1] over the applied bias voltage [0].
    '''
    def errorFunction(yLO,ySIS,dummy):
        '''The error function given by Skalare equation 7.
        
        inputs
        ------ 
        yLO: 2 element array, real and complex value
            The parameter which requires fit to the admittance of the circuit
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0]
        
        returns
        -------
        float:
            The error of the function
        '''
        #iLO = np.multiply(ySIS[0],np.sqrt(ySS)) 
        iLO = yLO[2] # TODO is LO a set value or is it an array
        yLO = yLO[0]+1j*yLO[1]
        #SS -> Sum Squared
        ySS = np.add(np.square(yLO.real+ySIS[1].real) , np.square(yLO.imag+ySIS[1].imag))
        errorfunc = []
        errorfunc.append( np.sum(np.square(ySIS[0]))) # single value
        errorfunc.append( np.square(np.abs(iLO))*np.sum(np.reciprocal(ySS)))
        errorfunc.append( -2*np.abs(iLO)*np.sum(np.divide(ySIS[0],ySS)))
        print(errorfunc)
        return np.sum(errorfunc)
    guess = [50,50,1000] # Does not take complex value as input
    return fmin(errorFunction,guess,args=(ySIS,1))

yEmb = findYemb(ySISarray)







    
    
    
    
    
    
    
    
    