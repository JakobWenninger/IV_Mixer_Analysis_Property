#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:48:08 2019

@author: wenninger
"""

from Mixer import Mixer,kwargs_Mixer_John
tuckerSummationIndex = 10

def cost_vLO(params,iLO,yLO,iVunpumped):
#    params: vLO
    #all values ar at a single bias voltage.
    
    
    return iLO-iAC-vLO*yLO





def iACSISRe_Calc(self,unpumped):
    '''The real AC current computed from the unpumped DC IV curve and the pumping level at each bias voltage alpha.
        
    inputs
    ------
    unpumped: 2d array
        The IV data of the unpumped IV curve extended to allow computation of V0+n*Vph.
    
    returns:
    --------
    array of shape V0
        The real part of the AC current
    '''
    n= np.arange(-self.tuckerSummationIndex-1,self.tuckerSummationIndex+2) # accont for one extra bessel function in each direction
    ifreqRe = []
    for i in range(len(self.pumping_Levels[0])):
        bessel = jv(n,self.pumping_Levels[1,i])
        unpumpedOffseted = []
        for nx in n[1:-1]:
            unpumpedOffseted.append(unpumped[1,(np.abs(unpumped[0]-(self.pumping_Levels[0,i]+nx*self.vPh))).argmin()])
#            ifreqRe.append([unpumped[0,(np.abs(unpumped[0]-(self.pumping_Levels[0,i]))).argmin()],
#                                              np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
        ifreqRe.append([self.pumping_Levels[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])

    return np.array(ifreqRe).T