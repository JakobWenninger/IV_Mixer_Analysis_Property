#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:46:17 2019

@author: wenninger
"""

'''
Situation:
    There are problems with the result of the impedance recovery.
    My Results      15.5-12.9j
    Johns Results   6.3-4j
    
Further there are problems by changing the maximum and minimum voltage of the bins. 

These issues are debugged within this script.    
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import os
from Mixer import Mixer,kwargs_Mixer,kwargs_IV_Response_John
from plotxy import plot

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
#def legend_settings(fig,ax):
#    legend = ax.legend(loc='best', shadow=False,ncol=1)
#    leg = fig.gca().get_legend()
#    ltext  = legend.get_texts()  # all the text.Text instance in the legend
#    llines = legaxget_lines()  # all the lines.Line2D instance in the legend
#    fig.setp(ltext, fontsize='small')
#    fig.setp(llines, linewidth=1.5)      # the legend linewidth
#    fig.tight_layout()


directory = 'Impedance_Recovery/Different_Impedance_Recovery_Parameter_Test_2019_12_06/'
if not os.path.exists(directory):
        os.makedirs(directory)

kwargs_Mixer_John = {**kwargs_Mixer,**kwargs_IV_Response_John}
kwargs_Mixer_John['steps_ImpedanceRecovery']=1
#set kwargs
kwargs = []

kwargs_2001_6 = kwargs_Mixer_John.copy()
kwargs_2001_6['descriptionMixer']='2001 6'
kwargs.append(kwargs_2001_6)
kwargs_1001_6 = kwargs_Mixer_John.copy()
kwargs_1201_6 = kwargs_Mixer_John.copy()
kwargs_1201_6['numberOfBins']=1201
kwargs_1201_6['descriptionMixer']='1201 6'
kwargs.append(kwargs_1201_6)
kwargs_1001_6['numberOfBins']=1001
kwargs_1001_6['descriptionMixer']='1001 6'
kwargs.append(kwargs_1001_6)
kwargs_801_6 = kwargs_Mixer_John.copy()
kwargs_801_6['numberOfBins']=801
kwargs_801_6['descriptionMixer']=' 801 6'
kwargs.append(kwargs_801_6)

kwargs_2001_10 = kwargs_Mixer_John.copy()
kwargs_2001_10['vmin']=-10
kwargs_2001_10['vmax']=10
kwargs_2001_10['descriptionMixer']='2001 10'
kwargs.append(kwargs_2001_10)
kwargs_1201_10 = kwargs_Mixer_John.copy()
kwargs_1201_10['numberOfBins']=1201
kwargs_1201_10['vmin']=-10
kwargs_1201_10['vmax']=10
kwargs_1201_10['descriptionMixer']='1201 10'
kwargs.append(kwargs_1201_10)
kwargs_1001_10 = kwargs_Mixer_John.copy()
kwargs_1001_10['numberOfBins']=1001
kwargs_1001_10['vmin']=-10
kwargs_1001_10['vmax']=10
kwargs_1001_10['descriptionMixer']='1001 10'
kwargs.append(kwargs_1001_10)
kwargs_801_10 = kwargs_Mixer_John.copy()
kwargs_801_10['numberOfBins']=801
kwargs_801_10['vmin']=-10
kwargs_801_10['vmax']=10
kwargs_801_10['descriptionMixer']=' 801 10'
kwargs.append(kwargs_801_10)

steps_ImpedanceRecovery = [1,2,3]
numberOfBins = [2001,1401,1201,1101,1001,901,801,701,601]
vlimit = [10,6]
#For testing
#steps_ImpedanceRecovery = [1]
#numberOfBins = [2001]
#vlimit = [10]

kwargs =[]
for s in steps_ImpedanceRecovery:
    for v in vlimit:
        for n in numberOfBins:
            kwargs.append(kwargs_Mixer_John.copy())
            kwargs[-1]['numberOfBins']=n
            kwargs[-1]['vmin']=-v
            kwargs[-1]['vmax']=v
            kwargs[-1]['descriptionMixer']='%d %04d %02d'%(s,n,v)


#initiate Mixers
files = ['DummyData/John/Unpumped.csv','DummyData/John/Pumped.csv']
Ms=[]#Mixers
for i in kwargs:
    Ms.append(Mixer(files[0],files[1],**i))

def unpumped():
    #plot unpumped IV curves
    subject='Unpumped'
    f0 = plt.figure()
    for i in Ms:
        try:
            ax0 = f0.add_subplot(1, 1, 1)
            ax0.plot(i.Unpumped.offsetCorrectedBinedIVData[0],i.Unpumped.offsetCorrectedBinedIVData[1],label=i.descriptionMixer)
        except:
            print('Exception in Mixer:')
            print(i.descriptionMixer)
    legend = ax0.legend(loc='best', shadow=False,ncol=2)
    f0.savefig(directory+subject+'.pdf')
    f0.show()

def pumped():
    #plot pumped IV curves
    subject='Pumped'
    f1 = plt.figure()
    for i in Ms:
        ax0 = f1.add_subplot(1, 1, 1)
        ax0.plot(i.Pumped.offsetCorrectedBinedIVData[0],i.Pumped.offsetCorrectedBinedIVData[1],label=i.descriptionMixer)
    legend = ax0.legend(loc='best', shadow=False,ncol=2)
    f1.savefig(directory+subject+'.pdf')
    f1.show()    
    
def pumping_Levels():
    #plot Pumping Levels
    subject='Pumping_Levels'
    f2 = plt.figure()
    for i in Ms:
        ax0 = f2.add_subplot(1, 1, 1)
        ax0.plot(i.pumping_Levels[0],i.pumping_Levels[1],label=i.descriptionMixer)
    legend = ax0.legend(loc='best', shadow=False,ncol=2)
    f2.savefig(directory+subject+'.pdf')
    f2.show()    

def real_AC_Current():
    #plot Real AC Current
    subject='Real_AC_Current'
    f3 = plt.figure()
    for i in Ms:
        ax0 = f3.add_subplot(1, 1, 1)
        ax0.plot(i.iACSISRe[0],i.iACSISRe[1],label=i.descriptionMixer)
    legend = ax0.legend(loc='best', shadow=False,ncol=2)
    f3.savefig(directory+subject+'.pdf')
    f3.show() 

def imaginary_AC_Current():
    #plot imaginary AC Current
    subject='Imaginary_AC_Current'
    f4 = plt.figure()
    for i in Ms:
        ax0 = f4.add_subplot(1, 1, 1)
        ax0.plot(i.iACSISIm[0],i.iACSISIm[1],label=i.descriptionMixer)
    legend = ax0.legend(loc='best', shadow=False,ncol=2)
    f4.savefig(directory+subject+'.pdf')
    f4.show() 

def admittance_SIS_Junction():
    #plot Admittance SIS junction
    subject='Admittance_SIS_Junction'
    f5 = plt.figure()
    for i in Ms:
        ax0 = f5.add_subplot(1, 1, 1)
        ax0.plot(i.ySIS[0],i.ySIS[1],label=i.descriptionMixer)
    ax0.set_ylim(-.1,.1)
    legend = ax0.legend(loc='best', shadow=False,ncol=2)
    f5.savefig(directory+subject+'.pdf')
    f5.show() 

def admittance_SIS_Junction_Masked():
    #plot Admittance SIS junction
    subject='Masked_ySIS_'
    fmask=[]
    for i in Ms:
        fmask.append(plt.figure())
        i.plot_mask_steps()
        plt.plot(i.ySIS[0],i.ySIS[1]*1000)
        plt.ylim(-400,400)
        fmask[-1].savefig(directory+subject+i.descriptionMixer.replace(' ','_')+'.pdf')
        fmask[-1].show() 

def embeddingImpedance():     
    #Embedding Impedances
    output = ''
    for i in Ms:
        #print embedding impedance
        zEmb =i.zEmb
        output+=('Embedding Impedance '+i.descriptionMixer+'\t Ratio Voltage/Bin '+'%.4f \t %.2f, %.2f Ohm'%(i.Unpumped.binWidth,zEmb[0],zEmb[1]))
    print(output)
    Log = open(directory+'Embedding_Impedance_log.txt','w')
    Log.write(output)
    Log.close()


