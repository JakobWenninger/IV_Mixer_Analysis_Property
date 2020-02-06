import numpy as np
import datetime 
import matplotlib.pyplot as plt
from scipy import integrate,special
from scipy.misc import derivative
from scipy.special import expit
import pandas as pd
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

def plotsettings(name=None):
    '''This functions is used to set the plot layout and show the plot. 
    In case the name is not None, the plot is saved.
    ------
    inputs
    ------
    name: str
        Filename (without .pdf ending) used to save the plot to a file.
    '''
    plt.grid(True)
    plt.rcParams.update({'font.size': 12})
    #legend = plt.legend(loc='best', shadow=False,ncol=1)
    #leg = plt.gca().get_legend()
    #ltext  = leg.get_texts()  # all the text.Text instance in the legend
    #llines = leg.get_lines()  # all the lines.Line2D instance in the legend
    #plt.setp(ltext, fontsize='small')
    #plt.setp(llines, linewidth=1.5)      # the legend linewidth
    plt.tight_layout()
    if not name == None:
        plt.savefig(name+'.pdf', bbox_inches="tight")
    plt.show()

kB=1.3806e-23 # J/K
elec=-1.6022e-19 # C

deltaCooperPair = lambda temp,tempCrit : np.multiply(3.528/2*kB,np.multiply(tempCrit,np.sqrt(1-temp/tempCrit)))
deltaCooperPair.__doc__='''
This function returns the Cooper Pair binding energy according to BCS theory.
Tinkham equ. 3.54
------
inputs
------
temp: float or int or 1D array
    The actual temperature of the junction
tempCrit: float or int or 1D array
    The criticl temperture of the superconductor
 '''
def vgap_over_Temperature(te=np.arange(0,12,.01),teC=10):
    '''Plots the temperature dependence of the gap voltage
    ------
    inputs
    ------
    temp: float or int or 1D array
        The actual temperature of the junction
    tempCrit: float or int or 1D array
        The criticl temperture of the superconductor
    '''
    cond=te<=teC # Vgap = 0 for te>teC
    vgap=np.zeros_like(te) 
    vgap[cond]=2000/elec*deltaCooperPair(te[cond],teC)
    plt.plot(te,vgap)
    plt.xlabel('Temperature [K]')
    plt.ylabel('Voltage [mV]')
    plotsettings('Vgap_over_Temperature_at_Tc_%r_K'%teC)

def info(te,teC):
    '''This function prints information of the Cooper pair binding energy and critical voltage.
    ------
    inputs
    ------
    temp: float or int
        The actual temperature of the junction
    tempCrit: float or int
        The criticl temperture of the superconductor
    '''
    print('Cooper Pair Binding Energy (Delta) %r J'%deltaCooperPair(te,teC))
    print('Critical Voltage %r V'%(2*deltaCooperPair(te,teC)/elec))

def nS_over_nN0(x,te,teC):
    '''Returns the number of Cooper pairs over the number of electrons in normal state at 0 K, dependent on the energy.
    \frac{N_\text{S}(T)}{N_\text{N}(T=0\,\text{K})} Tinkham equ. 3.73
    ------
    inputs
    ------
    x: 1D array
        The energy relative to the fermi surface in J
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor
    '''
    x=np.array(x)
    dCP = deltaCooperPair(te,teC)
    ret = np.zeros_like(x)
    cond=(x>dCP) | (x<-dCP) # the condition of values not zero
    ret[cond] = np.abs(x[cond])/np.sqrt(np.subtract(np.multiply(x[cond],x[cond]),dCP*dCP))
    return ret

def _nS_over_nN0_Test(energyRange=np.arange(-.01*elec,.01*elec,.0001*elec),te=4,teC=10):
    '''Test function of :fun: nS_over_nN0.
    ------
    inputs
    ------
    energyRange: 1D array
        energy values relative to the fermi surface in J tested
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor
    '''
    plt.plot(energyRange,nS_over_nN0(energyRange,te,teC))
    plt.axvline(deltaCooperPair(te,teC))
    plt.axvline(-deltaCooperPair(te,teC))
    plt.xlabel('Energy [J]')
    plt.ylabel('Cooper Pair Density [m$^{-3}$]')
    plt.show()

def _expit_Test(energyRange=np.arange(-.01*elec,.01*elec,.0001*elec),te=4):
    '''Test function of scipy.special.expit. This function represents the Fermi Dirac Distribution.
    The probability to find an electron below the fermi surface is 1 while it is 0 above (at 0K)
     ------
    inputs
    ------
    energyRange: 1D array
        energy values relative to the fermi surface in J tested
    te: float or int
        The actual temperature of the junction
    '''
    plt.plot(energyRange,expit(-energyRange/(kB*te)))
    plt.show()

def _expit_Difference_Test(energyRange=np.arange(-.01*elec,.01*elec,.0001*elec),v0=0,te=4):
    '''Test function of scipy.special.expit. It tests the difference of two expit function subtracted from each other.
    ------
    inputs
    ------
    energyRange: 1D array
        energy values relative to the fermi surface in J tested
    v0: float
        Bias voltage applied to the junction
    te: float or int
        The actual temperature of the junction
    '''
    plt.plot(energyRange,np.subtract(expit(-np.divide(np.add(energyRange,np.multiply(elec,v0)),(kB*te))),expit(-np.divide(energyRange,(kB*te)))))
    plt.show()

funcToInt = lambda x,v0,te,teC : np.multiply(np.multiply(nS_over_nN0(x,te,teC),
                                                         nS_over_nN0(np.add(x,np.multiply(elec,v0)),te,teC)),
                                     np.subtract(expit(-np.divide(np.add(x,np.multiply(elec,v0)),(kB*te))),
                                                 expit(-np.divide(x,(kB*te)))))
funcToInt.__doc__='''
The function which needs to be integrated to compute the current through the SIS junction.
Tinkham equation 3.82
------
inputs
------
x: 1D array
    energy values relative to the fermi surface in J tested. Over this energy is integrated.
v0: float or int
    Bias voltage applied to the junction
te: float or int
    The actual temperature of the junction
teC: float or int
    The criticl temperture of the superconductor   
'''

def singularities(vtest,te=4,teC=10):
    '''This function computes the singularities occuring in :func: funcToInt.
    The singularities need to be specified in the scipy.integrate.quad to achieve proper integration.
    ------
    inputs
    ------
    vtest: float or int
        Bias voltage, which is applied to the junction
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor  
    ------
    return: np.array 1D with 6 values
        List of the singularities occuring  
    ####
    Singularities appear at +-(deltaCooperPair(te,teC)+elec*v0) and +-(deltaCooperPair(te,teC)-elec*v0)
    [J]		        [V]	
    2.435e-22	    all	    Cooper Pair energy
    -4.0375e-22	    .001 	deltaCooperPair(4,10)+elec*.001
    -7.24195e-22	.003	deltaCooperPair(4,10)+elec*.003
    minor peak decaying into increasing negative energy
    -8.844e-22	    .004	deltaCooperPair(4,10)+elec*.004
    -10.446e-22	    .005	deltaCooperPair(4,10)+elec*.005
    Until here the peaks are very small compared to the following peaks
    -2.53525e-22	>=.004	Cooper Pair energy
    -3.9735e-22	    .004	deltaCooperPair(4,10)-elec*.004
    -5.5757e-22	    .005	deltaCooperPair(4,10)-elec*.005
    -13.4867e-22	.01	    deltaCooperPair(4,10)-elec*.01
    '''
    pnts=deltaCooperPair(te,teC)
    pnts=np.abs(np.array([pnts,pnts-elec*vtest,pnts+elec*vtest]))
    return np.hstack([pnts,np.negative(pnts)])

def _funcToInt_Test(energyRange=np.arange(-.01*elec,.01*elec,.00001*elec),v0=np.arange(0,.01,0.0001),te=4,teC=10,hidden=False):
    '''Test function of :fun: funcToInt. It computes and plots the function over an energy range for several bias voltges.
    ------
    inputs
    ------
    energyRange: 1D array
        energy values relative to the fermi surface in J tested. Over this energy is integrated later.
    v0: 1D np.array
        Bias voltages applied to the junction which are computed
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor   
    hidden: bool
        Skip plotting the result
    ------
    return: 2D np.array
        The values of the function over the energy values (second index). For each bias voltage a subarray is returned (first index)
    ####
    Singularities appear at +-(deltaCooperPair(te,teC)+elec*v0) and +-(deltaCooperPair(te,teC)-elec*v0)
    [J]		        [V]	
    2.435e-22	    all	    Cooper Pair energy
    -4.0375e-22	    .001 	deltaCooperPair(4,10)+elec*.001
    -7.24195e-22	.003	deltaCooperPair(4,10)+elec*.003
    minor peak decaying into increasing negative energy
    -8.844e-22	    .004	deltaCooperPair(4,10)+elec*.004
    -10.446e-22	    .005	deltaCooperPair(4,10)+elec*.005
    Until here the peaks are very small compared to the following peaks
    -2.53525e-22	>=.004	Cooper Pair energy
    -3.9735e-22	    .004	deltaCooperPair(4,10)-elec*.004
    -5.5757e-22	    .005	deltaCooperPair(4,10)-elec*.005
    -13.4867e-22	.01	    deltaCooperPair(4,10)-elec*.01
    '''
    result=[]
    #vlines=np.array([])
    for i in v0:
        result.append(funcToInt(energyRange,i,te,teC))
        #vlines=np.hstack([vlines,singularities(te,teC,i)])
    result=np.array(result)
    if not hidden:
        plots=plt.plot(energyRange,result.T)
        #plt.vlines(vlines,-100,100)
        plt.legend(plots,v0,loc='best')
        plt.xlabel('Energy [J]')
        #plt.ylabel('')
        plt.show()
    return result

def _funcToInt_Test_temperature_sweep(energyRange=np.arange(-.01*elec,.01*elec,.00001*elec),v0=np.arange(0,.01,0.001),te=[.001,.1,1,4,7],teC=10):
    '''Test function of :fun: funcToInt. It computes and plots the function over an energy range for several bias voltges and over several temperatures.
    ------
    inputs
    ------
    energyRange: 1D array
        energy values relative to the fermi surface in J tested. Over this energy is integrated later.
    v0: 1D np.array
        Bias voltages applied to the junction which are computed
    te: 1D np.array
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor   
     '''
    result=[]
    fig=plt.figure()
    for temperatures in te:
        a=[]
        vlines=np.array([])
        for i in v0:
            a.append(funcToInt(energyRange,i,temperatures,teC))
            vlines=np.hstack([vlines,singularities(vtest,te,teC)])
        result.append(np.sum(a,1))
        plt.plot(v0,result[-1],label='t %r'%temperatures)
        plt.vlines(vlines,-100,100)
    plt.legend()
    plt.show()

def current_Quad(vrange=np.arange(-.01,.01,0.0002),intboundry=elec*10,te=4,teC=10,rN=100,hidden=False):
    '''This function integrates the :func: funcToInt to compute the current for different bias voltages with scipy.integrate.quad requires the location of singularities to work properly.
    Tinkham equation 3.82 or Nicols et al 1960
    ------
    inputs
    ------
    vrange: 1D np.array
        Bias voltages applied to the junction which are computed
    intboundry: float 
        The energy range (from -intboundry to +intboundry) which is integrated over. In the literature this is infinity, but a limited integration is sufficient anyway. 
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor
    rN: float or int
        The normal resistivity.
    hidden: bool
        Skip plotting the result   
    -------
    returns 1D np.array
        The currents evaluated
     '''
    current=[]
    for vtest in vrange: 
        intres = integrate.quad(funcToInt,-intboundry,intboundry,args=(vtest,te,teC),points=singularities(vtest,te,teC))
        current.append(intres[0]/(elec*rN))
    current=np.array(current)
    if not hidden:
        plt.plot(vrange,current)
        plt.xlabel('Voltage [V]')
        plt.ylabel('Current [A]')
        plt.show()
    return current

def current_Fixed(vrange=np.arange(-.01,.01,0.0002),sampledEnergies=np.arange(-.01*elec,.01*elec,.00001*elec), te=4,teC=10,rN=100,hidden=False):
    '''This function integrates the :func: funcToInt to compute the current for different bias voltages with discrete values. The descrete vaues are equispaced. 
    The distance between the sampled energies is not accounted for. This function is more for testing reasons of :func: funcToInt, while integrate.quad did not work
    ------
    inputs
    ------
    vrange: 1D np.array
        Bias voltages applied to the junction which are computed
    sampledEnergies: 1D np.array 
        The energy range (from -intboundry to +intboundry in descrete steps) which is integrated over. 
    te: 1D np.array
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor
    rN: float or int
        The normal resistivity.
    hidden: bool
        Skip plotting the result   
     '''
    current=[]
    for vtest in vrange: 
        res=funcToInt(sampledEnergies,vtest,te,teC)
        current.append(np.divide(np.sum(res,axis=0),elec*rN))
        #current.append(intres[0]/(elec*rN))
    if not hidden:
        plt.plot(vrange,current)
        plt.xlabel('Voltage [V]')
        plt.ylabel('Current [A]')
        plt.show()
    return current


def current_over_temperature(te=[.01,.1,.4,1,4,9],teC=10,
                             vrange=np.arange(-.01,.01,0.00001), intboundry=elec*10,rN=100): 
    '''This function plots the current over different temperatures.
    ------
    inputs
    ------
    te: 1D np.array
        The actual temperature of the junction to be plotted
    teC: float or int
        The criticl temperture of the superconductor
    vrange: 1D np.array
        Bias voltages applied to the junction which are plotted
    intboundry: float 
        The energy range (from -intboundry to +intboundry) which is integrated over. In the literature this is infinity, but a limited integration is sufficient anyway. 
    rN: float or int
        The normal resistivity.
    '''
    currents=[]
    for tet in te:
        print('Integrate %r K'%tet)
        currents.append(current_Quad(vrange=vrange,intboundry=intboundry,te=tet,teC=teC,rN=rN,hidden=True))
    plots=plt.plot(vrange,np.array(currents).T)
    plt.legend(plots,te,loc='best')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A]')
    now = datetime.datetime.now()
    plotsettings('Current_over_Temperature_at_Tc_%r_K_%s'%(teC,now.strftime('%Y_%m_%d_%H_%M')))

def current_over_rN(rN=np.logspace(1.5,3,7),te=4,teC=10,
                             vrange=np.arange(-.01,.01,0.00001), intboundry=elec*10,): 
    '''This function plots the current over different normal resistances.
    ------
    inputs
    ------
    rN: 1D array
        The normal resistivity which are plotted
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor
    vrange: 1D np.array
        Bias voltages applied to the junction which are plotted
    intboundry: float 
        The energy range (from -intboundry to +intboundry) which is integrated over. In the literature this is infinity, but a limited integration is sufficient anyway. 
    '''
    currents=[]
    for rNt in rN:
        print('Integrate %r $\Omega$'%rNt)
        currents.append(current_Quad(vrange=vrange,intboundry=intboundry,te=te,teC=teC,rN=rNt,hidden=True))
    plots=plt.plot(vrange,np.array(currents).T)
    plt.legend(plots,rN,loc='best')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A]')
    now = datetime.datetime.now()
    plotsettings('Current_over_Normal_Resistance_%s'%(now.strftime('%Y_%m_%d_%H_%M')))
    
    
current_Quad(np.arange(-.01,.01,0.00002)) #computation of IV curve

#raw_input("="*20+"Done"+"="*20)   

