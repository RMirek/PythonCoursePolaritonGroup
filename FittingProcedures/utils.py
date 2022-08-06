# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy.optimize import curve_fit
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker
plt.rcParams.update({'font.size': 15}) # font size

def valueIndex(array, val):
    '''
    Finding the closest value to the 'val' argument. 
    Gives value and the index in array. Working ONLY in SORTED array!!!
    
    
    Parameters
    ----------
    array : np array 
        sorted data.
    value : float
        selected value searched in an array.
            

    Returns:
    --------    
    value : float
        value closest to 'val' in array
    idx : int
        position/index of given value in array    
    '''
    # array = np.asarray(array)
    idx = (np.abs(array - val)).argmin()
    return array[idx], array.tolist().index(array[idx])


def joinModes(modes_pos_left, modes_pos_right, angles, zakp2, zakk):
    """ 
    Merging the modes found in negative and positive angles"


    Parameters
    ----------
    modes_pos_left : list
        list of modes energies for negative angles 
    modes_pos_left : list
        list of modes energies for positive angles 
    angles : numpy array
        list of angles
    zakp2, zakk : int, int
        range of the angles
    Returns:
    --------    
    modes_pos : list
        merged list of the modes positions       
    """
    modes_pos_left = np.flipud(modes_pos_left)
    modes_pos_right = np.array(modes_pos_right)    
    modes_pos = [*modes_pos_left,*modes_pos_right] 
    modes_pos = np.array(modes_pos)    
    modes_pos = np.column_stack(([np.array(angles[zakp2:zakk]), modes_pos]))
    return modes_pos
   
def plotMapWithModes(angles, enereV, data, modes_pos):
    """ 
    Plots the solutions of Schroedinger equation for Hamiltonian 5x5 together with the pseudocolor plot"


    Parameters
    ----------
    angles : numpy array
        list of angles
    enereV : numpy array
        list of energies
    data : numpy array
        data for a pseudocolor plot
    modes_pos : numpy array
        energies of the calculated modes
    """
    cmap = plt.get_cmap('Spectral_r')  # skala kolorów
    fig, ax = plt.subplots(1,1, figsize=(6, 8), facecolor='w', edgecolor='k')        
    draw_map = ax.pcolormesh(angles, enereV, data.T, cmap=cmap,  alpha=1, shading ='gouraud')
    for i in range(len(modes_pos[0])):
        ax.plot(modes_pos[:,0], modes_pos[:,i], 'r--')
    ax.axis([-28, 28, enereV[-1], enereV[0]])
    # Osie
    colorRPL = 'black'
    ax.set_xlabel('Angle ($^\circ$)',fontsize=15)
    ax.set_ylabel('Energy (eV)',fontsize=15)  
    ax.tick_params(axis='both', direction = "in", which='major', labelsize = 15,
       right = True, top = True, left = True)
    # Colorbar
    cbaxes = inset_axes(ax, width="5%", height="30%", loc = 4)
    cb = fig.colorbar(draw_map, cax=cbaxes, orientation='vertical')
    cb.set_label('Intensity (arb. units)', color=colorRPL, fontsize = 10)
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.ax.tick_params(color=colorRPL, direction = "in",left = True, right = True )
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.outline.set_edgecolor(colorRPL)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=colorRPL)
    cb.outline.set_visible(False)
    cb.solids.set_edgecolor("face")    
    # fig.savefig(save_name, bbox_inches='tight', dpi=200)
    # plt.close('all')     
   
    
def FG1_c(x, a1, b1, x1, sigma1):
    '''
    Gaussian function with constant background
    y = a1 + b1 * np.exp(-(x - x1)**2 / (2 * sigma1**2)) 
    
    
    Parameters
    ----------
    x : list 
        arguments in a list. 
    a1 : float
        constant background
    b1 : float
        amplitude of first gaussian peak
    x1 : float
        position of first gaussian maximum 
    sigma1 : float
        width of first gaussian shape        

    Returns:
    --------    
    Gauss : list
        list of intensities for given x            
    '''
    return a1 + b1 * np.exp(-(x - x1)**2 / (2 * sigma1**2))

   
def fit_FG1_c(xdata, ydata, a1_init, b1_init, x1_init, sigma1_init,
                            fitrange = None, bounds = None, 
                            pltresult=0, pltinit=0, pltsave=None):
    """
    function fitting FG1_c function to data
    -----------------------
    xdata, ydata: data to fit 
    a_init, b_init, x0_init, sigma_init: initial parameters
    bounds: gives boundary conditions if needed
    """
    
    if fitrange:
        xmin, xmin_idx = valueIndex(xdata, fitrange[0])
        xmax, xmax_idx = valueIndex(xdata, fitrange[1])
        if xmin_idx > xmax_idx: # in case when vector will be inverted and index of max value will be lower than index of min value
            xdata = xdata[xmax_idx:xmin_idx]
            ydata = ydata[xmax_idx:xmin_idx]
        else:
            xdata = xdata[xmin_idx:xmax_idx]
            ydata = ydata[xmin_idx:xmax_idx]
    popt, pcov = curve_fit(FG1_c, xdata, ydata, [a1_init, b1_init, x1_init, sigma1_init], bounds)
    return popt, pcov    

def setDataLength(data):
    """ 
    sets the data length as a global variable   
    """
    global dataLength 
    dataLength = len(data)

def solveH2x2(E1, Om1, c, d):
    """ 
    Calculates the solutions of Schroedinger equation for Hamiltonian 2x2"


    Parameters
    ----------
    E1 : float
        energy of first oscillator
    Om1 : float
        coupling strength between photon and first oscillator    
    c : float
        photon energy        
    d : float
        photon dispersion parameter      
    Returns:
    --------    
    lp1 : list
        energy of first coupled mode
    up : list
        energy of second coupled mode
    exc1 : list
        energy of first oscillator
    ph : list
        photon energy   
    """
    lp1, up = [], []
    exc1, ph = [], []
    
    for i in np.arange(-25, 25, 50/dataLength):     
        H = np.array([[c+d*i*i, Om1/2.],
                      [Om1/2., E1]])
        wart,wekt=la.eigh(H)
        lp1.append(wart[0])
        up.append(wart[1])
        exc1.append(E1)
        ph.append(c+d*i*i)

    return lp1, up, exc1, ph     

def solveH5x5(E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d):
    """ 
    Calculates the solutions of Schroedinger equation for Hamiltonian 5x5"


    Parameters
    ----------
    E1 : float
        energy of first oscillator
    E2 : float
        energy of second oscillator
    E3 : float
        energy of third oscillator
    E4 : float
        energy of fourth oscillator        
    Om1 : float
        coupling strength between photon and first oscillator 
    Om2 : float
        coupling strength between photon and second oscillator
    Om3 : float
        coupling strength between photon and third oscillator
    Om4 : float
        coupling strength between photon and fourth oscillator        
    c : float
        photon energy        
    d : float
        photon dispersion parameter      
    Returns:
    --------    
    lp1 : list
        energy of first coupled mode
    lp2 : list
        energy of second coupled mode
    lp3 : list
        energy of third coupled mode
    lp4 : list
        energy of fourth coupled mode       
    up : list
        energy of fifth coupled mode
    exc1 : list
        energy of first oscillator
    exc2 : list
        energy of second oscillator
    exc3 : list
        energy of third oscillator        
    exc4 : list
        energy of fourth oscillator
    ph : list
        photon energy   
    """
    lp1, lp2, lp3, lp4, up = [], [], [], [], []
    exc1, exc2, exc3, exc4, ph = [], [], [], [], []
    
    for i in np.arange(-25, 25, 50/dataLength):     
        H = np.array([[c+d*i*i, Om1/2., Om2/2., Om3/2., Om4/2.],
                      [Om1/2., E1, 0, 0, 0],
                      [Om2/2., 0, E2, 0, 0],
                      [Om3/2., 0, 0, E3, 0],
                      [Om4/2., 0, 0, 0, E4],
                      ])
        wart,wekt=la.eigh(H)
        lp1.append(wart[0])
        lp2.append(wart[1])
        lp3.append(wart[2])
        lp4.append(wart[3])
        up.append(wart[4])
        exc1.append(E1)
        exc2.append(E2)
        exc3.append(E3)
        exc4.append(E4)
        ph.append(c+d*i*i)
        
    return lp1, lp2, lp3, lp4, up, exc1, exc2, exc3, exc4, ph        
        

def plotInitParams2x2(data, E1, Om1, c, d, ylim = None):
    """ 
    Plots the solutions of Schroedinger equation for Hamiltonian 2x2 together with the data"


    Parameters
    ----------
    data : numpy array
        data containing modes energies
    E1 : float
        energy of first oscillator
    Om1 : float
        coupling strength between photon and first oscillator 
    c : float
        photon energy        
    d : float
        photon dispersion parameter   
    ylim : vector
        the y-axis range. If None, suitable values are automatically chosen.    
    """
    
    lp1, up, exc1, ph = solveH2x2(E1, Om1, c, d)
       
    fig, ax = plt.subplots(1,1, figsize=(8,6), facecolor='w', edgecolor='k')  
    ax.plot(data[:,1], 'r-')
    ax.plot(data[:,2], 'b-') 
    ax.plot(lp1, 'r-')
    ax.plot(up, 'b-')
    ax.plot(exc1, 'k--')
    ax.plot(ph, 'k--')
    ax.set_xlabel('Angle (px)')
    ax.set_ylabel('Energy (eV)')
    if ylim:
        ax.set_ylim(ylim)
    plt.show()
        
    
def plotInitParams5x5(data, E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d):
    """ 
    Plots the solutions of Schroedinger equation for Hamiltonian 5x5 together with the data "


    Parameters
    ----------
    data : numpy array
        data containing modes energies
    E1 : float
        energy of first oscillator
    E2 : float
        energy of second oscillator
    E3 : float
        energy of third oscillator
    E4 : float
        energy of fourth oscillator        
    Om1 : float
        coupling strength between photon and first oscillator 
    Om2 : float
        coupling strength between photon and second oscillator
    Om3 : float
        coupling strength between photon and third oscillator
    Om4 : float
        coupling strength between photon and fourth oscillator        
    c : float
        photon energy        
    d : float
        photon dispersion parameter      
    ylim : vector
        the y-axis range. If None, suitable values are automatically chosen.    
    """
    
    lp1, lp2, lp3, lp4, up, exc1, exc2, exc3, exc4, ph = solveH5x5(E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d)
        
    fig, ax = plt.subplots(1,1, figsize=(8,6), facecolor='w', edgecolor='k')  
    ax.plot(data[:,1], 'r-')
    ax.plot(data[:,2], 'm-')
    ax.plot(data[:,3], 'y-')
    ax.plot(data[:,4], 'g-')
    ax.plot(data[:,5], 'b-')
    ax.plot(lp1, 'r-')
    ax.plot(lp2, 'm-')
    ax.plot(lp3, 'y-')
    ax.plot(lp4, 'g-')
    ax.plot(up, 'b-')
    ax.plot(exc1, 'k--')
    ax.plot(exc2, 'k--')
    ax.plot(exc3, 'k--')
    ax.plot(exc4, 'k--')
    ax.plot(ph, 'k--')
    ax.set_xlabel('Angle (px)')
    ax.set_ylabel('Energy (eV)')
    plt.show()
    
    
def combinedF2x2(data, E1, Om1, c, d):
    """ 
    Gives merged solutions of Schroedinger equation for Hamiltonian 2x2."


    Parameters
    ----------
    data : numpy array
        needed to perform the fitting  
    E1 : float
        energy of first oscillator
    Om1 : float
        coupling strength between photon and first oscillator 
    c : float
        photon energy        
    d : float
        photon dispersion parameter        

    Returns:
    --------    
    modes : numpy array
        array containing two coupled modes
    """
    
    mode1 = solveH2x2(E1, Om1, c, d)[0]
    mode2 = solveH2x2(E1, Om1, c, d)[1]

    modes = np.concatenate([mode1, mode2])

    return modes    

def combinedF5x5(data, E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d):
    """ 
    Gives merged solutions of Schroedinger equation for Hamiltonian 5x5."


    Parameters
    ----------
    data : numpy array
        needed to perform the fitting  
    E1 : float
        energy of first oscillator
    E2 : float
        energy of second oscillator
    E3 : float
        energy of third oscillator
    E4 : float
        energy of fourth oscillator        
    Om1 : float
        coupling strength between photon and first oscillator 
    Om2 : float
        coupling strength between photon and second oscillator
    Om3 : float
        coupling strength between photon and third oscillator
    Om4 : float
        coupling strength between photon and fourth oscillator        
    c : float
        photon energy        
    d : float
        photon dispersion parameter        

    Returns:
    --------    
    modes : numpy array
        array containing all five coupled modes
    """
    
    mode1 = solveH5x5(E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d)[0]
    mode2 = solveH5x5(E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d)[1]
    mode3 = solveH5x5(E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d)[2]
    mode4 = solveH5x5(E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d)[3]
    mode5 = solveH5x5(E1, E2, E3, E4, Om1, Om2, Om3, Om4, c, d)[4]

    modes = np.concatenate([mode1, mode2, mode3, mode4, mode5])

    return modes

def angleVec(k0=256, NA=0.55):
    """ 
    Gives exemplary vector with angles for reciprocal space."


    Parameters
    ----------
    k0 : int 
        number of pixel with zero angle.  
    NA : float
        numerical aperture of used objective 


    Returns:
    --------    
    angles : numpy array
        arrays of angles centered at k0   
    """
    angles = []
    for point in range(0, 512):
        angles.append(180/math.pi*math.atan((point - k0)*math.tan(math.asin(NA))/390 ))            
    angles = np.array(angles) 
    return angles

def nmtoeV(vec) : 
    """
    Changes 'nm' to 'eV' or 'eV' to 'nm'


    Parameters
    ----------
    vec : list 
        list of energies in eV/nm.  


    Returns:
    --------    
    result : list
        list of energies in nm/eV
    """
    h = 6.62606957e-34;
    e = 1.602176565e-19;
    c = 299792458;
    result = h*c*1e9/e/vec
    return result

def normalizeToUnity(data):
    """
    Normalizing the data
    
    
    Parameters
    ----------
    data : np array 
        data to normalize.  

    Returns:
    --------    
    data : np array
        normalized data
    """
    
    return (data - np.min(data))/(np.max(data)-np.min((data)))


#%% HOMEWORK: find the problems with the functions below and solve them

def FindMaximaFirst2(data, start_px=0, stop_px=-1):
    """
    Finding the two local maxima in a specified range of given data
    
    
    Parameters
    ----------
    data : np array 
        data with local maxima  
    start_px, stop_px : int, int
        range containing the first maximum 

    Returns:
    --------    
    max_px, max_px2 : int, int
        indices of the found local maxima
    """
    max_px = valueIndex(data, np.max(data[start_px:stop_px]))[1]
    max2_px = valueIndex(data, np.max(data[:max_px-100]))[1]
    return max_px, max2_px

def FindMaximaNext2(data, guess0, guess1, width = 10):
    """
    Finding the two local minima in a specified ranges of given data
    
    
    Parameters
    ----------
    data : np array 
        data with local maxima  
    guess0 : int
        center of the range containing the first maximum 
    guess1 : int
        center of the range containing the second maximum 
    width : int
        width of the specified ranges

    Returns:
    --------    
    max_px, max_px2 : int, int
        indices of the found local maxima
    """
    max_px = valueIndex(data, np.max(data[guess0-width:guess0+width]))[1]
    max2_px = valueIndex(data, np.max(data[guess1-width:guess1+width]))[1]
    return max_px, max2_px

def FindMinimaFirst5(data, start_px=0, stop_px=-1):
    """
    Finding the five local minima in a specified range of given data
    
    
    Parameters
    ----------
    data : np array 
        data with local maxima  
    start_px, stop_px : ?, ? 
        ?

    Returns:
    --------    
    max_px, max_px2 : int, int
        indices of the found local maxima
    """
    min_px = valueIndex(data, np.min(data[510:stop_px]))[1]
    min2_px = valueIndex(data, np.min(data[start_px:]))[1]
    min3_px = valueIndex(data, np.min(data[start_px:min2_px-50]))[1]
    min4_px = valueIndex(data, np.min(data[start_px:min3_px-50]))[1]
    min5_px = valueIndex(data, np.min(data[start_px:min4_px-50]))[1]
    return min_px, min2_px, min3_px, min4_px, min5_px

def FindMinimaNext5(data, guess0, guess1, guess2, guess3, guess4, width = 5):
    """
    Finding the two local minima in a specified ranges of given data
    
    
    Parameters
    ----------
    data : np array 
        data with local maxima  
    guess0 : int
        center of the range containing the first maximum 
    guess1 : int
        center of the range containing the second maximum 
    guess2 : int
        center of the range containing the third maximum 
    guess3 : int
        center of the range containing the fourth maximum 
    guess4 : int
        center of the range containing the fifth maximum 
    width : int
        width of the specified ranges

    Returns:
    --------    
    max_px, max_px2 : int, int
        indices of the found local maxima
    """
    min_px = valueIndex(data, np.min(data[guess0-width:guess0+width]))[1]
    min2_px = valueIndex(data, np.min(data[guess1-width:guess1+width]))[1]
    min3_px = valueIndex(data, np.min(data[guess2-width:guess2+width]))[1]
    min4_px = valueIndex(data, np.min(data[guess3-width:guess3+width]))[1]
    min5_px = valueIndex(data, np.min(data[guess4-width:guess4+width]))[1]
    return min_px, min2_px, min3_px, min4_px, min5_px
