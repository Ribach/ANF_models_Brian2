# =============================================================================
# This script provides some functions that are useful for certain calculations
# and transformations.
# =============================================================================
from brian2 import *
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

# =============================================================================
#  Get soma diameters to approximate spherical
# =============================================================================
def get_soma_diameters(nof_segments,
                       dendrite_diameter,
                       soma_diameter,
                       axon_diameter):
    """This function calculates the diameters for each compartment, for a segmentation
    of the soma into multiple compartments.

    Parameters
    ----------
    nof_segments : integer
        Number of segments into which soma will be devided.
    dendrite_diameter : measure of lengths
        Diameter of the dendrite.
    soma_diameter : measure of lengths
        Maximum diameter of the soma.
    axon_diameter : measure of lengths
        Diameter of the axon.
                
    Returns
    -------
    diameter vector
        Gives back a vector of start and end diameters for each segment of soma
        i.e. a vector of length nof_segments+1
    """
    
    ##### length of compartment
    soma_comp_len = soma_diameter/(nof_segments)
    
    ##### initialize diameter array
    soma_comp_diameters = np.zeros((1,nof_segments+1))*meter
    
    if nof_segments == 1:
        soma_comp_diameters = [soma_diameter, soma_diameter]
    elif nof_segments%2==0:
        ##### index of one central diameter
        center_of_soma = int(nof_segments/2)
        
        ##### diameters left part
        soma_comp_diameters[0,0:center_of_soma] = [2*np.sqrt(((soma_diameter/2)**2)-(soma_comp_len*i)**2) for i in range(center_of_soma,0,-1)]
        # no diameters smaller than dendrite diameter
        soma_comp_diameters[0,0:center_of_soma][np.where(soma_comp_diameters[0,0:center_of_soma] < dendrite_diameter)] = dendrite_diameter
        
        ##### diameter center
        soma_comp_diameters[0,center_of_soma] = soma_diameter
        
        ##### diameter right part
        soma_comp_diameters[0,center_of_soma+1:] = [2*np.sqrt(((soma_diameter/2)**2)-(soma_comp_len*i)**2) for i in range(1,center_of_soma+1)]
        # no diameters smaller than axon diameter
        soma_comp_diameters[0,center_of_soma:][np.where(soma_comp_diameters[0,center_of_soma:] < axon_diameter)] = axon_diameter
    else:
        ##### indexes of the two central diameters
        center_of_soma = [int(np.floor((nof_segments)/2)), int(np.ceil((nof_segments)/2))]
    
        ##### diameters left part
        soma_comp_diameters[0,0:center_of_soma[0]] = [2*np.sqrt(((soma_diameter/2)**2)-(soma_comp_len*(i+0.5))**2) for i in range(center_of_soma[0],0,-1)]
        # no diameters smaller than dendrite diameter
        soma_comp_diameters[0,0:center_of_soma[0]][np.where(soma_comp_diameters[0,0:center_of_soma[0]] < dendrite_diameter)] = dendrite_diameter
        
        ##### diameter center
        soma_comp_diameters[0,center_of_soma] = soma_diameter
        
        ##### diameter right part
        soma_comp_diameters[0,center_of_soma[1]+1:] = [2*np.sqrt(((soma_diameter/2)**2)-(soma_comp_len*(i+0.5))**2) for i in range(1,center_of_soma[1])]
        # no diameters smaller than axon diameter
        soma_comp_diameters[0,center_of_soma[1]+1:][np.where(soma_comp_diameters[0,center_of_soma[1]+1:] < axon_diameter)] = axon_diameter
    
    return soma_comp_diameters

# =============================================================================
#  Split pandas datframe column with lists to multiple rows
# =============================================================================
def explode(df, lst_cols, fill_value=''):
    """This function reshapes a pandas dataframe that contains a column with lists,
    that the resulting dataframe has one row for each list element.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe to be reshaped.
    lst_cols : list of strings 
        All strings in the list have to be column names of df. If they include lists,
        the dataframe will be reshaped in the mentioned way.
    fill_value : string
        This argument defines, how Na values are shown/repaced in the resulting dataframe
                
    Returns
    -------
    pandas dataframe
        Gives back the reshaped pandas dataframe
    """

    ##### make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    ##### all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    ##### calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        ##### all lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        ##### at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

# =============================================================================
#  Break down 3D coordinates to 1D
# =============================================================================
def coordinates_to_1D(x, y, z):
    """This function calculates the distances of given points of the
    threedimensional space to the first point.

    Parameters
    ----------
    x : float vector
        Coordinates in x direction.
    y : float vector
        Coordinates in y direction.
    z : float vector
        Coordinates in z direction.
                
    Returns
    -------
    float vector
        Gives back the distances between the points.
    """
    
    ##### initialize distances vector
    d = np.zeros_like(x)
    
    ##### calculate distances
    for ii in range(1,len(x)):
        d[ii] = d[ii-1] + np.sqrt((x[ii] - x[ii-1])**2 + (y[ii] - y[ii-1])**2 + (z[ii] - z[ii-1])**2)
    
    return d

# =============================================================================
#  Interpolate potentials
# =============================================================================
def interpolate_potentials(potentials,
                           pot_distances,
                           comp_distances,
                           comp_lenghts,
                           method = "linear"):
    """This function interpolates values of a given potential distribution.

    Parameters
    ----------
    potentials : float vector
        Potential values at pot_distances.
    pot_distances : float vector
        Distances of given potentials to the location of the first potential.
    comp_distances : float vector
        Distances of model compartment middle points to the peripheral terminal.
    method : string
        Method that is used for interpolation. Spline and linear are possible.
                
    Returns
    -------
    float vector
        Gives back a vector of potentials at the compartment middle points
    """
    
    ##### linear interpolation
    if method == "linear":
            
        # initialize vector for interpolated potentials at compartments
        comp_potentials = np.zeros_like(comp_distances)
        
        # fill comp_potentials with last potential
        comp_potentials[:] = potentials[-1]
        
        # initialize variable that saves the last index of the actual compartment
        last_index = 0
        
        # loop over compartments
        for ii in range(len(comp_distances[comp_distances <= max(pot_distances)])):
            
            # get indexes of potentials in range of compartment
            pot_indexes = np.where(np.logical_and(pot_distances >= comp_distances[ii]-0.5*comp_lenghts[ii],
                                                        pot_distances <= comp_distances[ii]+0.5*comp_lenghts[ii]))[0]
            
            # get number of potentials in range of compartment
            nof_pots = len(pot_indexes)
            
            # update last_index
            if nof_pots > 0:
                last_index = max(pot_indexes)
            
            # distinguish between zero, one ore more potentials within the compartment range
            if nof_pots == 0:
                
                comp_potentials[ii] = np.interp(x = comp_distances[ii],
                               xp = [pot_distances[last_index], pot_distances[last_index+1]],
                               fp = [potentials[last_index], potentials[last_index+1]])
                
            elif nof_pots == 1:
                comp_potentials[ii] = potentials[last_index]
            
            else:
                comp_potentials[ii] = np.mean(potentials[pot_indexes])

    ##### spline interpolation
    if method == "spline":
        knot_points = interpolate.splrep(pot_distances, potentials, s=0)
        comp_potentials = interpolate.splev(comp_distances, knot_points, der=0)

    return comp_potentials

# =============================================================================
#  Get shifted colormap
# =============================================================================
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''

    
#    def display_cmap(cmap):
#        plt.imshow(np.linspace(0, 100, 256)[None, :],  aspect=25,    interpolation='nearest', cmap=cmap) 
#        plt.axis('off')

#    display_cmap(cmap)
    
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
