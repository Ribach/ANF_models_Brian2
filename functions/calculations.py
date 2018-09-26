from brian2 import *
import numpy as np
import pandas as pd

# =============================================================================
#  Get soma diameters to approximate spherical
# =============================================================================
def get_soma_diameters(nof_segments,
                       dendrite_diameter,
                       soma_diameter,
                       axon_diameter):
    """This function calculates the stimulus current at the current source for
    a single biphasic pulse stimulus at each point of time

    Parameters
    ----------
    nof_segments : integer
        Number of segments into which soma will be devided.
    dendrite_diameter : measure of lengths
        Diameter of the dendrite.
    soma_diameter : measure of lengths
        Diameter of the soma.
    axon_diameter : measure of lengths
        Diameter of the axon.
                
    Returns
    -------
    current vector
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
        ##### no diameters smaller than dendrite diameter
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
        ##### no diameters smaller than dendrite diameter
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
    """This function calculates the stimulus current at the current source for
    a single biphasic pulse stimulus at each point of time

    Parameters
    ----------
    name : integer
        Number of segments into which soma will be devided.
    values : measure of lengths
        Diameter of the dendrite.
                
    Returns
    -------
    display name
        Gives back a vector of start and end diameters for each segment of soma
        i.e. a vector of length nof_segments+1
    """

    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
    
    
    
    
    
        