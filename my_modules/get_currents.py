from brian2 import *
import numpy as np

# =============================================================================
#  get_currents_for_external_stimulation
# =============================================================================
def get_currents_for_external_stimulation(compartment_lengths,
                                          nof_timesteps,
                                          stimulus_current_vector,
                                          stimulated_compartment,
                                          electrode_distance,
                                          rho_out,
                                          axoplasmatic_resistances):
    """This function calculates the currents for each compartment and timestep
    due to an external stimulation. At first, the activation function is
    calculated which is the numerical second derivative of the external potentials.
    Dividing this by the axoplasmatic resistance gives the wanted currents.

    Parameters
    ----------
    compartment_lengths : measure of length
        Lengths of compartments.
    nof_timesteps : integer
        Number of timesteps in whole simulation.
    stimulus_current_vector : current
        Current amplitudes at electrode for each time step.
    stimulated_compartment : integer
        Index of compartment of electrode location.
    electrode_distance : measure of length
        Shortest distance between electrode and axon.
    rho_out : resistance*measure of length
        Extracellular resistivity.
    axoplasmatic_resistances : resistance
        Axoplasmatic resistances of each compartment.
        
    Returns
    -------
    current matrix
        Gives back a matrix of currents for each compartment and timestep

    """
    
    ##### Number of compartments
    nof_comps = len(compartment_lengths)        
        
    ##### calculate electrode distance for all compartments (center)
    distance_x = np.zeros((1,nof_comps))
    
    if stimulated_compartment > 0:
        for ii in range(stimulated_compartment-1,-1,-1):
            distance_x[0,ii] = 0.5* compartment_lengths[stimulated_compartment] + np.sum(compartment_lengths[stimulated_compartment-1:ii:-1]) + 0.5* compartment_lengths[ii]
    
    if stimulated_compartment < nof_comps:
        for ii in range(stimulated_compartment+1,nof_comps,1):
            distance_x[0,ii] = 0.5* compartment_lengths[stimulated_compartment] + np.sum(compartment_lengths[stimulated_compartment+1:ii:1]) + 0.5* compartment_lengths[ii]
            
    distance = np.sqrt((distance_x*meter)**2 + electrode_distance**2)
    
    ##### calculate axoplasmatic resistances (always for the two halfs of neightbouring compartments)
    R_a = np.zeros((1,nof_comps))*ohm
    
    if stimulated_compartment > 0:
        for i in range(0,stimulated_compartment):
            R_a[0,i] = 0.5* axoplasmatic_resistances[i] + 0.5* axoplasmatic_resistances[i+1]
            
    R_a[0,stimulated_compartment] = axoplasmatic_resistances[stimulated_compartment]
    
    if stimulated_compartment < nof_comps:
        for i in range(stimulated_compartment+1,nof_comps):
            R_a[0,i] = 0.5* axoplasmatic_resistances[i-1] + 0.5* axoplasmatic_resistances[i]
            
    ##### Calculate activation functions
    V_ext = np.zeros((nof_comps,nof_timesteps))*mV
    E_ext = np.zeros((nof_comps,nof_timesteps))*mV
    A_ext = np.zeros((nof_comps,nof_timesteps))*mV
    
    for ii in range(0,nof_timesteps):
        V_ext[:,ii] = (rho_out*stimulus_current_vector[0,ii]) / (4*np.pi*distance)
        E_ext[0:-1,ii] = -np.diff(V_ext[:,ii])
        A_ext[1:-1,ii] = -np.diff(E_ext[0:-1,ii])
    
    ##### Calculate currents
    I_ext = A_ext/np.transpose(np.tile(R_a, (nof_timesteps,1)))
    
    return I_ext


# =============================================================================
#  get_currents_for_internal_stimulation
# =============================================================================
def get_currents_for_internal_stimulation(nof_compartments,
                                          nof_timesteps,
                                          stimulus_current_vector,
                                          stimulated_compartments):
    """This function calculates the currents for each compartment and timestep
    due to an internal stimulation.

    Parameters
    ----------
    nof_compartments : integer
        Number of compartments.
    nof_timesteps : integer
        Number of timesteps in whole simulation.
    stimulus_current_vector : current
        Current amplitudes for each time step (row vector with lenth nof_timesteps).
    stimulated_compartment : integer
        Index of compartment of electrode location.
        
    Returns
    -------
    current matrix
        Gives back a matrix of currents for each compartment and timestep

    """
    
    ##### initialize current matrix
    I_int = np.zeros((nof_compartments,nof_timesteps))*mA
    
    ##### calculate currents (loop over stimulated compartments)
    for ii in range(0,len(stimulated_compartments)):
        I_int[stimulated_compartments[ii],:] = stimulus_current_vector
    
    return I_int
