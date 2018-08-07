from brian2 import *
import numpy as np

# =============================================================================
#  Calculate stimulus currents for each compartment and timestep
# =============================================================================
def get_stimulus_current(model,
                         dt = 5*ms,
                         stimulation_type = "external",
                         pulse_form = "mono",
                         stimulated_compartment = 4,
                         time_before = 0*ms,
                         time_after = 1*ms,
                         nof_pulses = 1,
                         add_noise = False,
                         amp_mono = 1.5*uA,
                         duration_mono = 200*us,
                         amps_bi = [-2,2]*uA,
                         durations_bi = [100,0,100]*us,
                         inter_pulse_gap = 1*ms):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    dt : time
        Lenght of one time step.
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    time_before : time
        Time until (first) pulse starts.
    time_after : time
        Time which is still simulated after the end of the last pulse
    nof_pulses : integer
        Number of pulses.
    add_noise : boolean
        Defines, if noise is added
    amp_mono : current
        Amplitude of current stimulus in case it is monophasic.
    duration_mono : time
        Duration of stimulus in case it is monophasic.
    amps_bi : curent vector
        Vector of length two, which includes the amplitudes of the first and the second phase, resepectively.
    durations_bi : time vector
        Vector of length three, which includes the durations of the first phase, the interphase gap and the second phase, resepectively.
    inter_pulse_gap : time
        Time interval between pulses for nof_pulses > 1.
    compartment_lengths : measure of length vector
        Contains the lengths of all compartments.
    stimulated_compartment : integer
        Index of compartment which is stimulated.
    electrode_distance : measure of length
        Shortest distance between electrode and axon (for external stimulation).
    rho_out : resistance*measure of length
        Extracellular resistivity (for external stimulation) .
    axoplasmatic_resistances : resistance vector
        Axoplasmatic resistances of each compartment (for external stimulation).
    k_noise : amp/[noise_term]
        Defines the noise strength if add_noise = True.
    noise_term : amp/[k_noise]
        Is multiplied with k_noise.
                
    Returns
    -------
    current matrix
        Gives back a vector of currents for each timestep
    runtime
        Gives back the duration of the simulation
    """
    
    ##### calculate runtime
    if pulse_form == "mono":
        runtime = time_before + nof_pulses*duration_mono + (nof_pulses-1)*inter_pulse_gap + time_after
    elif pulse_form == "bi":
        runtime = time_before + nof_pulses*sum(durations_bi) + (nof_pulses-1)*inter_pulse_gap + time_after
    else:
        print("Just 'mono' and 'bi' are allowed for pulse_form")
        return
    
    ##### calculate number of timesteps
    nof_timesteps = int(np.ceil(runtime/dt))
    
    ##### initialize current stimulus vector
    I_elec = np.zeros((1,nof_timesteps))*mA
    
    ##### create current vector for one pulse
    if pulse_form == "mono":
        timesteps_pulse = int(duration_mono/dt)
        I_pulse = np.zeros((1,timesteps_pulse))*mA
        I_pulse[:] = amp_mono
    else:
        timesteps_pulse = int(sum(durations_bi)/dt)
        I_pulse = np.zeros((1,timesteps_pulse))*mA
        end_index_first_phase = round(durations_bi[0]/dt)
        start_index_second_phase = round(end_index_first_phase + durations_bi[1]/dt)
        I_pulse[0,:end_index_first_phase] = amps_bi[0]
        I_pulse[0,start_index_second_phase:] = amps_bi[1]
    
    ##### Fill stimulus current vector
    if nof_pulses == 1:
        I_elec[0,round(time_before/dt):round(time_before/dt)+timesteps_pulse] = I_pulse
    elif nof_pulses > 1:
        I_inter_pulse_gap = np.zeros((1,round(inter_pulse_gap/dt)))*amp
        I_pulse_train = np.append(np.tile(np.append(I_pulse, I_inter_pulse_gap), nof_pulses-1),I_pulse)*amp
        I_elec[0,round(time_before/dt):round(time_before/dt)+len(I_pulse_train)] = I_pulse_train
        
    ##### number of compartments
    nof_comps = len(model.compartment_lengths)
    
    ##### external stimulation
    if stimulation_type == "extern":
        
        # calculate electrode distance for all compartments (center)
        distance_x = np.zeros((1,nof_comps))
        
        if stimulated_compartment > 0:
            for ii in range(stimulated_compartment-1,-1,-1):
                distance_x[0,ii] = 0.5* model.compartment_lengths[stimulated_compartment] + np.sum(model.compartment_lengths[stimulated_compartment-1:ii:-1]) + 0.5* model.compartment_lengths[ii]
        
        if stimulated_compartment < nof_comps:
            for ii in range(stimulated_compartment+1,nof_comps,1):
                distance_x[0,ii] = 0.5* model.compartment_lengths[stimulated_compartment] + np.sum(model.compartment_lengths[stimulated_compartment+1:ii:1]) + 0.5* model.compartment_lengths[ii]
                
        distance = np.sqrt((distance_x*meter)**2 + model.electrode_distance**2)
        
        # calculate axoplasmatic resistances (always for the two halfs of neightbouring compartments)
        R_a = np.zeros((1,nof_comps))*ohm
        
        if stimulated_compartment > 0:
            for i in range(0,stimulated_compartment):
                R_a[0,i] = 0.5* model.R_a[i] + 0.5* model.R_a[i+1]
                
        R_a[0,stimulated_compartment] = model.R_a[stimulated_compartment]
        
        if stimulated_compartment < nof_comps:
            for i in range(stimulated_compartment+1,nof_comps):
                R_a[0,i] = 0.5* model.R_a[i-1] + 0.5* model.R_a[i]
                
        # Calculate activation functions
        V_ext = np.zeros((nof_comps,nof_timesteps))*mV
        E_ext = np.zeros((nof_comps,nof_timesteps))*mV
        A_ext = np.zeros((nof_comps,nof_timesteps))*mV
        
        for ii in range(0,nof_timesteps):
            V_ext[:,ii] = (model.rho_out*I_elec[0,ii]) / (4*np.pi*distance)
            E_ext[0:-1,ii] = -np.diff(V_ext[:,ii])
            A_ext[1:-1,ii] = -np.diff(E_ext[0:-1,ii])
        
        # Calculate currents
        I_stim = A_ext/np.transpose(np.tile(R_a, (nof_timesteps,1)))
        
    ##### internal stimulation
    elif stimulation_type == "intern":
        
        # initialize current matrix
        I_stim = np.zeros((nof_comps,nof_timesteps))*mA
        
        # fill current matrix
        I_stim[stimulated_compartment,:] = I_elec

    ##### wrong entry
    else:
        print("Just 'extern' and 'intern' are allowed for stimulation_type")
        return
    
    ##### add noise
    if add_noise:
        I_stim = I_stim + np.transpose(np.transpose(np.random.normal(0, 1, np.shape(I_stim)))*model.k_noise*model.noise_term)
        
    return I_stim, runtime
