from brian2 import *
import numpy as np

# =============================================================================
#  Calculate stimulus currents for each compartment and timestep
# =============================================================================
def get_stimulus_current(model,
                         dt = 5*ms,
                         pulse_form = "mono",
                         stimulation_type = "extern",
                         stimulated_compartment = None,
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
    model : module
        Module that contains all parameters for a certain model
    dt : time
        Sampling timestep.
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    stimulated_compartment : integer
        Index of compartment which is stimulated.
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
                
    Returns
    -------
    current matrix
        Gives back a matrix of stumulus currents with one row for each compartment
        and one column for each timestep.
    runtime
        Gives back the duration of the simulation        
    """
    
    ##### use second node for stimulation if no index is given
    if stimulated_compartment is None:
        stimulated_compartment = np.where(model.structure == 2)[0][2]
    
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
    I_elec = np.zeros(nof_timesteps)*mA
    
    ##### create current vector for one pulse
    if pulse_form == "mono":
        timesteps_pulse = int(duration_mono/dt)
        I_pulse = np.zeros(timesteps_pulse)*mA
        I_pulse[:] = amp_mono
    else:
        timesteps_pulse = int(sum(durations_bi)/dt)
        I_pulse = np.zeros(timesteps_pulse)*mA
        end_index_first_phase = round(durations_bi[0]/dt)
        start_index_second_phase = round(end_index_first_phase + durations_bi[1]/dt)
        I_pulse[:end_index_first_phase] = amps_bi[0]
        I_pulse[start_index_second_phase:] = amps_bi[1]
    
    ##### Fill stimulus current vector
    if nof_pulses == 1:
        I_elec[round(time_before/dt):round(time_before/dt)+timesteps_pulse] = I_pulse
    elif nof_pulses > 1:
        I_inter_pulse_gap = np.zeros(round(inter_pulse_gap/dt))*amp
        I_pulse_train = np.append(np.tile(np.append(I_pulse, I_inter_pulse_gap), nof_pulses-1),I_pulse)*amp
        I_elec[round(time_before/dt):round(time_before/dt)+len(I_pulse_train)] = I_pulse_train
        
    ##### number of compartments
    nof_comps = len(model.compartment_lengths)
    
    ##### external stimulation
    if stimulation_type == "extern":
        
        # calculate electrode distance for all compartments (center)
        distance_x = np.zeros(nof_comps)
        
        if stimulated_compartment > 0:
            # loop over all compartments before the stimulated one
            for ii in range(stimulated_compartment-1,-1,-1):
                distance_x[ii] = 0.5* model.compartment_lengths[stimulated_compartment] + \
                np.sum(model.compartment_lengths[stimulated_compartment-1:ii:-1]) + 0.5* model.compartment_lengths[ii]
        
        if stimulated_compartment < nof_comps:
            # loop over all compartments after the stimulated one
            for ii in range(stimulated_compartment+1,nof_comps,1):
                distance_x[ii] = 0.5* model.compartment_lengths[stimulated_compartment] + \
                np.sum(model.compartment_lengths[stimulated_compartment+1:ii:1]) + 0.5* model.compartment_lengths[ii]
                
        distance = np.sqrt((distance_x*meter)**2 + model.electrode_distance**2)
        
        # calculate axoplasmatic resistances (always for the two halfs of neightbouring compartments)
        R_a = np.zeros(nof_comps)*ohm
        
        if stimulated_compartment > 0:
            for i in range(0,stimulated_compartment):
                R_a[i] = 0.5* model.R_a[i] + 0.5* model.R_a[i+1]
                
        R_a[stimulated_compartment] = model.R_a[stimulated_compartment]
        
        if stimulated_compartment < nof_comps:
            for i in range(stimulated_compartment+1,nof_comps):
                R_a[i] = 0.5* model.R_a[i-1] + 0.5* model.R_a[i]
                
        # Calculate activation functions
        V_ext = np.zeros((nof_comps,nof_timesteps))*mV
        E_ext = np.zeros((nof_comps,nof_timesteps))*mV
        A_ext = np.zeros((nof_comps,nof_timesteps))*mV
        
        for ii in range(0,nof_timesteps):
            V_ext[:,ii] = (model.rho_out*I_elec[ii]) / (4*np.pi*distance)
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
        np.random.seed()
        I_stim = I_stim + np.transpose(np.transpose(np.random.normal(0, 1, np.shape(I_stim)))*model.k_noise*model.noise_term)
        
    return I_stim, runtime
