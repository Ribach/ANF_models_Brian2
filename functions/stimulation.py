from brian2 import *
import numpy as np

# =============================================================================
#  Calculate stimulus currents for each compartment and timestep
# =============================================================================
def get_stimulus_current(model,
                         dt = 5*us,
                         pulse_form = "mono",
                         stimulation_type = "extern",
                         stimulated_compartment = None,
                         time_before = 0*ms,
                         time_after = 1*ms,
                         nof_pulses = 1,
                         add_noise = False,
                         amp_mono = 200*uA,
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
                
        # Calculate activation function (=stimulus current)
        V_ext = np.zeros((nof_comps,nof_timesteps))*volt
        I_stim = np.zeros((nof_comps,nof_timesteps))*amp
        
        for ii in range(nof_comps):
            V_ext[ii,:] = (model.rho_out*I_elec) / (4*np.pi*distance[ii])

        for ii in range(nof_comps):
            if ii == 0:
                I_stim[0,:] = (V_ext[1,:] - V_ext[0,:])/(0.5*model.R_a[ii+1] + 0.5*model.R_a[ii])
            elif ii == nof_comps-1:
                I_stim[-1,:] = (V_ext[-2,:] - V_ext[-1,:])/(0.5*model.R_a[ii-1] + 0.5*model.R_a[ii])
            else:
                I_stim[ii,:] = (V_ext[ii-1,:] - V_ext[ii,:])/(0.5*model.R_a[ii-1] + 0.5*model.R_a[ii]) + (V_ext[ii+1,:] - V_ext[ii,:])/(0.5*model.R_a[ii+1] + 0.5*model.R_a[ii])

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

# =============================================================================
#  Calculate stimulus currents for given potential distribution
# =============================================================================
def get_stim_current_for_given_potentials(model,
                                          dt,
                                          V,
                                          pulse_form = "bi",
                                          time_before = 1*ms,
                                          time_after = 2*ms,
                                          nof_pulses = 20,
                                          add_noise = True,
                                          duration_mono = 50*us,
                                          durations_bi = [40,20,40]*us,
                                          inter_pulse_gap = 1*ms):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    model : module
        Module that contains all parameters for a certain model
    dt : time
        Sampling timestep.
    V : voltage vector
        One voltage per compartment has to be provided. It defines the resulting
        potentials at the compartments after a stimulation.
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
    duration_mono : time
        Duration of stimulus in case it is monophasic.
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
    
    ##### calculate runtime
    if pulse_form == "mono":
        runtime = time_before + nof_pulses*duration_mono + (nof_pulses-1)*inter_pulse_gap + time_after
    elif pulse_form == "bi":
        runtime = time_before + nof_pulses*sum(durations_bi) + (nof_pulses-1)*inter_pulse_gap + time_after
    else:
        print("Just 'mono' and 'bi' are allowed for pulse_form")
        #return
    
    ##### number of compartments
    nof_comps = len(model.compartment_lengths)
    
    ##### calculate number of timesteps
    nof_timesteps = int(np.ceil(runtime/dt))
    
    ##### initialize potential matrix (one row per compartment and one column per timestep)
    V_ext = np.zeros((nof_comps, nof_timesteps))*volt
    
    ##### create potential matrix for one pulse
    if pulse_form == "mono":
        timesteps_pulse = int(duration_mono/dt)
        V_pulse = np.zeros((nof_comps, timesteps_pulse))*volt
        V_pulse[:] = np.tile(-V, (timesteps_pulse,1)).transpose()
    else:
        timesteps_pulse = int(sum(durations_bi)/dt)
        V_pulse = np.zeros((nof_comps, timesteps_pulse))*volt
        end_index_first_phase = round(durations_bi[0]/dt)
        start_index_second_phase = round(end_index_first_phase + durations_bi[1]/dt)
        V_pulse[:,:end_index_first_phase] = np.tile(-V, (end_index_first_phase,1)).transpose()
        V_pulse[:,start_index_second_phase:] = np.tile(V, (timesteps_pulse-start_index_second_phase,1)).transpose()
    
    ##### Fill stimulus current vector
    if nof_pulses == 1:
        V_ext[:,round(time_before/dt):round(time_before/dt)+timesteps_pulse] = V_pulse
    elif nof_pulses > 1:
        V_inter_pulse_gap = np.zeros((nof_comps,round(inter_pulse_gap/dt)))*volt
        V_pulse_train = np.concatenate((np.tile(np.concatenate((V_pulse, V_inter_pulse_gap), axis = 1), nof_pulses-1),V_pulse), axis = 1)*volt
        V_ext[:,round(time_before/dt):round(time_before/dt) + np.shape(V_pulse_train)[1]] = V_pulse_train
            
    ##### Calculate activation function (=stimulus current)
    I_stim = np.zeros((nof_comps,nof_timesteps))*amp
    
    for ii in range(nof_comps):
        if ii == 0:
            I_stim[0,:] = (V_ext[1,:] - V_ext[0,:])/(0.5*model.R_a[ii+1] + 0.5*model.R_a[ii])
        elif ii == nof_comps-1:
            I_stim[-1,:] = (V_ext[-2,:] - V_ext[-1,:])/(0.5*model.R_a[ii-1] + 0.5*model.R_a[ii])
        else:
            I_stim[ii,:] = (V_ext[ii-1,:] - V_ext[ii,:])/(0.5*model.R_a[ii-1] + 0.5*model.R_a[ii]) + (V_ext[ii+1,:] - V_ext[ii,:])/(0.5*model.R_a[ii+1] + 0.5*model.R_a[ii])

    
    ##### add noise
    if add_noise:
        np.random.seed()
        I_stim = I_stim + np.transpose(np.transpose(np.random.normal(0, 1, np.shape(I_stim)))*model.k_noise*model.noise_term)
        
    return I_stim, runtime

# =============================================================================
#  Calculate stimulus currents for sinusodial stimulation
# =============================================================================
def get_stimulus_current_for_sinus(model,
                                   dt = 5*us,
                                   stimulation_type = "extern",
                                   stimulated_compartment = None,
                                   time_before = 0*ms,
                                   time_after = 1*ms,
                                   stim_length = 5*ms,
                                   add_noise = False,
                                   amplitude = 200*uA,
                                   frequency = 1000*Hz):
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
    runtime = time_before + stim_length + time_after
    
    ##### calculate number of timesteps
    nof_timesteps = int(np.ceil(runtime/dt))
    
    ##### initialize current stimulus vector
    I_elec = np.zeros(nof_timesteps)*mA
    
    ##### create current vector for stimulus
    timesteps_pulse = int(stim_length/dt)
    time_vector = np.arange(timesteps_pulse)*dt
    I_pulse = amplitude* np.sin(2*np.pi*frequency*time_vector)
    
    ##### Fill stimulus current vector
    I_elec[round(time_before/dt):round(time_before/dt)+timesteps_pulse] = I_pulse
        
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
                
        # Calculate activation function (=stimulus current)
        V_ext = np.zeros((nof_comps,nof_timesteps))*volt
        I_stim = np.zeros((nof_comps,nof_timesteps))*amp
        
        for ii in range(nof_comps):
            V_ext[ii,:] = (model.rho_out*I_elec) / (4*np.pi*distance[ii])

        for ii in range(nof_comps):
            if ii == 0:
                I_stim[0,:] = (V_ext[1,:] - V_ext[0,:])/(0.5*model.R_a[ii+1] + 0.5*model.R_a[ii])
            elif ii == nof_comps-1:
                I_stim[-1,:] = (V_ext[-2,:] - V_ext[-1,:])/(0.5*model.R_a[ii-1] + 0.5*model.R_a[ii])
            else:
                I_stim[ii,:] = (V_ext[ii-1,:] - V_ext[ii,:])/(0.5*model.R_a[ii-1] + 0.5*model.R_a[ii]) + (V_ext[ii+1,:] - V_ext[ii,:])/(0.5*model.R_a[ii+1] + 0.5*model.R_a[ii])

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
