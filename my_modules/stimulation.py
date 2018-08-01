from brian2 import *
import numpy as np

# =============================================================================
#  Single monophasic pulse stimulus
# =============================================================================
def single_monophasic_pulse_stimulus(nof_timesteps,
                                     dt,
                                     current_amplitude,
                                     time_before_pulse,
                                     stimulus_duration):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    nof_timesteps : integer
        Number of timesteps of whole simulation.
    dt : time
        Lenght of one time step.
    current_amplitude : current
        Amplitude of current stimulus.
    time_before_pulse : time
        Time until pulse starts.
    stimulus_duration : time
        Duration of stimulus.
                
    Returns
    -------
    current vector
        Gives back a vector of currents for each timestep
    """

    ##### initialize current stimulus matrix
    I_stim = np.zeros((1,nof_timesteps))*mA
    
    ##### calculate current for each timestep
    I_stim[0, round(time_before_pulse/dt):round(round(time_before_pulse/dt) + stimulus_duration/dt)+1] = current_amplitude
    
    return I_stim

# =============================================================================
#  Single biphasic pulse stimulus
# =============================================================================
def single_biphasic_pulse_stimulus(nof_timesteps,
                                   dt,
                                   current_amplitude_first_phase,
                                   current_amplitude_second_phase,
                                   time_before_pulse,
                                   duration_first_phase,
                                   duration_second_phase,
                                   duration_interphase_gap):
    """This function calculates the stimulus current at the current source for
    a single biphasic pulse stimulus at each point of time

    Parameters
    ----------
    nof_timesteps : integer
        Number of timesteps of whole simulation.
    dt : time
        Lenght of one time step.
    current_amplitude_first_phase : current
        Amplitude of first phase.
    current_amplitude_second_phase : current
        Amplitude of second phase.
    time_before_pulse : time
        Time until pulse starts.
    duration_first_phase : time
        Duration of first phase.
    duration_second_phase : time
        Duration of second phase.
    duration_interphase_gap : time
        Duration of interphase gap.
                
    Returns
    -------
    current vector
        Gives back a vector of currents for each timestep
    """

    ##### initialize current stimulus matrix
    I_stim = np.zeros((1,nof_timesteps))*mA
    
    ##### time indexes
    start_index_first_phase = round(time_before_pulse/dt)
    end_index_first_phase = round(start_index_first_phase + duration_first_phase/dt)+1
    start_index_second_phase = round(end_index_first_phase + duration_interphase_gap/dt)
    end_index_second_phase = round(start_index_second_phase + duration_second_phase/dt)+1
    
    ##### calculate current for each timestep
    I_stim[0, start_index_first_phase:end_index_first_phase] = current_amplitude_first_phase
    I_stim[0, start_index_second_phase:end_index_second_phase] = current_amplitude_second_phase

    return I_stim

# =============================================================================
#  Pulse train
# =============================================================================
def pulse_train_stimulus(nof_timesteps,
                         dt,
                         current_vector,
                         time_before_pulse_train,
                         nof_pulses,
                         inter_pulse_gap):
    """This function calculates the stimulus current at the current source for
    a pulse train at each point of time

    Parameters
    ----------
    nof_timesteps : integer
        Number of timesteps of whole simulation.
    dt : time
        Lenght of one time step.
    current_vector : current
        Current vector of one pulse.
    time_before_pulse_train : time
        Time until pulse train starts.
    nof_pulses : integer
        Number of pulses.
    inter_pulse_gap : time
        Time period between two pulses.
        
    Returns
    -------
    current vector
        Gives back a vector of currents for each timestep
    """

    ##### generate inter_pulse_gap vector
    I_inter_pulse_gap = np.zeros((1,round(inter_pulse_gap/dt)))*amp
    
    ##### cut leading and trailing zeros of current_vector
    current_vector = np.trim_zeros(current_vector[0])
    
    ##### generate current vector with: time_before_pulse - current_vector - inter_pulse_gap - current_vector - inter_pulse_gap ....
    I_pulse_train = np.append(np.tile(np.append(current_vector, I_inter_pulse_gap), nof_pulses-1),current_vector)
    
    ##### initialize current stimulus matrix    
    I_stim = np.zeros((1,nof_timesteps))*amp
    
    ##### fill whole current vector
    I_stim[0, 0:round(time_before_pulse_train/dt)+1] = 0*amp
    I_stim[0, round(time_before_pulse_train/dt)+1:len(I_pulse_train)+1] = I_pulse_train*amp
    
    return I_stim