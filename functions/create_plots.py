from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  Voltage course lines
# =============================================================================
def voltage_course_lines(plot_name,
                         time_vector,
                         voltage_matrix,
                         comps_to_plot,
                         distance_comps_middle,
                         length_neuron,
                         V_res):
    """This function plots the membrane potential of all compartments over time
    as voltage course lines spaced according the real compartment distances

    Parameters
    ----------
    time_vector : integer
        Number of timesteps of whole simulation.
    voltage_matrix : time
        Lenght of one time step.
    distance_comps_middle : current
        Amplitude of current stimulus.
    length_neuron : time
        Time until pulse starts.
    V_res : time
        Duration of stimulus.
                
    Returns
    -------
    current vector
        Gives back a vector of currents for each timestep
    """

    ##### factor to define voltage-amplitude heights
    v_amp_factor = 1/(50)
    
    ##### distances between lines and x-axis
    offset = np.cumsum(distance_comps_middle)/meter
    offset = (offset/max(offset))*10
    
    plt.close(plot_name)
    voltage_course = plt.figure(plot_name)
    for ii in comps_to_plot:
        plt.plot(time_vector/ms, offset[ii] - v_amp_factor*(voltage_matrix[ii, :]-V_res)/mV, "#000000")
    plt.yticks(np.linspace(0,10, int(length_neuron/mm)+1),range(0,int(length_neuron/mm)+1,1))
    plt.xlabel('Time/ms', fontsize=16)
    plt.ylabel('Position/mm [major] V/mV [minor]', fontsize=16)
    plt.gca().invert_yaxis() # inverts y-axis => - v_amp_factor*(.... has to be written above
    plt.show(plot_name)
    #voltage_course.savefig('voltage_course_lines.png')
    
    return voltage_course

# =============================================================================
#  Voltage course colors
# =============================================================================
def voltage_course_colors(plot_name,
                          time_vector,
                          voltage_matrix,
                          distance_comps_middle):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    time_vector : integer
        Number of timesteps of whole simulation.
    voltage_matrix : time
        Lenght of one time step.
    distance_comps_middle : current
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
    
    plt.close(plot_name)
    voltage_course = plt.figure(plot_name)
    plt.set_cmap('hot')
    plt.pcolormesh(np.array(time_vector/ms),np.cumsum(distance_comps_middle)/mm,np.array((voltage_matrix)/mV))
    clb = plt.colorbar()
    clb.set_label('V/mV')
    plt.xlabel('t/ms')
    plt.ylabel('Position/mm')
    plt.show(plot_name)
    
    return voltage_course

# =============================================================================
#  Single node response
# =============================================================================
def single_node_response(plot_name,
                         time_vector,
                         voltage_matrix,
                         parameter_vector,
                         parameter_unit,
                         V_res):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    time_vector : integer
        Number of timesteps of whole simulation.
    voltage_matrix : time
        Lenght of one time step.
    distance_comps_middle : current
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
    
    nof_test_parameter = len(parameter_vector)
    nof_runs_per_test_parameter = round(np.shape(voltage_matrix)[0]/nof_test_parameter)
     
    plt.close(plot_name)
    single_node_response = plt.figure(plot_name)
    for ii in range(0,nof_test_parameter):
        plt.text(0.04, 100*ii+V_res/mV+10, f"{parameter_vector[ii]} {parameter_unit}")
        for jj in range(0, nof_runs_per_test_parameter):
            plt.plot(time_vector/ms, 100*ii + voltage_matrix[nof_runs_per_test_parameter*ii+jj,:], "#000000")     
    plt.xlabel('Time/ms', fontsize=16)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.show(plot_name)
    
    return single_node_response
    
# =============================================================================
#  Strength duration curve
# =============================================================================
def strength_duration_curve(plot_name,
                            durations,
                            stimulus_amps):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    time_vector : integer
        Number of timesteps of whole simulation.
    voltage_matrix : time
        Lenght of one time step.
    distance_comps_middle : current
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
        
    plt.close(plot_name)
    strength_duration_curve = plt.figure(plot_name)
    plt.plot(durations[np.where(stimulus_amps != 0)]/us, stimulus_amps[np.where(stimulus_amps != 0)]/uA, "#000000")
    plt.xlabel('Stimulus duration / us', fontsize=16)
    plt.ylabel('Stimulus amplitude required / uA', fontsize=16)
    plt.show(plot_name)
    
    return strength_duration_curve
    