from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)

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
#  Single node response bar plot
# =============================================================================
def single_node_response_bar_plot(data):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    time_vector : integer
        Number of timesteps of whole simulation.
                
    Returns
    -------
    current vector
        Gives back a vector of currents for each timestep
    """
    
    colnames = data.columns.values[2:]
    
    data = data.melt(id_vars = data.columns.values[0:2], var_name = "observation")
        
    if "run" in data.columns.values:
        hue = None
    else:
        hue=data.columns.values[1]
        
    single_node_response_bar_plot = sns.catplot(x=data.columns.values[0],
                                                y=data.columns.values[3],
                                                hue=hue,
                                                kind="bar",
                                                data=data,
                                                col = data.columns.values[2],
                                                sharex=False,
                                                sharey=False,
                                                height = 2.8,
                                                col_wrap=2)
    
    for ax, title in zip(single_node_response_bar_plot.axes.flat, colnames):
        ax.set_title(title, fontsize=15)
    
    return single_node_response_bar_plot

# =============================================================================
#  Single node response voltage course plot
# =============================================================================
def single_node_response_voltage_course(voltage_data):
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
     
    ##### define number of columns
    nof_param_1_values = len(np.unique(voltage_data[voltage_data.columns.values[0]]))
    col_wrap = 1
    if nof_param_1_values > 3:
        col_wrap = 2

    if nof_param_1_values > 6:
        col_wrap = 3
    
    ##### define color palette and legend
    nof_param_2_values = len(np.unique(voltage_data[voltage_data.columns.values[1]]))
    if "run" in voltage_data.columns.values:
        palette = sns.color_palette("Blues_d", n_colors = nof_param_2_values)
        legend = False
    else:
        palette = sns.color_palette(n_colors = nof_param_2_values)
        legend = "full"
    
        
    single_node_response = sns.relplot(x="time",
                                       y="value",
                                       hue=voltage_data.columns.values[1],
                                       col=voltage_data.columns.values[0],
                                       col_wrap = col_wrap,
                                       kind="line",
                                       data=voltage_data,
                                       height = 2.8,
                                       legend=legend,
                                       palette = palette)
    
    if voltage_data.columns.values[0] == 'model name':
        _,index = np.unique(voltage_data[voltage_data.columns.values[0]], return_index=True)
        colnames = voltage_data[voltage_data.columns.values[0]][np.sort(index)]
        for ax, title in zip(single_node_response.axes.flat, colnames):
            ax.set_title(title, fontsize=15) 
    
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

# =============================================================================
#  refractory curve
# =============================================================================
def refractory_curve(plot_name,
                     inter_pulse_intervalls,
                     stimulus_amps,
                     threshold):
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
    plt.plot(inter_pulse_intervalls[np.where(stimulus_amps != 0)]/ms, (stimulus_amps[np.where(stimulus_amps != 0)]/threshold)/uA, "#000000")
    plt.xlabel('Inter pulse intervall / ms', fontsize=16)
    plt.ylabel('Stimulus amplitude required / uA', fontsize=16)
    plt.show(plot_name)
    
    return strength_duration_curve
    

# =============================================================================
#  post-stimulus-time-histogram
# =============================================================================
def post_stimulus_time_histogram(plot_name,
                                 bin_edges,
                                 bin_heigths,
                                 bin_width):
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
    post_stimulus_time_histogram = plt.figure(plot_name)
    plt.bar(bin_edges[:-1]*1000,
            bin_heigths,
            width = bin_width)
    plt.xlabel('Time / ms', fontsize=16)
    plt.ylabel('Spikes per second', fontsize=16)
    plt.show(plot_name)
    
    return post_stimulus_time_histogram