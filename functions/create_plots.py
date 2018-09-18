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
def single_node_response_voltage_course(plot_name,
                                        voltage_data,
                                        col_wrap = 0,
                                        height = 2):
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
    if col_wrap == 0:
        nof_param_1_values = len(np.unique(voltage_data[voltage_data.columns.values[0]]))
        col_wrap = 1
        if nof_param_1_values > 3:
            col_wrap = 2
    
        if nof_param_1_values > 7:
            col_wrap = 3
    
    ##### define color palette and legend
    nof_param_2_values = len(np.unique(voltage_data[voltage_data.columns.values[1]]))
    if "run" in voltage_data.columns.values:
        palette = sns.color_palette("Blues_d", n_colors = nof_param_2_values)
        legend = False
    else:
        palette = sns.color_palette(n_colors = nof_param_2_values)
        legend = "full"
    
    ##### plot voltage courses
    single_node_response = sns.relplot(x="time / ms",
                                       y="membrane potential / mV",
                                       hue=voltage_data.columns.values[1],
                                       col=voltage_data.columns.values[0],
                                       col_wrap = col_wrap,
                                       kind="line",
                                       data=voltage_data,
                                       height = height,
                                       legend=legend,
                                       palette = palette)
    
    ##### delete label in certain cases
    if voltage_data.columns.values[0] in ['model name', 'stimulation info']:
        _,index = np.unique(voltage_data[voltage_data.columns.values[0]], return_index=True)
        colnames = voltage_data[voltage_data.columns.values[0]][np.sort(index)]
        for ax, title in zip(single_node_response.axes.flat, colnames):
            ax.set_title(title, fontsize=15)
            
    single_node_response = single_node_response.fig
        
    return single_node_response
    
# =============================================================================
#  Strength duration curve
# =============================================================================
def strength_duration_curve(plot_name,
                            threshold_matrix,
                            rheobase = 0,
                            chronaxie = 0):
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
    
    threshold_matrix = threshold_matrix[threshold_matrix["threshold"].notnull()]
        
    plt.close(plot_name)
    strength_duration_curve = plt.figure(plot_name)
    plt.plot(threshold_matrix["phase duration"]*1e6, threshold_matrix["threshold"]*1e6, label = '_nolegend_')
    if not rheobase == 0 and not chronaxie == 0:
        plt.hlines(y=rheobase/uA, xmin=-0, xmax=max(threshold_matrix["phase duration"]/us), linestyles="dashed", label = f"rheobase: {round(rheobase/uA, 2)} uA")
        plt.scatter(x=chronaxie/us, y=2*rheobase/uA, label = f"chronaxie: {round(chronaxie/us)} us")
        plt.legend()
    plt.xlabel('Stimulus duration / us', fontsize=16)
    plt.ylabel('Stimulus amplitude required / uA', fontsize=16)
    plt.show(plot_name)
    
    return strength_duration_curve

# =============================================================================
#  relative spread
# =============================================================================
def relative_spread(plot_name,
                    threshold_matrix):
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
    
    ##### delete column "run"
    threshold_matrix = threshold_matrix.drop(columns = "run")
    
    ##### round phase durations and thresholds
    threshold_matrix["phase duration"] = round(threshold_matrix["phase duration"]*1000000).astype(int)
    threshold_matrix["threshold"] = round(threshold_matrix["threshold"]*1000000,3)
    
    ##### add pulse form info to phase duration
    #threshold_matrix["phase duration"] = f'{threshold_matrix["phase duration"]} us'
    
    ##### plot thresholds
    plt.close(plot_name)
    relative_spread_plot = plt.figure(plot_name)
    sns.set_style("whitegrid")
    sns.boxplot(data=threshold_matrix, x="phase duration", y="threshold", hue="pulse form", showfliers=False, dodge=False)
    sns.stripplot(x='phase duration', y='threshold',
                   data=threshold_matrix, 
                   jitter=True, 
                   marker='o', 
                   alpha=0.6,
                   color='black')
    plt.xlabel('Phase duration / us', fontsize=16)
    plt.ylabel('Threshold / uA', fontsize=16)
    plt.show(plot_name)
        
    return relative_spread_plot

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
    
    inter_pulse_intervalls = inter_pulse_intervalls[np.where(stimulus_amps != 0)]/ms
    thresholds_second_spike = np.round(stimulus_amps[np.where(stimulus_amps != 0)]/threshold, 2)
        
    plt.close(plot_name)
    refractory_curve = plt.figure(plot_name)
    plt.plot(inter_pulse_intervalls, thresholds_second_spike, "#000000")
    plt.xlabel('Inter pulse intervall / ms', fontsize=16)
    plt.ylabel('threshold 2nd stimulus / threshold', fontsize=16)
    plt.show(plot_name)
    
    return refractory_curve
    

# =============================================================================
#  post-stimulus time histogram
# =============================================================================
def post_stimulus_time_histogram(plot_name,
                                 psth_dataset):
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
    
    psth_dataset["spike times"] = (psth_dataset["spike times"]*1000)
    nof_bins = round(max(psth_dataset["spike times"]))
    
    grid = sns.FacetGrid(psth_dataset, row="amplitude", col="pulse rate", margin_titles=True)
    grid.map(plt.hist, "spike times", bins = nof_bins)
    
    post_stimulus_time_histogram = grid.fig
    
#    plt.close(plot_name)
#    post_stimulus_time_histogram = plt.figure(plot_name)
#    plt.bar(bin_edges[:-1]*1000,
#            bin_heigths,
#            width = bin_width)
#    plt.xlabel('Time / ms', fontsize=16)
#    plt.ylabel('Spikes per second', fontsize=16)
#    plt.show(plot_name)
    
    return post_stimulus_time_histogram

# =============================================================================
#  inter-stimulus intervall histogram
# =============================================================================
def inter_stimulus_intervall_histogram(plot_name,
                                       isi_dataset):
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
    
    isi_dataset["spike times"] = (isi_dataset["spike times"]*1000)
    nof_bins = round(max(isi_dataset["spike times"]))
    
    grid = sns.FacetGrid(isi_dataset, row="amplitude", col="pulse rate", margin_titles=True)
    grid.map(plt.hist, "spike times", bins = nof_bins)
    
    inter_stimulus_intervall_histogram = grid.fig
    
    return inter_stimulus_intervall_histogram














