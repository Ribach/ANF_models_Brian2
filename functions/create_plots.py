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
    threshold_matrix = threshold_matrix.loc[threshold_matrix["threshold"]/amp != 0]
        
    plt.close(plot_name)
    strength_duration_curve = plt.figure(plot_name)
    plt.plot(threshold_matrix["phase_duration"]/us, threshold_matrix["threshold"]/uA, label = '_nolegend_')
    if not rheobase == 0 and not chronaxie == 0:
        plt.hlines(y=rheobase/uA, xmin=-0, xmax=max(threshold_matrix["phase_duration"]/us), linestyles="dashed", label = "rheobase: {} uA".format(round(rheobase/uA, 2)))
        plt.scatter(x=chronaxie/us, y=2*rheobase/uA, label = "chronaxie: {} us".format(round(chronaxie/us)))
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
    
    ##### round phase durations and thresholds
    threshold_matrix["phase duration"] = round(threshold_matrix["phase duration"]/second*1000000).astype(int)
    threshold_matrix["threshold"] = round(threshold_matrix["threshold"]/amp*1000000,3)
    
    ##### add pulse form info to phase duration
    #threshold_matrix["phase duration"] = '{} us'.format(threshold_matrix["phase duration"])
    
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
                     inter_pulse_intervals,
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
    
    ##### remove inter_pulse_intervals where no second spikes were obtained
    inter_pulse_intervals = inter_pulse_intervals[np.where(stimulus_amps != 0)[0]]/ms
    
    ##### calculate the ratio of the threshold of the second spike and the masker
    thresholds_second_spike = stimulus_amps[np.where(stimulus_amps != 0)[0]]/threshold
     
    ###### plot refractory curve
    plt.close(plot_name)
    refractory_curve = plt.figure(plot_name)
    plt.plot(inter_pulse_intervals, thresholds_second_spike, "#000000")
    plt.xlabel('Inter pulse interval / ms', fontsize=16)
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
    
    ##### convert spike times to ms
    psth_dataset["spike times (us)"] = np.ceil(psth_dataset["spike times (us)"]/1000).astype(int)
    psth_dataset = psth_dataset.rename(index = str, columns={"spike times (us)" : "spike times (ms)"})
    
    ##### get amplitude levels and pulse rates
    amplitudes = psth_dataset["amplitude"].unique().tolist()
    pulse_rates = psth_dataset["pulse rate"].unique().tolist()
    
    ##### get number of different pulse rates and stimulus amplitudes
    nof_amplitudes = len(amplitudes)
    nof_pulse_rates = len(pulse_rates)
    
    ##### get number of runs and bins
    nof_runs = max(psth_dataset["run"])+1
    nof_bins = max(psth_dataset["spike times (ms)"])

    ##### get bin edges
    bin_edges = list(range(nof_bins))

    fig, axes = plt.subplots(nof_amplitudes, nof_pulse_rates, sharex=True, sharey=True, figsize=(2*nof_pulse_rates, 2*nof_amplitudes))

    for ii in range(nof_amplitudes):
        for jj in range(nof_pulse_rates):
            
            #### building a subset of the relevant rows
            current_data = psth_dataset[psth_dataset["amplitude"] == amplitudes[ii]][psth_dataset["pulse rate"] == pulse_rates[jj]]

            ##### calculating the bin heights
            bin_heights = [round(sum(current_data["spike times (ms)"] == kk)*1000/nof_runs).astype(int) for kk in range(1,nof_bins+1)]
            
            ##### create barplot
            axes[ii][jj].bar(x = bin_edges, height = bin_heights, width = 1, color = "black")
            
            ##### remove top and right lines
            axes[ii][jj].spines['top'].set_visible(False)
            axes[ii][jj].spines['right'].set_visible(False)
            
            ##### define y-achses range and tick numbers
            axes[ii][jj].set_ylim([0,1500])
            axes[ii][jj].set_yticks([0,500,1000])
            
            ##### write stimulus amplitude in plots
            axes[ii][jj].text(np.ceil(nof_bins/2), 1250, "i={}mA".format(current_data["stimulus amplitude (uA)"][0]))
            
            ##### remove ticks
            axes[ii][jj].tick_params(axis = 'both', left = 'off', bottom = 'off')     
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    
    ##### use the pulse rates as column titles
    for ax, columtitle in zip(axes[0], ["{} pps".format(pulse_rates[ii]) for ii in range(nof_pulse_rates)]):
        ax.set_title(columtitle)
    
    ##### use ticks in the leftmost column
    for ax in axes[:,0]:
        ax.tick_params(axis = 'both', left = True, bottom = 'off')
        
    ##### use ticks in the bottommost row
    for ax in axes[nof_amplitudes-1]:
        ax.tick_params(axis = 'both', left = 'off', bottom = True)
    
    ##### get labels for the axes
    fig.text(0.5, 0.02, 'Time after pulse train onset (ms)', ha='center')
    fig.text(0.02, 0.5, 'Response Rate (spikes/s)', va='center', rotation='vertical')
    
    return post_stimulus_time_histogram


