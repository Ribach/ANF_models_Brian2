# =============================================================================
# This script includes all plot functions for the tests that are part of the test
# battery. Each plot shows just test results for a single model
# =============================================================================
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
    plot_name : string
        This defines how the plot window will be named.
    time_vector : list of time values
        Vector contains the time points, that correspond to the voltage values
        of the voltage matrix.
    voltage_matrix : matrix of mambrane potentials
        Matrix has one row for each compartment and one columns for each time
        step. Number of columns has to be the same as the length of the time vector
    comps_to_plot : vector of integers
        This vector includes the numbers of the compartments, which should be part of the plot.
    distance_comps_middle : list of lengths
        This list contains the distances of the compartments, which allows that
        the lines in the plot are spaced according to the real compartment distances
    length_neuron : length
        Defines the total length of the neuron.
    V_res : voltage
        Defines the resting potential of the model.
                
    Returns
    -------
    figure with voltage course plot
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
    
    ##### no grid
    plt.grid(False)
    plt.show(plot_name)
    
    return voltage_course

# =============================================================================
#  Voltage course colors
# =============================================================================
def voltage_course_colors(plot_name,
                          time_vector,
                          voltage_matrix,
                          distance_comps_middle):
    """This function plots the membrane potential of all compartments for each
    time step. The coltages are depicted as colors.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    time_vector : list of time values
        Vector contains the time points, that correspond to the voltage values
        of the voltage matrix.
    voltage_matrix : matrix of mambrane potentials
        Matrix has one row for each compartment and one columns for each time
        step. Number of columns has to be the same as the length of the time vector
    distance_comps_middle : list of lengths
        This list contains the distances of the compartments, which allows that
        the color boxes in the plot are spaced according to the real compartment distances
                
    Returns
    -------
    figure with voltage course plot (colored version)
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
#  Single node response voltage course plot
# =============================================================================
def single_node_response_voltage_course(plot_name,
                                        voltage_data):
    """This function plots the membrane voltage course of one compartment split
    in multiple plots for different phase durations, pulse forms and stimulus amplitudes.
    For stochastic stimulation, multiple lines are shown in each plot

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    voltage_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "phase duration (us)" 
        - "pulse form"
        - "membrane potential (mV)"
        - "time (ms)"
        - "amplitude level"
                
    Returns
    -------
    figure with single node response plot
    """
    
    ##### get amplitude levels and phase durations
    phase_durations = voltage_data["phase duration (us)"].unique().tolist()
    amplitudes = voltage_data["amplitude level"].unique().tolist()
    
    ##### get number of different stimulus amplitudes and phase durations
    nof_phase_durations = len(phase_durations)
    nof_amplitudes = len(amplitudes)    
    
    ##### get number of runs
    nof_runs = max(voltage_data["run"])+1
    
    ##### get achses ranges
    y_min = min(voltage_data["membrane potential (mV)"]) - 5
    y_max = max(voltage_data["membrane potential (mV)"]) + 5
    x_max = max(voltage_data["time (ms)"])
    
    ##### get y-ticks
    y_ticks = [np.round(voltage_data["membrane potential (mV)"].iloc[0]).astype(int), np.round(y_max-5).astype(int)]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_phase_durations, nof_amplitudes, sharex=True, sharey=True, num = plot_name, figsize=(3*nof_amplitudes, 3*nof_phase_durations))
    
    ##### create plots    
    for ii in range(nof_phase_durations):
        for jj in range(nof_amplitudes):
            for kk in range(nof_runs):
            
                #### building a subset of the relevant rows
                current_data = voltage_data[voltage_data["phase duration (us)"] == phase_durations[ii]]\
                                           [voltage_data["amplitude level"] == amplitudes[jj]]\
                                           [voltage_data["run"] == kk]
        
                ##### create plot
                axes[ii][jj].plot(current_data["time (ms)"], current_data["membrane potential (mV)"], color = "black")
                
            ##### remove top and right lines
            axes[ii][jj].spines['top'].set_visible(False)
            axes[ii][jj].spines['right'].set_visible(False)
            
            ##### define achses ranges
            axes[ii][jj].set_ylim([y_min,y_max])
            axes[ii][jj].set_xlim([0,x_max])
            
            ##### set y-ticks
            axes[ii][jj].set_yticks(y_ticks)
            
            ##### remove ticks
            axes[ii][jj].tick_params(axis = 'both', left = 'off', bottom = 'off')
                
            ##### add right side y label
            if jj == nof_amplitudes-1:
                axes[ii][jj].yaxis.set_label_position("right")
                axes[ii][jj].set_ylabel("{} ({} us)".format(current_data["pulse form"].iloc[0], phase_durations[ii]), rotation=-90)
                
            ##### no grid
            axes[ii][jj].grid(False)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    
    ##### use the pulse rates as column titles
    for ax, columtitle in zip(axes[0], ["i = {}".format(amplitudes[ii]) for ii in range(nof_amplitudes)]):
        ax.set_title(columtitle, fontsize=13)
    
    ##### use ticks in the leftmost column
    for ax in axes[:,0]:
        ax.tick_params(axis = 'both', left = True, bottom = 'off')
        
    ##### use ticks in the bottommost row
    for ax in axes[nof_phase_durations-1]:
        ax.tick_params(axis = 'both', left = 'off', bottom = True)
    
    ##### get labels for the axes
    fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Membrane potential (mV)', va='center', rotation='vertical', fontsize=14)
        
    return fig
    
# =============================================================================
#  Strength duration curve
# =============================================================================
def strength_duration_curve(plot_name,
                            threshold_data,
                            rheobase = None,
                            chronaxie = None):
    """This function plots the theshold depending on the phase length of the stimulus.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    threshold_data : pandas dataframe
        The dataframe has to contain the following columns:
        - "phase duration (us)" 
        - "threshold (uA)"
    rheobase : current
        Optional attribute that containes the rheobase, which is marked in the figure
    chronaxie : time
        Optional attribute that containes the chronaxie, which is marked in the figure
                
    Returns
    -------
    figure with single node response plot
    """
    
    ##### exclude values, where no threshold was found
    threshold_data = threshold_data.loc[threshold_data["threshold (uA)"] != 0]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig = plt.figure(plot_name)
    axes = fig.add_subplot(1, 1, 1)
    
    ##### no grid
    axes.grid(False)
    
    ##### plot strength duration curve    
    axes.plot(threshold_data["phase duration (us)"], threshold_data["threshold (uA)"], color = "black", label = "_nolegend_")
    
    ##### mark chronaxie and rheobase    
    if rheobase is not None and chronaxie is not None:
        axes.hlines(y=rheobase/uA, xmin=-0, xmax=max(threshold_data["phase duration (us)"]), linestyles="dashed", label = "rheobase: {} uA".format(round(rheobase/uA, 2)))
        axes.scatter(x=chronaxie/us, y=2*rheobase/uA, color = "blue", label = "chronaxie: {} us".format(round(chronaxie/us)))
        plt.legend()
    
    ##### add labels to the axes    
    axes.set_xlabel('Stimulus duration / us', fontsize=16)
    axes.set_ylabel('Stimulus amplitude required / uA', fontsize=16)
    
    return fig

# =============================================================================
#  relative spread
# =============================================================================
def relative_spread(plot_name,
                    threshold_data):
    """This function provides boxplots, showing the relative spread of thresholds
    for a model for different phase durations and pulse forms.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    threshold_data : pandas dataframe
        The dataframe has to contain the following columns:
        - "phase duration (us)" 
        - "threshold (uA)"
        - "pulse form"
                
    Returns
    -------
    figure with single relative spreads boxplot
    """

    ##### round thresholds
    threshold_data["threshold"] = round(threshold_data["threshold"]/amp*1e6,3)

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig = plt.figure(plot_name)
    
    ##### create boxplots
    sns.set_style("whitegrid")
    sns.boxplot(data=threshold_data, x="phase duration (us)", y="threshold", hue="pulse form", showfliers=False, dodge=False)
    sns.stripplot(x='phase duration (us)', y='threshold',
                   data=threshold_data, 
                   jitter=True, 
                   marker='o', 
                   alpha=0.6,
                   color='black')
    plt.xlabel('Phase duration / us', fontsize=16)
    plt.ylabel('Threshold / uA', fontsize=16)
        
    return fig

# =============================================================================
#  refractory curve
# =============================================================================
def refractory_curve(plot_name,
                     refractory_table):
    """This function plots the refractory curve of a model, which shows the
    minimum required current amplitudes (thresholds) for a second stimulus to
    elicit a second action potential.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    refractory_table : pandas dataframe
        The dataframe has to contain the following columns:
        - "interpulse interval" 
        - "threshold ratio"
        - "minimum required amplitude"
                
    Returns
    -------
    figure with single relative spreads boxplot
    """
    
    ##### remove rows where no second spikes were obtained
    refractory_table = refractory_table[refractory_table["minimum required amplitude"] != 0]
        
    ##### calculate the ratio of the threshold of the second spike and the masker
    refractory_table["threshold ratio"] = refractory_table["minimum required amplitude"]/refractory_table["threshold"]
    
    ##### convert interpulse intervals to ms
    refractory_table["interpulse interval"] = refractory_table["interpulse interval"]*1e3
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig = plt.figure(plot_name)
    axes = fig.add_subplot(1, 1, 1)

    ###### plot refractory curve
    axes.plot(refractory_table["interpulse interval"], refractory_table["threshold ratio"], color = "black")
    
    ##### show points
    axes.scatter(refractory_table["interpulse interval"], refractory_table["threshold ratio"], color = "blue")
    
    ##### add line at threshold level
    axes.hlines(y=1, xmin=-0, xmax=max(refractory_table["interpulse interval"]), linestyles="dashed")
    
    ##### define axes ranges
    axes.set_xlim([0,max(refractory_table["interpulse interval"])+0.5])
    axes.set_ylim([0,max(refractory_table["threshold ratio"])+0.5])
    
    ##### axes labels
    axes.set_xlabel('Inter pulse interval / ms', fontsize=14)
    axes.set_ylabel('threshold 2nd stimulus / threshold', fontsize=14)
    
    ##### no grid
    axes.grid(False)
    
    return fig
    

# =============================================================================
#  post-stimulus time histogram
# =============================================================================
def post_stimulus_time_histogram(plot_name,
                                 psth_dataset,
                                 plot_style = "firing_efficiency"):
    """This function plots the poststimulus-time histograms one for each amplitude
    and pulse rate.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    refractory_table : pandas dataframe
        The dataframe has to contain the following columns:
        - "spike times (us)" 
        - "amplitude"
        - "run"
        - "pulse rate"
    plot_style : string
        Describes, what the y-achses show. Either "firing_efficiency" or "spikes_per_timebin"
        are possible.
                
    Returns
    -------
    figure with poststimulus-time histograms
    """
    
    ##### convert spike times to ms
    psth_dataset["spike times (us)"] = np.ceil(list(psth_dataset["spike times (us)"]*1000)).astype(int)
    psth_dataset = psth_dataset.rename(index = str, columns={"spike times (us)" : "spike times (ms)"})
    
    ##### get amplitude levels and pulse rates
    amplitudes = psth_dataset["amplitude"].unique().tolist()
    pulse_rates = psth_dataset["pulse rate"].unique().tolist()
    
    ##### get number of different pulse rates and stimulus amplitudes
    nof_amplitudes = len(amplitudes)
    nof_pulse_rates = len(pulse_rates)
    
    ##### specify bin width (in ms)
    bin_width = 10
    
    ##### get number of runs and bins
    nof_runs = max(psth_dataset["run"])+1
    nof_bins = int((max(psth_dataset["spike times (ms)"])+1) / bin_width)

    ##### get bin edges
    bin_edges = [ii*bin_width+0.5*bin_width for ii in range(nof_bins+1)]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_amplitudes, nof_pulse_rates, sharex=True, sharey=True, num = plot_name, figsize=(3*nof_pulse_rates, 3*nof_amplitudes))
    
    ##### initialize maximum bin height
    max_bin_height = 0
    
    ##### create plots
    for ii in range(nof_amplitudes):
        for jj in range(nof_pulse_rates):
            
            #### building a subset of the relevant rows
            current_data = psth_dataset[psth_dataset["amplitude"] == amplitudes[ii]][psth_dataset["pulse rate"] == pulse_rates[jj]]

            ##### calculating the bin heights
            bin_heights = [sum((bin_width*kk < current_data["spike times (ms)"]) & (current_data["spike times (ms)"] < bin_width*kk+bin_width))/nof_runs for kk in range(0,nof_bins+1)]
            if plot_style == "firing_efficiency":
                bin_heights = [height / (current_data["pulse rate"].iloc[0]/second * bin_width*ms) for height in bin_heights]
            
            ##### create barplot
            axes[ii][jj].bar(x = bin_edges, height = bin_heights, width = bin_width, color = "black", linewidth=0.3)
            
            ##### remove top and right lines
            axes[ii][jj].spines['top'].set_visible(False)
            axes[ii][jj].spines['right'].set_visible(False)
            
            ##### update max_bin_height
            if round(max(bin_heights)) > max_bin_height:
                max_bin_height = round(max(bin_heights))
            
            ##### define x-achses range and tick numbers
            axes[ii][jj].set_xlim([-10,max(bin_edges)*1.1])
            axes[ii][jj].set_xticks([0,max(bin_edges)-0.5*bin_width])
                        
            ##### remove ticks
            axes[ii][jj].tick_params(axis = 'both', left = 'off', bottom = 'off')
            
            ##### add right side y label
            if jj == nof_pulse_rates-1:
                axes[ii][jj].yaxis.set_label_position("right")
                axes[ii][jj].set_ylabel("i={}".format(current_data["amplitude"].iloc[0]), rotation=-90)
                axes[ii][jj].yaxis.set_label_coords(1.1,0.5)
                
            ##### no grid
            axes[ii][jj].grid(False) 
    
    ##### further adjustments
    for ii in range(nof_amplitudes):
        for jj in range(nof_pulse_rates):
            
            #### building a subset of the relevant rows
            current_data = psth_dataset[psth_dataset["amplitude"] == amplitudes[ii]][psth_dataset["pulse rate"] == pulse_rates[jj]]
            
            if plot_style == "firing_efficiency":
                ##### define y-achses range and tick numbers
                axes[ii][jj].set_ylim([0,1.25])
                axes[ii][jj].set_yticks([0,0.5,1])
            
                ##### Write spiking efficiences as percentage
                vals = (axes[ii][jj].get_yticks() * 100).astype(int)
                axes[ii][jj].set_yticklabels(['{}%'.format(x) for x in vals])
            
            ##### write stimulus amplitdues in plots
            axes[ii][jj].text(np.ceil(max(bin_edges)/3.2), max_bin_height+0.1, "i={}mA".format(current_data["stimulus amplitude (uA)"][0]))

    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    ##### use the pulse rates as column titles
    for ax, columtitle in zip(axes[0], ["{} pps".format(pulse_rates[ii]) for ii in range(nof_pulse_rates)]):
        ax.set_title(columtitle, y = 1.1)
    
    ##### use ticks in the leftmost column
    for ax in axes[:,0]:
        ax.tick_params(axis = 'both', left = True, bottom = 'off')
        
    ##### use ticks in the bottommost row
    for ax in axes[nof_amplitudes-1]:
        ax.tick_params(axis = 'both', left = 'off', bottom = True)
    
    ##### get labels for the axes
    fig.text(0.5, 0.05, 'Time after pulse train onset (ms)', ha='center', fontsize=14)
    if plot_style == "firing_efficiency":
        fig.text(0.06, 0.5, 'firing efficiency', va='center', rotation='vertical', fontsize=14)
    else:
        fig.text(0.08, 0.5, 'Spikes per timebin ({} ms)'.format(bin_width), va='center', rotation='vertical')
    
    return fig
