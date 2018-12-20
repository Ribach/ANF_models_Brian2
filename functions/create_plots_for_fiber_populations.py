# =============================================================================
# This script includes all plot functions for the tests that are part of the 
# "Model analyses" script and for the comparisons in the "Model_comparison"
# sript. The plots usually compare the results of multiple models. In some cases
# the plots show also experimental data.
# =============================================================================
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)

##### import functions
import functions.calculations as calc

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14
import models.Rudnicki_2018 as rudnicki_18
import models.trials.Rattay_adap_2001 as rattay_adap_01
import models.trials.Briaire_adap_2005 as briaire_adap_05
import models.trials.Imennov_adap_2009 as imennov_adap_09
import models.trials.Negm_ANF_2014 as negm_ANF_14

# =============================================================================
#  Raster plot, showing spiketimes of fibers (for one model)
# =============================================================================
def raster_plot(plot_name,
                spike_trains):
    """This function plots the refractory curves which show the minimum required
    current amplitudes (thresholds) for a second stimulus to elicit a second
    action potential. There is one plot for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    refractory_curves : pandas dataframe
        This dataframe has to contain the following columns:
        - "interpulse interval" 
        - "minimum required amplitude"
        - "threshold"
        - "model"
                
    Returns
    -------
    figure with refractory curve comparison
    """
    
    ##### initializations
    bin_width = 1
    nof_bins = max(spike_trains["neuron_number"])
    
    ##### get y-axes range
    y_min = min(spike_trains["neuron_number"])-5
    y_max = max(spike_trains["neuron_number"])+5
    
    ##### delete rows whith no spike
    spike_trains = spike_trains[pd.to_numeric(spike_trains['spikes'], errors='coerce').notnull()].copy()
        
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, (ax0, ax1) = plt.subplots(1,2, sharex=False, sharey=False, num = plot_name, gridspec_kw = {'width_ratios':[3, 1]}, figsize=(12, 7))

    ##### define figure title
    model_name = eval("{}.display_name".format(spike_trains["model_name"].iloc[0]))
    fig.suptitle('Spike times and firing rate; {}'.format(model_name), fontsize=16)
    
    ##### raster plot
    # no grid
    ax0.grid(False)
    # define x-axes range
    x_min = 0
    x_max = int(max(spike_trains["duration"])*1e3)
    ax0.set_xlim([x_min,x_max])
    # define y axes ranges
    ax0.set_ylim([y_min,y_max])
    # plot spikes
    ax0.scatter(spike_trains["spikes"]*1e3, spike_trains["neuron_number"], color = "black", s = 2)
    # get labels for the axes    
    ax0.set_xlabel("Time / ms", fontsize=14)
    ax0.set_ylabel("Nerve fiber number", fontsize=14)
    
    ##### discharge rate plot
    # no grid
    ax1.grid(False)
    # calculate bin edges
    bin_edges = [ii*bin_width+0.5*bin_width for ii in range(0,nof_bins+1)]
    # calculate bin heights
    bin_heights = [len(spike_trains[spike_trains["neuron_number"] == ii]) / spike_trains["nof_pulses"].iloc[0] for ii in range(nof_bins+1)]
    # define x-axes range
    x_min = 0
    x_max = max(bin_heights)*1.1
    ax1.set_xlim([x_min,x_max])
    ##### define y axes ranges
    ax1.set_ylim([y_min,y_max])
    # create barplot
    ax1.barh(y = bin_edges, width = bin_heights, height = bin_width, color = "black", linewidth=0.3, edgecolor = "none")
    # get labels for the axes
    ax1.set_xlabel("Firing efficiency", fontsize=14)
    ##### Write spiking efficiences as percentage
    vals = (ax1.get_xticks() * 100).astype(int)
    ax1.set_xticklabels(['{}%'.format(x) for x in vals])

    ##### bring subplots close to each other.
    fig.subplots_adjust(wspace=0.15)
    
    return fig
    
# =============================================================================
#  Plot number of spiking fibers over stimulus amplitudes
# =============================================================================
def nof_spikes_over_stim_amp(plot_name,
                             spike_table):
    """This function plots the refractory curves which show the minimum required
    current amplitudes (thresholds) for a second stimulus to elicit a second
    action potential. There is one plot for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    refractory_curves : pandas dataframe
        This dataframe has to contain the following columns:
        - "interpulse interval" 
        - "minimum required amplitude"
        - "threshold"
        - "model"
                
    Returns
    -------
    figure with refractory curve comparison
    """
        
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(8, 5))

    ##### define figure title
    model_name = eval("{}.display_name".format(spike_table["model_name"].iloc[0]))
    fig.suptitle('{}'.format(model_name), fontsize=16)
    
    ##### no grid
    axes.grid(True)
    
    ###### plot number of spikes over stimulus amplitudes
    axes.plot(spike_table["stim_amp"]*1e3, spike_table["nof_spikes"], color = "black")
    
    ##### write dynamic range in plots
    stim_amp_min_spikes = max(spike_table["stim_amp"][spike_table["nof_spikes"] == min(spike_table["nof_spikes"])])
    stim_amp_max_spikes = min(spike_table["stim_amp"][spike_table["nof_spikes"] == max(spike_table["nof_spikes"])])
    if stim_amp_min_spikes != 0:
        dynamic_range = np.round(10*np.log10(stim_amp_max_spikes/stim_amp_min_spikes),1)
        axes.text(0, max(spike_table["nof_spikes"])-25, "Dynamic range: {} dB".format(dynamic_range), fontsize=12)
    
    ##### get labels for the axes    
    axes.set_xlabel("stimulus amplitude / mA", fontsize=14)
    axes.set_ylabel("Number of spiking fibers", fontsize=14)
    
    return fig

# =============================================================================
#  Plot dB above threshold over distance along spiral lamina (colored)
# =============================================================================
def spikes_color_plot(plot_name,
                         spike_table):
    """This function plots the refractory curves which show the minimum required
    current amplitudes (thresholds) for a second stimulus to elicit a second
    action potential. There is one plot for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    refractory_curves : pandas dataframe
        This dataframe has to contain the following columns:
        - "interpulse interval" 
        - "minimum required amplitude"
        - "threshold"
        - "model"
                
    Returns
    -------
    figure with refractory curve comparison
    """
    
    ##### get model module
    model = eval(spike_table["model_name"].iloc[0])
    
    ##### calculate spikes per fiber
    spikes_per_fiber = spike_table.groupby(["model_name","stim_amp"])["spike"].sum().reset_index()
    spikes_per_fiber = spikes_per_fiber.rename(index = str, columns={"spike" : "nof_spikes"})
    
    ##### calculate dynamic range values
    stim_amp_min_spikes = max(spikes_per_fiber["stim_amp"][spikes_per_fiber["nof_spikes"] == min(spikes_per_fiber["nof_spikes"])])
    spike_table["dynamic_range"] = 10*np.log10(spike_table["stim_amp"]/stim_amp_min_spikes)
        
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(12, 8))

    ##### define figure title
    model_name =model.display_name
    fig.suptitle('{}'.format(model_name), fontsize=16)
    
    ##### no grid
    axes.grid(False)
    
    if hasattr(model, "index_soma"):
        ##### create color map
        basic_cols=['#006837', '#ffffbf', '#a50026']
        cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
        
        ##### adjust cmap that middle of diverging colors is at soma
        endpoint = max(spike_table["first_spike_dist"]) #model.length_neuron/mm
        midpoint = (np.cumsum(model.compartment_lengths)[model.middle_comp_soma]/mm)/endpoint
        cmap = calc.shiftedColorMap(cmap, midpoint=midpoint, name='shifted')
        
        ##### give soma an extra color
        color_res = cmap.N # resolution of cmap
        if hasattr(model, "length_soma"):
            soma_length = model.length_soma
        else:
            soma_length = model.diameter_soma / mm
        soma_range = int(np.ceil(soma_length/max(spike_table["first_spike_dist"])*color_res))
        start_point = int((np.cumsum(model.compartment_lengths)[model.start_index_soma]/mm)/endpoint*color_res)
        for ii in range(start_point, start_point + soma_range):
            cmap_list = [cmap(i) for i in range(cmap.N)]
            cmap_list[ii] = LinearSegmentedColormap.from_list('mycmap', ['#feff54','#feff54'])(0)
            cmap = cmap.from_list('Custom cmap', cmap_list, cmap.N)
        
    else:
        cmap = 'YlGnBu'
    
    ##### create x and y mesh
    dynamic_ranges = pd.unique(spike_table["dynamic_range"].sort_values())
    distances_sl = pd.unique(spike_table["dist_along_sl"].sort_values())
    xmesh, ymesh = np.meshgrid(distances_sl, dynamic_ranges)
    
    ##### get the corresponding first spike distance for each x and y value
    distances = spike_table.pivot_table(index="dynamic_range", columns="dist_along_sl", values="first_spike_dist", fill_value=0).as_matrix()
    distances[distances == 0] = 'nan'
    
    ###### show spiking fibers depending on stimulus amplitude
    color_mesh = axes.pcolormesh(xmesh, ymesh, distances, cmap = cmap, norm = Normalize(vmin = 0, vmax = max(spike_table["first_spike_dist"])),linewidth=0,rasterized=True)
    clb = fig.colorbar(color_mesh)
    
    ##### define axes ranges
    axes.set_xlim([0,max(spike_table["dist_along_sl"])])
    
    ##### change y-achses to dynamic range
    axes.set_yticklabels(['{} dB'.format(y) for y in axes.get_yticks()])
    
    if hasattr(model, "index_soma"):
        
        ##### change clb ticks
        soma = endpoint*midpoint
        dendrite = soma*0.25
        axon = soma + (endpoint-soma)*0.75
        clb.set_ticks([dendrite, soma, axon])
        clb.set_ticklabels(["dendrite","soma","axon"])
        clb.set_label('Position of first spike')
        
    else:
        clb.set_label('Distance from peripheral terminal / mm')
    
    ##### get labels for the axes    
    axes.set_xlabel("Distance along spiral lamina / mm", fontsize=14)
    axes.set_ylabel("dB above threshold", fontsize=14)
    
    return fig

# =============================================================================
#  Plot dB above threshold over distance along spiral lamina and mark latencies
# =============================================================================
def latencies_color_plot(plot_name,
                         spike_table):
    """This function plots the refractory curves which show the minimum required
    current amplitudes (thresholds) for a second stimulus to elicit a second
    action potential. There is one plot for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    refractory_curves : pandas dataframe
        This dataframe has to contain the following columns:
        - "interpulse interval" 
        - "minimum required amplitude"
        - "threshold"
        - "model"
                
    Returns
    -------
    figure with refractory curve comparison
    """
    
    ##### get model module
    model = eval(spike_table["model_name"].iloc[0])
    
    ##### calculate spikes per fiber
    spikes_per_fiber = spike_table.groupby(["model_name","stim_amp"])["spike"].sum().reset_index()
    spikes_per_fiber = spikes_per_fiber.rename(index = str, columns={"spike" : "nof_spikes"})
    
    ##### calculate dynamic range values
    stim_amp_min_spikes = max(spikes_per_fiber["stim_amp"][spikes_per_fiber["nof_spikes"] == min(spikes_per_fiber["nof_spikes"])])
    spike_table["dynamic_range"] = 10*np.log10(spike_table["stim_amp"]/stim_amp_min_spikes)
        
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(12, 8))

    ##### define figure title
    model_name =model.display_name
    fig.suptitle('{}'.format(model_name), fontsize=16)
    
    ##### no grid
    axes.grid(False)
    
    ##### define color map
    cmap = 'YlGnBu'
    
    ##### create x and y mesh
    dynamic_ranges = pd.unique(spike_table["dynamic_range"].sort_values())
    distances_sl = pd.unique(spike_table["dist_along_sl"].sort_values())
    xmesh, ymesh = np.meshgrid(distances_sl, dynamic_ranges)
    
    ##### get the corresponding first spike distance for each x and y value
    latencies = spike_table.pivot_table(index="dynamic_range", columns="dist_along_sl", values="latency", fill_value=0).as_matrix().astype(float)
    latencies[latencies == 0] = 'nan'
    
    ###### show spiking fibers depending on stimulus amplitude
    color_mesh = axes.pcolormesh(xmesh, ymesh, latencies, cmap = cmap,linewidth=0,rasterized=True)
    clb = fig.colorbar(color_mesh)
    
    ##### define axes ranges
    axes.set_xlim([0,max(spike_table["dist_along_sl"])])
    
    ##### change y-achses to dynamic range
    axes.set_yticklabels(['{} dB'.format(y) for y in axes.get_yticks()])

    ##### get labels for the axes    
    axes.set_xlabel("Distance along spiral lamina / mm", fontsize=14)
    axes.set_ylabel("dB above threshold", fontsize=14)
    clb.set_label('Spike latency / us')
    
    return fig

# =============================================================================
#  Plots dB above threshold over distance along spiral lamina for cathodic, anodic and biphasic pulses
# =============================================================================
def compare_pulse_forms(plot_name,
                        spike_table):
    """This function plots dB above threshold (of all pulse forms) over distance
    along spiral lamina and compares different pulse forms. There is one plot
    for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    spike_table : pandas dataframe
        This dataframe has to contain the following columns:
        - "model_name" 
        - "neuron_number"
        - "stim_amp"
        - "pulse_form"
                
    Returns
    -------
    figure with comparison of spiking behaviour for different pulse forms
    """
    
    ##### get model names
    models = spike_table["model_name"].unique().tolist()
    
    ##### get pulse forms
    pulse_forms = np.sort(spike_table["pulse_form"].unique()).tolist()
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of rows
    nof_rows = np.ceil(len(models)/nof_cols).astype(int)
    
    ##### get number of plots
    nof_plots = len(models)
    
    ##### list electrode positions
    electrode_positions = [4.593, 7.435, 9.309, 11.389, 13.271, 15.164, 16.774, 18.522, 20.071, 21.364, 22.629, 23.649]
    
    ##### get index of stimulus electrode
    elec_nr = spike_table["elec_nr"].iloc[0]
    
    ##### get axes ranges
    y_min = min(spike_table["dynamic_range"])
    y_max = max(spike_table["dynamic_range"])
    x_min = 0
    x_max = max(spike_table["dist_along_sl"])
    
    ##### define colors
    colors = ["red", "black", "blue"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=False, num = plot_name, figsize=(6*nof_cols, 4*nof_rows))
    
    ##### create plots  
    for ii in range(nof_rows*nof_cols):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        col = ii-row*nof_cols
        
        ##### define axes ranges
        axes[row][col].set_ylim([y_min,y_max])
        axes[row][col].set_xlim([x_min,x_max])
        
        ##### turn off x-labels for all but the bottom plots
        if (nof_plots - ii) > nof_cols:
             plt.setp(axes[row][col].get_xticklabels(), visible=False)
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot thresholds
        if ii < nof_plots:
            
            model = models[ii]
                
            ##### building a subset
            current_data = spike_table[spike_table["model_name"] == model]
            
            ##### loop over pulse forms
            for jj,pulse_form in enumerate(pulse_forms):
                
                current_pulse_form = current_data[current_data["pulse_form"] == pulse_form]
                
                axes[row][col].plot(current_pulse_form["dist_along_sl"], current_pulse_form["dynamic_range"], color = colors[jj], label = pulse_form)
                
            ##### remove top and right lines
            axes[row][col].spines['top'].set_visible(False)
            axes[row][col].spines['right'].set_visible(False)
                
            ##### write model name in plots
            axes[row][col].text(x_max*0.3, y_max+0.3, eval("{}.display_name".format(model)))
            
            ##### add electrode position
            axes[row][col].scatter(electrode_positions[elec_nr], -0.2, color = "black", marker = "^", label = "_nolegend_", clip_on=False)
                
            ##### change y-achses to dynamic range
            axes[row][col].set_yticklabels(['{} dB'.format(y) for y in axes[row][col].get_yticks()])
            
            ##### add legend
            axes[row][col].legend()
                
            ##### horizontal grid
            #axes[row][col].yaxis.grid(True)
            axes[row][col].grid(True)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.15, wspace=0.2)
    
    ##### get labels for the axes
    fig.text(0.5, 0.05, 'Distance along spiral lamina / mm', ha='center', fontsize=14)
    fig.text(0.05, 0.5, 'dB above threshold', va='center', rotation='vertical', fontsize=14)
        
    return fig
