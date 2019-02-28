# =============================================================================
# This script includes all plot functions for fiber populations, both for single
# models and model comparisons
# =============================================================================
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from string import ascii_uppercase as letters
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
#  Raster plot, showing spike times of fibers (for one model)
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
        dynamic_range = np.round(20*np.log10(stim_amp_max_spikes/stim_amp_min_spikes),1)
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
    spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/stim_amp_min_spikes)
        
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
    spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/stim_amp_min_spikes)
        
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

# =============================================================================
#  Dynamic range comparison
# =============================================================================
def nof_spikes_over_stim_amp_comparison(plot_name,
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
    
    ##### get model names
    models = spike_table["model_name"].unique().tolist()
    models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]
    
    ##### get electrodes
    electrodes = spike_table["elec_nr"].unique().tolist()
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of rows
    nof_rows = np.ceil(len(models)/nof_cols).astype(int)
    
    ##### get number of plots
    nof_plots = len(models)
    
    ##### initialize maximum value for dB above threshold
    max_dB = 0
    
    ##### define colors
    colors = ["#1500ff","#5a3ee7","#705cd3","#7b73c1","#7b73c1","#8086b1","#8195a5","#81a29a","#7eb28a","#78c379","#69da5e","#45f52e"]
    #colors = ["#0026ff","#4123f6","#611eea","#761adf","#950fc7","#b200a9","#c30092","#d50075","#dc0068","#e60054","#f0003b","#f70026"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=True, sharey=True, num = plot_name, figsize=(5*nof_cols, 2.3*nof_rows))
    
    ##### create plots  
    for ii in range(nof_rows*nof_cols):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        col = ii-row*nof_cols
        
        ##### turn off x-labels for all but the bottom plots
        if (nof_plots - ii) > nof_cols:
             plt.setp(axes[row][col].get_xticklabels(), visible=False)
             axes[row][col].tick_params(axis = "both", bottom = "off")
        
        ##### turn off y-labels for all but the bottom plots
        if (col != 0) and (ii < nof_plots):  
             plt.setp(axes[row][col].get_yticklabels(), visible=False)
             axes[row][col].tick_params(axis = "both", left = "off")
            
        ##### remove further subplots that are not needed
        if ii > nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot number of spiking fibers over stim amp
        if ii < nof_plots:
            
            model = models[ii]
                
            ##### building subsets
            current_data = spike_table[spike_table["model_name"] == model]
            
            ##### loop over electrodes
            for jj, electrode in enumerate(electrodes):
                
                ##### build subset
                current_data = spike_table[(spike_table["model_name"] == model) & (spike_table["elec_nr"] == electrode)]
                
                ##### calculate dB above threshold
                stim_amp_min_spikes = max(current_data["stim_amp"][current_data["nof_spikes"] == min(current_data["nof_spikes"])])
                current_data["dB_above_thr"] = 20*np.log10(current_data["stim_amp"]/stim_amp_min_spikes)
                
                ##### update max_dB
                max_dB = max(max_dB, max(current_data["dB_above_thr"]))
                
                ##### plot curves
                axes[row][col].plot(current_data["dB_above_thr"], current_data["nof_spikes"], color = colors[jj], label = electrode+1)
                
                ##### mark dynamic range
                db_all_fibers_spike = current_data["dB_above_thr"][current_data["nof_spikes"] == 400]
                if len(db_all_fibers_spike) > 0:
                    axes[row][col].scatter(min(db_all_fibers_spike), 400, color = colors[jj], marker = "|", label = "_nolegend_")
                
            ##### remove top and right lines
            axes[row][col].spines['top'].set_visible(False)
            axes[row][col].spines['right'].set_visible(False)
                
            ##### write model name in plots
            axes[row][col].text(2, 430, eval("{}.display_name".format(model)))
                
            ##### no grid
            axes[row][col].grid(True, alpha = 0.5)
            
        ##### add legend to first plots per column
        if ii == 0:
            legend = axes[row][col].legend(ncol=2 ,title='Electrode Number:', fontsize=8.5)
            plt.setp(legend.get_title(),fontsize=9.5)
    
    ##### define axes ranges
    #plt.gca().set_xlim(left = 0)
    plt.gca().set_xlim([0,max_dB])
    plt.gca().set_ylim([0,470])
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    ##### get labels for the axes
    fig.text(0.5, 0.0, 'dB above threshold', ha='center', fontsize=13)
    fig.text(0.053, 0.5, 'Number of spiking fibers', va='center', rotation='vertical', fontsize=13)
    
    return fig

# =============================================================================
#  Raster plot comparison
# =============================================================================
def raster_plot_comparison(plot_name,
                           spike_table):
    """This function plots thresholds for pulse trains over different durations
    and pulse rates. There is one plot for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    threshold_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "threshold (uA)" 
        - "number of pulses"
        - "pulses per second"
        - "model"
                
    Returns
    -------
    figure with thresholds per pulse train comparison
    """
    
    ##### get lenth of spiral lamina
    length_lamina = max(spike_table["dist_along_sl"])
    
    ##### list electrode positions
    electrode_positions = [4.593, 7.435, 9.309, 11.389, 13.271, 15.164, 16.774, 18.522, 20.071, 21.364, 22.629, 23.649]
    
    ##### initializations
    nof_bins = max(spike_table["neuron_number"])
    bin_width = length_lamina / nof_bins
    
    ##### get x-axes range
    x_min = 0
    x_max = spike_table["duration"].iloc[0]*1e3 *1.05
    
    ##### get y-axes range
    y_min = -2
    y_max = spike_table["max_dist_along_sl"].iloc[0]*1.02

    ##### get model names
    models = spike_table["model_name"].unique().tolist()
    models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]
    
    ##### get pulse rates
    pulse_rates = spike_table["pulse_rate"].unique().tolist()   
    
    ##### get electrode number
    elec_nr = spike_table["elec_nr"].iloc[0]
    
    ##### get number of rows
    nof_rows = len(models)
              
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, 5, sharex = "col", sharey=True, num = plot_name,
                             gridspec_kw = {'width_ratios':[3,1,0.6,3,1]}, figsize=(8.5, 2.4*nof_rows))
    
    ##### loop over figure type
    for ii, pulse_rate in enumerate(pulse_rates):
    
        ##### loop over models 
        for jj, model in enumerate(models):
            
            ##### building a subset for current model and pulse rate
            current_model = spike_table[(spike_table["model_name"] == model) & (spike_table["pulse_rate"] == pulse_rate)]
            
            ##### turn off x-labels for all but the bottom plots
            if jj < nof_rows-1:
                 plt.setp(axes[jj][ii*3].get_xticklabels(), visible=False)
                 axes[jj][ii*3].tick_params(axis = "both", bottom = "off")
                 axes[jj][ii*3+1].tick_params(axis = "both", bottom = "off")
            
            ##### raster plot
            # no grid
            axes[jj][ii*3].grid(False)
            # define x-axes range
            axes[jj][ii*3].set_xlim([x_min,x_max])
            # define y axes ranges
            axes[jj][ii*3].set_ylim([y_min,y_max])
            # plot spikes
            axes[jj][ii*3].scatter(current_model["spikes"]*1e3, current_model["dist_along_sl"], color = "black", s = 0.1)
            # add labels to second raster plot
            if ii==1: axes[jj][ii*3].tick_params(axis = 'y', left = 'on', right = "off", labelleft = True)
            
            ##### firing efficiency plot
            # no grid
            axes[jj][ii*3+1].grid(False)
            # calculate bin edges
            bin_edges = [ii*bin_width+0.5*bin_width for ii in range(0,nof_bins+1)]
            # normalize bin edges for length of lamina
            bin_edges = [ii/max(bin_edges)*length_lamina for ii in bin_edges]
            # calculate bin heights
            bin_heights = [len(current_model[current_model["neuron_number"] == ii]) / current_model["nof_pulses"].iloc[0] * 0.1/spike_table["duration"].iloc[0] for ii in range(nof_bins+1)]
            # define x-axes range
            x_min_fire_eff = 0
            x_max_fire_eff = 1.1 #max(bin_heights)*1.1
            axes[jj][ii*3+1].set_xlim([x_min_fire_eff,x_max_fire_eff])
            # define y axes ranges
            axes[jj][ii*3+1].set_ylim([y_min,y_max])
            # create barplot
            axes[jj][ii*3+1].barh(y = bin_edges, width = bin_heights, height = bin_width, color = "black", linewidth=0.3, edgecolor = "none")
            # write spiking efficiences as percentage
            vals = (axes[jj][ii*3+1].get_xticks() * 100).astype(int)
            axes[jj][ii*3+1].set_xticklabels(['{}%'.format(x) for x in vals])
            # no ticks and label on right side
            axes[jj][ii*3+1].tick_params(axis = 'y', left = 'off', right = "off")
            
            ##### add electrode position
            axes[jj][ii*3].scatter(-1.5, electrode_positions[elec_nr], color = "black", marker = ">", label = "_nolegend_", clip_on=False)
            
    ##### further adjustments
    for ii in range(nof_rows):
        ##### remove subplots in the middle
        axes[ii][2].set_axis_off()
        ##### add letters for models in free space
        axes[ii][1].yaxis.set_label_position("right")
        axes[ii][1].set_ylabel(letters[ii], fontsize=15, fontweight = "bold", rotation = 0)
        axes[ii][1].yaxis.set_label_coords(1.28,0.58)
        ##### defining y ticks
        axes[ii][0].set_yticks([0,5,10,15,20])
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0, wspace=0.1)
    
    ##### write pulse rate over plots
    fig.text(0.28, 0.9, '100 pps', va='center', fontsize=14)
    fig.text(0.68, 0.9, '1000 pps', va='center', fontsize=14)
    
    ##### get labels for the axes
    axes[nof_rows-1][0].set_xlabel('Time / ms', fontsize=12)
    axes[nof_rows-1][3].set_xlabel('Time / ms', fontsize=12)
    axes[nof_rows-1][1].set_xlabel('Firing efficiency', fontsize=12)
    axes[nof_rows-1][4].set_xlabel('Firing efficiency', fontsize=12)
    fig.text(0.06, 0.5, 'Distance along spiral lamina / mm', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
# Plots dB above threshold over distance along spiral lamina for cathodic,
# anodic and biphasic pulses and compare responses for mulitplie electrodes
# =============================================================================
def compare_pulse_forms_for_multiple_electrodes(plot_name,
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
    models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]
    
    ##### get electrodes
    electrodes = spike_table["elec_nr"].unique().tolist()
    
    ##### get pulse forms
    pulse_forms = np.sort(spike_table["pulse_form"].unique()).tolist()
    
    ##### define number of columns and rows
    nof_cols = len(models)
    nof_rows = len(electrodes)
    
    ##### list electrode positions
    electrode_positions = [4.593, 7.435, 9.309, 11.389, 13.271, 15.164, 16.774, 18.522, 20.071, 21.364, 22.629, 23.649]
    
    ##### define y-axis range
    y_min = 0
    y_max = max(spike_table["dist_along_sl"])
    x_min = min(spike_table["dynamic_range"])-0.2
    x_max = max(spike_table["dynamic_range"])
    
    ##### define colors
    colors = ["red", "black", "blue"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex="col", sharey=True, num = plot_name, figsize=(7, 9.5))
    
    ##### loop over models and electrodes
    for ii, elec_nr in enumerate(electrodes):
        for jj, model in enumerate(models):
            
            ##### define x-axis range
            x_min = min(spike_table["dynamic_range"][spike_table["model_name"] == model])-0.2
            x_max = max(spike_table["dynamic_range"][spike_table["model_name"] == model])
            
            ##### set axes ranges
            axes[ii][jj].set_ylim([y_min,y_max])
            axes[ii][jj].set_xlim([x_min,x_max])
            
            ##### loop over pulse forms
            for kk,pulse_form in enumerate(pulse_forms):
                
                ##### build a subset
                current_data = spike_table[(spike_table["model_name"] == model) & (spike_table["elec_nr"] == elec_nr) & (spike_table["pulse_form"] == pulse_form)]
                
                ##### plot graphs
                axes[ii][jj].plot(current_data["dynamic_range"], current_data["dist_along_sl"], color = colors[kk], label = pulse_form)
            
            ##### add electrode position
            axes[ii][jj].scatter(-1/20 * max(current_data["dynamic_range"]), electrode_positions[elec_nr], color = "black", marker = ">", label = "_nolegend_", clip_on=False, s = 40)
            
            ##### define x and y ticks
            axes[ii][jj].set_xticks([0,10,20])
            axes[ii][jj].set_yticks([0,5,10,15,20])
            
#            ##### change y-achses to dynamic range
#            axes[ii][jj].set_yticklabels(['{} dB'.format(y) for y in axes[ii][jj].get_yticks()])
            
            ##### add grid
            axes[ii][jj].grid(True)
            
            ##### write model name as column headers
            if ii == 0:
                axes[ii][jj].set_title(eval("{}.display_name".format(model)), fontsize = 9)
    
    #### add legend above plots
    axes[0][0].legend(ncol=3, loc=(0.55,1.15))
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.15)
    
    ##### get labels for the axes
    fig.text(0.5, 0.057, 'dB above total threshold', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Distance along spiral lamina / mm', va='center', rotation='vertical', fontsize=12)
    
    return fig

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked location of spike initiation for one electrode
# =============================================================================
def spikes_color_plot_comparison(plot_name,
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
    models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]
    
    ##### get electrode number
    elec_nr = spike_table["elec_nr"].iloc[0]
    
    ##### define number rows
    nof_rows = len(models)
    
    ##### list electrode positions
    electrode_positions = [4.593, 7.435, 9.309, 11.389, 13.271, 15.164, 16.774, 18.522, 20.071, 21.364, 22.629, 23.649]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, 2, sharex="col", sharey=False, num = plot_name, gridspec_kw = {'width_ratios':[25,1]}, figsize=(7, 2.2*nof_rows))
    
    ##### loop over models and electrodes
    for ii, model_name in enumerate(models):
            
            ##### build a subset
            current_data = spike_table[spike_table["model_name"] == model_name]
        
            ##### get model module
            model = eval(current_data["model_name"].iloc[0])
            
            if hasattr(model, "index_soma"):
                ##### create color map
                basic_cols=['#006837', '#ffffbf', '#a50026']
                cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
                
                ##### adjust cmap that middle of diverging colors is at soma
                endpoint = max(current_data["first_spike_dist"]) #model.length_neuron/mm
                midpoint = (np.cumsum(model.compartment_lengths)[model.middle_comp_soma]/mm)/endpoint
                cmap = calc.shiftedColorMap(cmap, midpoint=midpoint, name='shifted')
                
                ##### give soma an extra color
                color_res = cmap.N # resolution of cmap
                if hasattr(model, "length_soma"):
                    soma_length = model.length_soma
                else:
                    soma_length = model.diameter_soma / mm
                soma_range = int(np.ceil(soma_length/max(current_data["first_spike_dist"])*color_res))
                start_point = int((np.cumsum(model.compartment_lengths)[model.start_index_soma]/mm)/endpoint*color_res)
                for jj in range(start_point, start_point + soma_range):
                    cmap_list = [cmap(kk) for kk in range(cmap.N)]
                    cmap_list[jj] = LinearSegmentedColormap.from_list('mycmap', ['#feff54','#feff54'])(0)
                    cmap = cmap.from_list('Custom cmap', cmap_list, cmap.N)
                
            else:
                midpoint = max(current_data["first_spike_dist"]) / 2
                cmap = LinearSegmentedColormap.from_list('mycmap', ['#ffffbf', '#a50026'])
            
            ##### create x and y mesh
            dynamic_ranges = pd.unique(current_data["dynamic_range"].sort_values())
            distances_sl = pd.unique(current_data["dist_along_sl"].sort_values())
            xmesh, ymesh = np.meshgrid(distances_sl, dynamic_ranges)
            
            ##### get the corresponding first spike distance for each x and y value
            distances = current_data.pivot_table(index="dynamic_range", columns="dist_along_sl", values="first_spike_dist", fill_value=0).as_matrix()
            distances[distances == 0] = 'nan'
            
            ###### show spiking fibers depending on stimulus amplitude
            color_mesh = axes[ii][0].pcolormesh(xmesh, ymesh, distances, cmap = cmap, norm = Normalize(vmin = 0, vmax = max(current_data["first_spike_dist"])),linewidth=0,rasterized=True)
            
            ##### show colorbar
            clb = plt.colorbar(color_mesh, cax = axes[ii][1])
            
            ##### define axes ranges
            axes[ii][0].set_xlim([0,max(current_data["dist_along_sl"])])
            axes[ii][0].set_ylim([0,max(current_data["dynamic_range"])])
            
            #### add electrode position
            axes[ii][0].scatter(electrode_positions[elec_nr], -0.6, clip_on=False, color = "black", marker = "^", label = "_nolegend_", s = 40)        
            
            ##### change clb ticks and labels
            if hasattr(model, "index_soma"):
                soma = endpoint*midpoint
                dendrite = soma*0.25
                axon = soma + (endpoint-soma)*0.75
                clb.set_ticks([dendrite, soma, axon])
                clb.set_ticklabels(["dendrite","soma","axon"])
            else:
                clb.set_ticks([midpoint])
                clb.set_ticklabels(["axon"])
                  
            ##### write model names in plot
            if elec_nr >= 5: x_pos = 0.5
            else: x_pos = 15
            axes[ii][0].text(x_pos, 1, model.display_name, fontsize=11)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    ##### get labels for the axes
    axes[nof_rows-1][0].set_xlabel('Distance along spiral lamina / mm', fontsize=14)
#    axes[nof_rows-1][1].set_xlabel('Position of first spike', fontsize=14)
    fig.text(0.04, 0.5, 'dB above threshold', va='center', rotation='vertical', fontsize=14)
    
    return fig

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked location of spike initiation for multiple electrodes
# =============================================================================
def spikes_color_plot_comparison_multiple_electrodes(plot_name,
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
    models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]
    
    ##### get electrode number
    electrodes = spike_table["elec_nr"].unique().tolist()
    
    ##### define number rows
    nof_cols = len(models)
    
    ##### define number rows
    nof_rows = len(electrodes)
    
    ##### list electrode positions
    electrode_positions = [4.593, 7.435, 9.309, 11.389, 13.271, 15.164, 16.774, 18.522, 20.071, 21.364, 22.629, 23.649]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows+2, nof_cols, sharex=False, sharey="row", num = plot_name, gridspec_kw = {'height_ratios':[25]*nof_rows + [5] + [1]}, figsize=(7, 9.5))
    
    ##### loop over models and electrodes
    for ii, elec_nr in enumerate(electrodes):
        for jj, model_name in enumerate(models):
            
            ##### build a subset for current model
            current_model = spike_table[(spike_table["model_name"] == model_name) & (spike_table["elec_nr"] == elec_nr)]
            
            ##### define x-axis range
            x_max = max(current_model["dynamic_range"])
            
            ##### build a subset for current electrode
            current_data = current_model[current_model["elec_nr"] == elec_nr]
        
            ##### get model module
            model = eval(current_data["model_name"].iloc[0])
            
            if hasattr(model, "index_soma"):
                ##### create color map
                basic_cols=['#006837', '#feff54', '#a50026'] #006837 #ffffbf #a50026
                cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
                
                ##### adjust cmap that middle of diverging colors is at soma
                endpoint = max(current_data["first_spike_dist"]) #model.length_neuron/mm
                midpoint = (np.cumsum(model.compartment_lengths)[model.middle_comp_soma]/mm)/endpoint
                cmap = calc.shiftedColorMap(cmap, midpoint=midpoint, name='shifted')
                
                ##### give soma an extra color
                color_res = cmap.N # resolution of cmap
                if hasattr(model, "length_soma"):
                    soma_length = model.length_soma
                else:
                    soma_length = model.diameter_soma / mm
                soma_range = int(np.ceil(soma_length/max(current_data["first_spike_dist"])*color_res))
                start_point = int((np.cumsum(model.compartment_lengths)[model.start_index_soma]/mm)/endpoint*color_res)
                for kk in range(start_point, start_point + soma_range):
                    cmap_list = [cmap(ll) for ll in range(cmap.N)]
                    cmap_list[kk] = LinearSegmentedColormap.from_list('mycmap', ['#FFFF00','#FFFF00'])(0) #feff54 #feff54
                    cmap = cmap.from_list('Custom cmap', cmap_list, cmap.N)
                
            else:
                midpoint = max(current_data["first_spike_dist"]) / 2
                cmap = LinearSegmentedColormap.from_list('mycmap', ['#feff54', '#a50026'])
            
            ##### create x and y mesh
            dynamic_ranges = pd.unique(current_data["dynamic_range"].sort_values())
            distances_sl = pd.unique(current_data["dist_along_sl"].sort_values())
            xmesh, ymesh = np.meshgrid(distances_sl, dynamic_ranges)
            
            ##### get the corresponding first spike distance for each x and y value
            distances = current_data.pivot_table(index="dynamic_range", columns="dist_along_sl", values="first_spike_dist", fill_value=0).as_matrix()
            distances[distances == 0] = 'nan'
            
            ###### show spiking fibers depending on stimulus amplitude
            color_mesh = axes[ii][jj].pcolormesh(ymesh, xmesh, distances, cmap = cmap, norm = Normalize(vmin = 0, vmax = max(current_data["first_spike_dist"])),linewidth=0,rasterized=True)
            
            if ii == 0:
                ##### show colorbar
                clb = plt.colorbar(color_mesh, cax = axes[nof_rows+1][jj], orientation = "horizontal")
                
                ##### change clb ticks and labels
                if hasattr(model, "index_soma"):
                    soma = endpoint*midpoint
                    dendrite = soma*0.25
                    axon = soma + (endpoint-soma)*0.75
                    clb.set_ticks([dendrite, soma, axon])
                    clb.ax.set_xticklabels(["dendrite","soma","axon"], rotation=45, fontsize=10)
                    clb.ax.tick_params(axis='both', which='major', pad=-3)
                else:
                    clb.set_ticks([midpoint])
                    clb.ax.set_xticklabels(["axon"], rotation=45, fontsize=10)
                    clb.ax.tick_params(axis='both', which='major', pad=-3)
                
                ##### write model names in plot
                axes[ii][jj].set_title(model.display_name, fontsize=9)
                
                ##### remove subplots before colormap (was just used to get space)
                axes[nof_rows][jj].set_axis_off()
            
            ##### define axes ranges
            axes[ii][jj].set_xlim([0,x_max])
            axes[ii][jj].set_ylim([0,max(current_data["dist_along_sl"])-0.1])
            
            ##### turn off x-labels for all but the bottom plots
            if ii != nof_rows-1:
                 plt.setp(axes[ii][jj].get_xticklabels(), visible=False)
                 axes[ii][jj].tick_params(axis = "both", bottom = "off")
            
            #### add electrode position
            axes[ii][jj].scatter(-1/20 * max(current_data["dynamic_range"]), electrode_positions[elec_nr], clip_on=False, color = "black", marker = ">", label = "_nolegend_", s = 40)        
            
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.15)
    
    ##### get labels for the axes
    fig.text(0.5, 0.134, 'dB above threshold', ha='center', fontsize=12)
    fig.text(0.5, 0.044, 'Location of first AP', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Distance along spiral lamina / mm', va='center', rotation='vertical', fontsize=12)
    
    return fig

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked latency of spike initiation for one electrode
# =============================================================================
def latencies_color_plot_comparions(plot_name,
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
    models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]
    
    ##### get electrode number
    elec_nr = spike_table["elec_nr"].iloc[0]
    
    ##### define number rows
    nof_cols = len(models)
    
    ##### list electrode positions
    electrode_positions = [4.593, 7.435, 9.309, 11.389, 13.271, 15.164, 16.774, 18.522, 20.071, 21.364, 22.629, 23.649]
    
    ##### define color map, discretisize it and cut of last color (almost white)
    cmap = plt.cm.get_cmap("CMRmap_r",20)
    cmaplist = [cmap(ii) for ii in range(3,cmap.N-3)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N-6)
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(3, nof_cols, sharex=False, sharey="row", num = plot_name, gridspec_kw = {'height_ratios':[32,7,1.2]}, figsize=(7, 4))
    fig.subplots_adjust(bottom=0.15)
    
    ##### loop over models and electrodes
    for ii, model in enumerate(models):
        
        ##### build a subset
        current_data = spike_table[spike_table["model_name"] == model]
        
        ##### create x and y mesh
        dynamic_ranges = pd.unique(current_data["dynamic_range"].sort_values())
        distances_sl = pd.unique(current_data["dist_along_sl"].sort_values())
        xmesh, ymesh = np.meshgrid(distances_sl, dynamic_ranges)
        
        ##### get the corresponding first spike distance for each x and y value
        latencies = current_data.pivot_table(index="dynamic_range", columns="dist_along_sl", values="latency", fill_value=0).as_matrix().astype(float)
        latencies[latencies == 0] = 'nan'
        
        ###### show spiking fibers depending on stimulus amplitude
        color_mesh = axes[0][ii].pcolormesh(ymesh, xmesh, latencies, cmap = cmap, linewidth=0, vmax = max(current_data["latency"]), rasterized=True)
        
        ##### define axes ranges
        axes[0][ii].set_xlim([0,max(current_data["dist_along_sl"])])
        
        ##### show colorbar
        clb = plt.colorbar(color_mesh, cax = axes[2][ii], orientation = "horizontal")
        clb.ax.locator_params(nbins=3)
        
        ##### write model names in plot
        axes[0][ii].set_title(eval("{}.display_name".format(model)), fontsize=9)
        
        ##### remove subplots before colormap (was just used to get space)
        axes[1][ii].set_axis_off()
        
        ##### define axes ranges
        axes[0][ii].set_xlim([0,max(current_data["dynamic_range"])])
        axes[0][ii].set_ylim([0,max(current_data["dist_along_sl"])-0.1])
        
        #### add electrode position
        axes[0][ii].scatter(-1/20 * max(current_data["dynamic_range"]), electrode_positions[elec_nr], clip_on=False, color = "black", marker = ">", label = "_nolegend_", s = 40)        
        
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.15)
    
    ##### get labels for the axes
    fig.text(0.5, 0.2, 'dB above threshold', ha='center', fontsize=12)
    fig.text(0.5, 0.03, 'AP latency / ms', ha='center', fontsize=12)
    fig.text(0.041, 0.58, 'Distance along spiral lamina / mm', va='center', rotation='vertical', fontsize=12)
    
    return fig




