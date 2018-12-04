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
import models.Rattay_2001 as rattay_01
import models.Rattay_adap_2001 as rattay_adap_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Briaire_adap_2005 as briaire_adap_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Imennov_adap_2009 as imennov_adap_09
import models.Negm_2014 as negm_14
import models.Negm_ANF_2014 as negm_ANF_14
import models.Rudnicki_2018 as rudnicki_18

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

# =============================================================================
#  Plot dynamic range over fiber indexes (colored)
# =============================================================================
def dyn_range_color_plot(plot_name,
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
    model_name = eval("{}.display_name".format(spike_table["model_name"].iloc[0]))
    fig.suptitle('{}'.format(model_name), fontsize=16)
    
    ##### no grid
    axes.grid(False)
    
    if 'soma_middle_dist' in spike_table.columns:
        ##### create color map
        basic_cols=['#006837', '#ffffbf', '#a50026']
        cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
        
        ##### adjust cmap that middle of diverging colors is at soma
        endpoint = max(spike_table["first_spike_dist"])
        midpoint = spike_table["soma_middle_dist"].iloc[0]/endpoint
        cmap = calc.shiftedColorMap(cmap, midpoint=midpoint, name='shifted')
        
    else:
        cmap = 'YlGnBu'
    
    ##### create x and y mesh
    dynamic_ranges = pd.unique(spike_table["dynamic_range"].sort_values())
    neuron_numbers = pd.unique(spike_table["neuron_number"].sort_values())
    xmesh, ymesh = np.meshgrid(neuron_numbers, dynamic_ranges)
    
    ##### get the corresponding first spike distance for each x and y value
    distances = spike_table.pivot_table(index="dynamic_range", columns="neuron_number", values="first_spike_dist", fill_value=0).as_matrix()
    distances[distances == 0] = 'nan'
    
    ###### show spiking fibers depending on stimulus amplitude
    color_mesh = axes.pcolormesh(xmesh, ymesh, distances, cmap = cmap, norm = Normalize(vmin = 0, vmax = max(spike_table["first_spike_dist"])))
    clb = fig.colorbar(color_mesh)
    
    ##### define axes ranges
    axes.set_xlim([0,400])
    
    ##### change y-achses to dynamic range
    axes.set_yticklabels(['{} dB'.format(y) for y in axes.get_yticks()])
    
    if 'soma_middle_dist' in spike_table.columns:
        
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
    axes.set_xlabel("Fiber index", fontsize=14)
    axes.set_ylabel("Dynamic range", fontsize=14)

