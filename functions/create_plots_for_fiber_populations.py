# =============================================================================
# This script includes all plot functions for the tests that are part of the 
# "Model analyses" script and for the comparisons in the "Model_comparison"
# sript. The plots usually compare the results of multiple models. In some cases
# the plots show also experimental data.
# =============================================================================
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)

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
    fig, (ax0, ax1) = plt.subplots(1,2, sharex=False, sharey=True, num = plot_name, gridspec_kw = {'width_ratios':[3, 1.25]}, figsize=(12, 7))
    
    ##### define axes ranges
    ax0.set_ylim([y_min,y_max])
    
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
    # plot spikes
    ax0.scatter(spike_trains["spikes"]*1e3, spike_trains["neuron_number"], color = "black", s = 2)
    # get labels for the axes    
    ax0.set_xlabel("Time / ms", fontsize=14)
    ax0.set_ylabel("Nerve fiber number", fontsize=14)
    
    ##### discharge rate plot
    # no grid
    ax1.grid(False)
    # calculate bin edges
    bin_edges = [ii*bin_width+0.5*bin_width for ii in range(-1,nof_bins)]
    # calculate bin heights
    bin_heights = [len(spike_trains[spike_trains["neuron_number"] == ii]) / max(spike_trains["duration"]) for ii in range(nof_bins+1)]
    # define x-axes range
    x_min = 0
    x_max = int(max(bin_heights)) + 2
    ax1.set_xlim([x_min,x_max])
    # create barplot
    ax1.barh(y = bin_edges, width = bin_heights, height = bin_width, color = "black", linewidth=0.3)
    # get labels for the axes
    ax1.set_xlabel("Firing rate (discharges/s)", fontsize=14)
    # no y-achses
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.tick_params(axis = 'both', left = 'off')
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(wspace=0)
    
# =============================================================================
#  Raster plot, showing spiketimes of fibers
# =============================================================================
def raster_plot_comparison(plot_name,
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
    
    ##### get model names
    models = spike_trains["model_name"].unique().tolist()
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of rows
    nof_rows = np.ceil(len(models)/nof_cols).astype(int)
    
    ##### get number of plots
    nof_plots = len(models)
    
    ##### get axes ranges
    y_min = min(spike_trains["neuron_number"])-5
    y_max = max(spike_trains["neuron_number"])+5
    x_min = 0
    x_max = int(max(spike_trains["duration"])*1e3)
    
    ##### delete rows whith no spike
    spike_trains = spike_trains[pd.to_numeric(spike_trains['spikes'], errors='coerce').notnull()]
        
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=False, num = plot_name, figsize=(6*nof_cols, 6*nof_rows))
    
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
             
        ##### turn off y-ticks and labels for all but the left plots
        if (col != 0) and (ii < nof_plots):  
             plt.setp(axes[row][col].get_yticklabels(), visible=False)
             axes[row][col].tick_params(axis = 'both', left = 'off')
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot voltage courses
        if ii < nof_plots:
            
            model = models[ii]
                
            ##### building a subset
            current_data = spike_trains[spike_trains["model_name"] == model]
                                      
            ##### plot spikes
            axes[row][col].scatter(current_data["spikes"]*1e3, current_data["neuron_number"], color = "black", s = 0.1)
                
            ##### no grid
            axes[row][col].grid(False)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    
    ##### get labels for the axes
    fig.text(0.5, 0.06, 'Time / ms', ha='center', fontsize=14)
    fig.text(0.05, 0.5, 'Nerve fiber number', va='center', rotation='vertical', fontsize=14)
        
    return fig