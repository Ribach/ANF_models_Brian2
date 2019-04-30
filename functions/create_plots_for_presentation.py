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
#  Raster plot comparison
# =============================================================================
def raster_plot_comparison_presentation(plot_name,
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
    
    ##### get electrode number
    elec_nr = spike_table["elec_nr"].iloc[0]
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of rows
    nof_rows = np.ceil(len(models)/nof_cols).astype(int)
              
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols*2+1, sharex = "col", sharey=True, num = plot_name,
                             gridspec_kw = {'width_ratios':[3,1,0.2,3,1]}, figsize=(12,7))
    
    ##### loop over models 
    for ii, model in enumerate(models):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        if ii/nof_cols == np.round(ii/nof_cols):
            col = 0
        else:
            col = 3
        
        ##### building a subset for current model
        current_model = spike_table[spike_table["model_name"] == model]
        
        ##### raster plot
        # no grid
        axes[row][col].grid(False)
        # define x-axes range
        axes[row][col].set_xlim([x_min,x_max])
        # define y axes ranges
        axes[row][col].set_ylim([y_min,y_max])
        # plot spikes
        axes[row][col].scatter(current_model["spikes"]*1e3, current_model["dist_along_sl"], color = "black", s = 0.1)
        # add labels to second raster plot
        if col==2: axes[row][col].tick_params(axis = 'y', left = 'on', right = "off", labelleft = True)
        
        ##### firing efficiency plot
        # add grid
        axes[row][col+1].grid(True, axis = "x", alpha = 0.5)
        # calculate bin edges
        bin_edges = [ll*bin_width+0.5*bin_width for ll in range(0,nof_bins+1)]
        # normalize bin edges for length of lamina
        bin_edges = [ll/max(bin_edges)*length_lamina for ll in bin_edges]
        # calculate bin heights
        bin_heights = [len(current_model[current_model["neuron_number"] == ll]) / current_model["nof_pulses"].iloc[0] * 0.1/spike_table["duration"].iloc[0] for ll in range(nof_bins+1)]
        # define x-axes range
        x_min_fire_eff = 0
        x_max_fire_eff = 1.1 #max(bin_heights)*1.1
        axes[row][col+1].set_xlim([x_min_fire_eff,x_max_fire_eff])
        # set x-ticks
        axes[row][col+1].set_xticks([0,0.5,1])
        # define y-axes ranges
        axes[row][col+1].set_ylim([y_min,y_max])
        # create barplot
        axes[row][col+1].barh(y = bin_edges, width = bin_heights, height = bin_width, color = "black", linewidth=0.3, edgecolor = "none")
        # write spiking efficiences as percentage
        vals = (axes[row][col+1].get_xticks() * 100).astype(int)
        axes[row][col+1].set_xticklabels(['{}%'.format(x) for x in vals])
        # no ticks and label on right side
        axes[row][col+1].tick_params(axis = 'y', left = 'off', right = "off")
        
        ##### add electrode position
        axes[row][col].scatter(-1/40 * max(current_model["spikes"])*1e3, electrode_positions[elec_nr], color = "black", marker = ">", label = "_nolegend_", clip_on=False, s=100)
        
        ##### add model name
        axes[row][col].text((x_max-x_min)/2.5, y_max + 1, eval("{}.display_name".format(model)), fontsize=14)
        
    ##### further adjustments
    for ii in range(nof_rows):
        ##### remove subplots in the middle
        axes[ii][2].set_axis_off()
        ##### defining y ticks
        axes[ii][0].set_yticks([0,5,10,15,20])
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    
    ##### get labels for the axes
    axes[nof_rows-1][0].set_xlabel('Time / ms', fontsize=14)
    axes[nof_rows-1][3].set_xlabel('Time / ms', fontsize=14)
    axes[nof_rows-1][1].set_xlabel('Firing efficiency', fontsize=14)
    axes[nof_rows-1][4].set_xlabel('Firing efficiency', fontsize=14)
    fig.text(0.07, 0.5, 'Distance along spiral lamina / mm', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked location of spike initiation for multiple electrodes
# =============================================================================
def spikes_color_plot_comparison_presentation(plot_name,
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
    fig, axes = plt.subplots(nof_rows+2, nof_cols, sharex=False, sharey="row", num = plot_name, gridspec_kw = {'height_ratios':[25]*nof_rows + [6] + [1]}, figsize=(12, 7))
    
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
                    clb.ax.set_xticklabels(["dendrite","soma","axon"], rotation=45, fontsize=12)
                    clb.ax.tick_params(axis='both', which='major', pad=-3)
                else:
                    clb.set_ticks([midpoint])
                    clb.ax.set_xticklabels(["axon"], rotation=45, fontsize=12)
                    clb.ax.tick_params(axis='both', which='major', pad=-3)
                
                ##### write model names in plot
                axes[ii][jj].set_title(model.display_name, fontsize=12)
                
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
            axes[ii][jj].scatter(-1/20 * max(current_data["dynamic_range"]), electrode_positions[elec_nr], clip_on=False, color = "black", marker = ">", label = "_nolegend_", s = 70)        
            
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.15)
    
    ##### get labels for the axes
    fig.text(0.5, 0.15, 'dB above threshold', ha='center', fontsize=14)
    fig.text(0.5, 0.002, 'Location of first AP', ha='center', fontsize=14)
    fig.text(0.08, 0.5, 'Distance along spiral lamina / mm', va='center', rotation='vertical', fontsize=14)

    return fig

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked latency of spike initiation for one electrode
# =============================================================================
def latencies_color_plot_comparions_presentation(plot_name,
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

# =============================================================================
#  Single node response voltage course model comparison
# =============================================================================
def single_node_response_comparison_presentation(plot_name,
                                                 voltage_data):
    """This function plots voltage courses for a certain stimulation with one
    plot for each model in the voltage_data dataframe. For more than one run
    per model several lines will be shown in each plot.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    voltage_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "membrane potential (mV)" 
        - "time (ms)"
        - "model"
        - "run
                
    Returns
    -------
    figure with single node response comparison plot
    """
    
    ##### get model names
    models = voltage_data["model"].unique().tolist()
    models = ['Rattay et al. (2001)', 'Briaire and Frijns (2005)', 'Smit et al. (2010)', 'Imennov and Rubinstein (2009)']
    
    ##### define number of columns
    nof_models = len(models)
    
    ##### get axes ranges
    y_min = min(voltage_data["membrane potential (mV)"]) - 5
    y_max = max(voltage_data["membrane potential (mV)"]) + 15
    x_max = 1.4
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, nof_models, sharex=False, sharey=False, num = plot_name, figsize=(12,3))
    fig.subplots_adjust(bottom=0.22)
    
    ##### create plots  
    for ii,model in enumerate(models):
        
        ##### define axes ranges
        axes[ii].set_ylim([y_min,y_max])
        axes[ii].set_xlim([0,x_max])
        
        ##### turn off y-labels for all but the left plots
        if ii != 0:  
             plt.setp(axes[ii].get_yticklabels(), visible=False)
             axes[ii].tick_params(axis = "both", left = "off")
        
        ##### building subsets
        current_data_1th = voltage_data[(voltage_data["model"] == model) & (voltage_data["amplitude level"] == "1*threshold")]
        current_data_2th = voltage_data[(voltage_data["model"] == model) & (voltage_data["amplitude level"] == "2*threshold")]         
        
        ##### plot lines
        axes[ii].plot(current_data_1th["time (ms)"], current_data_1th["membrane potential (mV)"], color = "black", label = r"$1 \cdot I_{\rm{th}}$")
        axes[ii].plot(current_data_2th["time (ms)"], current_data_2th["membrane potential (mV)"], color = "red", label = r"$2 \cdot I_{\rm{th}}$")
        
#        ##### put legend next to plots
#        if ii == len(models)-1:
#            axes[ii].legend(loc = (0.75, 0.4), shadow = False, title = "stimulus amplitude")
            
        ##### remove top and right lines
        axes[ii].spines['top'].set_visible(False)
        axes[ii].spines['right'].set_visible(False)
            
        ##### write model name in plots
        axes[ii].text(x_max*0.05, y_max-10, model, fontsize=12)
            
        ##### no grid
        axes[ii].grid(False)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    ##### get labels for the axes
    fig.text(0.5, 0.04, 'Time / ms', ha='center', fontsize=14)
    axes[0].set_ylabel('Membrane potential / mV', fontsize=14)
    
    return fig

# =============================================================================
#  Strength duration curve model comparison
# =============================================================================
def strength_duration_curve_comparison_presentation(plot_name,
                                                    threshold_data_cat,
                                                    threshold_data_ano,
                                                    strength_duration_table = None):
    """This function plots the model thresholds over the phase length of the stimulus.
    There is one line for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    threshold_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "threshold (uA)" 
        - "phase duration (us)"
        - "model"
    strength_duration_table : pandas dataframe
        This dataframe is optional and marks the chronaxie and rheobase values of
        the models in the plots. If defined, it has to contain the following columns:
        - "chronaxie (us)" 
        - "rheobase (uA)"
        - "model"
                
    Returns
    -------
    figure with conduction velocity comparison
    """
    
    ##### get model names
    models = ['Rattay et al. (2001)', 'Briaire and Frijns (2005)', 'Smit et al. (2010)', 'Imennov and Rubinstein (2009)']
    
    ##### exclude rows, where no threshold was found
    threshold_data_cat = threshold_data_cat.loc[threshold_data_cat["threshold (uA)"] != 0]
    threshold_data_ano = threshold_data_ano.loc[threshold_data_ano["threshold (uA)"] != 0]
    
    ##### exclude rows, whith thresholds higher than 1000 uA
    threshold_data_cat = threshold_data_cat.loc[threshold_data_cat["threshold (uA)"] <= 1000]
    threshold_data_ano = threshold_data_ano.loc[threshold_data_ano["threshold (uA)"] <= 1000]
    
    ##### get y range
    y_min = -0.5
    y_max = 1100
    
    ##### define colors and markers
    colors = ["black","black","red","red"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","red","red"]
    line_styles = [":","-","-",":","-.","-",":","-.","-",":","-.","-"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 2, sharey=True, num = plot_name, figsize=(10, 5))
    fig.subplots_adjust(bottom=0.16)
    
    for ii,polarity in enumerate(["cathodic","anodic"]):
        
        ##### ad grid
        axes[ii].grid(True, which='both', alpha = 0.5)
            
        ##### cathodic pulses:
        for jj, model in enumerate(models):
            
            ##### building a subset
            if polarity == "cathodic":
                current_data = threshold_data_cat[threshold_data_cat["model"] == model]
                label = model
            else:
                current_data = threshold_data_ano[threshold_data_ano["model"] == model]
                label = "_nolegend_"
    
            ##### plot strength duration curve    
            axes[ii].semilogx(current_data["phase duration (us)"], current_data["threshold (uA)"],
                color = colors[jj], linestyle = line_styles[jj], label = "_nolegend_", basex=10)
            
            axes[ii].scatter(current_data["phase duration (us)"].iloc[0], current_data["threshold (uA)"].iloc[0],
                color = colors[jj], marker = markers[jj], s = 40, edgecolor = edgecolors[jj], label = label)
        
            ##### define y axes range
            axes[ii].set_ylim([y_min,y_max])
    
            ##### add labels to x-axes    
            axes[ii].set_xlabel(r'Phase duration / $\rm{\mu s}$', fontsize=14)
    
        ##### use normal values for axes (no powered numbers)
        for axis in [axes[ii].xaxis, axes[ii].yaxis]:
            axis.set_major_formatter(ScalarFormatter())
            
    ##### show legend
    fig.legend(loc = (0.35, 0.64), shadow = False)
    
    ##### show ticks and labels of right plot on right side
    axes[1].tick_params(axis = 'y', left = 'off', right = "on", labelright = True)
    
    #### add titles
    axes[0].set_title("cathodic stimulation")
    axes[1].set_title("anodic stimulation")
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(wspace=0)
    
    ##### add y-axis label
    axes[0].set_ylabel(r'Threshold / $\rm{\mu A}$', fontsize=14)
    
    return fig
        
# =============================================================================
#  Voltage course comparison
# =============================================================================
def voltage_course_comparison_plot_presentation(plot_name,
                                                model_names,
                                                time_vector,
                                                max_comp,
                                                voltage_courses):
    """This function plots the membrane potential of all compartments over time
    as voltage course lines spaced according the real compartment distances. There
    will be one plot for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    model_names : list of strings
        List with strings with the model names in the format of the imported
        modules on top of the script
    time_vector : list of time values
        Vector contains the time points, that correspond to the voltage values
        of the voltage matrices.
    max_comp : list of integers
        defines the maximum number of compartments to show for each model
    voltage_courses : list of matrices of mambrane potentials
        There is one matrix per model. Each matrix has one row for each compartment
        and one columns for each time step. Number of columns has to be the same
        as the length of the time vector
                
    Returns
    -------
    figure with voltage course plots for each model
    """
    
    ##### get models
    models = [eval(model) for model in model_names]
    
    ##### define number of columns
    nof_models = len(models)
     
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, nof_models, sharex=False, sharey=True, num = plot_name, figsize=(14,5))
    fig.subplots_adjust(bottom=0.16)
    
    ##### create plots  
    for ii, model in enumerate(models):
        
        ##### turn off y-labels
        if ii != 0:
            plt.setp(axes[ii].get_yticklabels(), visible=False)
            axes[ii].tick_params(axis = 'both', left = 'off')
        
        ##### get array of compartments to plot
        comps_to_plot = model.comps_to_plot[model.comps_to_plot < max_comp[ii]]
        
        ##### get voltage courses for current model
        voltage_matrix = voltage_courses[ii]
        
        ##### distances between lines and x-axis
        offset = np.cumsum(model.distance_comps_middle)/meter
        offset = (offset/max(offset))*10
        
        ##### plot lines
        for jj in comps_to_plot:
            axes[ii].plot(time_vector/ms, offset[jj] - 1/(30)*(voltage_matrix[jj, :]-model.V_res)/mV, color = "black", linewidth = 0.6)
        
        ##### write model name above plots
        axes[ii].set_title(model.display_name_plots, fontsize=13)
            
        ##### no grid
        axes[ii].grid(False)
    
    ##### invert y-achses
    axes[0].invert_yaxis()
    
    ##### 
    axes[0].set_yticks([6,4,2,0])
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0, wspace=0)
    
    ##### get labels for the axes
    fig.text(0.5, 0.025, 'Time / ms', ha='center', fontsize=14)
    fig.text(0.07, 0.5, 'Position along fiber [major]', va='center', rotation='vertical', fontsize=14)
    fig.text(0.09, 0.5, 'membrane potential [minor]', va='center', rotation='vertical', fontsize=14)
    
    return fig

# =============================================================================
#  PSTH comparison
# =============================================================================
def psth_comparison_presentation(plot_name,
                                 psth_data,
                                 amplitudes = None,
                                 pulse_rates = None,
                                 plot_style = "firing_efficiency"):
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
    models = psth_data["model"].unique().tolist()
    models = ['Rattay et al. (2001)', 'Briaire and Frijns (2005)', 'Smit et al. (2010)', 'Imennov and Rubinstein (2009)']
    
    ##### get amplitude levels and pulse rates
    if amplitudes is None: amplitudes = psth_data["amplitude"].unique().tolist()
    if pulse_rates is None: pulse_rates = psth_data["pulse rate"].unique().tolist()

    ##### get number of different models, pulse rates and stimulus amplitudes
    nof_models = len(models)
    nof_amplitudes = len(amplitudes)
    nof_pulse_rates = len(pulse_rates)
    
    ##### specify bin width (in ms)
    bin_width = 10
    
    ##### get number of runs and bins
    nof_runs = max(psth_data["run"])+1
    nof_bins = int((max(psth_data["spike times (ms)"])+1) / bin_width)

    ##### get bin edges
    bin_edges = [ii*bin_width+0.5*bin_width for ii in range(nof_bins+1)]

    ##### initialize maximum bin height
    max_bin_height = 0
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_models, nof_amplitudes*nof_pulse_rates, sharex=True, sharey=True, num = plot_name, figsize=(12, 7))
    
    ##### loop over models 
    for ii, model in enumerate(models):
        
        ##### loop over amplitudes and pulse rates
        for jj, amplitude in enumerate(amplitudes):
            for kk, pulse_rate in enumerate(pulse_rates):
                
                ##### get number of current column
                col = jj*nof_pulse_rates + kk
                
                ##### turn off x-labels for all but the bottom plots
                if ii < nof_models-1:
                     plt.setp(axes[ii][col].get_xticklabels(), visible=False)
                     axes[ii][col].tick_params(axis = "both", bottom = "off")
                
                ##### turn off y-labels for all but the left plots
                if jj+kk > 0:  
                     plt.setp(axes[ii][col].get_yticklabels(), visible=False)
                     axes[ii][col].tick_params(axis = "both", left = "off")
                
                ##### building a subset of the relevant rows
                current_data = psth_data[(psth_data["amplitude"] == amplitude) & (psth_data["pulse rate"] == pulse_rate) & (psth_data["model"] == model)]
        
                ##### calculating the bin heights
                bin_heights = [sum((bin_width*kk < current_data["spike times (ms)"]) & (current_data["spike times (ms)"] < bin_width*kk+bin_width))/nof_runs for kk in range(0,nof_bins+1)]
                if plot_style == "firing_efficiency":
                    bin_heights = [height / (current_data["pulse rate"].iloc[0]/second * bin_width*ms) for height in bin_heights]
                
                ##### create barplot
                axes[ii][col].bar(x = bin_edges, height = bin_heights, width = bin_width, color = "black", linewidth=0.3)
                
                ##### remove top and right lines
                axes[ii][col].spines['top'].set_visible(False)
                axes[ii][col].spines['right'].set_visible(False)
                
                ##### update max_bin_height
                if round(max(bin_heights)) > max_bin_height:
                    max_bin_height = round(max(bin_heights))
                
                ##### define x-achses range and tick numbers
                axes[ii][col].set_xlim([-10,max(bin_edges)*1.1])
                axes[ii][col].set_xticks([0,max(bin_edges)-0.5*bin_width])
                
                ##### no grid
                axes[ii][col].grid(False) 
            
    ##### further adjustments
    for ii, model in enumerate(models):
        for jj, amplitude in enumerate(amplitudes):
            for kk, pulse_rate in enumerate(pulse_rates):
                
                ##### get number of current column
                col = jj*nof_pulse_rates + kk
                
                #### building a subset of the relevant rows
                current_data = psth_data[(psth_data["amplitude"] == amplitude) & (psth_data["pulse rate"] == pulse_rate) & (psth_data["model"] == model)]
                
                if plot_style == "firing_efficiency":
                    ##### define y-achses range and tick numbers
                    axes[ii][col].set_ylim([0,1.25])
                    axes[ii][col].set_yticks([0,0.5,1])
                
                    ##### Write spiking efficiences as percentage
                    vals = (axes[ii][col].get_yticks() * 100).astype(int)
                    axes[ii][col].set_yticklabels(['{}%'.format(x) for x in vals])
                    
                else:
                    ##### define y-achses range and tick numbers
                    axes[ii][col].set_ylim([0,max_bin_height*1.35])
            
            ##### write model name in first suplot
            axes[ii][jj].text(np.ceil(max(bin_edges)/100), max_bin_height*1.1, model, fontsize=12)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1, wspace=0.15)
    
    ##### use the pulse rates as column titles
    for ax, columtitle in zip(axes[0], ["{} pps".format(pulse_rates[ii]) for ii in range(nof_pulse_rates)]):
        ax.set_title(columtitle, y = 1.1)
    
    ##### get labels for the axes
    fig.text(0.5, 0.045, 'Time after pulse-train onset / ms', ha='center', fontsize=14)
    
    if plot_style == "firing_efficiency":
        fig.text(0.03, 0.5, 'firing efficiency', va='center', rotation='vertical', fontsize=14)
    else:
        fig.text(0.065, 0.5, 'APs per timebin ({} ms)'.format(bin_width), va='center', rotation='vertical', fontsize=14)

    return fig

# =============================================================================
#  Refractory curve comparison
# =============================================================================
def refractory_curves_comparison_presentation(plot_name,
                                              refractory_curves,
                                              model_names):
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
    
    ##### get models
    models = [eval(model) for model in model_names]
    
    ##### define number of columns
    nof_models = len(models)
    
    ##### get axes ranges
    y_max = max(refractory_curves["threshold ratio"]) + 6
    x_min = 0
    x_max = max(refractory_curves["interpulse interval"]) + 0.2
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, nof_models, sharex=False, sharey=True, num = plot_name, figsize=(12, 3))
    fig.subplots_adjust(bottom=0.18)
    
    ##### create plots  
    for ii, model in enumerate(models):
        
        ##### define axes ranges
        axes[ii].set_xlim([x_min,x_max])
             
        ##### turn off y-ticks and labels for all but the left plots
        if ii != 0:  
             plt.setp(axes[ii].get_yticklabels(), visible=False)
             axes[ii].tick_params(axis = 'both', left = 'off')
                
        ##### building a subset
        current_data = refractory_curves[refractory_curves["model"] == model.display_name_plots]
                                  
        ##### plot threshold curve
        axes[ii].set_yscale('log', basey=2)
        axes[ii].plot(current_data["interpulse interval"], current_data["threshold ratio"], color = "black", linewidth = 2)
        
        ##### add line at threshold level
        axes[ii].hlines(y=1, xmin=x_min, xmax=x_max, linestyles="dashed", color = "black")
        
        ##### show points
        axes[ii].scatter(current_data["interpulse interval"], current_data["threshold ratio"], color = "black", marker = "o", alpha  = 0.5, s = 15)
        
        ##### defining y ticks
        axes[ii].set_yticks([1,2,4,8,16])
        
        ##### use normal values for axes (no powered numbers)
        for axis in [axes[ii].xaxis, axes[ii].yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        ##### remove top and right lines
        axes[ii].spines['top'].set_visible(False)
        axes[ii].spines['right'].set_visible(False)
            
        ##### write model name above plots
        axes[ii].set_title(model.display_name_plots, fontsize=13)
            
        ##### no grid
        axes[ii].grid(False)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.2, wspace=0.05)
    
    ##### get labels for the axes
    fig.text(0.5, 0.02, "IPI / ms", ha="center", fontsize=14)
    fig.text(0.07, 0.5, r"$I_{\rm{th}}$ (2nd stimulus) / $I_{\rm{th}}$ (masker)", va="center", rotation="vertical", fontsize=14)

    return fig

# =============================================================================
# Compare stochastic properties for different k_noise values
# =============================================================================
def stochastic_properties_presentation(plot_name,
                                       stochasticity_table):
    """This function plots the relative spread of thresholds over the jitter.
    There is one line for each model connecting the measured points for different
    noise levels (different amounts of noise). An aria in the plot is colored,
    showing the experimental range of measured values.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    stochasticity_table : pandas dataframe
        This dataframe has to contain the following columns:
        - "relative spread (%)" 
        - "jitter (us)"
        - "model"
                
    Returns
    -------
    figure a comparison of the stochastic properties
    """
    
    ##### get model names
    models = stochasticity_table["model"].unique().tolist()
    models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]
    
    ##### define colors and markers
    colors = ["black","black","red","red"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","red","red"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(10,5))
    
    ##### plot experimental range
    axes.fill_between([80,190],[5,5],[12,12], facecolor = "white", hatch = "///", edgecolor="black", label = "Experimental range")
    
    ##### create plots  
    for ii, model in enumerate(models):
                        
        ##### building a subset
        current_data = stochasticity_table[stochasticity_table["model"] == model]
                                  
        ##### plot threshold curve
        axes.plot(current_data["jitter (us)"], current_data["relative spread (%)"], color = colors[ii], label = "_nolegend_")
        
        ##### show points
        axes.scatter(current_data["jitter (us)"], current_data["relative spread (%)"],
                                  color = colors[ii], marker = markers[ii], edgecolor = edgecolors[ii], label = "{}".format(eval("{}.display_name_plots".format(model))))
    
    ##### define axes ranges
    axes.set_ylim([0,30])
    axes.set_xlim([0,200])
                
    ##### add legend
    plt.legend(loc = (0.35,0.69))
    
    ##### Write relative spreads as percentage
    vals = axes.get_yticks().astype(int)
    axes.set_yticklabels(['{}%'.format(x) for x in vals])
        
    ##### no grid
    axes.grid(False)
    
    ##### get labels for the axes
    fig.text(0.5, 0.02, r'Jitter / $\rm{\mu s}$', ha='center', fontsize=14)
    fig.text(0.03, 0.5, 'Relative spread of thresholds', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
#  Dynamic range comparison
# =============================================================================
def nof_spikes_over_stim_amp_presentation(plot_name,
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
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=True, sharey=True, num = plot_name, figsize=(10, 5))
    
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
    fig.text(0.058, 0.5, 'Number of spiking fibers', va='center', rotation='vertical', fontsize=13)
    
    return fig

# =============================================================================
# Plots dB above threshold over distance along spiral lamina for cathodic,
# anodic and biphasic pulses and compare responses for mulitplie electrodes
# =============================================================================
def compare_pulse_forms_for_multiple_electrodes_presentation(plot_name,
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
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex="col", sharey=True, num = plot_name, figsize=(10, 5))
    
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
    fig.text(0.5, 0.014, 'dB above total threshold', ha='center', fontsize=12)
    fig.text(0.06, 0.5, 'Distance along spiral lamina / mm', va='center', rotation='vertical', fontsize=12)
    
    return fig