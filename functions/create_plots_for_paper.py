# =============================================================================
# This script includes all plot functions for the tests that are part of the 
# "Model analyses" script and for the comparisons in the "Model_comparison"
# sript. The plots usually compare the results of multiple models. In some cases
# the plots show also experimental data.
# =============================================================================
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from string import ascii_uppercase as letters

##### import models
import models.Rattay_2001 as rattay_01
import models.Briaire_2005 as briaire_05
import models.Smit_2010 as smit_10

# =============================================================================
#  Conductance velocity comparison
# =============================================================================
def conduction_velocity_comparison(plot_name,
                                   velocity_data):
    """This function plots conduction velocities over ANF diameters. The lines
    for measurements of two differents experiments are shown as well as points
    for the model values.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    model_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "velocity (m/s)" 
        - "outer diameter (um)"
        - "section"
        - "model_name"
                
    Returns
    -------
    figure with conduction velocity comparison
    """
    
    models = ["Rattay et al. (2001)", "Briaire and Frijns (2005)", "Smit et al. (2010)"]
    
    ##### change strings to float
    velocity_data["velocity (m/s)"] = velocity_data["velocity (m/s)"].astype(float)
    velocity_data["outer diameter (um)"] = velocity_data["outer diameter (um)"].astype(float)
    
    ##### experimental data of Hursh 1939
    x_Hursh = np.array([2,6])
    y_Hursh = x_Hursh*6
    
    ##### experimental data of Boyd and Kalu 1979
    x_Boyd_1 = np.array([3,6])
    y_Boyd_1 = x_Boyd_1*4.6
    
    ##### define colors and markers
    colors = ["black","red","blue"]
    edgecolors = ["black","red","blue"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(9, 4))
    fig.subplots_adjust(bottom=0.15)
    
    ##### no grid
    axes.grid(False)
    
    ##### Plot AP rise time values of models
    for ii, model in enumerate(models):
        
        ##### plot dendrite point
        current_model = velocity_data[(velocity_data["model_name"] == model) & (velocity_data["section"] == "dendrite")]
        axes.scatter(current_model["outer diameter (um)"], current_model["velocity (m/s)"], color = colors[ii], marker = "o", edgecolor = edgecolors[ii],label = "{} {}".format(model, "dendrite"))
        
        ##### plot axon point
        current_model = velocity_data[(velocity_data["model_name"] == model) & (velocity_data["section"] == "axon")]
        axes.scatter(current_model["outer diameter (um)"], current_model["velocity (m/s)"], color = colors[ii], marker = "v", edgecolor = edgecolors[ii],label = "{} {}".format(model, "axon"))
    
    ##### Plot lines for the experiments
    axes.plot(x_Hursh,y_Hursh, 'k--', label = "Hursh (1939)")
    axes.plot(x_Boyd_1,y_Boyd_1, 'k:', label = "Boyd and Kalu (1979)")
    
    ##### define axes ranges
    axes.set_ylim([0,50])

    ##### show legend
    plt.legend(ncol=2)
    
    ##### add labels to the axes
    axes.set_xlabel(r"$D$ / $\rm{\mu m}$", fontsize=14)
    axes.set_ylabel(r"$v_{\rm{c}}$ / $\rm{ms}^{-1}$", fontsize=14)
    
    return fig

# =============================================================================
#  Single node response voltage course model comparison
# =============================================================================
def single_node_response_comparison(plot_name,
                                    voltage_data,
                                    presentation = False):
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
    models = ['Rattay et al. (2001)', 'Briaire and Frijns (2005)', 'Smit et al. (2010)']
    
    ##### define number of columns
    nof_models = len(models)
    
    ##### get axes ranges
    y_min = min(voltage_data["membrane potential (mV)"]) - 5
    y_max = max(voltage_data["membrane potential (mV)"]) + 15
    x_max = 1.4
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, nof_models, sharex=False, sharey=False, num = plot_name, figsize=(10,3))
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
        
        ##### put legend next to plots
        if ii == len(models)-2:
            axes[ii].legend(loc = (0.42, 0.4), shadow = False, title = "stimulus amplitude", fontsize=10)
            
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
#  Plot rise and fall time comparison in one figure
# =============================================================================
def rise_and_fall_time_comparison(plot_name,
                                  model_data):
    """This function plots conduction velocities over ANF diameters. The lines
    for measurements of two differents experiments are shown as well as points
    for the model values.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    model_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "velocity (m/s)" 
        - "outer diameter (um)"
        - "section"
        - "model_name"
                
    Returns
    -------
    figure with conduction velocity comparison
    """
    
    ##### define x-vector for curves
    x_values = np.linspace(0,80,100)
    
    #### rise time and fall time curve
    # Get points of AP duration curve of Paintal 1965 (with software Engague Digitizer)
    velocity = [10.61,13.989,19.143,24.295,28.907,34.58,42.199,53.531,63.795,72.116,80.08]
    AP_duration_paintal = [598.678,565.683,520.758,480.049,454.792,428.841,402.906,374.891,356.704,343.42,330.133]
    # Interpolate AP duration curve with 4. order polynomial
    paintal_AP_duration = np.poly1d(np.polyfit(velocity,AP_duration_paintal,4))
    # Get points of AP fall time curve of Paintal 1965 (with software Engague Digitizer)
    velocity = [16,64]
    AP_fall_time_paintal = [350,263]
    # Interpolate AP fall time curve linearly
    paintal_fall_time = np.poly1d(np.polyfit(velocity,AP_fall_time_paintal,1))
    # Get AP rise time curve
    paintal_rise_time = paintal_AP_duration - paintal_fall_time
    
    ##### initialize handles and labels
    handles, labels = (0, 0)
    
    ##### get model names
    models = ['Rattay et al. (2001)', 'Briaire and Frijns (2005)', 'Smit et al. (2010)']
    
    ##### define colors and markers
    colors = ["black","red","blue"]
    edgecolors = ["black","red","blue"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, num = plot_name, figsize=(11, 5), gridspec_kw = {'height_ratios':[3, 1.1]})
    
    ##### no grid
    axes[0][0].grid(False)
        
    ##### remove last subplot
    fig.delaxes(axes[1][1])
    
    ##### Plot AP rise and fall time curves of Paintal
    axes[0][0].plot(x_values,paintal_rise_time(x_values), color = "black", label = "Experimental data from Paintal (1966)")
    axes[0][1].plot(x_values,paintal_fall_time(x_values), color = "black", label = "Experimental data from Paintal (1966)")
    
    ##### Plot rise and fall time values of models
    for ii, model in enumerate(models):
        
        ##### plot dendrite points
        current_data = model_data[model_data["model_name"] == "{} dendrite".format(model)]
        axes[0][0].scatter(current_data["conduction velocity dendrite (m/s)"],current_data["rise time (us)"],
                                      color = colors[ii], marker = "o", edgecolor = edgecolors[ii], label = model)
        axes[0][1].scatter(current_data["conduction velocity dendrite (m/s)"],current_data["fall time (us)"],
                                      color = colors[ii], marker = "o", edgecolor = edgecolors[ii], label = model)         
        
        ##### plot axon points
        current_data = model_data[model_data["model_name"] == "{} axon".format(model)]
        axes[0][0].scatter(current_data["conduction velocity axon (m/s)"],current_data["rise time (us)"],
                                      color = colors[ii], marker = "v", edgecolor = edgecolors[ii], label = model)
        axes[0][1].scatter(current_data["conduction velocity axon (m/s)"],current_data["fall time (us)"],
                                      color = colors[ii], marker = "v", edgecolor = edgecolors[ii], label = model)
    
    ##### put legend in first subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[1][0].set_axis_off()
    axes[1][0].legend(handles, labels, loc=(0.05,-0.3), ncol=3, fontsize=11.8)
    
    ##### set lower y limit to zero
    axes[0][0].set_ylim(bottom=0)
    axes[0][1].set_ylim(bottom=0)
    
    ##### adjust subplot positions.
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    
    ##### add labels to the axes    
    axes[0][0].set_ylabel(r'Rise time / $\rm{\mu s}$', fontsize=14)
    axes[0][1].set_ylabel(r'Fall time / $\rm{\mu s}$', fontsize=14)
    axes[0][0].set_xlabel(r'$v_{\rm{c}}$ / $\rm{ms}^{-1}$', fontsize=14)
    axes[0][1].set_xlabel(r'$v_{\rm{c}}$ / $\rm{ms}^{-1}$', fontsize=14)
    
    return fig

# =============================================================================
#  Strength duration curve model comparison
# =============================================================================
def strength_duration_curve_comparison(plot_name,
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
    models = ['Rattay et al. (2001)', 'Briaire and Frijns (2005)', 'Smit et al. (2010)']
        
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
    colors = ["black","red","blue"]
    edgecolors = ["black","red","blue"]
    markers = ["o","v","s"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 2, sharey=True, num = plot_name, figsize=(8, 3.5))
    fig.subplots_adjust(bottom=0.15)
    
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
                             color = colors[jj], label = "_nolegend_", basex=10)
            
            axes[ii].scatter(current_data["phase duration (us)"].iloc[0], current_data["threshold (uA)"].iloc[0],
                            color = colors[jj], marker = markers[jj], edgecolor = edgecolors[jj], label = label)
        
            ##### define y axes range
            axes[ii].set_ylim([y_min,y_max])
            axes[ii].set_xlim([8,550])
    
            ##### add labels to x-axes    
            axes[ii].set_xlabel(r'Phase duration / $\rm{\mu s}$', fontsize=14)
    
        ##### use normal values for axes (no powered numbers)
        for axis in [axes[ii].xaxis, axes[ii].yaxis]:
            axis.set_major_formatter(ScalarFormatter())
            
    ##### show legend
#    fig.legend(loc = (0.38, 0.55), shadow = False)
    axes[0].legend()
    
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
#  Refractory curve comparison
# =============================================================================
def refractory_curves_comparison(plot_name,
                                 refractory_curves):
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
    model_names = ["rattay_01", "briaire_05", "smit_10"]
    models = [eval(model) for model in model_names]
    
    ##### define number of columns
    nof_models = len(models)
    
    ##### get axes ranges
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
#  PSTH comparison
# =============================================================================
def psth_comparison(plot_name,
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
    models = ['Briaire and Frijns (2005)', 'Rattay et al. (2001)', 'Smit et al. (2010)']
        
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
    fig, axes = plt.subplots(nof_models, nof_amplitudes*nof_pulse_rates, sharex=True, sharey=True, num = plot_name, figsize=(nof_amplitudes*nof_pulse_rates*2, nof_models*1.4))
    fig.subplots_adjust(bottom=0.13)
    
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
                
                ##### add letter on right side to identify model
                if col == nof_amplitudes*nof_pulse_rates-1:
                    axes[ii][col].yaxis.set_label_position("right")
                    axes[ii][col].set_ylabel(letters[ii], fontsize=16, rotation = 0)
                    axes[ii][col].yaxis.set_label_coords(1.1,0.5)
                    
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
                    
                    ##### write stimulus amplitdues in plots
                    axes[ii][col].text(np.ceil(max(bin_edges)/8), 1.1, r"$I={}$".format(current_data["stimulus amplitude (uA)"][0]) + r"$\rm{\mu A}$", fontsize=10)
                    
                else:
                    ##### define y-achses range and tick numbers
                    axes[ii][col].set_ylim([0,max_bin_height*1.35])
                    axes[ii][col].set_yticks([0,np.floor(max_bin_height/2),max_bin_height])
                
                    ##### write stimulus amplitdues in plots
                    axes[ii][col].text(np.ceil(max(bin_edges)/8), max_bin_height*1.1, r"$I={}$".format(current_data["stimulus amplitude (uA)"][0]) + r"$\rm{\mu A}$", fontsize=10)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1, wspace=0.15)
    
    ##### use the pulse rates as column titles
    for ax, columtitle in zip(axes[0], ["{} pps".format(pulse_rates[ii]) for ii in range(nof_pulse_rates)]):
        ax.set_title(columtitle, y = 1.1)
    
    ##### get labels for the axes
    fig.text(0.5, 0.01, 'Time after pulse-train onset / ms', ha='center', fontsize=14)
    if plot_style == "firing_efficiency":
        fig.text(0.03, 0.5, 'firing efficiency', va='center', rotation='vertical', fontsize=14)
    else:
        fig.text(0.065, 0.5, 'APs per timebin ({} ms)'.format(bin_width), va='center', rotation='vertical', fontsize=14)

    return fig

# =============================================================================
#  Voltage course comparison
# =============================================================================
def voltage_course_comparison_plot(plot_name,
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
    nof_cols = 3
    
    ##### get number of rows
    nof_rows = np.ceil(len(models)/nof_cols).astype(int)
    
    ##### get number of plots
    nof_plots = len(models)
        
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=True, num = plot_name, figsize=(3.2*nof_cols, 2.7*nof_rows))
    fig.subplots_adjust(bottom=0.16)
    
    ##### create plots  
    for ii in range(nof_rows*nof_cols):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        col = ii-row*nof_cols
        
        ##### turn off y-labels
        plt.setp(axes[col].get_yticklabels(), visible=False)
        axes[col].tick_params(axis = 'both', left = 'off')
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[col])
        
        ##### plot voltage courses
        if ii < nof_plots:
            
            model = models[ii]
            
            ##### get array of compartments to plot
            comps_to_plot = model.comps_to_plot[model.comps_to_plot < max_comp[ii]]
            
            ##### get voltage courses for current model
            voltage_matrix = voltage_courses[ii]
            
            ##### distances between lines and x-axis
            offset = np.cumsum(model.distance_comps_middle)/meter
            offset = (offset/max(offset))*10
            
            ##### plot lines
            for jj in comps_to_plot:
                axes[col].plot(time_vector/ms, offset[jj] - 1/(30)*(voltage_matrix[jj, :]-model.V_res)/mV, "black", linewidth = 0.9)
            
            ##### write model name in plots
            axes[col].text(0.5, -3, "{}".format(model.display_name_plots), fontsize=11)
                
            ##### no grid
            axes[col].grid(False)
    
    ##### invert y-achses
    axes[col].invert_yaxis()
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0, wspace=0)
    
    ##### get labels for the axes
    fig.text(0.5, 0.004, 'Time / ms', ha='center', fontsize=12)
    fig.text(0.08, 0.5, 'Position along fiber [major]', va='center', rotation='vertical', fontsize=12)
    fig.text(0.1, 0.5, 'membrane potential [minor]', va='center', rotation='vertical', fontsize=12)
    
    return fig

# =============================================================================
# Compare stochastic properties for different k_noise values
# =============================================================================
def stochastic_properties_comparison(plot_name,
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
    models = ["rattay_01", "briaire_05", "smit_10"]
    
    ##### define colors and markers
    colors = ["black","red","blue"]
    edgecolors = ["black","red","blue"]
    markers = ["o","v","s"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(7,3.5))
    fig.subplots_adjust(bottom=0.15)
    
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
    plt.legend(ncol=2)
    
    ##### Write relative spreads as percentage
    vals = axes.get_yticks().astype(int)
    axes.set_yticklabels(['{}%'.format(x) for x in vals])
        
    ##### no grid
    axes.grid(False)
    
    ##### get labels for the axes
    fig.text(0.5, 0.01, r'Jitter / $\rm{\mu s}$', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Relative spread of thresholds', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
#  Plot thresholds for pulse trains
# =============================================================================
def thresholds_for_pulse_trains(plot_name,
                                pulse_train_thr_over_rate,
                                pulse_train_thr_over_dur):
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

    ##### get model names
    models = pulse_train_thr_over_rate["model"].unique().tolist()
    
    ##### get number of rows
    nof_rows = len(models)
    
    ##### define colors and markers for rate plots
    colors_rate = ["black","red","blue"]
    edgecolors_rate = ["black","red","blue"]
    markers_rate = ["o","v","s"]
    
    ##### define colors and markers for duration plots
    colors_dur = ["black","red","blue"]
    markers_dur = ["v","s","o"]
    edgecolors_dur = ["black","red","blue"]
              
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, 2, sharex = "col", sharey=True, num = plot_name, figsize=(9, 1.5*nof_rows))
    
    ##### loop over figure type
    for ii, method in enumerate(["thr_over_rate", "thr_over_dur"]):
        
        if method == "thr_over_rate":
            threshold_data = pulse_train_thr_over_rate
        else:
            threshold_data = pulse_train_thr_over_dur
    
        ##### loop over models 
        for jj, model in enumerate(models):
            
            ##### building a subset for current model
            current_model = threshold_data[threshold_data["model"] == model]
            
            ##### use single pulse threshold as reference and calculate thresholds in dB
            threshold_single_pulse = current_model["threshold (uA)"][current_model["number of pulses"] == 1].iloc[0]
            current_model["dB_below_threshold"] = 20*np.log10(current_model["threshold (uA)"]/threshold_single_pulse)
            
            if method == "thr_over_rate":
                
                for kk, dur in enumerate(np.sort(pulse_train_thr_over_rate["pulse train durations (ms)"].unique()).tolist()):
                    
                    ##### built subset for current train duration
                    current_data = current_model[current_model["pulse train durations (ms)"] == dur]
                    
                    ##### plot latency curve
                    axes[jj][ii].plot(current_data["pulses per second"], current_data["dB_below_threshold"], color = colors_rate[kk], label = "_nolegend_")
                    
                    ##### show points
                    axes[jj][ii].scatter(current_data["pulses per second"], current_data["dB_below_threshold"],
                        color = colors_rate[kk], marker = markers_rate[kk], edgecolor = edgecolors_rate[kk], label = "{} ms".format(dur))
                    
            else:
                
                for kk, pps in enumerate(np.sort(pulse_train_thr_over_dur["pulses per second"].unique()).tolist()):
                    
                    ##### built subset for current train duration
                    current_data = current_model[current_model["pulses per second"] == pps]
                    
                    ##### plot latency curve
                    axes[jj][ii].plot(current_data["pulse train durations (ms)"], current_data["dB_below_threshold"], color = colors_dur[kk], label = "_nolegend_")
                    
                    ##### show points
                    axes[jj][ii].scatter(current_data["pulse train durations (ms)"], current_data["dB_below_threshold"],
                        color = colors_dur[kk], marker = markers_dur[kk], edgecolor = edgecolors_dur[kk], label = "{} pps".format(pps))
                    
            ##### add legend to first plots per column
            if jj == 0:
                if method == "thr_over_rate":
                    legend = axes[jj][ii].legend(ncol=3 ,title='Train duration:', fontsize=10, loc=(0,1.05))
                    plt.setp(legend.get_title(),fontsize=11)
                else:
                    legend = axes[jj][ii].legend(ncol=3 ,title='Pulse rate:', fontsize=10, loc=(0,1.05))
                    plt.setp(legend.get_title(),fontsize=11)
                
            ##### logarithmic achses
            axes[jj][ii].set_xscale('log')
            
            ##### use normal values for axes (no powered numbers)
            for axis in [axes[jj][ii].xaxis, axes[jj][ii].yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            
            #### define x axis range and x ticks and write model names in plots
            if method == "thr_over_rate":
                axes[jj][ii].set_xlim([100,20000])
                axes[jj][ii].set_xticks([100,1000,10000])
                ##### add letter on right side to identify model
                axes[jj][ii].yaxis.set_label_position("right")
                axes[jj][ii].set_ylabel(letters[jj], fontsize=14, rotation = 0)
                axes[jj][ii].yaxis.set_label_coords(1.048,0.57)
            else:
                axes[jj][ii].set_xlim([0.09,30])
                axes[jj][ii].set_xticks([0.1,1,10])
                axes[jj][ii].tick_params(axis = 'y', left = 'off', right = "on", labelright = True)
                
            ##### set y ticks
            axes[jj][ii].set_yticks([-2,-1,0,1])
            
            ##### no grid
            axes[jj][ii].grid(True, which='both', axis='both', linestyle='--', alpha = 0.5)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0, wspace=0.1)
    
    ##### get labels for the axes
    axes[nof_rows-1][0].set_xlabel('Pulse rate (pps)', fontsize=12)
    axes[nof_rows-1][1].set_xlabel('Pulse-train duration / ms', fontsize=12)
    fig.text(0.07, 0.5, 'Threshold (dB re single pulse threshold)', va='center', rotation='vertical', fontsize=12)
        
    return fig

# =============================================================================
#  Plot thresholds for sinus stimulation
# =============================================================================
def thresholds_for_sinus(plot_name,
                         sinus_thresholds):
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
    
    ##### remove combinations of durations and frequencies, where no complete sinus period could be measured
    sinus_thresholds = sinus_thresholds[(sinus_thresholds["stimulus length (ms)"] != 0.5) | (sinus_thresholds["frequency"] > 1)]
    sinus_thresholds = sinus_thresholds[(sinus_thresholds["stimulus length (ms)"] != 2) | (sinus_thresholds["frequency"] >= 0.5)]
    
    ##### only frequencies higher than 0.125 kHz
    sinus_thresholds = sinus_thresholds[sinus_thresholds["frequency"] >= 1]
    
    ##### get model names
    models = sinus_thresholds["model"].unique().tolist()
    
    ##### get frequencies
    frequencies = sinus_thresholds["frequency"].unique().tolist()
    
    ##### get number of plots
    nof_plots = len(models)
    
    ##### define colors and markers
    colors = ["blue","black","red","blue","black","red"]
    markers = ["o","s","v","o","s","v"]
    edgecolors = ["blue","black","red","blue","black","red"]
              
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, 3, sharex = False, sharey=True, num = plot_name, figsize=(10,3))
    fig.subplots_adjust(bottom=0.18)
    
    ##### create plots  
    for ii in range(3):
        
        ##### turn off y-labels for all but the left plots
        if ii != 0:  
             plt.setp(axes[ii].get_yticklabels(), visible=False)
             axes[ii].tick_params(axis = "both", left = "off")
        
        model = models[ii]
        
        ##### building a subset for current model
        current_model = sinus_thresholds[sinus_thresholds["model"] == model]
        
        ##### use threshold of max frequency as reference and calculate thresholds in dB
        threshold_max_frequ = current_model["threshold (uA)"][current_model["frequency"] == max(frequencies)].iloc[0]
        current_model["dB_below_threshold"] = 20*np.log10(current_model["threshold (uA)"]/threshold_max_frequ)
        
        ##### loop over stimulus lengths
        for kk, dur in enumerate(np.sort(current_model["stimulus length (ms)"].unique()).tolist()):
            
            ##### built subset for current train duration
            current_data = current_model[current_model["stimulus length (ms)"] == dur]
            
            ##### plot data
            axes[ii].plot(current_data["frequency"], current_data["dB_below_threshold"], color = colors[kk], label="_nolegend_")
            
            ##### show points
            axes[ii].scatter(current_data["frequency"], current_data["dB_below_threshold"],
                color = colors[kk], marker = markers[kk], edgecolor = edgecolors[kk], label = "{} ms".format(dur))
            
        ##### logarithmic achses
        axes[ii].set_xscale('log', basex=2)
        
        ##### use normal values for axes (no powered numbers)
        for axis in [axes[ii].xaxis, axes[ii].yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        #### define x axis range and x ticks and write model names in plots
        axes[ii].set_xlim([0.8,20])
        axes[ii].set_xticks(np.array([1,2,4,8,16]).round(2))
        axes[ii].text(1, 0,"{}".format(eval("{}.display_name_plots".format(model))), fontsize=12)
        
        ##### set y ticks
        axes[ii].set_yticks([0, -5,-10,-15,-20,-25])
        
        ##### add grid
        axes[ii].grid(True, which='both', axis='both', linestyle='--', alpha = 0.5)
        
        ##### save handles and labels for the plot before the one with the legend
        if ii == 1:
            axes[ii].legend(loc=(0.01,0.05), title = "stimulus duration", ncol=2)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0, wspace=0)
    
    ##### get labels for the axes
    fig.text(0.5, 0.01, 'frequency / kHz', ha='center', fontsize=12)
    fig.text(0.045, 0.5, 'Threshold', va='center', rotation='vertical', fontsize=12)
    fig.text(0.065, 0.5, '(dB re threshold for 16 kHz)', va='center', rotation='vertical', fontsize=12)
        
    return fig
