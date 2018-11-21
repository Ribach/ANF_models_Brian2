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
#  Voltage course comparison
# =============================================================================
def voltage_course_comparison_plot(plot_name,
                                   model_names,
                                   time_vector,
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
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=True, num = plot_name, figsize=(4.5*nof_cols, 4*nof_rows))
    
    ##### create plots  
    for ii in range(nof_rows*nof_cols):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        col = ii-row*nof_cols
        
        ##### turn off x-labels for all but the bottom plots
        if (nof_plots - ii) > nof_cols:
             plt.setp(axes[row][col].get_xticklabels(), visible=False)
             axes[row][col].tick_params(axis = 'both', bottom = 'off')
        
        ##### turn off y-labels
        plt.setp(axes[row][col].get_yticklabels(), visible=False)
        axes[row][col].tick_params(axis = 'both', left = 'off')
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot voltage courses
        if ii < nof_plots:
            
            model = models[ii]
            
            ##### get voltage courses for current model
            voltage_matrix = voltage_courses[ii]
            
            ##### distances between lines and x-axis
            offset = np.cumsum(model.distance_comps_middle)/meter
            offset = (offset/max(offset))*10
            
            ##### plot lines
            for ii in model.comps_to_plot:
                axes[row][col].plot(time_vector/ms, offset[ii] - 1/(30)*(voltage_matrix[ii, :]-model.V_res)/mV, "#000000")
            
            ##### write model name in plots
            axes[row][col].text(0.45, -3, "{}".format(model.display_name), fontsize=13.5)
                
            ##### no grid
            axes[row][col].grid(False)
    
    ##### invert y-achses
    axes[row][col].invert_yaxis()
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0, wspace=0)
    
    ##### get labels for the axes
    fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
    fig.text(0.1, 0.5, 'Position [major] membrane potential [minor]', va='center', rotation='vertical', fontsize=14)
    
    return fig
    
# =============================================================================
#  Conductance velocity comparison
# =============================================================================
def conduction_velocity_comparison(plot_name,
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
    
    ##### change strings to float
    model_data["velocity (m/s)"] = model_data["velocity (m/s)"].astype(float)
    model_data["outer diameter (um)"] = model_data["outer diameter (um)"].astype(float)
    model_data["section"][model_data["section"] == "fiber"] = ""
    
    ##### experimental data of Hursh 1939
    x_Hursh = np.array([2,20])
    y_Hursh = x_Hursh*6
    
    ##### experimental data of Boyd and Kalu 1979
    x_Boyd_1 = np.array([3,12])
    y_Boyd_1 = x_Boyd_1*4.6
    
    x_Boyd_2 = np.array([10,20])
    y_Boyd_2 = x_Boyd_2*5.7
    
    ##### define colors and markers
    colors = ["black","black","black","red","red","red","yellow","yellow","yellow","blue","blue","blue"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","black","red","red","red","black","black","black","blue","blue","blue"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    #fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(7, 6))
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(9, 5.5))
    
    ##### no grid
    axes.grid(False)
    
    ##### Plot AP rise time values of models
    for ii in range(len(model_data)):
        
        ##### plot point
        axes.scatter(model_data["outer diameter (um)"][ii], model_data["velocity (m/s)"][ii], color = colors[ii], marker = markers[ii], edgecolor = edgecolors[ii],
                                label = "{} {}".format(model_data["model_name"][ii], model_data["section"][ii]))
    
    ##### Plot lines for the experiments
    axes.plot(x_Hursh,y_Hursh, 'k--', label = "Hursh 1939")
    axes.plot(x_Boyd_1,y_Boyd_1, 'k:', label = "Boyd and Kalu 1979")
    axes.plot(x_Boyd_2,y_Boyd_2, 'k:', label = "_nolegend_")

    ##### show legend
    plt.legend(ncol=2)

    ##### add labels to the axes    
    axes.set_xlabel('Outer fiber diameter / um', fontsize=16)
    axes.set_ylabel('Conduction velocity / (m/s)', fontsize=16)  
    
    return fig

# =============================================================================
#  Single node response voltage course model comparison
# =============================================================================
def single_node_response_comparison(plot_name,
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
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of rows
    nof_rows = np.ceil(len(models)/nof_cols).astype(int)
    
    ##### get number of plots
    nof_plots = len(models)
    
    ##### get number of runs
    nof_runs = max(voltage_data["run"])+1
    
    ##### get axes ranges
    y_min = min(voltage_data["membrane potential (mV)"]) - 5
    y_max = max(voltage_data["membrane potential (mV)"]) + 15
    x_max = max(voltage_data["time (ms)"])
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=False, num = plot_name, figsize=(4.5*nof_cols, 3*nof_rows))
    
    ##### create plots  
    for ii in range(nof_rows*nof_cols):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        col = ii-row*nof_cols
        
        ##### define axes ranges
        axes[row][col].set_ylim([y_min,y_max])
        axes[row][col].set_xlim([0,x_max])
        
        ##### turn off x-labels for all but the bottom plots
        if (nof_plots - ii) > nof_cols:
             plt.setp(axes[row][col].get_xticklabels(), visible=False)
             axes[row][col].tick_params(axis = 'both', bottom = 'off')
        
        ##### turn off y-labels for all but the bottom plots
        if (col != 0) and (ii < nof_plots):  
             plt.setp(axes[row][col].get_yticklabels(), visible=False)
             axes[row][col].tick_params(axis = 'both', left = 'off')
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot voltage courses
        if ii < nof_plots:
            
            model = models[ii]
                
            ##### loop over runs
            for jj in range(nof_runs):
                
                ##### building a subset
                current_data = voltage_data[voltage_data["model"] == model][voltage_data["run"] == jj]
                                          
                ##### create plot
                axes[row][col].plot(current_data["time (ms)"], current_data["membrane potential (mV)"], color = "black")
                
            ##### remove top and right lines
            axes[row][col].spines['top'].set_visible(False)
            axes[row][col].spines['right'].set_visible(False)
                
            ##### write model name in plots
            axes[row][col].text(x_max*0.05, y_max-12.5, "{}".format(model))
                
            ##### no grid
            axes[row][col].grid(False)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    
    ##### get labels for the axes
    fig.text(0.5, 0.06, 'Time (ms)', ha='center', fontsize=14)
    fig.text(0.05, 0.5, 'Membrane potential (mV)', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
#  Plot rise-time over conduction velocity according to Paintal 1966
# =============================================================================
def paintal_rise_time_curve(plot_name,
                            model_data):
    """This function plots rise times over the conduction velocity. The lines
    for measurements of two differents experiments are shown as well as points
    for the model values.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    model_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "conduction velocity (m/s)" 
        - "rise time (us)"
        - "model_name"
                
    Returns
    -------
    figure with rise time comparison
    """
    
    ##### define x-vector for curves
    x_values = np.linspace(0,80,100)
    
    ##### Get points of AP duration curve of Paintal 1965 (with software Engague Digitizer)
    velocity = [10.61,13.989,19.143,24.295,28.907,34.58,42.199,53.531,63.795,72.116,80.08]
    AP_duration_paintal = [598.678,565.683,520.758,480.049,454.792,428.841,402.906,374.891,356.704,343.42,330.133]
    
    ##### Interpolate AP duration curve with 4. order polynomial
    paintal_AP_duration = np.poly1d(np.polyfit(velocity,AP_duration_paintal,4))
    
    ##### Get points of AP fall time curve of Paintal 1965 (with software Engague Digitizer)
    velocity = [16,64]
    AP_fall_time_paintal = [350,263]
    
    ##### Interpolate AP fall time curve linearly
    paintal_fall_time = np.poly1d(np.polyfit(velocity,AP_fall_time_paintal,1))
    
    ##### Get AP rise time curve
    paintal_rise_time = paintal_AP_duration - paintal_fall_time
    
    ##### get model names
    models = model_data["model_name"].tolist()
    
    ##### define colors and markers
    colors = ["black","black","black","red","red","red","yellow","yellow","yellow","blue","blue","blue"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","black","red","red","red","black","black","black","blue","blue","blue"]

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(10.5, 6))
    
    ##### no grid
    axes.grid(False)
    
    ##### Plot AP rise time curve of Paintal
    axes.plot(x_values,paintal_rise_time(x_values), color = "black", label = "Experimental data from Paintal 1965")
    
    ##### Plot AP rise time values of models
    for ii,model in enumerate(models):
        
        ##### building a subset
        current_data = model_data[model_data["model_name"] == model]
        
        ##### plot point
        axes.scatter(current_data["conduction velocity (m/s)"],current_data["rise time (us)"],
                                  color = colors[ii], marker = markers[ii], edgecolor = edgecolors[ii], label = "{}".format(model))

    ##### show legend
    plt.legend()

    ##### add labels to the axes    
    axes.set_xlabel('Conduction velocity / (m/s)', fontsize=16)
    axes.set_ylabel('Rise time / us', fontsize=16)  
    
    return fig

# =============================================================================
#  Plot fall-time over conduction velocity according to Paintal 1966
# =============================================================================
def paintal_fall_time_curve(plot_name,
                            model_data):
    """This function plots fall times over the conduction velocity. The lines
    for measurements of two differents experiments are shown as well as points
    for the model values.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    model_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "conduction velocity (m/s)" 
        - "fall time (us)"
        - "model_name"
                
    Returns
    -------
    figure with fall time comparison
    """
    
    ##### define x-vector for curves
    x_values = np.linspace(0,80,100)
    
    ##### Get points of AP fall time curve of Paintal 1965 (with software Engague Digitizer)
    velocity = [16,64]
    AP_fall_time_paintal = [350,263]
    
    ##### Interpolate AP fall time curve linearly
    paintal_fall_time = np.poly1d(np.polyfit(velocity,AP_fall_time_paintal,1))
    
    ##### get model names
    models = model_data["model_name"].tolist()
    
    ##### define colors and markers
    colors = ["black","black","black","red","red","red","yellow","yellow","yellow","blue","blue","blue"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","black","red","red","red","black","black","black","blue","blue","blue"]

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(10.5, 6))
    
    ##### no grid
    axes.grid(False)
    
    ##### Plot AP rise time curve of Paintal
    axes.plot(x_values,paintal_fall_time(x_values), color = "black", label = "Experimental data from Paintal 1965")
    
    ##### Plot AP rise time values of models
    for ii,model in enumerate(models):
        
        ##### building a subset
        current_data = model_data[model_data["model_name"] == model]
        
        ##### plot point
        axes.scatter(current_data["conduction velocity (m/s)"],current_data["fall time (us)"],
                                  color = colors[ii], marker = markers[ii], edgecolor = edgecolors[ii], label = "{}".format(model))
            
    ##### show legend
    plt.legend()

    ##### add labels to the axes    
    axes.set_xlabel('Conduction velocity / (m/s)', fontsize=16)
    axes.set_ylabel('Fall time / us', fontsize=16)  
    
    return fig

# =============================================================================
#  Strength duration curve model comparison
# =============================================================================
def strength_duration_curve_comparison(plot_name,
                                       threshold_data,
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
    models = threshold_data["model"].unique().tolist()
    
    ##### exclude values, where no threshold was found
    threshold_data = threshold_data.loc[threshold_data["threshold (uA)"] != 0]
    
    ##### get y range
    y_min = -0.5
    y_max = max(threshold_data["threshold (uA)"]) + 1
    
    ##### define colors and markers
    colors = ["black","black","black","red","red","red","blue","blue","blue","yellow","yellow","yellow","blue","blue","blue"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","black","red","red","red","blue","blue","blue","black","black","black","blue","blue","blue"]
    line_styles = [":","-.","-",":","-.","-",":","-.","-",":","-.","-"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(10, 6))
    
    ##### no grid
    axes.grid(False)
    
    ##### loop over models
    for ii, model in enumerate(models):
        
        ##### building a subset
        current_data = threshold_data[threshold_data["model"] == model]

        ##### plot strength duration curve    
        axes.semilogx(current_data["phase duration (us)"], current_data["threshold (uA)"],
                                   color = colors[ii], linestyle = line_styles[ii], label = "_nolegend_")
        
        axes.scatter(current_data["phase duration (us)"].iloc[0], current_data["threshold (uA)"].iloc[0],
                                  color = colors[ii], marker = markers[ii], edgecolor = edgecolors[ii], label = model)
        
#        ##### mark chronaxie
#        if strength_duration_table is not None:
#        axes.vlines(x=strength_duration_table["chronaxie (us)"][model],
#                    ymin = y_min,
#                    ymax=2*strength_duration_table["rheobase (uA)"][model],
#                    linestyles="dashed", label = "_nolegend_")
    
    ##### define y axes range
    axes.set_ylim([y_min,y_max])
            
    ##### show legend
    plt.legend()
    
    ##### add labels to the axes    
    axes.set_xlabel('Stimulus duration / us', fontsize=16)
    axes.set_ylabel('Stimulus amplitude required / uA', fontsize=16)
    
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
    
    ##### remove rows where no second spikes were obtained
    refractory_curves = refractory_curves[refractory_curves["minimum required amplitude"] != 0]
        
    ##### calculate the ratio of the threshold of the second spike and the masker
    refractory_curves["threshold ratio"] = refractory_curves["minimum required amplitude"]/refractory_curves["threshold"]
    
    ##### convert interpulse intervals to ms
    refractory_curves["interpulse interval"] = refractory_curves["interpulse interval"]*1e3
    
    ##### get model names
    models = refractory_curves["model"].unique().tolist()
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of rows
    nof_rows = np.ceil(len(models)/nof_cols).astype(int)
    
    ##### get number of plots
    nof_plots = len(models)
    
    ##### get axes ranges
    y_min = 0
    y_max = max(refractory_curves["threshold ratio"]) + 1.5
    x_min = 0
    x_max = max(refractory_curves["interpulse interval"]) + 0.2
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=False, num = plot_name, figsize=(4.5*nof_cols, 3*nof_rows))
    
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
        
        ##### defining y ticks
        axes[row][col].set_yticks([1,5,10])
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot voltage courses
        if ii < nof_plots:
            
            model = models[ii]
                
            ##### building a subset
            current_data = refractory_curves[refractory_curves["model"] == model]
                                      
            ##### plot threshold curve
            axes[row][col].plot(current_data["interpulse interval"], current_data["threshold ratio"], color = "black")
            
            ##### add line at threshold level
            axes[row][col].hlines(y=1, xmin=x_min, xmax=x_max, linestyles="dashed", color = "black")
            
            ##### show points
            axes[row][col].scatter(current_data["interpulse interval"], current_data["threshold ratio"], color = "black", marker = "o", alpha  = 0.5)
                
            ##### remove top and right lines
            axes[row][col].spines['top'].set_visible(False)
            axes[row][col].spines['right'].set_visible(False)
                
            ##### write model name in plots
            axes[row][col].text(x_max*0.1, y_max-1, "{}".format(model))
                
            ##### no grid
            axes[row][col].grid(False)
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    
    ##### get labels for the axes
    fig.text(0.5, 0.06, 'Inter pulse interval / ms', ha='center', fontsize=14)
    fig.text(0.05, 0.5, 'threshold 2nd stimulus / threshold', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
#  relative spread comparison
# =============================================================================
def relative_spread_comparison(plot_name,
                               threshold_data):
    """This function provides boxplots, showing the relative spread of thresholds
    for a model for different noise levels (different amounts of noise).

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    threshold_data : pandas dataframe
        This dataframe has to contain the following columns:
        - "phase duration (us)" 
        - "threshold"
        - "noise level"
                
    Returns
    -------
    figure with relative spread comparison
    """

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig = plt.figure(plot_name)
    
    ##### create boxplots
    sns.set_style("whitegrid")
    sns.boxplot(data=threshold_data, x="phase duration (us)", y="threshold", hue="noise level", showfliers=False, dodge=True)
    plt.xlabel('Phase duration / us', fontsize=16)
    plt.ylabel('Threshold / uA', fontsize=16)
        
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
    
    ##### define colors and markers
    colors = ["black","black","black","red","red","red","blue","blue","blue","yellow","yellow","yellow","blue","blue","blue"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","black","red","red","red","blue","blue","blue","black","black","black"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(8.5,5.5))
    
    ##### create plots  
    for ii, model in enumerate(models):
                        
        ##### building a subset
        current_data = stochasticity_table[stochasticity_table["model"] == model]
                                  
        ##### plot threshold curve
        axes.plot(current_data["jitter (us)"], current_data["relative spread (%)"], color = colors[ii], label = "_nolegend_")
        
        ##### show points
        axes.scatter(current_data["jitter (us)"], current_data["relative spread (%)"],
                                  color = colors[ii], marker = markers[ii], edgecolor = edgecolors[ii], label = "{}".format(eval("{}.display_name".format(model))))
    
    ##### define axes ranges
    axes.set_ylim([0,45])
    axes.set_xlim([0,150])
    
    ##### plot experimental range
    axes.fill_between([80,120],[5,5],[10,10], alpha = 0.4, edgecolor="black", label = "experimental range")
                
    ##### add legend
    plt.legend()
    
    ##### Write relative spreads as percentage
    vals = axes.get_yticks().astype(int)
    axes.set_yticklabels(['{}%'.format(x) for x in vals])
        
    ##### no grid
    axes.grid(False)
    
    ##### get labels for the axes
    fig.text(0.5, 0.02, 'jitter / us', ha='center', fontsize=14)
    fig.text(0.03, 0.5, 'relative spread of thresholds', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
# Plot latencies over stimulus duration (old version in one plot)
# =============================================================================
def latencies_over_stimulus_duration_old(plot_name,
                                         latency_models,
                                         latency_measurements = None):
    """This function plots the latencies over stimulus durations and compares
    data from the models with experimental measurements. All is shown in one plot.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    latency_models : pandas dataframe
        This dataframe includes the data of the models and has to contain
        the following columns:
        - "latency (ms)" 
        - "amplitude level"
        - "model"
    latency_measurements : pandas dataframe
        This dataframe includes the data of measurements (e.g. ABR) and has to
        contain the following columns:
        - "latency (ms)" 
        - "amplitude level"
        - "subject"
        - "ear"
                
    Returns
    -------
    figure with latency over stimulus amplitude comparison
    """
    
    ##### get model names
    models = latency_models["model"].unique().tolist()
    
    ##### define colors and markers
    colors = ["black","black","black","red","red","red","blue","blue","blue","yellow","yellow","yellow","blue","blue","blue"]
    markers = ["o","v","s","o","v","s","o","v","s","o","v","s"]
    edgecolors = ["black","black","black","red","red","red","blue","blue","blue","black","black","black"]
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(8.5,5.5))
    
    ##### plot data from models  
    for ii, model in enumerate(models):
                        
        ##### building a subset
        current_data = latency_models[latency_models["model"] == model]
                                  
        ##### plot latency curve
        axes.plot(current_data["amplitude level"], current_data["latency (ms)"], color = colors[ii], label = "_nolegend_")
        
        ##### show points
        axes.scatter(current_data["amplitude level"], current_data["latency (ms)"],
                                  color = colors[ii], marker = markers[ii], edgecolor = edgecolors[ii], label = "{}".format(eval("{}.display_name".format(model))))
    
    ##### plot data from measurements
    if latency_measurements is not None:
        
        ##### get subjects
        subjects = latency_measurements[["subject", "ear"]].drop_duplicates()
                
        ##### loop over subjects
        for ii in range(len(subjects)):
                    
            ##### building a subset
            current_data = latency_measurements[latency_measurements["subject"] == subjects["subject"].tolist()[ii]]
            current_data = current_data[current_data["ear"] == subjects["ear"].tolist()[ii]]
            
            ##### plot latency curve
            axes.plot(current_data["amplitude level"], current_data["latency (ms)"], color = colors[ii+len(models)], label = "_nolegend_")
            
            ##### show points
            axes.scatter(current_data["amplitude level"], current_data["latency (ms)"],
                                      color = colors[ii+len(models)], marker = markers[ii+len(models)], edgecolor = edgecolors[ii+len(models)],
                                      label = "{}, {} ear".format(subjects["subject"].tolist()[ii],subjects["ear"].tolist()[ii]))
                
    ##### add legend
    plt.legend()
    
    ##### no grid
    axes.grid(False)
    
    ##### get labels for the axes
    axes.set_xlabel('Stimulus amplitude / threshold', fontsize=16)
    axes.set_ylabel('latency / ms', fontsize=16)  
        
    return fig

# =============================================================================
#  Plot latencies over stimulus duration for different electrode distances
# =============================================================================
def latencies_over_stimulus_duration(plot_name,
                                     latency_models,
                                     latency_measurements = None):
    """This function plots the latencies over stimulus durations for different
    electrode distances and compares data from the models with experimental
    measurements. There is one plot for the measured data and one for each model.

    Parameters
    ----------
    plot_name : string
        This defines how the plot window will be named.
    latency_models : pandas dataframe
        This dataframe includes the data of the models and has to contain
        the following columns:
        - "latency (ms)" 
        - "amplitude level"
        - "electrode distance (um)"
        - "model"
    latency_measurements : pandas dataframe
        This dataframe includes the data of measurements (e.g. ABR) and has to
        contain the following columns:
        - "latency (ms)" 
        - "amplitude level"
        - "subject"
        - "ear"
                
    Returns
    -------
    figure with latency over stimulus amplitude comparison
    """
    
    ##### get model names
    models = latency_models["model"].unique().tolist()
    
    ##### get electrode distances
    electrode_distances = latency_models["electrode distance (um)"].unique().tolist()
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of plots
    nof_plots = len(models)
    if latency_measurements is not None:
        nof_plots = nof_plots + 1
    
    ##### get number of rows
    nof_rows = np.ceil(nof_plots/nof_cols).astype(int)
    
    ##### get x- axes ranges
    x_min = 0.8
    x_max = max(latency_models["amplitude level"]) + 0.2
    
    ##### define colors and markers
    colors_ABR = ["black","red","blue","black","red","blue","black","red","blue","black","red","blue"]
    markers_ABR = ["o","o","o","v","v","v","s","s","s","o","o","o","v","v","v","s","s","s"]
    edgecolors_ABR = ["black","red","blue","black","red","blue","black","red","blue","black","red","blue"]
    
    colors = ["#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"]
    markers = ["o","v","s","d","o","v","s","d","o","v","s","d"]
    edgecolors = ["#ffeda0","#ffeda0","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"]

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=False, num = plot_name, figsize=(8*nof_cols, 5*nof_rows))
    
    ##### plot experimental results if provided
    if latency_measurements is not None:
        
        ##### define y axis range
        y_min = min(latency_measurements["latency (ms)"]) - (max(latency_measurements["latency (ms)"]) - min(latency_measurements["latency (ms)"]))*0.1
        y_max = max(latency_measurements["latency (ms)"]) + (max(latency_measurements["latency (ms)"]) - min(latency_measurements["latency (ms)"]))*0.2
        axes[0][0].set_ylim([y_min,y_max])
        
        ##### define axes ranges
        axes[0][0].set_xlim([x_min,x_max])
        
        ##### turn off x-labels for all but the bottom plots
        if (nof_plots - 0) > nof_cols:
             plt.setp(axes[0][0].get_xticklabels(), visible=False)
                
        ##### get subjects
        subjects = latency_measurements[["subject", "ear"]].drop_duplicates()
                
        ##### loop over subjects
        for ii in range(len(subjects)):
                    
            ##### building a subset
            current_data = latency_measurements[latency_measurements["subject"] == subjects["subject"].tolist()[ii]]
            current_data = current_data[current_data["ear"] == subjects["ear"].tolist()[ii]]
            
            ##### plot latency curve
            axes[0][0].plot(current_data["amplitude level"], current_data["latency (ms)"], color = colors_ABR[ii], label = "_nolegend_")
            
            ##### show points
            axes[0][0].scatter(current_data["amplitude level"], current_data["latency (ms)"],
                              color = colors_ABR[ii], marker = markers_ABR[ii], edgecolor = edgecolors_ABR[ii],
                              label = "{}, {} ear".format(subjects["subject"].tolist()[ii],subjects["ear"].tolist()[ii]))
            
            ##### add legend
            axes[0][0].legend()
            
        ##### write description in plots
        axes[0][0].text(1.2, y_max-0.1*(max(latency_measurements["latency (ms)"]) - min(latency_measurements["latency (ms)"])), "ABR measurements", fontsize=14)
    
        ##### remove top and right lines
        axes[0][0].spines['top'].set_visible(False)
        axes[0][0].spines['right'].set_visible(False)
        
        ##### define start plot number for model plots
        model_plot_start = 1
        
    else:
        model_plot_start = 0
    
    ##### loop over plots 
    for ii in range(model_plot_start, nof_rows*nof_cols):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        col = ii-row*nof_cols
        
        ##### turn off x-labels for all but the bottom plots
        if (nof_plots - ii) > nof_cols:
             plt.setp(axes[row][col].get_xticklabels(), visible=False)
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot latencies
        if ii < nof_plots:
            
            ##### building a subset for current model
            model = models[ii-model_plot_start]
            current_model = latency_models[latency_models["model"] == model]
            
            ##### define y axis range
            y_min = min(current_model["latency (ms)"]) - (max(current_model["latency (ms)"]) - min(current_model["latency (ms)"]))*0.05
            y_max = max(current_model["latency (ms)"]) + (max(current_model["latency (ms)"]) - min(current_model["latency (ms)"]))*0.2
            axes[row][col].set_ylim([y_min,y_max])
            
            ##### define x-axis ranges
            axes[row][col].set_xlim([x_min,x_max])
            
            ##### loop over electrode distances
            for jj, electrode_distance in enumerate(electrode_distances):
                
                ##### built subset for current electrode distance
                current_data = current_model[current_model["electrode distance (um)"] == electrode_distance]
                
                ##### plot latency curve
                axes[row][col].plot(current_data["amplitude level"], current_data["latency (ms)"], color = colors[jj], label = "_nolegend_")
                
                ##### show points
                axes[row][col].scatter(current_data["amplitude level"], current_data["latency (ms)"],
                                          color = colors[jj], marker = markers[jj], edgecolor = edgecolors[jj], label = "{} um".format(electrode_distance))
                
            ##### remove top and right lines
            axes[row][col].spines['top'].set_visible(False)
            axes[row][col].spines['right'].set_visible(False)
                
            ##### write model name in plots
            axes[row][col].text(1.2, y_max-0.1*(max(current_model["latency (ms)"]) - min(current_model["latency (ms)"])), "{}".format(eval("{}.display_name".format(model))), fontsize=14)
                
            ##### no grid
            axes[row][col].grid(False)
            
            ##### add legend
            axes[row][col].legend(title = "electrode distance:")
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    
    ##### get labels for the axes
    fig.text(0.5, 0.055, 'stimulus amplitude / threshold', ha='center', fontsize=14)
    fig.text(0.08, 0.5, 'latency / ms', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
#  Plot thresholds for pulse trains
# =============================================================================
def thresholds_for_pulse_trains(plot_name,
                                threshold_data):
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
    models = threshold_data["model"].unique().tolist()
    
    ##### get pulse rates
    pulse_rates = threshold_data["pulses per second"].unique().tolist()
    
    ##### define number of columns
    nof_cols = 2
    
    ##### get number of plots
    nof_plots = len(models)
    
    ##### get number of rows
    nof_rows = np.ceil(nof_plots/nof_cols).astype(int)
    
    ##### define colors and markers
    colors = ["black","red","blue","#33a02c","black","red","blue","#33a02c"]
    markers = ["o","v","s","d","o","v","s","d"]
    edgecolors = ["black","red","blue","#33a02c","black","red","blue","#33a02c"]
              
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### create figure
    fig, axes = plt.subplots(nof_rows, nof_cols, sharex=False, sharey=False, num = plot_name, figsize=(8*nof_cols, 5*nof_rows))
    
    ##### loop over plots 
    for ii in range(nof_rows*nof_cols):
        
        ##### get row and column number
        row = np.floor(ii/nof_cols).astype(int)
        col = ii-row*nof_cols
        
        ##### turn off x-labels for all but the bottom plots
        if (nof_plots - ii) > nof_cols:
             plt.setp(axes[row][col].get_xticklabels(), visible=False)
        
        ##### remove not needed subplots
        if ii >= nof_plots:
            fig.delaxes(axes[row][col])
        
        ##### plot thresholds
        if ii < nof_plots:
            
            ##### building a subset for current model
            model = models[ii]
            current_model = threshold_data[threshold_data["model"] == model]
            
            ##### define y axis range
            y_min = min(current_model["threshold (uA)"]) - (max(current_model["threshold (uA)"]) - min(current_model["threshold (uA)"]))*0.05
            y_max = max(current_model["threshold (uA)"]) + (max(current_model["threshold (uA)"]) - min(current_model["threshold (uA)"]))*0.2
            axes[row][col].set_ylim([y_min,y_max])
            
            ##### loop over electrode distances
            for jj, pps in enumerate(pulse_rates):
                
                ##### built subset for current electrode distance
                current_data = current_model[current_model["pulses per second"] == pps]
                
                ##### plot latency curve
                axes[row][col].plot(current_data["number of pulses"], current_data["threshold (uA)"], color = colors[jj], label = "_nolegend_")
                
                ##### show points
                axes[row][col].scatter(current_data["number of pulses"], current_data["threshold (uA)"],
                    color = colors[jj], marker = markers[jj], edgecolor = edgecolors[jj], label = pps)
                
            ##### logarithmic achses
            axes[row][col].set_xscale('log')
            axes[row][col].set_yscale('log')
            
            ##### remove top and right lines
            axes[row][col].spines['top'].set_visible(False)
            axes[row][col].spines['right'].set_visible(False)
                
            ##### write model name in plots
            axes[row][col].text(1, y_max-0.1*(max(current_model["threshold (uA)"]) - min(current_model["threshold (uA)"])),
                "{}".format(eval("{}.display_name".format(model))), fontsize=14)
                
            ##### no grid
            axes[row][col].grid(False)
            
            ##### add legend
            axes[row][col].legend(title = "Pulses per second:")
    
    ##### bring subplots close to each other.
    fig.subplots_adjust(hspace=0.05, wspace=0.1)
    
    ##### get labels for the axes
    fig.text(0.5, 0.055, 'number of pulses', ha='center', fontsize=14)
    fig.text(0.07, 0.5, 'threshold / uA', va='center', rotation='vertical', fontsize=14)
        
    return fig
