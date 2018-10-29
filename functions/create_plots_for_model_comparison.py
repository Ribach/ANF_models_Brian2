from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)

##### import functions
import functions.create_plots as plot

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

# =============================================================================
#  Voltage course comparison
# =============================================================================
def voltage_course_comparison_plot(plot_name,
                                   model_names,
                                   time_vector,
                                   voltage_courses):
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
    
    ##### change strings to float
    model_data["velocity (m/s)"] = model_data["velocity (m/s)"].astype(float)
    model_data["outer diameter (um)"] = model_data["outer diameter (um)"].astype(float)
    
    ##### experimental data of Hursh 1939
    x_Hursh = np.array([2,20])
    y_Hursh = x_Hursh*6
    
    ##### experimental data of Boyd and Kalu 1979
    x_Boyd_1 = np.array([3,12])
    y_Boyd_1 = x_Boyd_1*4.6
    
    x_Boyd_2 = np.array([10,20])
    y_Boyd_2 = x_Boyd_2*5.7
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(7, 6))
    
    ##### no grid
    axes.grid(False)
    
    ##### Plot AP rise time values of models
    for ii in range(len(model_data)):
        
        ##### plot point
        axes.scatter(model_data["outer diameter (um)"][ii], model_data["velocity (m/s)"][ii],
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
    
    fig.text(0.5, 0.001, 'Time (ms)', ha='center', fontsize=14)
    fig.text(0.03, 0.5, 'Membrane potential (mV)', va='center', rotation='vertical', fontsize=14)
        
    return fig

# =============================================================================
#  Plot rise-time over conduction velocity according to Paintal 1966
# =============================================================================
def paintal_rise_time_curve(plot_name,
                            model_data):
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

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(7, 5))
    
    ##### no grid
    axes.grid(False)
    
    ##### Plot AP rise time curve of Paintal
    axes.plot(x_values,paintal_rise_time(x_values), color = "black", label = "Experimental data from Paintal 1965")
    
    ##### Plot AP rise time values of models
    for model in models:
        
        ##### building a subset
        current_data = model_data[model_data["model_name"] == model]
        
        ##### plot point
        axes.scatter(current_data["conduction velocity (m/s)"],current_data["rise time (us)"], label = "{}".format(model))

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
    
    ##### define x-vector for curves
    x_values = np.linspace(0,80,100)
    
    ##### Get points of AP fall time curve of Paintal 1965 (with software Engague Digitizer)
    velocity = [16,64]
    AP_fall_time_paintal = [350,263]
    
    ##### Interpolate AP fall time curve linearly
    paintal_fall_time = np.poly1d(np.polyfit(velocity,AP_fall_time_paintal,1))
    
    ##### get model names
    models = model_data["model_name"].tolist()

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(7, 5))
    
    ##### no grid
    axes.grid(False)
    
    ##### Plot AP rise time curve of Paintal
    axes.plot(x_values,paintal_fall_time(x_values), color = "black", label = "Experimental data from Paintal 1965")
    
    ##### Plot AP rise time values of models
    for model in models:
        
        ##### building a subset
        current_data = model_data[model_data["model_name"] == model]
        
        ##### plot point
        axes.scatter(current_data["conduction velocity (m/s)"],current_data["fall time (us)"], label = "{}".format(model))
            
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
                                       threshold_matrix,
                                       strength_duration_table):
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
    
    ##### get model names
    models = threshold_matrix["model"].unique().tolist()
    
    ##### exclude values, where no threshold was found
    threshold_matrix = threshold_matrix.loc[threshold_matrix["threshold (uA)"] != 0]
    
    ##### get y range
    y_min = -0.5
    y_max = max(threshold_matrix["threshold (uA)"]) + 1
    
    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig, axes = plt.subplots(1, 1, num = plot_name, figsize=(7, 6))
    
    ##### no grid
    axes.grid(False)
    
    ##### loop over models
    for model in models:
        
        ##### building a subset
        current_data = threshold_matrix[threshold_matrix["model"] == model]

        ##### plot strength duration curve    
        axes.semilogx(current_data["phase duration (us)"], current_data["threshold (uA)"], label = model)
        
#        ##### mark chronaxie
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

    ##### close possibly open plots
    plt.close(plot_name)
    
    ##### generate figure
    fig = plt.figure(plot_name)
    
    ##### create boxplots
    sns.set_style("whitegrid")
    sns.boxplot(data=threshold_matrix, x="phase duration (us)", y="threshold", hue="noise level", showfliers=False, dodge=True)
    plt.xlabel('Phase duration / us', fontsize=16)
    plt.ylabel('Threshold / uA', fontsize=16)
        
    return fig

