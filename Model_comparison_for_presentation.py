# =============================================================================
# This script collects the fiber population results of all (!) models and generates
# and saves plots that compare the results among each other.
# =============================================================================
##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import itertools as itl
import h5py

##### import models
import models.Rattay_2001 as rattay_01
import models.Briaire_2005 as briaire_05
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09

##### import functions
import functions.stimulation as stim
import functions.create_plots_for_presentation as plot

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### list of all models
models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]

##### save plots
save_plots = True
presentation_image_path = "C:/Users/Richard/Documents/Studium/Master Elektrotechnik/Semester 4/Masterarbeit/Abschlussvortrag/Bilder"

##### define path of potential distribution data
h5py_path = "Measurements/Potential_distributions/original_mdl.h5"

##### define range of simulated neurons
neuron_range = range(0,400)

##### define range of considered electrodes
electrode_range = range(0,12)

# =============================================================================
# Calculate the distances of the peripheral terminal of the neurons from the origin
# =============================================================================
##### open h5py file
with h5py.File(h5py_path, 'r') as potential_data:
    
    ##### initialize dataframe
    distances = pd.DataFrame(np.zeros((len(neuron_range),2)), columns = ["neuron_number", "dist_along_sl"])
    distances["neuron_number"] = neuron_range
    
    ##### calculate distances along the spiral lamina
    for ii in range(1,len(neuron_range)):
        
        coord = potential_data['neuron{}'.format(ii)]["coordinates"][0,:]
        if ii == 1: last_coord = potential_data['neuron{}'.format(ii-1)]["coordinates"][0,:]
        
        distances["dist_along_sl"].iloc[ii] = distances["dist_along_sl"].iloc[ii-1] + \
            np.sqrt((coord[0] - last_coord[0])**2 + (coord[1] - last_coord[1])**2 + (coord[2] - last_coord[2])**2)*1e3
        
        last_coord = coord
    
    ##### set minimum distance to zero
    distances["dist_along_sl"] = distances["dist_along_sl"] - min(distances["dist_along_sl"])

# =============================================================================
# Raster plot comparison
# =============================================================================
##### define electrode number
elec_nr = 5

##### get data
for ii, pulse_rate in enumerate([100,1000]):
    
    ##### load table with spike times
    data = pd.read_csv("results/Fiber_population/spike_trains/190204_spike_trains_elec{}_{}pps.csv".format(elec_nr, pulse_rate))
    
    ##### add electrode number
    data["pulse_rate"] = pulse_rate
    
    ##### connect data of all pulse rates in one dataframe
    if ii == 0:
        spike_table = data
    else:
        ##### add column with AP shape data of current model
        spike_table = pd.concat([spike_table, data])

##### delete rows whith no spike
spike_table = spike_table[pd.to_numeric(spike_table['spikes'], errors='coerce').notnull()]

##### only consider first 50 ms
spike_table["duration"] = 0.05
spike_table = spike_table[spike_table["spikes"] <= spike_table["duration"].iloc[0]]

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])
spike_table["max_dist_along_sl"] = max(distances["dist_along_sl"])

##### add electrode number
spike_table["elec_nr"] = elec_nr

##### plot comparison
spike_table_100pps = spike_table[spike_table["pulse_rate"] == 100].copy()
raster_plot_comparison_100pps = plot.raster_plot_comparison_presentation(plot_name = "Raster plot comparison 100 pps",
                                                                         spike_table = spike_table_100pps)
spike_table_1000pps = spike_table[spike_table["pulse_rate"] == 1000].copy()
raster_plot_comparison_1000pps = plot.raster_plot_comparison_presentation(plot_name = "Raster plot comparison 1000 pps",
                                                                          spike_table = spike_table_1000pps)
        
##### save plot
if save_plots:
    raster_plot_comparison_100pps.savefig("{}/raster_plot_comparison_100pps.png".format(presentation_image_path), bbox_inches='tight')
    raster_plot_comparison_1000pps.savefig("{}/raster_plot_comparison_1000pps.png".format(presentation_image_path), bbox_inches='tight')

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked location of spike initiation for multiple electrodes
# =============================================================================
##### define which electrodes to show
electrodes = [0,11]

##### get data
for ii, electrode in enumerate(electrodes):
    
    ##### load tables with spike times
    data = pd.read_csv("results/Fiber_population/dynamic_ranges/190201_spikes_per_stim_amp_elec{}.csv".format(electrode))
    
    ##### add electrode information
    data["elec_nr"] = electrode
    
    ##### loop over models
    for jj, model_name in enumerate(models):
        
        ##### get model module
        model = eval(model_name)
        
        ##### build subset for current model
        data_model = data[data["model_name"] == model_name].copy()
        
        ##### remove rows without spike
        data_model = data_model[(data_model["spike"] == 1) & (data_model["spike_at_last_comp"] == False)]
        
        ##### add distance of compartments to terminal
        data_model["first_spike_dist"] = [np.cumsum(model.compartment_lengths)[int(data_model["first_spike_comp"].iloc[kk])]/mm for kk in range(data_model.shape[0])]
        
        ##### connect data of all electrodes and models in one dataframe
        if (ii == 0) & (jj==0):
            spike_table = data_model
        else:
            ##### add column with AP shape data of current model
            spike_table = pd.concat([spike_table, data_model])

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])

##### calculate dynamic range values
dyn_range = spike_table.groupby(["model_name", "elec_nr"])["stim_amp"].min().reset_index()
dyn_range = dyn_range.rename(index = str, columns={"stim_amp" : "min_amp_spike"})
spike_table = pd.merge(spike_table, dyn_range, on=["model_name", "elec_nr"])
spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/spike_table["min_amp_spike"])

##### generate plot
first_spike_color_plot_comparison_multiple_electrodes = plot.spikes_color_plot_comparison_presentation(plot_name = "dB above threshold over distance along spiral lamina plot comparison",
                                                                                                       spike_table = spike_table)

##### save plot
if save_plots:
    first_spike_color_plot_comparison_multiple_electrodes.savefig("{}/first_spike_color_plot_comparison_multiple_electrodes.png".format(presentation_image_path), bbox_inches='tight')

# =============================================================================
# Single node response plot
# =============================================================================
##### initialize list of dataframes to save voltage courses
voltage_courses = [pd.DataFrame()]*len(models)

##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"

##### loop over models
for ii,model_name in enumerate(models):
    
    model = eval(model_name)
    
    ##### get voltage course of model
    voltage_course_dataset = pd.read_csv("results/{}/Single_node_response_plot_data_deterministic {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    voltage_course_dataset["model"] = model.display_name_plots
    
    ##### shift voltage courses to the left
    if model_name == "rattay_01":
        voltage_course_dataset["time (ms)"] = voltage_course_dataset["time (ms)"] - 0.4
    elif model_name == "briaire_05":
        voltage_course_dataset["time (ms)"] = voltage_course_dataset["time (ms)"] - 0.12
    elif model_name == "smit_10":
        voltage_course_dataset["time (ms)"] = voltage_course_dataset["time (ms)"] - 0.2        
    elif model_name == "imennov09":
        voltage_course_dataset["time (ms)"] = voltage_course_dataset["time (ms)"] - 0.05
    
    ##### write subset of dataframe in voltage courses list
    voltage_courses[ii] = voltage_course_dataset[["model", "membrane potential (mV)","time (ms)", "amplitude level"]]\
                                                [voltage_course_dataset["pulse form"] == pulse_form]\
                                                [voltage_course_dataset["phase duration (us)"] == phase_duration/us]

##### connect dataframes to one dataframe
voltage_courses = pd.concat(voltage_courses,ignore_index = True)

##### delete rows with negative time
voltage_courses = voltage_courses[voltage_courses["time (ms)"] >= 0]

##### show only stimulation with threshold
voltage_courses = voltage_courses[voltage_courses["amplitude level"] == "1*threshold"]

##### plot voltage courses
single_node_response = plot.single_node_response_comparison_presentation(plot_name = "Voltage courses model comparison",
                                                                         voltage_data = voltage_courses)

##### save plot
if save_plots:
    single_node_response.savefig("{}/single_node_response comparison.png".format(presentation_image_path), bbox_inches='tight')

# =============================================================================
# PSTHs
# =============================================================================
##### initialize list of dataframes to save psth data for each model
psth_data = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model_name in enumerate(models):
    
    model = eval(model_name)
    
    ##### get psth data of model
    psth_data[ii] = pd.read_csv("results/{}/PSTH_table {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    psth_data[ii]["model"] = model.display_name_plots

##### connect dataframes to one dataframe
psth_data = pd.concat(psth_data,ignore_index = True)

##### convert spike times to ms
psth_data["spike times (us)"] = np.ceil(list(psth_data["spike times (us)"]*1000)).astype(int)
psth_data = psth_data.rename(index = str, columns={"spike times (us)" : "spike times (ms)"})

##### plot PSTH comparison
psth_plot = plot.psth_comparison_presentation(plot_name = "PSTH model comparison",
                                              psth_data = psth_data,
                                              amplitudes = ['1*threshold'],
                                              pulse_rates = [400, 800, 2000, 5000],
                                              plot_style = "spikes_per_time_bin")

##### save plot
if save_plots:
    psth_plot.savefig("{}/psth_plot_comparison_thr.png".format(presentation_image_path), bbox_inches='tight')

# =============================================================================
# Strength duration curve
# =============================================================================
##### initialize list of dataframes to save strength duration curves
stength_duration_curves_cat = [pd.DataFrame()]*len(models)
stength_duration_curves_ano = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model_name in enumerate(models):
    
    model = eval(model_name)
    
    ##### read strength duration curves
    stength_duration_curves_cat[ii] = pd.read_csv("results/{}/Strength_duration_plot_table_cathodic {}.csv".format(model.display_name,model.display_name))
    stength_duration_curves_ano[ii] = pd.read_csv("results/{}/Strength_duration_plot_table_anodic {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    stength_duration_curves_cat[ii]["model"] = model.display_name_plots
    stength_duration_curves_ano[ii]["model"] = model.display_name_plots

##### connect list of dataframes to one dataframe
stength_duration_curves_cat = pd.concat(stength_duration_curves_cat,ignore_index = True)
stength_duration_curves_ano = pd.concat(stength_duration_curves_ano,ignore_index = True)

##### plot strength duration curve
strength_duration_curve = plot.strength_duration_curve_comparison_presentation(plot_name = "Strength duration curve model comparison",
                                                                               threshold_data_cat = stength_duration_curves_cat,
                                                                               threshold_data_ano = stength_duration_curves_ano)

##### save plot
if save_plots:
    strength_duration_curve.savefig("{}/strength_duration_curve comparison.png".format(presentation_image_path), bbox_inches='tight')
    
# =============================================================================
# Plot voltage course for all models
# =============================================================================
stim_amps = [0.2, 1.5, 0.3,0.1]
max_node = [14,24,55,23]
max_comp = [0,0,0,0]

##### initialize list to save voltage courses
voltage_courses =  [ [] for i in range(len(models)) ]

for ii, model_name in enumerate(models):
    
    ##### get model
    model = eval(model_name)
    
    ##### just save voltage values for a certain compartment range
    max_comp[ii] = np.where(model.structure == 2)[0][max_node[ii]]
    
    ##### set up the neuron
    neuron, model = model.set_up_model(dt = 5*us, model = model)
    
    ##### record the membrane voltage
    M = StateMonitor(neuron, 'v', record=True)
    
    ##### save initialization of the monitor(s)
    store('initialized')

    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = 5*us,
                                                pulse_form = "mono",
                                                stimulation_type = "intern",
                                                time_before = 0.2*ms,
                                                time_after = 1.5*ms,
                                                stimulated_compartment = np.where(model.structure == 2)[0][1],
                                                ##### monophasic stimulation
                                                amp_mono = stim_amps[ii]*nA,
                                                duration_mono = 100*us)
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = 5*us)
            
    ##### run simulation
    run(runtime)
    
    ##### save M.v in voltage_courses
    voltage_courses[ii] = M.v[:max_comp[ii],:]

##### Plot membrane potential of all compartments over time
voltage_course_comparison = plot.voltage_course_comparison_plot_presentation(plot_name = "Voltage courses all models2",
                                                                             model_names = models,
                                                                             time_vector = M.t,
                                                                             max_comp = max_comp,
                                                                             voltage_courses = voltage_courses)

##### save plot
if save_plots:
    voltage_course_comparison.savefig("{}/voltage_course_comparison_plot.png".format(presentation_image_path), bbox_inches='tight')

# =============================================================================
# Refractory curves
# =============================================================================
##### initialize list of dataframes to save voltage courses
refractory_curves = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model_name in enumerate(models):
    
    model = eval(model_name)
    
    ##### get voltage course of model
    refractory_curves[ii] = pd.read_csv("results/{}/Refractory_curve_table {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    refractory_curves[ii]["model"] = model.display_name_plots

##### connect dataframes to one dataframe
refractory_curves = pd.concat(refractory_curves,ignore_index = True)

##### remove rows where no second spikes were obtained
refractory_curves = refractory_curves[refractory_curves["minimum required amplitude"] != 0]
    
##### calculate the ratio of the threshold of the second spike and the masker
refractory_curves["threshold ratio"] = refractory_curves["minimum required amplitude"]/refractory_curves["threshold"]

##### convert interpulse intervals to ms
refractory_curves["interpulse interval"] = refractory_curves["interpulse interval"]*1e3

##### plot voltage courses
refractory_curves_plot = plot.refractory_curves_comparison_presentation(plot_name = "Refractory curves model comparison",
                                                                        refractory_curves = refractory_curves)

##### save plot
if save_plots:
    refractory_curves_plot.savefig("{}/refractory_curves_plot comparison.png".format(presentation_image_path), bbox_inches='tight')

# =============================================================================
# Stochastic responses
# =============================================================================
relative_spreads = pd.read_csv("results/Analyses/relative_spreads_k_noise_comparison.csv")
single_node_response_table = pd.read_csv("results/Analyses/single_node_response_table_k_noise_comparison.csv")

##### Combine relative spread and jitter information and exclude rows with na values
stochasticity_table = pd.merge(relative_spreads, single_node_response_table, on=["model","knoise ratio"]).dropna()

##### Exclude relative spreads bigger than 30% and jitters bigger than 200 us
stochasticity_table = stochasticity_table[(stochasticity_table["relative spread (%)"] < 30) & (stochasticity_table["jitter (us)"] < 200)]

##### plot table
stochasticity_plot = plot.stochastic_properties_presentation(plot_name = "Comparison of stochastic properties",
                                                             stochasticity_table = stochasticity_table)

##### save plot
stochasticity_plot.savefig("{}/stochasticity_plot.png".format(presentation_image_path), bbox_inches='tight')

# =============================================================================
# Dynamic range comparison (among electrodes and models)
# =============================================================================
##### get dynamic range plot comparison for all models and electrodes
for ii, electrode in enumerate(electrode_range):
    
    ##### get strength duration data
    data = pd.read_csv("results/Fiber_population/dynamic_ranges/181206_spikes_per_stim_amp_elec{}.csv".format(electrode))
    
    ##### add electrode number
    data["elec_nr"] = electrode
    
    ##### connect data of all electrodes in one dataframe
    if ii == 0:
        dyn_range_table = data
    else:
        ##### add column with AP shape data of current model
        dyn_range_table = pd.concat([dyn_range_table, data])
    
##### calculate spikes per fiber
dyn_range_table = dyn_range_table.groupby(["model_name","stim_amp", "elec_nr"])["spike"].sum().reset_index()
dyn_range_table = dyn_range_table.rename(index = str, columns={"spike" : "nof_spikes"})

##### plot comparison
dyn_range_comparison = plot.nof_spikes_over_stim_amp_presentation(plot_name = "Comparison of dynamic ranges",
                                                                spike_table = dyn_range_table)

##### save plot
if save_plots:
    dyn_range_comparison.savefig("{}/dyn_range_comparison.png".format(presentation_image_path), bbox_inches='tight')
    
# =============================================================================
# Comparison of excitation differences of cathodic, anodic and biphasic pulses
# =============================================================================
##### define which electrodes to show
electrodes = [5,11]

##### get data
for ii, electrode in enumerate(electrodes):
    
    ##### load tables with spike times
    spike_table_bi = pd.read_csv("results/Fiber_population/dynamic_ranges/190201_spikes_per_stim_amp_elec{}.csv".format(electrode))
    spike_table_cat = pd.read_csv("results/Fiber_population/dynamic_ranges/190131_spikes_per_stim_amp_elec{}_cat.csv".format(electrode))
    spike_table_ano = pd.read_csv("results/Fiber_population/dynamic_ranges/190204_spikes_per_stim_amp_elec{}_ano.csv".format(electrode))
    
    ##### add column with pulse form information
    spike_table_bi["pulse_form"] = "biphasic"
    spike_table_cat["pulse_form"] = "cathodic"
    spike_table_ano["pulse_form"] = "anodic"
    
    ##### connect dataframes
    data = pd.concat([spike_table_bi, spike_table_cat], ignore_index = True)
    data = pd.concat([data, spike_table_ano], ignore_index = True)
    
    ##### add electrode information
    data["elec_nr"] = electrode
    
    ##### connect data of all electrodes in one dataframe
    if ii == 0:
        spike_table = data
    else:
        ##### add column with AP shape data of current model
        spike_table = pd.concat([spike_table, data])

##### remove rows without spike
spike_table = spike_table[spike_table["spike"] == 1]
spike_table = spike_table[spike_table["spike_at_last_comp"] == False]

##### get absolute values of stimulus amplitudes
spike_table["stim_amp"] = abs(spike_table["stim_amp"])

##### calculate dynamic range values
dyn_range = spike_table.groupby(["model_name", "elec_nr"])["stim_amp"].min().reset_index()
dyn_range = dyn_range.rename(index = str, columns={"stim_amp" : "min_amp_spike"})
spike_table = pd.merge(spike_table, dyn_range, on=["model_name", "elec_nr"])
spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/spike_table["min_amp_spike"])

##### get minimum amplitudes with spikes for each fiber
spike_table = spike_table.groupby(["model_name", "elec_nr", "neuron_number", "pulse_form"])["dynamic_range"].min().reset_index()

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])

##### generate plot
pulse_form_comparison_plot = plot.compare_pulse_forms_for_multiple_electrodes_presentation(plot_name = "Compare excation of anodic, cathodic and biphasic pulses for different electrodes",
                                                                                           spike_table = spike_table)

##### save plot
if save_plots:
    pulse_form_comparison_plot.savefig("{}/pulse_form_comparison_plot.png".format(presentation_image_path), bbox_inches='tight')