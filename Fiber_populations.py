# =============================================================================
# This script shows the responses of a population of fibers to a stimulation
# with the realistic potential distribution of a 3D model of a implanted human
# cochlear.
# The potentials are given to a distance of at least 5.5 mm from the fiber terminals.
# =============================================================================
##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import thorns as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import datetime

##### import functions and parameters
import functions.fiber_population_tests as fptest
import functions.create_plots_for_fiber_populations as plot
import functions.create_plots_for_model_comparison as plotcomp
import functions.calculations as calc
import parameters.stim_amp_ranges_dynamic_range as param
import parameters.stim_amp_ranges_dynamic_range_all_elec_same_val as param

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

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### define path of potential distribution data
h5py_path = "Measurements/Potential_distributions/original_mdl.h5"

##### initialize clock
dt = 5*us

##### define way of processing
backend = "serial"

##### define which tests to run
measure_spike_trains_with_threshold = False
measure_spike_trains_with_dynamic_range = False
dynamic_range_test = True

##### define if plots should be generated
generate_plots = False

##### get date
now = datetime.datetime.now()
date = now.strftime("%y%m%d")

##### define which models to observe
models_deterministic = ["briaire_05", "smit_10", "imennov_09"]
models_stochastic = ["rattay_01"]

##### define pulse train
pulse_form = "bi"
phase_duration = 40*us
pulse_rate = 1000/second
pulse_train_duration = 100*ms
nof_pulses = np.floor(pulse_train_duration*pulse_rate).astype(int)
inter_pulse_gap = 1/pulse_rate - 2*phase_duration

##### define stimulus level (multiplied with threshold)
if measure_spike_trains_with_threshold:
    stim_level = 1.5
elif measure_spike_trains_with_dynamic_range:
    desired_number_of_spiking_fibers = 100

##### define number stimulated electrode
elec_nr = 4

##### define range of simulated neurons
neuron_range = range(0,400)

# =============================================================================
# Measure nearest fiber to selected electrode (the one with the highest potenital value)
# =============================================================================
##### open h5py file
with h5py.File(h5py_path, 'r') as potential_data:
    
    max_potentials = [max(potential_data['neuron{}'.format(ii)]["potentials"][:,elec_nr]) for ii in range(0,400)]
    
    max_value = max(max_potentials)
    nearest_fiber = max_potentials.index(max_value)

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

if measure_spike_trains_with_threshold:
    # =============================================================================
    # Measure threshold current (deterministic models) or current for 50% firing
    # efficiency (stochastic models)
    # =============================================================================
    ##### get thresholds for deterministic models
    params = {"model_name" : models_deterministic}
    
    thresholds1 = th.util.map(func = fptest.get_threshold_for_pot_dist,
                             space = params,
                             backend = backend,
                             cache = "yes",
                             kwargs = {"dt" : dt,
                                       "h5py_path" : h5py_path,
                                       "elec_nr" : elec_nr,
                                       "neuron_number" : nearest_fiber,
                                       "phase_duration" : phase_duration,
                                       "nof_pulses" : 10, # further pulses don't change much
                                       "inter_pulse_gap" : inter_pulse_gap,
                                       "delta" : 0.1*uA,
                                       "reference_amp" : 1*mA,
                                       "upper_border" : 2*mA,
                                       "pulse_form" : pulse_form,
                                       "add_noise" : False})
    
    thresholds1.reset_index(inplace=True)
    
    ##### get currents for 50% firing efficiency for stochastic models
    params = {"model_name" : models_stochastic}
    
    thresholds2 = th.util.map(func = fptest.get_threshold_for_fire_eff,
                             space = params,
                             backend = backend,
                             cache = "yes",
                             kwargs = {"dt" : dt,
                                       "h5py_path" : h5py_path,
                                       "elec_nr" : elec_nr,
                                       "neuron_number" :  nearest_fiber,
                                       "fire_eff_desired" : 0.5,
                                       "phase_duration" : phase_duration,
                                       "nof_pulses" : nof_pulses,
                                       "inter_pulse_gap" : inter_pulse_gap,
                                       "delta" : 0.1*uA,
                                       "reference_amp" : 1*mA,
                                       "upper_border" : 2*mA,
                                       "pulse_form" : pulse_form,
                                       "add_noise" : True})
    
    thresholds2.reset_index(inplace=True)
    
    threshold_table = pd.concat([thresholds1, thresholds2], ignore_index = True)

if measure_spike_trains_with_dynamic_range:
    # =============================================================================
    # Calculate stimulus amplitudes for desired number of spiking fibers
    # =============================================================================
    ##### load spike table
    spike_table = pd.read_csv("results/Fiber_population/dynamic_ranges/190201_spikes_per_stim_amp_elec{}.csv".format(elec_nr))
    
    ##### calculate spikes per fiber
    spike_table = spike_table.groupby(["model_name","stim_amp"])["spike"].sum().reset_index()
    spike_table = spike_table.rename(index = str, columns={"spike" : "nof_spikes"})
    
    ##### get model names
    model_names = np.unique(spike_table["model_name"])
    
    ##### initialize dataframe to safe dynamic ranges
    stim_amps = pd.DataFrame(np.zeros((len(model_names),2)), columns=["model_name","stim_amp"])
    
    ##### calculate desired number of spiking fibers
    for ii,model in enumerate(model_names):
        spike_table_model = spike_table[spike_table["model_name"] == model].copy()
        stim_amps["model_name"].iloc[ii] = model        
        stim_amps["stim_amp"].iloc[ii] = spike_table_model["stim_amp"].iloc[(spike_table_model['nof_spikes']-desired_number_of_spiking_fibers).abs().argsort()[:1]].iloc[0]

if measure_spike_trains_with_threshold or measure_spike_trains_with_dynamic_range:
    # =============================================================================
    # Calculate spike times for fiber population
    # =============================================================================
    if measure_spike_trains_with_threshold:
        params = [{"model_name" : model,
                   "add_noise" : True if model in models_stochastic else False,
                   "potential_multiplier" : threshold_table["threshold"][threshold_table["model_name"] == model].iloc[0]/(1*mA) * stim_level,
                   "neuron_number" : ii}
                    for model in models_deterministic + models_stochastic\
                    for ii in neuron_range]
    
    if measure_spike_trains_with_dynamic_range:
        params = [{"model_name" : model,
                   "add_noise" : True if model in models_stochastic else False,
                   "potential_multiplier" : stim_amps["stim_amp"][stim_amps["model_name"] == model].iloc[0]*amp/(1*mA),
                   "neuron_number" : ii}
                    for model in models_deterministic + models_stochastic\
                    for ii in neuron_range]
    
    spike_trains = th.util.map(func = fptest.get_spike_trains,
                               space = params,
                               backend = backend,
                               cache = "no",
                               kwargs = {"dt" : dt,
                                         "h5py_path" : h5py_path,
                                         "elec_nr" : elec_nr,
                                         "phase_duration" : phase_duration,
                                         "nof_pulses" : nof_pulses,
                                         "inter_pulse_gap" : inter_pulse_gap,
                                         "pulse_form" : pulse_form,
                                         "time_before" : 2*ms,
                                         "time_after" : 10*ms})
    
    spike_trains.reset_index(inplace=True)
    spike_trains = spike_trains[["model_name","neuron_number","duration","spikes"]]
    
    ##### split lists in spike times column to multiple rows
    spike_trains = calc.explode(spike_trains, ["spikes"])
    
    ##### add column with pulse rate
    spike_trains["nof_pulses"] = nof_pulses
    
    ##### save dataframe as csv    
    spike_trains.to_csv("results/Fiber_population/spike_trains/{}_spike_trains_elec{}_{}pps.csv".format(date,elec_nr,int(pulse_rate)), index=False, header=True)
    
    if generate_plots:
        # =============================================================================
        # Generate raster plots
        # =============================================================================
        ##### load table with spike times
        spike_trains = pd.read_csv("results/Fiber_population/spike_trains/190204_spike_trains_elec4_1000pps.csv")
        
        ##### generate raster plot for all models
        for model in models_deterministic + models_stochastic:
            
            spike_trains_model = spike_trains[spike_trains["model_name"] == model].copy()
            raster_plot = plot.raster_plot(plot_name = "Raster plot {}2".format(eval("{}.display_name".format(model))),
                                           spike_trains = spike_trains_model)

if dynamic_range_test:
    # =============================================================================
    # Measure number of spiking fibers for different stimulus amplitudes
    # =============================================================================
    for electrode in range(12):
        
        ##### load dict with stimulus amplitude ranges for selected electrode
        stim_amps = param.stim_amps[electrode]
        
        ##### measure if there was a spike for each model, fiber and stimulus amplitude
        params = [{"model_name" : model,
                   "stim_amp" : stim_amp*1e-3,
                   "neuron_number" : ii}
                    for model in models_deterministic + models_stochastic\
                    for stim_amp in stim_amps[model]\
                    for ii in neuron_range]
        
        spike_table = th.util.map(func = fptest.measure_spike,
                                 space = params,
                                 backend = backend,
                                 cache = "no",
                                 kwargs = {"dt" : dt,
                                           "h5py_path" : h5py_path,
                                           "elec_nr" : electrode,
                                           "phase_duration" : phase_duration,
                                           "nof_pulses" : 1,
                                           "pulse_form" : pulse_form,
                                           "add_noise" : False,
                                           "reference_amp" : 1*mA,
                                           "measure_first_spike_location" : True,
                                           "measure_latency" : True})
    
        spike_table.reset_index(inplace=True)
    
        ##### save dataframe as csv    
        spike_table.to_csv("results/Fiber_population/dynamic_ranges/{}_spikes_per_stim_amp_elec{}.csv".format(date,electrode), index=False, header=True)

    if generate_plots:
        # =============================================================================
        # Plot number of spiking fibers over stimulus amplitudes (dynamic range plot)
        # =============================================================================
        ##### define which electrode to show
        electrode = 0
        
        ##### load table with spike times
        spike_table = pd.read_csv("results/Fiber_population/dynamic_ranges/181206_spikes_per_stim_amp_elec{}.csv".format(electrode))
        
        ##### generate dynamic range plot for all models
        for model in models_deterministic + models_stochastic:
            
            spike_table_model = spike_table[spike_table["model_name"] == model].copy()
            
            ##### calculate spikes per fiber
            spike_table_model = spike_table_model.groupby(["model_name","stim_amp"])["spike"].sum().reset_index()
            spike_table_model = spike_table_model.rename(index = str, columns={"spike" : "nof_spikes"})
            
            dyn_range_plot = plot.nof_spikes_over_stim_amp(plot_name = "Dynamic range plot for {}".format(eval("{}.display_name".format(model))),
                                                           spike_table = spike_table_model)
        
        # =============================================================================
        # Plots dB above threshold over distance along spiral lamina and mark location of spike initiation
        # =============================================================================
        ##### define which electrode to show
        electrode = 3
        
        ##### load table with spike times
        spike_table = pd.read_csv("results/Fiber_population/dynamic_ranges/181206_spikes_per_stim_amp_elec{}.csv".format(electrode))
        
        ##### add distances of fibers from base
        spike_table = pd.merge(spike_table, distances, on=["neuron_number"])
        
        ##### generate plot for all models
        for model_name in models_deterministic + models_stochastic:
            
            ##### get model module
            model = eval(model_name)
            
            ##### build subset for current model
            spike_table_model = spike_table[spike_table["model_name"] == model_name].copy()
            spike_table_model = spike_table_model[spike_table_model["spike"] == 1]
            
            ##### add distance of compartments to terminal
            spike_table_model["first_spike_dist"] = [np.cumsum(model.compartment_lengths)[int(spike_table_model["first_spike_comp"].iloc[ii])]/mm for ii in range(spike_table_model.shape[0])]
            
            ##### generate plot
            dynamic_range_color_plot = plot.spikes_color_plot(plot_name = "dB above threshold over distance along spiral lamina plot for {}".format(eval("{}.display_name".format(model_name))),
                                                                 spike_table = spike_table_model)
            
            ##### save plots
#            dynamic_range_color_plot.savefig("results/{}/dynamic_ranges/dynamic_range_color_plot_elec{}.pdf".format(model.display_name,electrode),
#                                             bbox_inches='tight', dpi=300)
            
        # =============================================================================
        # Plots dB above threshold over distance along spiral lamina and mark latency
        # =============================================================================
        ##### define which electrode to show
        electrode = 6
        
        ##### load table with spike times
        spike_table = pd.read_csv("results/Fiber_population/dynamic_ranges/181206_spikes_per_stim_amp_elec{}.csv".format(electrode))
        
        ##### add distances of fibers from base
        spike_table = pd.merge(spike_table, distances, on=["neuron_number"])
        
        ##### generate plot for all models
        for model_name in models_deterministic + models_stochastic:
            
            ##### get model module
            model = eval(model_name)
            
            ##### build subset for current model
            spike_table_model = spike_table[spike_table["model_name"] == model_name].copy()
            spike_table_model = spike_table_model[spike_table_model["spike"] == 1]
            spike_table_model = spike_table_model[spike_table_model["spike_at_last_comp"] == False]
            
            ##### generate plot
            latency_color_plot = plot.latencies_color_plot(plot_name = "dB above threshold over distance along spiral lamina for {}".format(eval("{}.display_name".format(model_name))),
                                                           spike_table = spike_table_model)
        
        # =============================================================================
        # Plots dB above threshold over distance along spiral lamina for cathodic, anodic and biphasic pulses
        # =============================================================================
        ##### define which electrode to show
        electrode = 5
        
        ##### load tables with spike times
        spike_table_bi = pd.read_csv("results/Fiber_population/dynamic_ranges/181206_spikes_per_stim_amp_elec{}.csv".format(electrode))
        spike_table_cat = pd.read_csv("results/Fiber_population/dynamic_ranges/181211_spikes_per_stim_amp_elec{}_cat.csv".format(electrode))
        spike_table_ano = pd.read_csv("results/Fiber_population/dynamic_ranges/181217_spikes_per_stim_amp_elec{}_ano.csv".format(electrode))

        ##### add column with pulse form information
        spike_table_bi["pulse_form"] = "biphasic"
        spike_table_cat["pulse_form"] = "cathodic"
        spike_table_ano["pulse_form"] = "anodic"

        ##### connect dataframes
        spike_table = pd.concat([spike_table_bi, spike_table_cat], ignore_index = True)
        spike_table = pd.concat([spike_table, spike_table_ano], ignore_index = True)
        
        ##### remove rows without spike
        spike_table = spike_table[spike_table["spike"] == 1]
        spike_table = spike_table[spike_table["spike_at_last_comp"] == False]
        
        ##### get absolute values of stimulus amplitudes
        spike_table["stim_amp"] = abs(spike_table["stim_amp"])
        
        ##### calculate dynamic range values
        dyn_range = spike_table.groupby(["model_name"])["stim_amp"].min().reset_index()
        dyn_range = dyn_range.rename(index = str, columns={"stim_amp" : "min_amp_spike"})
        spike_table = pd.merge(spike_table, dyn_range, on=["model_name"])
        spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/spike_table["min_amp_spike"])
        
        ##### get minimum amplitudes with spikes for each fiber
        spike_table = spike_table.groupby(["model_name", "neuron_number", "pulse_form"])["dynamic_range"].min().reset_index()
        
        ##### add distances of fibers from base
        spike_table = pd.merge(spike_table, distances, on=["neuron_number"])
                
        ##### add electrode number
        spike_table["elec_nr"] = electrode
        
        ##### generate plot
        pulse_form_comparison_plot = plot.compare_pulse_forms(plot_name = "dB above threshold over distance along spiral lamina plot for different pulse forms",
                                                              spike_table = spike_table)
        