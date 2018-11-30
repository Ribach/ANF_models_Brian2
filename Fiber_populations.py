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
import functions.calculations as calc
import parameters.stim_amp_ranges_dynamic_range as param

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14
import models.Rudnicki_2018 as rudnicki_18

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
models_deterministic = ["briaire_05", "smit_10"]
models_stochastic = ["rattay_01", "imennov_09"]

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
                                       "amps_start_interval" : [0,2]*mA,
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
                                       "amps_start_interval" : [0,2]*mA,
                                       "pulse_form" : pulse_form,
                                       "add_noise" : True})
    
    thresholds2.reset_index(inplace=True)
    
    threshold_table = pd.concat([thresholds1, thresholds2], ignore_index = True)

if measure_spike_trains_with_dynamic_range:
    # =============================================================================
    # Calculate stimulus amplitudes for desired number of spiking fibers
    # =============================================================================
    ##### load spike table
    spike_table = pd.read_csv("test_battery_results/Fiber_population/dynamic_ranges/181129_spikes_per_stim_amp_elec4.csv")
    
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
    spike_trains.to_csv("test_battery_results/Fiber_population/spike_trains/{}_spike_trains_elec{}_{}pps.csv".format(date,elec_nr,int(pulse_rate)), index=False, header=True)
    
    if generate_plots:
        # =============================================================================
        # Generate raster plots
        # =============================================================================
        ##### load table with spike times
        spike_trains = pd.read_csv("test_battery_results/Fiber_population/spike_trains/181130_spike_trains_elec4_1000pps_new.csv")
        
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
                                           "measure_first_spike_location" : True})
    
        spike_table.reset_index(inplace=True)
    
        ##### save dataframe as csv    
        spike_table.to_csv("test_battery_results/Fiber_population/dynamic_ranges/{}_spikes_per_stim_amp_elec{}.csv".format(date,electrode), index=False, header=True)

    if generate_plots:
        # =============================================================================
        # Plot number of spiking fibers over stimulus amplitudes
        # =============================================================================
        ##### load table with spike times
        spike_table = pd.read_csv("test_battery_results/Fiber_population/dynamic_ranges/181129_spikes_per_stim_amp_elec4.csv")
        
        ##### generate raster plot for all models
        for model in models_deterministic + models_stochastic:
            
            spike_table_model = spike_table[spike_table["model_name"] == model].copy()
            
            ##### calculate spikes per fiber
            spike_table_model = spike_table_model.groupby(["model_name","stim_amp"])["spike"].sum().reset_index()
            spike_table_model = spike_table_model.rename(index = str, columns={"spike" : "nof_spikes"})
            
            raster_plot = plot.nof_spikes_over_stim_amp(plot_name = "Dynamic range plot for {}".format(eval("{}.display_name".format(model))),
                                                        spike_table = spike_table_model)
        
#        # =============================================================================
#        # Plots dynamic range over fiber indexes
#        # =============================================================================
#        ##### load table with spike times
#        spike_table = pd.read_csv("test_battery_results/Fiber_population/dynamic_ranges/181129_spikes_per_stim_amp_elec4.csv")
#        
#        ##### generate plot for all models
#        for model_name in models_deterministic + models_stochastic:
#            
#            ##### get model module
#            model = eval(model_name)
#            
#            ##### build subset for current model
#            spike_table_model = spike_table[spike_table["model_name"] == model_name].copy()
#            spike_table_model = spike_table_model[spike_table_model["spike"] == 1]
#            ##### define color depending on compartment
#            spike_table_model["color_index"] = 0
#            
#            min(spike_table_model["first_spike_comp"])
#            
#            model.start_index_soma
#            
#            
#            
#            spike_table_model["color_index"][spike_table_model["first_spike_comp"] < model.start_index_soma]
#            
#            spike_table_model["first_spike_comp"][spike_table_model["first_spike_comp"] < model.start_index_soma] / max(spike_table_model["first_spike_comp"][spike_table_model["first_spike_comp"] < model.start_index_soma])
#            
#            
#            raster_plot = plot.nof_spikes_over_stim_amp(plot_name = "Dynamic range plot for {}".format(eval("{}.display_name".format(model))),
#                                                        spike_table = spike_table_model)














