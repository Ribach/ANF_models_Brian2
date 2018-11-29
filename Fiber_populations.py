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

##### import functions
import functions.fiber_population_tests as fptest
import functions.create_plots_for_fiber_populations as plot
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
spike_train_test = False
dynamic_range_test = True

##### define if plots should be generated
generate_plots = False

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
stim_level = 1.5

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

if spike_train_test:
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
    
    # =============================================================================
    # Calculate spike times for fiber population
    # =============================================================================
    params = [{"model_name" : model,
               "add_noise" : True if model in models_stochastic else False,
               "potential_multiplier" : threshold_table["threshold"][threshold_table["model_name"] == model].iloc[0]/(1*mA) * stim_level,
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
    spike_trains.to_csv("test_battery_results/Fiber_population/spike_trains.csv", index=False, header=True)
    
    if generate_plots:
        # =============================================================================
        # Generate raster plots
        # =============================================================================
        ##### load table with spike times
        spike_trains = pd.read_csv("test_battery_results/Fiber_population/181128_spike_trains_elec2_new.csv")
        
        ##### generate raster plot for all models
        for model in models_deterministic + models_stochastic:
            
            spike_trains_model = spike_trains[spike_trains["model_name"] == model].copy()
            raster_plot = plot.raster_plot(plot_name = "Raster plot {}".format(eval("{}.display_name".format(model))),
                                           spike_trains = spike_trains_model)

if dynamic_range_test:
    # =============================================================================
    # Measure number of spiking fibers for different stimulus amplitudes
    # =============================================================================
    ##### define stimulus amplitudes in mA (different ranges for the models)
    stim_amps = {"rattay_01" : np.linspace(0,5,50),
                 "briaire_05" : np.linspace(0,4,40),
                 "smit_10" : np.linspace(0,7,70),
                 "imennov_09" : np.linspace(0,2,30)}
    
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
                                       "elec_nr" : elec_nr,
                                       "phase_duration" : phase_duration,
                                       "nof_pulses" : 1,
                                       "pulse_form" : pulse_form,
                                       "add_noise" : False,
                                       "reference_amp" : 1*mA})

    spike_table.reset_index(inplace=True)
    
    ##### calculate spikes per fiber
    spike_table = spike_table.groupby(["model_name","stim_amp"])["spike"].sum().reset_index()
    spike_table = spike_table.rename(index = str, columns={"spike" : "nof_spikes"})

    ##### save dataframe as csv    
    spike_table.to_csv("test_battery_results/Fiber_population/spikes_per_stim_amp.csv", index=False, header=True)

    if generate_plots:
        # =============================================================================
        # Plot number of spiking fibers over stimulus amplitudes
        # =============================================================================
        ##### load table with spike times
        spike_table = pd.read_csv("test_battery_results/Fiber_population/spikes_per_stim_amp.csv")
        
        ##### generate raster plot for all models
        for model in models_deterministic + models_stochastic:
            
            spike_table_model = spike_table[spike_table["model_name"] == model].copy()
            raster_plot = plot.nof_spikes_over_stim_amp(plot_name = "Dynamic range plot for {}".format(eval("{}.display_name".format(model))),
                                                        spike_table = spike_table_model)








