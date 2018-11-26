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

##### import potential values
potential_data = h5py.File('Measurements/Potential_distributions/original_mdl.h5', 'r')

# =============================================================================
# Initializations
# =============================================================================
##### initialize clock
dt = 1*us

##### define way of processing
backend = "serial"

##### define if plots should be generated
generate_plots = True

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
stim_level = 1.3

##### define number stimulated electrode
elec_nr = 2

##### define range of simulated neurons
neuron_range = range(0,200)

# =============================================================================
# Measure threshold current (deterministic models) or current for 50% firing
# efficiency (stochastic models)
# =============================================================================
##### get thresholds for deterministic models
params = {"model_name" : models_deterministic}

thresholds1 = th.util.map(func = fptest.get_threshold_for_pot_dist,
                         space = params,
                         backend = backend,
                         cache = "no",
                         kwargs = {"dt" : dt,
                                   "potential_data" : potential_data,
                                   "elec_nr" : elec_nr,
                                   "neuron_number" : 100,
                                   "phase_duration" : phase_duration,
                                   "nof_pulses" : 10, # further pulses don't change much
                                   "inter_pulse_gap" : inter_pulse_gap,
                                   "delta" : 0.1*uA,
                                   "start_amp" : 1*mA,
                                   "amps_start_interval" : [0,2]*mA,
                                   "pulse_form" : pulse_form,
                                   "add_noise" : False})

thresholds1.reset_index(inplace=True)

##### get currents for 50% firing efficiency for stochastic models
params = {"model_name" : models_stochastic}

thresholds2 = th.util.map(func = fptest.get_threshold_for_fire_eff,
                         space = params,
                         backend = backend,
                         cache = "no",
                         kwargs = {"dt" : dt,
                                   "potential_data" : potential_data,
                                   "elec_nr" : elec_nr,
                                   "neuron_number" :  100,
                                   "fire_eff_desired" : 0.5,
                                   "phase_duration" : phase_duration,
                                   "nof_pulses" : 20, # further pulses don't change much
                                   "inter_pulse_gap" : inter_pulse_gap,
                                   "delta" : 0.1*uA,
                                   "start_amp" : 1*mA,
                                   "amps_start_interval" : [0,2]*mA,
                                   "pulse_form" : pulse_form,
                                   "add_noise" : False})

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
                                     "potential_data" : potential_data,
                                     "elec_nr" : elec_nr,
                                     "phase_duration" : phase_duration,
                                     "nof_pulses" : nof_pulses,
                                     "inter_pulse_gap" : inter_pulse_gap,
                                     "pulse_form" : pulse_form,
                                     "time_before" : 2*ms,
                                     "time_after" : 10*ms})

spike_trains.reset_index(inplace=True)
spike_trains = spike_trains[["model_name","neuron_number","duration","spikes"]]

# =============================================================================
# Generate raster plot
# =============================================================================
#th.plot_raster(spike_trains)















