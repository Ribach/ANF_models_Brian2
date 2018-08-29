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

##### import functions
import functions.stimulation as stim
import functions.create_plots as plot
import functions.model_tests as test

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Frijns_2005 as frijns_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Definition of neuron and initialization of state monitor
# =============================================================================
##### choose model
model = rattay_01

##### initialize clock
dt = 5*us

##### set up the neuron
neuron, param_string, model = model.set_up_model(dt = dt, model = model)

##### load the parameters of the differential equations in the workspace
exec(param_string)

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

##### save initialization of the monitor(s)
store('initialized')

# =============================================================================
# Get chronaxie and rheobase
# =============================================================================
rheobase_matrix = test.get_thresholds(model = model,
                                      dt = dt,
                                      phase_durations = 1*ms,
                                      amps_start_intervall = [0,20]*uA,
                                      delta = 0.0001*uA,
                                      stimulation_type = "extern",
                                      pulse_form = "mono")

rheobase = rheobase_matrix["threshold"][0]*amp

chronaxie = test.get_chronaxie(model = model,
                               dt = dt,
                               rheobase = rheobase,
                               phase_duration_start_intervall = [0,1000]*us,
                               delta = 1*us,
                               stimulation_type = "extern",
                               pulse_form = "mono")

# =============================================================================
# Get strength-duration curve
# =============================================================================
##### define phase durations
phase_durations = np.round(np.logspace(1, 9, num=20, base=2.0))*us

##### calculate deterministic thresholds for the given phase durations and monophasic stimulation
strength_duration_matrix = test.get_thresholds(model = model,
                                               dt = 1*us,
                                               phase_durations = phase_durations,
                                               amps_start_intervall = [0,20]*uA,
                                               delta = 0.01*uA,
                                               nof_runs = 1,
                                               stimulation_type = "extern",
                                               pulse_form = "mono")

##### plot strength duration curve
strength_duration_curve = plot.strength_duration_curve(plot_name = f"Strength duration curve {model.display_name}",
                                                       threshold_matrix = strength_duration_matrix,
                                                       rheobase = rheobase,
                                                       chronaxie = chronaxie)



# =============================================================================
# Get thresholds for certain stimulation types and stimulus durations
# =============================================================================
##### define phase durations to test (in us)
phase_durations_mono = [40,50,100]
phase_durations_bi = [20,40,50,200,400]

##### generate lists with test parameters
phase_duration = [i*us for i in phase_durations_mono + phase_durations_bi]
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))

##### calculate thresholds
threshold = [0]*len(phase_duration)
for ii in range(0,len(threshold)):
    threshold[ii] = test.get_thresholds(model = model,
                                        dt = dt,
                                        phase_durations = phase_duration[ii],
                                        amps_start_intervall = [0,20]*uA,
                                        delta = 0.01*uA,
                                        stimulation_type = "extern",
                                        pulse_form = pulse_form[ii])["threshold"][0]*amp

##### Save values in dataframe
threshold_table = pd.DataFrame(
        {'phase duration': phase_duration,
         'pulse form': pulse_form,
         'threshold': threshold
         })

# =============================================================================
# Get the relative spread of thresholds for certain stimulation types and stimulus durations
# =============================================================================
##### define phase durations to test (in us)
phase_durations_mono = [40,100]
phase_durations_bi = [200,400]

##### generate lists with test parameters
phase_duration = [i*us for i in phase_durations_mono + phase_durations_bi]
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))

##### calculate relative spreads
relative_spread = [0]*len(phase_duration)
for ii in range(0,len(relative_spread)):
    thresholds = test.get_thresholds(model = model,
                                     dt = dt,
                                     phase_durations = phase_duration[ii],
                                     amps_start_intervall = [0,20]*uA,
                                     delta = 0.005*uA,
                                     nof_runs = 20,
                                     stimulation_type = "extern",
                                     pulse_form = pulse_form[ii])
    
    relative_spread[ii] = f'{round(np.std(thresholds["threshold"])/np.mean(thresholds["threshold"])*100, 2)}%'
    
##### Save values in dataframe
relative_spreads = pd.DataFrame(
    {'phase duration': phase_duration,
     'pulse form': pulse_form,
     'relative spread': relative_spread
    })

# =============================================================================
# Measure conduction velocity
# =============================================================================
##### define compartments to start and end measurements
node_indexes = np.where(model.structure == 2)[0]

if hasattr(model, "index_soma"):
    node_indexes_dendrite = node_indexes[node_indexes < min(model.index_soma)[0]]
    node_indexes_axon = node_indexes[node_indexes > max(model.index_soma)[0]]
        
    conduction_velocity_dendrite = test.get_conduction_velocity(model,
                                                                dt,
                                                                stimulated_compartment = node_indexes_dendrite[0],
                                                                measurement_start_comp = node_indexes_dendrite[1],
                                                                measurement_end_comp = node_indexes_dendrite[-1],
                                                                stimulation_type = "extern",
                                                                pulse_form = "bi",
                                                                stim_amp = 2*uA,
                                                                phase_duration = 100*us)
    
    conduction_velocity_axon = test.get_conduction_velocity(model,
                                                            dt,
                                                            stimulated_compartment = node_indexes_dendrite[0],
                                                            measurement_start_comp = node_indexes_axon[0],
                                                            measurement_end_comp = node_indexes_axon[-3],
                                                            stimulation_type = "extern",
                                                            pulse_form = "bi",
                                                            stim_amp = 2*uA,
                                                            phase_duration = 100*us)




# =============================================================================
# Measure single node response properties
# =============================================================================
##### define phase durations to test (in us)
phase_durations_mono = [40,50,100]
phase_durations_bi = [200,400]

##### generate lists with test parameters
phase_duration = [i*us for i in phase_durations_mono + phase_durations_bi]
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))    

##### initialize dataset
column_names = ["phase duration (us)", "pulse form", "stimulus amplitude (uA)", "threshold (uA)", "AP height (mV)", "rise time (ms)",
                "fall time (ms)", "AP duration (ms)", "latency (ms)", "jitter (ms)"]
node_response_data_summary = pd.DataFrame(np.zeros((len(phase_duration), len(column_names))), columns = column_names)


for ii in range(0,len(phase_duration)):
    
    ##### look up threshold for actual stimulation
    threshold = threshold_table["threshold"][threshold_table["pulse form"] == pulse_form[ii]]\
                                            [threshold_table["phase duration"]/us == phase_duration[ii]/us].iloc[0]
    
    stim_amp = threshold                                        
    
    ##### measure single node response properties
    voltage_data, node_response_data = \
    test.get_single_node_response(model = model,
                                  dt = dt,
                                  param_1 = "stim_amp",
                                  param_1_ratios = [1],
                                  param_2 = "stochastic_runs",
                                  stimulation_type = "extern",
                                  pulse_form = pulse_form[ii],
                                  stim_amp = stim_amp,
                                  phase_duration = phase_duration[ii],
                                  nof_runs = 10)
    
    node_response_data_summary.loc[ii,"phase duration (us)"] = phase_duration[ii]/us
    node_response_data_summary.loc[ii,"pulse form"] = pulse_form[ii]
    node_response_data_summary.loc[ii,"stimulus amplitude (uA)"] = stim_amp/uA
    node_response_data_summary.loc[ii,"threshold (uA)"] = threshold/uA
    node_response_data_summary.loc[ii,["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]] =\
    node_response_data[["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]].mean()
    node_response_data_summary.loc[ii,"jitter (ms)"] = node_response_data["latency (ms)"].std()
    node_response_data_summary.loc[ii,"AP duration (ms)"] = sum(node_response_data_summary.loc[ii,["rise time (ms)", "fall time (ms)"]])

##### plot voltage courses of single node
plot.single_node_response_voltage_course(voltage_data = voltage_data)

##### plot results in bar plot
plot.single_node_response_bar_plot(data = node_response_data_summary)

















    