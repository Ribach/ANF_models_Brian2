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
##### get rheobase
rheobase_matrix = test.get_thresholds(model = model,
                                      dt = dt,
                                      phase_durations = 1*ms,
                                      amps_start_intervall = [0,20]*uA,
                                      delta = 0.0001*uA,
                                      stimulation_type = "extern",
                                      pulse_form = "mono")

rheobase = rheobase_matrix["threshold"][0]*amp

##### get chronaxie
chronaxie = test.get_chronaxie(model = model,
                               dt = dt,
                               rheobase = rheobase,
                               phase_duration_start_intervall = [0,1000]*us,
                               delta = 1*us,
                               stimulation_type = "extern",
                               pulse_form = "mono")

##### round values
rheobase = np.round(rheobase/nA,1)*nA
chronaxie = np.round(chronaxie/us,1)*us

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

##### define test parameters
phase_duration = [i*us for i in phase_durations_mono + phase_durations_bi]
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))
runs_per_stimulus_type = 50

##### initialize threshold matrix and relative spread vector
threshold_matrix = np.zeros((len(phase_duration)*runs_per_stimulus_type,3))
relative_spread = [0]*len(phase_duration)

##### calculate relative spreads
for ii in range(0,len(relative_spread)):
    thresholds = test.get_thresholds(model = model,
                                     dt = dt,
                                     phase_durations = phase_duration[ii],
                                     amps_start_intervall = [0,20]*uA,
                                     delta = 0.005*uA,
                                     nof_runs = runs_per_stimulus_type,
                                     stimulation_type = "extern",
                                     pulse_form = pulse_form[ii])
    
    ##### write thresholds in threshold matrix
    threshold_matrix[ii*runs_per_stimulus_type:ii*runs_per_stimulus_type+runs_per_stimulus_type] = thresholds
    
    ##### write relative spreads in vector
    relative_spread[ii] = f'{round(np.std(thresholds["threshold"])/np.mean(thresholds["threshold"])*100, 2)}%'

##### save thresholds in dataframe
threshold_matrix = pd.DataFrame(threshold_matrix, columns = ["phase duration","run","threshold"])

threshold_matrix["pulse form"] = "monophasic"
threshold_matrix["pulse form"][round(1000000*threshold_matrix["phase duration"]).astype(int).isin(phase_durations_bi)] = "biphasic"

##### Save relative spread values in dataframe
relative_spreads = pd.DataFrame(
    {'phase duration': phase_duration,
     'pulse form': pulse_form,
     'relative spread': relative_spread
    })
    
##### plot relative spreads
relative_spread_plot = plot.relative_spread(plot_name = f"Relative spreads {model.display_name}",
                                            threshold_matrix = threshold_matrix)

# =============================================================================
# Measure conduction velocity
# =============================================================================
##### define compartments to start and end measurements
node_indexes = np.where(model.structure == 2)[0]

##### models with a soma
if hasattr(model, "index_soma"):
    
    ##### dendrite
    node_indexes_dendrite = node_indexes[node_indexes < min(model.index_soma)[0]]
        
    conduction_velocity_dendrite = test.get_conduction_velocity(model = model,
                                                                dt = dt,
                                                                stimulated_compartment = node_indexes_dendrite[0],
                                                                measurement_start_comp = node_indexes_dendrite[1],
                                                                measurement_end_comp = node_indexes_dendrite[-1])
    
    conduction_velocity_dendrite_ratio = round((conduction_velocity_dendrite/(meter/second))/(model.dendrite_outer_diameter/um),2)
    
    ##### axon
    node_indexes_axon = node_indexes[node_indexes > max(model.index_soma)[0]]

    conduction_velocity_axon = test.get_conduction_velocity(model = model,
                                                            dt = dt,
                                                            stimulated_compartment = node_indexes_dendrite[0],
                                                            measurement_start_comp = node_indexes_axon[0],
                                                            measurement_end_comp = node_indexes_axon[-3])
    
    conduction_velocity_axon_ratio = round((conduction_velocity_axon/(meter/second))/(model.axon_outer_diameter/um),2)

##### models without a soma 
else:
   conduction_velocity = test.get_conduction_velocity(model = model,
                                                      dt = dt,
                                                      stimulated_compartment = node_indexes[1],
                                                      measurement_start_comp = node_indexes[2],
                                                      measurement_end_comp = node_indexes[-3])
   
   conduction_velocity_ratio = round((conduction_velocity/(meter/second))/(model.fiber_outer_diameter/um),2)

# =============================================================================
# Measure single node response properties
# =============================================================================
##### define phase durations to test (in us)
phase_durations_mono = [40,50,100]
phase_durations_bi = [200]

##### generate lists with test parameters
phase_duration = [i*us for i in phase_durations_mono + phase_durations_bi]
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))    

##### define how many stimulus amplitudes to test
nof_stim_amps = 2

##### define number of stochastic runs
nof_stochastic_runs = 10

##### initialize summary dataset
column_names = ["phase duration (us)", "pulse form", "stimulus amplitude level", "AP height (mV)", "rise time (ms)",
                "fall time (ms)", "AP duration (ms)", "latency (ms)", "jitter (ms)"]
node_response_data_summary = pd.DataFrame(np.zeros((len(phase_duration*nof_stim_amps), len(column_names))), columns = column_names)

##### initialize list of datasets to save voltage course data
voltage_courses = [pd.DataFrame()]*len(phase_duration*nof_stim_amps)

##### loop over all phase durations to test
for ii in range(0,len(phase_duration)):
    
    ##### look up threshold for actual stimulation
    threshold = threshold_table["threshold"][threshold_table["pulse form"] == pulse_form[ii]]\
                                            [threshold_table["phase duration"]/us == phase_duration[ii]/us].iloc[0]
    
    ##### loop over stimulus amplitudes to test
    for jj in range(0,nof_stim_amps):
        
        ##### define stimulus amplitude
        stim_amp = (jj+1)*threshold                                        
        
        ##### measure single node response properties
        voltage_data, node_response_data = \
        test.get_single_node_response(model = model,
                                      dt = dt,
                                      param_1 = "stim_amp",
                                      param_1_ratios = [1],
                                      param_2 = "stochastic_runs",
                                      stimulation_type = "extern",
                                      pulse_form = pulse_form[ii],
                                      time_after_stimulation = 2*ms,
                                      stim_amp = stim_amp,
                                      phase_duration = phase_duration[ii],
                                      nof_runs = nof_stochastic_runs)
        
        ##### write voltage dataset in voltage courses list
        voltage_courses[nof_stim_amps*ii+jj] = voltage_data
        voltage_courses[nof_stim_amps*ii+jj] = voltage_courses[nof_stim_amps*ii+jj].drop(columns = "stimulus amplitude (uA)")
        voltage_courses[nof_stim_amps*ii+jj]["phase duration (us)"] = round(phase_duration[ii]/us)
        voltage_courses[nof_stim_amps*ii+jj]["pulse form"] = "monophasic" if pulse_form[ii]=="mono" else "biphasic"
        voltage_courses[nof_stim_amps*ii+jj]["stimulus amplitude level"] = "threshold" if jj==0 else f"{jj+1}*threshold"
        

        ##### write results in summary table
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"phase duration (us)"] = phase_duration[ii]/us
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"pulse form"] = "monophasic" if pulse_form[ii]=="mono" else "biphasic"
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"stimulus amplitude level"] = "threshold" if jj==0 else f"{jj+1}*threshold"
        node_response_data_summary.loc[nof_stim_amps*ii+jj,["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]] =\
        node_response_data[["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]].mean()
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"jitter (ms)"] = node_response_data["latency (ms)"].std()
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"AP duration (ms)"] = sum(node_response_data_summary.loc[ii,["rise time (ms)", "fall time (ms)"]])

##### combine list entries of voltage courses to one dataset
voltage_course_dataset = pd.concat(voltage_courses)
voltage_course_dataset["stimulation info"] = voltage_course_dataset["phase duration (us)"].map(str) + [" us; "] + voltage_course_dataset["pulse form"].map(str) +\
["; "] + voltage_course_dataset["stimulus amplitude level"].map(str)
voltage_course_dataset = voltage_course_dataset[["stimulation info", "run", "time / ms", "membrane potential / mV"]]

##### plot voltage courses of single node
plot.single_node_response_voltage_course(plot_name = f"Voltage courses {model.display_name}",
                                         voltage_data = voltage_course_dataset,
                                         col_wrap = 2,
                                         height = 3.5)

# =============================================================================
# Refractory properties
# =============================================================================
##### define phase durations to test (in us)
phase_durations_mono = [40,50,100]
phase_durations_bi = [200]

##### generate lists with test parameters
phase_duration = [i*us for i in phase_durations_mono + phase_durations_bi]
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))

##### initialize summary dataset
column_names = ["phase duration (us)", "pulse form", "absolute refractory period", "relative refractory period"]
refractory_table = pd.DataFrame(np.zeros((len(phase_duration), len(column_names))), columns = column_names)

##### define inter-pulse-intervalls
inter_pulse_intervalls = np.logspace(-1, 2.6, num=30, base=2)*ms

##### loop over all phase durations to test
for ii in range(0,len(phase_duration)):
    
    ##### look up threshold for actual stimulation
    threshold = threshold_table["threshold"][threshold_table["pulse form"] == pulse_form[ii]]\
                                                [threshold_table["phase duration"]/us == phase_duration[ii]/us].iloc[0]
        
    arp, rrp = test.get_refractory_periods(model = model,
                                           dt = dt,
                                           delta = 1*us,
                                           threshold = threshold,
                                           amp_masker = 1.5*threshold,
                                           stimulation_type = "extern",
                                           pulse_form = pulse_form[ii],
                                           phase_duration = phase_duration[ii])
    
    refractory_table.loc[ii,"phase duration (us)"] = phase_duration[ii]/us
    refractory_table.loc[ii,"pulse form"] = "monophasic" if pulse_form[ii]=="mono" else "biphasic"
    refractory_table.loc[ii,"absolute refractory period"] = arp
    refractory_table.loc[ii,"relative refractory period"] = rrp

##### generate refractory curve
threshold = threshold_table["threshold"][threshold_table["pulse form"] == pulse_form[3]]\
                             [threshold_table["phase duration"]/us == phase_duration[3]/us].iloc[0]  

min_required_amps, threshold = test.get_refractory_curve(model = model,
                                                         dt = dt,
                                                         inter_pulse_intervalls = inter_pulse_intervalls,
                                                         delta = 0.005*uA,
                                                         threshold = threshold,
                                                         amp_masker = 1.5*threshold,
                                                         stimulation_type = "extern",
                                                         pulse_form = pulse_form[3],
                                                         phase_duration = phase_duration[3])

##### plot refractory curve
refractory_curve = plot.refractory_curve(plot_name = f"Refractory curve {model.display_name}",
                                         inter_pulse_intervalls = inter_pulse_intervalls,
                                         stimulus_amps = min_required_amps,
                                         threshold = threshold)

# =============================================================================
# Post Stimulus Time Histogram
# =============================================================================
##### define bin_width
bin_width = 1*ms

##### calculate bin heigths and edges
bin_heigths, bin_edges = test.post_stimulus_time_histogram(model = model,
                                                    dt = 2*us,
                                                    nof_repeats = 50,
                                                    pulses_per_second = 2000,
                                                    stim_duration = 300*ms,
                                                    stim_amp = 1.5*uA,
                                                    stimulation_type = "extern",
                                                    pulse_form = "bi",
                                                    phase_duration = 40*us,
                                                    bin_width = bin_width)

##### plot post_stimulus_time_histogram
plot.post_stimulus_time_histogram(plot_name = f"PSTH {model.display_name}",
                                  bin_edges = bin_edges,
                                  bin_heigths = bin_heigths,
                                  bin_width = bin_width/ms)
























