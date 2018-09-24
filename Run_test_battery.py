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
import os

##### set working directory to folder of script
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)

##### import functions
import functions.stimulation as stim
import functions.create_plots as plot
import functions.model_tests as test
import functions.calculations as calc

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
model = frijns_05

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
                                      pulse_form = "mono",
                                      time_before = 2*ms)

rheobase = rheobase_matrix["threshold"][0]*amp

##### get chronaxie
chronaxie = test.get_chronaxie(model = model,
                               dt = dt,
                               rheobase = rheobase,
                               phase_duration_start_intervall = [0,1000]*us,
                               delta = 1*us,
                               stimulation_type = "extern",
                               pulse_form = "mono",
                               time_before = 2*ms)

##### round values
rheobase = np.round(rheobase/nA,1)*nA
chronaxie = np.round(chronaxie/us,1)*us

##### save values in dataframe
strength_duration_data = pd.DataFrame(np.array([[rheobase/uA], [chronaxie/us]]).T,
                                         columns = ["rheobase (uA)", "chronaxie (us)"])

##### Save table as csv    
strength_duration_data.to_csv("test_battery_results/{}/Strength_duration_data {}.csv".format(model.display_name,model.display_name), index=False, header=True)

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
                                               pulse_form = "mono",
                                               time_before = 2*ms)

##### plot strength duration curve
strength_duration_curve = plot.strength_duration_curve(plot_name = "Strength duration curve {}".format(model.display_name),
                                                       threshold_matrix = strength_duration_matrix,
                                                       rheobase = rheobase,
                                                       chronaxie = chronaxie)

##### save strength duration curve
strength_duration_curve.savefig("test_battery_results/{}/Strength_duration_curve {}.png".format(model.display_name,model.display_name))

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
                                        amps_start_intervall = [0,30]*uA,
                                        delta = 0.0001*uA,
                                        time_before = 2*ms,
                                        time_after = 3*ms,
                                        stimulation_type = "extern",
                                        pulse_form = pulse_form[ii])["threshold"][0]*amp

##### Save values in dataframe
threshold_table = pd.DataFrame(
        {'phase duration': phase_duration,
         'pulse form': pulse_form,
         'threshold': threshold
         })

##### Save table as csv    
threshold_table.to_csv("test_battery_results/{}/Threshold_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)

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
                                     delta = 0.0005*uA,
                                     time_before = 2*ms,
                                     time_after = 2*ms,
                                     nof_runs = runs_per_stimulus_type,
                                     stimulation_type = "extern",
                                     pulse_form = pulse_form[ii])
    
    ##### write thresholds in threshold matrix
    threshold_matrix[ii*runs_per_stimulus_type:ii*runs_per_stimulus_type+runs_per_stimulus_type] = thresholds
    
    ##### write relative spreads in vector
    relative_spread[ii] = '{}%'.format(round(np.std(thresholds["threshold"])/np.mean(thresholds["threshold"])*100, 2))

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

##### Save table as csv    
relative_spreads.to_csv("test_battery_results/{}/Relative_spreads {}.csv".format(model.display_name,model.display_name), index=False, header=True)    
    
##### plot relative spreads
relative_spread_plot = plot.relative_spread(plot_name = "Relative spreads {}".format(model.display_name),
                                            threshold_matrix = threshold_matrix)

##### save relative spreads plot
relative_spread_plot.savefig("test_battery_results/{}/Relative_spreads_plot {}.png".format(model.display_name,model.display_name))

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
    
    ##### save values in dataframe
    conduction_velocity_table = pd.DataFrame(np.array([[conduction_velocity_dendrite], [conduction_velocity_dendrite_ratio], [conduction_velocity_axon], [conduction_velocity_axon_ratio]]).T,
                                             columns = ["velocity dendrite (m/s)", "velocity/diameter dendrite", "velocity axon (m/s)", "velocity/diameter axon"])

##### models without a soma 
else:
   conduction_velocity = test.get_conduction_velocity(model = model,
                                                      dt = dt,
                                                      stimulated_compartment = node_indexes[1],
                                                      measurement_start_comp = node_indexes[2],
                                                      measurement_end_comp = node_indexes[-3])
   
   conduction_velocity_ratio = round((conduction_velocity/(meter/second))/(model.fiber_outer_diameter/um),2)
   
   ##### save values in dataframe
   conduction_velocity_table = pd.DataFrame(np.array([[conduction_velocity], [conduction_velocity_ratio]]).T,
                                            columns = ["velocity (m/s)", "velocity/diameter"])

##### Save table as csv    
conduction_velocity_table.to_csv("test_battery_results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)

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
nof_stochastic_runs = 20

##### initialize summary dataset
column_names = ["phase duration (us)", "pulse form", "stimulus amplitude level", "AP height (mV)", "rise time (ms)",
                "fall time (ms)", "AP duration (ms)", "latency (ms)", "jitter (ms)"]
node_response_data_summary = pd.DataFrame(np.zeros((len(phase_duration*nof_stim_amps), len(column_names))), columns = column_names)

##### initialize list of datasets to save voltage course data
voltage_courses = [pd.DataFrame()]*len(phase_duration)*nof_stim_amps

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
        voltage_courses[nof_stim_amps*ii+jj]["stimulus amplitude level"] = "threshold" if jj==0 else "{}*threshold".format(jj+1)
        

        ##### write results in summary table
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"phase duration (us)"] = phase_duration[ii]/us
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"pulse form"] = "monophasic" if pulse_form[ii]=="mono" else "biphasic"
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"stimulus amplitude level"] = "threshold" if jj==0 else "{}*threshold".format(jj+1)
        node_response_data_summary.loc[nof_stim_amps*ii+jj,["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]] =\
        node_response_data[["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]].mean()
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"latency (ms)"] = node_response_data_summary.loc[nof_stim_amps*ii+jj,"latency (ms)"]
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"jitter (ms)"] = node_response_data["latency (ms)"].std()
        node_response_data_summary.loc[nof_stim_amps*ii+jj,"AP duration (ms)"] = sum(node_response_data_summary.loc[ii,["rise time (ms)", "fall time (ms)"]])

##### round for 3 significant digits
for ii in ["AP height (mV)", "rise time (ms)", "fall time (ms)", "AP duration (ms)", "latency (ms)", "jitter (ms)"]:
    node_response_data_summary[ii] = ["%.3g" %node_response_data_summary[ii][jj] for jj in range(0,node_response_data_summary.shape[0])]

##### Save table as csv    
node_response_data_summary.to_csv("test_battery_results/{}/Node_response_data_summary {}.csv".format(model.display_name,model.display_name), index=False, header=True)

##### combine list entries of voltage courses to one dataset
voltage_course_dataset = pd.concat(voltage_courses)
voltage_course_dataset["stimulation info"] = voltage_course_dataset["phase duration (us)"].map(str) + [" us; "] + voltage_course_dataset["pulse form"].map(str) +\
["; "] + voltage_course_dataset["stimulus amplitude level"].map(str)
voltage_course_dataset = voltage_course_dataset[["stimulation info", "run", "time / ms", "membrane potential / mV"]]

##### plot voltage courses of single node
single_node_response = plot.single_node_response_voltage_course(plot_name = "Voltage courses {}".format(model.display_name),
                                                                voltage_data = voltage_course_dataset,
                                                                col_wrap = 2,
                                                                height = 3.5)

###### save voltage courses plot
single_node_response.savefig("test_battery_results/{}/Single_node_response {}.png".format(model.display_name,model.display_name))

# =============================================================================
# Refractory periods
# =============================================================================
##### define phase durations to test (in us)
phase_durations_mono = [40,50,100]
phase_durations_bi = [50,200]

##### generate lists with test parameters
phase_duration = [i*us for i in phase_durations_mono + phase_durations_bi]
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))

##### initialize summary dataset
column_names = ["phase duration (us)", "pulse form", "absolute refractory period (ms)", "relative refractory period (ms)"]
refractory_table = pd.DataFrame(np.zeros((len(phase_duration), len(column_names))), columns = column_names)

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
    refractory_table.loc[ii,"absolute refractory period (ms)"] = round(arp/ms, 3)
    refractory_table.loc[ii,"relative refractory period (ms)"] = round(rrp/ms, 3)
    
    ##### Save table as csv    
    refractory_table.to_csv("test_battery_results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)   

# =============================================================================
# Refractory curve
# =============================================================================
##### define inter-pulse-intervalls
inter_pulse_intervalls = np.logspace(-1.3, 2.9, num=80, base=2)*ms    

##### get thresholds for second spikes
min_required_amps, threshold = test.get_refractory_curve(model = model,
                                                         dt = dt,
                                                         inter_pulse_intervalls = inter_pulse_intervalls,
                                                         delta = 0.0001*uA,
                                                         stimulation_type = "extern",
                                                         pulse_form = "mono",
                                                         phase_duration = 100*us)

##### plot refractory curve
refractory_curve = plot.refractory_curve(plot_name = "Refractory curve {}".format(model.display_name),
                                         inter_pulse_intervalls = inter_pulse_intervalls,
                                         stimulus_amps = min_required_amps,
                                         threshold = threshold)

##### save refractory curve
refractory_curve.savefig("test_battery_results/{}/Refractory_curve {}.png".format(model.display_name,model.display_name))

# =============================================================================
# Post Stimulus Time Histogram
# =============================================================================
##### pulse rates to test
pulses_per_second = [250,1000,5000,10000]

##### phase durations for respective pulse rates
phase_duration = [40,40,40,20]*us

##### stimulus levels (will be multiplied with the threshold for a certain stimulation) 
stim_amp_level = [1,1.2,1.5]

##### initialize list of datasets to save bin heights end edges for each type of stimulation
psth_data = [pd.DataFrame()]*len(pulses_per_second)*len(stim_amp_level)

##### loop over pulse rates and phase duraions
for ii in range(0,len(pulses_per_second)):
    
    ##### look up threshold for actual stimulation
    threshold = threshold_table["threshold"][threshold_table["pulse form"] == "bi"]\
                               [threshold_table["phase duration"]/us == phase_duration[ii]/us].iloc[0]
    
    ##### loop over stimulus amplitudes
    for jj in range(0,len(stim_amp_level)):
                                        
        ##### calculate bin heigths and edges
        psth = test.post_stimulus_time_histogram(model = model,
                                                 dt = dt,
                                                 nof_repeats = 20,
                                                 pulses_per_second = pulses_per_second[ii],
                                                 stim_duration = 100*ms,
                                                 stim_amp = stim_amp_level[jj]*threshold,
                                                 stimulation_type = "extern",
                                                 pulse_form = "bi",
                                                 phase_duration = phase_duration[ii])
        
        ##### write the data in a list of dataframes
        psth_data[len(stim_amp_level)*ii+jj] = psth
        psth_data[len(stim_amp_level)*ii+jj]["amplitude"] = "{}*threshold".format(stim_amp_level[jj])
        
##### combine list entries of psth_data to one dataset
psth_dataset = pd.concat(psth_data)

##### save PSTH dataset to csv
psth_dataset.to_csv("test_battery_results/{}/PSTH_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)   

##### plot post_stimulus_time_histogram
post_stimulus_time_histogram = plot.post_stimulus_time_histogram(plot_name = "PSTH {}".format(model.display_name),
                                                                 psth_dataset = psth_dataset)

###### save post_stimulus_time_histogram
post_stimulus_time_histogram.savefig("test_battery_results/{}/PSTH {}.png".format(model.display_name,model.display_name))

## =============================================================================
## Inter-stimulus intervall Histogram
## =============================================================================
###### pulse rates to test
#pulses_per_second = [150,1000,5000]
#
###### phase durations for respective pulse rates
#phase_duration = [40,40,40]*us
#
###### stimulus levels (will be multiplied with the threshold for a certain stimulation) 
#stim_amp_level = [1,1.1,1.2]
#
###### initialize list of datasets to save bin heights end edges for each type of stimulation
#isi_data = [pd.DataFrame()]*len(pulses_per_second)*len(stim_amp_level)
#
###### loop over pulse rates and phase duraions
#for ii in range(0,len(pulses_per_second)):
#    
#    ##### look up threshold for actual stimulation
#    threshold = threshold_table["threshold"][threshold_table["pulse form"] == "bi"]\
#                               [threshold_table["phase duration"]/us == phase_duration[ii]/us].iloc[0]
#    
#    ##### loop over stimulus amplitudes
#    for jj in range(0,len(stim_amp_level)):
#                                        
#        ##### calculate bin heigths and edges
#        isi = test.inter_stimulus_intervall_histogram(model = model,
#                                                      dt = dt,
#                                                      nof_repeats = 5,
#                                                      pulses_per_second = pulses_per_second[ii],
#                                                      stim_duration = 30*ms,
#                                                      stim_amp = stim_amp_level[jj]*threshold,
#                                                      stimulation_type = "extern",
#                                                      pulse_form = "bi",
#                                                      phase_duration = phase_duration[ii])
#        
#        ##### write the data in a list of dataframes
#        isi_data[len(stim_amp_level)*ii+jj] = isi
#        isi_data[len(stim_amp_level)*ii+jj]["amplitude"] = "{}*threshold".format(stim_amp_level[jj])
#        
###### combine list entries of psth_data to one dataset
#isi_dataset = pd.concat(isi_data)
#
###### plot inter_stimulus_intervall_histogram
#plot.inter_stimulus_intervall_histogram(plot_name = "ISI {}".format(model.display_name),
#                                        isi_dataset = isi_dataset)
#
####### save inter_stimulus_intervall_histogram
#inter_stimulus_intervall_histogram.savefig("test_battery_results/{}/ISI {}.png".format(model.display_name,model.display_name))
