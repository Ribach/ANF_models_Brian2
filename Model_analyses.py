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
import itertools as itl

##### import functions
import functions.stimulation as stim
import functions.create_plots_for_model_comparison as plot
import functions.model_tests as test
import functions.tests_for_analyses as aly
import functions.calculations as calc

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### list of all models
models = ["rattay_01", "frijns_94", "briaire_05", "smit_09", "smit_10", "imennov_09", "negm_14"]

##### initialize clock
dt = 5*us

##### define way of processing
backend = "serial"

##### define if plots should be generated
generate_plots = False

##### define which tests to run
all_tests = False
voltage_courses_comparison = True
computational_efficiency_test = False
pulse_train_refractory_test = False
stochastic_properties_test = True

if any([all_tests, voltage_courses_comparison]):
    # =============================================================================
    # Plot voltage course for all models
    # =============================================================================
    models = ["rattay_01", "frijns_94", "briaire_05", "smit_09", "smit_10", "imennov_09"]
    
    ##### initialize list to save voltage courses
    voltage_courses =  [ [] for i in range(len(models)) ]
    
    for ii, model_name in enumerate(models):
        
        ##### get model
        model = eval(model_name)
        
        ##### set up the neuron
        neuron, param_string, model = model.set_up_model(dt = dt, model = model)
        
        ##### load the parameters of the differential equations in the workspace
        exec(param_string)
        
        ##### record the membrane voltage
        M = StateMonitor(neuron, 'v', record=True)
        
        ##### save initialization of the monitor(s)
        store('initialized')
    
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    pulse_form = "mono",
                                                    time_before = 0.2*ms,
                                                    time_after = 1.5*ms,
                                                    ##### monophasic stimulation
                                                    amp_mono = -3*uA,
                                                    duration_mono = 100*us)
        
        ##### get TimedArray of stimulus currents
        stimulus = TimedArray(np.transpose(I_stim), dt = dt)
                
        ##### run simulation
        run(runtime)
        
        ##### save M.v in voltage_courses
        voltage_courses[ii] = M.v
    
    ##### Plot membrane potential of all compartments over time
    voltage_course_comparison = plot.voltage_course_comparison_plot(plot_name = "Voltage courses all models",
                                                                    model_names = models,
                                                                    time_vector = M.t,
                                                                    voltage_courses = voltage_courses)
    
    ##### save plot
    voltage_course_comparison.savefig("test_battery_results/Analyses/voltage_course_comparison_plot {}.png", bbox_inches='tight')

if any([all_tests, stochastic_properties_test]):
    # =============================================================================
    # Get thresholds for certain stimulation types and stimulus durations
    # =============================================================================
    ##### define phase durations (in us) and pulse forms to test
    phase_durations = [100]
    pulse_forms = ["mono"]
    
    ##### define varied parameters 
    params = [{"model_name" : model,
               "phase_duration" : phase_durations[jj]*1e-6,
               "pulse_form" : pulse_forms[jj]}
                for model in models
                for jj in range(len(phase_durations))]
    
    ##### get thresholds
    threshold_table = th.util.map(func = test.get_threshold,
                                  space = params,
                                  backend = backend,
                                  cache = "no",
                                  kwargs = {"dt" : dt,
                                            "delta" : 0.0001*uA,
                                            "stimulation_type" : "extern",
                                            "amps_start_interval" : [0,30]*uA,
                                            "add_noise" : False})
    
    ##### change index to column
    threshold_table.reset_index(inplace=True)
    
    ##### change column names
    threshold_table = threshold_table.rename(index = str, columns={"model_name" : "model",
                                                                   "phase_duration" : "phase duration (us)",
                                                                   "pulse_form" : "pulse form",
                                                                   0:"threshold"})
    
    ##### add unit to phase duration
    threshold_table["phase duration (us)"] = [round(ii*1e6) for ii in threshold_table["phase duration (us)"]]
    
    ##### transpose dataframe    
    threshold_table = threshold_table.set_index("model").transpose()["rattay_01"]["threshold"]

if all_tests or computational_efficiency_test:
    # =============================================================================
    # Get computational efficiencies
    # =============================================================================
    ##### stimulus duration
    stimulus_duration = 50*ms
    
    ##### define runs per model
    nof_runs = 10
    
    ##### get computation times
    computation_times = aly.computational_efficiency_test(model_names = models,
                                                           dt = 1*us,
                                                           stimulus_duration = stimulus_duration,
                                                           nof_runs = nof_runs)
    
    ##### save dataframe to csv
    computation_times.to_csv("test_battery_results/Analyses/computational_efficiency.csv", index=False, header=True)

if all_tests or pulse_train_refractory_test:
    # =============================================================================
    # Get refractory periods for pulse trains
    # =============================================================================
    ##### define pulse rate of masker and second stimulus (in pulses per second)
    pulse_rate = 1200/second
    
    ##### define phase durations and inter_pulse_gap
    t_phase = 23*us
    t_ipg = 2*us
    
    ##### define pulse train duration
    t_pulse_train = 100*ms
    
    ##### calculate number of pulses
    nof_pulses = int(t_pulse_train * pulse_rate)
    
    ##### calculate inter pulse gap
    inter_pulse_gap = t_pulse_train/nof_pulses - 2*t_phase - t_ipg
    
    ##### define pulse rates
    pulse_rates = [1200,1500,18000,25000]
    
    ##### define varied parameters
    params = {"model_name" : models,
              "pulse_rate" : pulse_rates}
    
    ##### get thresholds
    refractory_table = th.util.map(func = aly.get_refractory_periods_for_pulse_trains,
                                   space = params,
                                   backend = backend,
                                   cache = "no",
                                   kwargs = {"dt" : 1*us,
                                             "delta" : 1*us,
                                             "stimulation_type" : "extern",
                                             "pulse_form" : "bi",
                                             "phase_durations" : [t_phase/us,t_ipg/us,t_phase/us]*us,
                                             "pulse_train_duration" : t_pulse_train})
    
    ##### change index to column
    refractory_table.reset_index(inplace=True)
    
    ##### change column names
    refractory_table = refractory_table.rename(index = str, columns={"model_name" : "model name",
                                                                     "pulse_rate" : "pulse rate",
                                                                     0 : "absolute refractory period (us)",
                                                                     1 : "relative refractory period (ms)"})
    
    ##### convert refractory periods to ms
    refractory_table["absolute refractory period (us)"] = refractory_table["absolute refractory period (us)"]*1e6
    refractory_table["relative refractory period (ms)"] = refractory_table["relative refractory period (ms)"]*1e3
    
    ##### round columns to 4 significant digits
    for ii in ["absolute refractory period (us)","relative refractory period (ms)"]:
        refractory_table[ii] = ["%.4g" %refractory_table[ii][jj] for jj in range(refractory_table.shape[0])]
    
    ##### Save dataframe as csv    
    refractory_table.to_csv("test_battery_results/Analyses/refractory_table_pulse_trains.csv", index=False, header=True)

# =============================================================================
# Get relative spread for different k_noise values
# =============================================================================
##### define k_noise values to test
k_noise_factor = np.append(np.round(np.arange(0.2,1,0.1),1), np.arange(1,5,0.5)).tolist()
k_noise_factor = [0.5,1,2]

##### define test parameters
phase_duration = 100*us
pulse_form = "mono"
runs_per_k_noise = 2

##### define varied parameters
params = {"model_name" : models,
          "parameter_ratio": k_noise_factor,
          "run_number" : [ii for ii in range(runs_per_k_noise)]}

##### get stochastic thresholds
relative_spreads = th.util.map(func = test.get_threshold_for_param,
                               space = params,
                               backend = backend,
                               cache = "no",
                               kwargs = {"dt" : dt,
                                         "parameter": "k_noise",
                                         "phase_duration" : phase_duration,
                                         "delta" : 0.0005*uA,
                                         "pulse_form" : pulse_form,
                                         "stimulation_type" : "extern",
                                         "amps_start_interval" : [0,20]*uA,
                                         "time_before" : 2*ms,
                                         "time_after" : 2*ms,
                                         "add_noise" : True})

##### change index to column
relative_spreads.reset_index(inplace=True)

##### change column names
relative_spreads = relative_spreads.rename(index = str, columns={"model_name" : "model",
                                                                 "parameter_ratio" : "knoise ratio",
                                                                 0:"threshold"})

##### exclude spontaneous APs
relative_spreads = relative_spreads[relative_spreads["threshold"] > 1e-9]

##### delete run_number column
relative_spreads = relative_spreads.drop(columns = ["run_number"])

##### calculate relative spread values
thresholds = relative_spreads.groupby(["model", "knoise ratio"])
relative_spreads = round(thresholds.std()/thresholds.mean()*100, 2)
relative_spreads.reset_index(inplace=True)
relative_spreads = relative_spreads.rename(index = str, columns={"threshold" : "relative spread (%)"})

##### Save relative spread dataframe as csv    
relative_spreads.to_csv("test_battery_results/Analyses/relative_spreads_k_noise_comparison.csv", index=False, header=True)

# =============================================================================
# Get jitter for different k_noise values
# =============================================================================
##### define k_noise values to test
k_noise_factor = np.append(np.round(np.arange(0.2,1,0.1),1), np.arange(1,5,0.5)).tolist()
k_noise_factor = [0.5,1,2]

##### define test parameters
phase_duration = 100*us
pulse_form = "mono"
runs_per_k_noise = 2

##### look up deterministic thresholds
threshold_table[threshold_table.index["phase duration (us)"] == phase_duration]


thresholds = threshold_table["model", "threshold"][threshold_table["pulse form"] == pulse_form]\
                                         [threshold_table["phase duration (us)"] == phase_duration/us]

##### define varied parameters 
params = [{"model_name" : model,
           "phase_duration" : phase_durations[jj]*1e-6,
           "pulse_form" : pulse_forms[jj]}
            for model in models
            for jj in range(len(phase_durations))]

##### define varied parameters
params = {"model_name" : models,
          "parameter_ratio": k_noise_factor,
          "run_number" : [ii for ii in range(runs_per_k_noise)]}

##### get single node response properties
single_node_response_table = th.util.map(func = test.get_single_node_response,
                                         space = params,
                                         backend = backend,
                                         cache = "no",
                                         kwargs = {"dt" : 1*us,
                                                   "phase_duration" : phase_duration,
                                                   "pulse_form" : pulse_form,
                                                   "stim_amp" : ,
                                                   "stimulation_type" : "extern",
                                                   "time_before" : 3*ms,
                                                   "time_after" : 2*ms,
                                                   "add_noise" : True})

##### change index to column
single_node_response_table.reset_index(inplace=True)

##### change column names
single_node_response_table = single_node_response_table.rename(index = str, columns={"phase_duration" : "phase duration (us)",
                                                                                     "stim_amp" : "stimulus amplitude (uA)",
                                                                                     "pulse_form" : "pulse form",
                                                                                     "run_number" : "run",
                                                                                     0 : "AP height (mV)",
                                                                                     1 : "rise time (us)",
                                                                                     2 : "fall time (us)",
                                                                                     3 : "latency (us)",
                                                                                     4 : "membrane potential (mV)",
                                                                                     5 : "time (ms)"})

##### add row with stimulus amplitude information
single_node_response_table["amplitude level"] = ["{}*threshold".format(stim_amp_levels[jj])
                                for ii in range(len(phase_durations))
                                for jj in range(len(stim_amp_levels))
                                for kk in range(nof_runs)]

##### change units from second to us and form amp to uA
single_node_response_table["phase duration (us)"] = round(single_node_response_table["phase duration (us)"]*1e6).astype(int)
single_node_response_table["stimulus amplitude (uA)"] = round(single_node_response_table["stimulus amplitude (uA)"]*1e6,2)
single_node_response_table["AP height (mV)"] = single_node_response_table["AP height (mV)"]*1e3
single_node_response_table["rise time (us)"] = single_node_response_table["rise time (us)"]*1e6
single_node_response_table["fall time (us)"] = single_node_response_table["fall time (us)"]*1e6
single_node_response_table["latency (us)"] = single_node_response_table["latency (us)"]*1e6

##### adjust pulse form column
single_node_response_table["pulse form"] = ["monophasic" if single_node_response_table["pulse form"][ii]=="mono" else "biphasic" for ii in range(np.shape(single_node_response_table)[0])]

##### calculate AP duration
single_node_response_table["AP duration (us)"] = single_node_response_table["rise time (us)"] + single_node_response_table["fall time (us)"]

##### build summary dataframe and exclude data where no action potential was elicited
single_node_response_summary = single_node_response_table[single_node_response_table["AP height (mV)"] > 60]

##### calculate jitter
jitter = single_node_response_summary.groupby(["phase duration (us)","stimulus amplitude (uA)","pulse form"])["latency (us)"].std().reset_index()
jitter = jitter.rename(index = str, columns={"latency (us)" : "jitter (us)"})
















