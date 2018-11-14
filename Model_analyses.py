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
models = ["rattay_01", "briaire_05", "imennov_09"]

##### initialize clock
dt = 5*us

##### define way of processing
backend = "serial"

##### define if plots should be generated
generate_plots = True

##### define which tests to run
all_tests = False
voltage_courses_comparison = False
computational_efficiency_test = False
pulse_train_refractory_test = False
stochastic_properties_test = False
latency_over_stim_amp_test = True

if any([all_tests, stochastic_properties_test]):
    # =============================================================================
    # Get thresholds for certain stimulation types and stimulus durations
    # =============================================================================
    ##### define phase durations (in us) and pulse forms to test
    phase_durations = [100]
    inter_phase_gap = [0]
    pulse_forms = ["mono"]
    
    ##### define varied parameters 
    params = [{"model_name" : model,
               "phase_duration" : phase_durations[jj]*1e-6,
               "inter_phase_gap" : inter_phase_gap[jj]*1e-6,
               "pulse_form" : pulse_forms[jj]}
                for model in models
                for jj in range(len(phase_durations))]
    
    ##### get thresholds
    threshold_table = th.util.map(func = test.get_threshold,
                                  space = params,
                                  backend = backend,
                                  cache = "no",
                                  kwargs = {"dt" : 5*us,
                                            "delta" : 0.001*uA,
                                            "stimulation_type" : "extern",
                                            "amps_start_interval" : [0,400]*uA,
                                            "add_noise" : False})
    
    ##### change index to column
    threshold_table.reset_index(inplace=True)
    
    ##### change column names
    threshold_table = threshold_table.rename(index = str, columns={"model_name" : "model",
                                                                   "phase_duration" : "phase duration (us)",
                                                                   "inter_phase_gap" : "inter phase gap (us)",
                                                                   "pulse_form" : "pulse form",
                                                                   0:"threshold"})
    
    ##### add unit to phase duration and inter phase gap
    threshold_table["phase duration (us)"] = [round(ii*1e6) for ii in threshold_table["phase duration (us)"]]
    threshold_table["inter phase gap (us)"] = [round(ii*1e6,1) for ii in threshold_table["inter phase gap (us)"]]

if all_tests or latency_over_stim_amp_test:
# =============================================================================
# Measure latencies for different stimulus amplitudes and electrode distances
# =============================================================================
##### define stimulus parameters
phase_duration = 45*us
inter_phase_gap = 2*us
stimulation_type = "extern"
pulse_form = "bi"
stim_node = 2

##### electrode distance ratios (to be multiplied with the models original electrode distance of 300*um)
electrode_distances = [200*um,300*um,500*um,700*um,1000*um,1500*um,2000*um,3000*um,5000*um]
electrode_distance_ratios = [electrode_distance/(300*um) for electrode_distance in electrode_distances]

##### stimulus amplitude levels (to be multiplied with threshold stimulus)
stim_amp_levels = np.linspace(1,5,20).tolist()

##### get thresholds of models for different electrode distances
params = [{"model_name" : model,
           "parameter_ratio" : electrode_distance_ratio}
            for model in models\
            for electrode_distance_ratio in electrode_distance_ratios]

thresholds = th.util.map(func = test.get_threshold,
                         space = params,
                         backend = backend,
                         cache = "no",
                         kwargs = {"dt" : 1*us,
                                   "parameter": "electrode_distance",
                                   "phase_duration" : phase_duration,
                                   "inter_phase_gap" : inter_phase_gap,
                                   "delta" : 0.0001*uA,
                                   "pulse_form" : pulse_form,
                                   "stimulation_type" : "extern",
                                   "amps_start_interval" : [0,10]*mA})

##### change index to column
thresholds.reset_index(inplace=True)

##### change column names
thresholds = thresholds.rename(index = str, columns={"model_name" : "model",
                                                     "parameter_ratio" : "parameter ratio",
                                                     0:"threshold (uA)"})

##### calculate electrode distance in um
thresholds["electrode distance (um)"] = [np.round(ratio*300).astype(int) for ratio in thresholds["parameter ratio"]]

##### calculate number of the node where certain latency can be measured for all models
params = [{"model_name" : model,
           "stim_amp" : thresholds["threshold (uA)"][thresholds["model"] == model][thresholds["parameter ratio"] == 1][0]*1.5}
            for model in models]

measurement_nodes = th.util.map(func = aly.get_node_number_for_latency,
                                space = params,
                                backend = backend,
                                cache = "no",
                                kwargs = {"dt" : 1*us,
                                          "latency_desired" : 2.5*ms,
                                          "phase_duration" : phase_duration,
                                          "inter_phase_gap" : inter_phase_gap,
                                          "delta" : 1,
                                          "numbers_start_interval" : [10,1000],
                                          "stimulation_type" : stimulation_type,
                                          "pulse_form" : "bi",
                                          "time_after" : 10*ms})

#model_name = "rattay_01"
#dt = 1*us
#latency_desired = 2.5*ms 
#stim_amp = thresholds["threshold (uA)"][thresholds["model"] == model][thresholds["parameter ratio"] == 1][0]
#phase_duration = 45*us
#delta = 1
#numbers_start_interval = [10,1000]
#inter_phase_gap = 2*us
#pulse_form = "bi"
#time_after = 5*ms

##### index of nodes, where latency is measured in order to obtain latency values near the measured values (is different for each model)
measurement_node = [60, 110, 800]

##### define varied parameters 
params = [{"model_name" : model,
           "stim_amp" : thresholds["threshold (uA)"][thresholds["model"] == model][thresholds["parameter ratio"] == electrode_distance_ratio][0] * stim_amp_level,
           "measurement_node" : measurement_node[ii],
           "electrode_distance" : thresholds["electrode distance (um)"][thresholds["model"] == model][thresholds["parameter ratio"] == electrode_distance_ratio][0]*1e-6}
            for ii, model in enumerate(models)\
            for electrode_distance_ratio in electrode_distance_ratios\
            for stim_amp_level in stim_amp_levels]

##### get latencies
latency_table = th.util.map(func = aly.get_latency,
                            space = params,
                            backend = backend,
                            cache = "no",
                            kwargs = {"dt" : 1*us,
                                      "phase_duration" : 45*us,
                                      "inter_phase_gap" : 2*us,
                                      "stimulus_node" : stim_node,
                                      "time_after" : 5*ms,
                                      "stimulation_type" : "extern",
                                      "pulse_form" : "bi"})

##### change index to column
latency_table.reset_index(inplace=True)

##### change column names
latency_table = latency_table.rename(index = str, columns={"model_name" : "model",
                                                           "stim_amp" : "stimulus amplitude (uA)",
                                                           "electrode_distance": "electrode distance (um)",
                                                           0:"latency (ms)"})

##### exclude rows where no AP was elicited
latency_table = latency_table[latency_table["latency (ms)"] != 0]

##### convert electrode distances to um
latency_table["electrode distance (um)"] = [np.round(distance*1e6).astype(int) for distance in latency_table["electrode distance (um)"]]

##### add amplitude level (factor by which threshold is multiplied)
latency_table["amplitude level"] = latency_table["stimulus amplitude (uA)"] / \
                                                [thresholds["threshold (uA)"][thresholds["model"] == latency_table["model"][ii]]\
                                                [thresholds["electrode distance (um)"] == latency_table["electrode distance (um)"][ii]][0]\
                                                for ii in range(len(latency_table))]

##### convert latency values to ms and stimulus amplitude to uA
latency_table["latency (ms)"] = [ii*1e3 for ii in latency_table["latency (ms)"]]
latency_table["stimulus amplitude (uA)"] = [ii*1e6 for ii in latency_table["stimulus amplitude (uA)"]]

##### Save dataframe as csv    
latency_table.to_csv("test_battery_results/Analyses/latency_table_models.csv", index=False, header=True)

##### get experimental data
latency_measurements = pd.read_csv("Measurements/Latency_data/latency_measurements.csv")

##### add stimulus amplitude levels to latency_measurements
latency_measurements["amplitude level"] = latency_measurements["stimulus amplitude (uA)"] / latency_measurements["threshold"]

##### plot latencies over stimulus amplitudes
latencies_over_stimulus_duration_plot = plot.latencies_over_stimulus_duration(plot_name = "Latencies over stimulus durations",
                                                                              latency_models = latency_table,
                                                                              latency_measurements = latency_measurements)

if generate_plots:
    ##### save plot
    latencies_over_stimulus_duration_plot.savefig("test_battery_results/Analyses/latencies_over_stimulus_duration_plot {}.png", bbox_inches='tight')

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

if all_tests or stochastic_properties_test:
    # =============================================================================
    # Get relative spread for different k_noise values
    # =============================================================================
    ##### define k_noise values to test
    k_noise_factor = np.append(np.round(np.arange(0.2,1,0.1),1), np.arange(1,5,0.5)).tolist()
    #k_noise_factor = [0.5,1,2]
    
    ##### define test parameters
    phase_duration = 100*us
    pulse_form = "mono"
    runs_per_k_noise = 100
    
    ##### define varied parameters
    params = {"model_name" : models,
              "parameter_ratio": k_noise_factor,
              "run_number" : [ii for ii in range(runs_per_k_noise)]}
    
    ##### get stochastic thresholds
    relative_spreads = th.util.map(func = test.get_threshold,
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
    #k_noise_factor = [0.5,1,2]
    
    ##### define test parameters
    phase_duration = 100*us
    pulse_form = "mono"
    runs_per_k_noise = 100
    
    ##### look up deterministic thresholds
    thresholds = threshold_table[threshold_table["phase duration (us)"] == phase_duration/us]
    thresholds = thresholds[threshold_table["pulse form"] == pulse_form][["model","threshold"]]
    
    ##### define varied parameters 
    params = [{"model_name" : model,
               "stim_amp" : threshold_table.set_index("model").transpose()[model]["threshold"],
               "parameter_ratio" : k_noise_factor[ii],
               "run_number" : jj}
                for model in models
                for ii in range(len(k_noise_factor))
                for jj in range(runs_per_k_noise)]
    
    ##### get single node response properties
    single_node_response_table = th.util.map(func = test.get_single_node_response,
                                             space = params,
                                             backend = backend,
                                             cache = "no",
                                             kwargs = {"dt" : 1*us,
                                                       "parameter": "k_noise",
                                                       "phase_duration" : phase_duration,
                                                       "pulse_form" : pulse_form,
                                                       "stimulation_type" : "extern",
                                                       "time_before" : 3*ms,
                                                       "time_after" : 2*ms,
                                                       "add_noise" : True})
    
    ##### change index to column
    single_node_response_table.reset_index(inplace=True)
    
    ##### change column names
    single_node_response_table = single_node_response_table.rename(index = str, columns={"model_name" : "model",
                                                                                         "run_number" : "run",
                                                                                         "parameter_ratio" : "knoise ratio",
                                                                                         0 : "AP height (mV)",
                                                                                         1 : "rise time (us)",
                                                                                         2 : "fall time (us)",
                                                                                         3 : "latency (us)",
                                                                                         4 : "membrane potential (mV)",
                                                                                         5 : "time (ms)"})
    
    ##### exclude data, where no action potential was elicited 
    single_node_response_table = single_node_response_table[single_node_response_table["AP height (mV)"] > 0.06]
    
    ##### build subset of relevant columns
    single_node_response_table = single_node_response_table[["model","knoise ratio","run","latency (us)"]]
    
    ##### change units from second to us
    single_node_response_table["latency (us)"] = single_node_response_table["latency (us)"]*1e6
    
    ##### calculate jitter
    single_node_response_table = single_node_response_table.groupby(["model","knoise ratio"])["latency (us)"].std().reset_index()
    single_node_response_table = single_node_response_table.rename(index = str, columns={"latency (us)" : "jitter (us)"})
    
    # =============================================================================
    # Plot relative spread over jitter for different models and k_noise values
    # =============================================================================
    relative_spreads = pd.read_csv("test_battery_results/Analyses/relative_spreads_k_noise_comparison.csv")
    single_node_response_table = pd.read_csv("test_battery_results/Analyses/single_node_response_table_k_noise_comparison.csv")
    
    
    ##### Combine relative spread and jitter information and exclude rows with na values
    stochasticity_table = pd.merge(relative_spreads, single_node_response_table, on=["model","knoise ratio"]).dropna()
    
    ##### plot table
    stochasticity_plot = plot.stochastic_properties_comparison(plot_name = "Comparison of stochastic properties",
                                                               stochasticity_table = stochasticity_table)
    
    ##### save plot
    stochasticity_plot.savefig("test_battery_results/Analyses/stochasticity_plot.png", bbox_inches='tight')

if all_tests or voltage_courses_comparison:
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
    
    if generate_plots:
        ##### save plot
        voltage_course_comparison.savefig("test_battery_results/Analyses/voltage_course_comparison_plot {}.png", bbox_inches='tight')







