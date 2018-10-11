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
# Initializations
# =============================================================================
##### choose model
model_name = "rattay_01"
model = eval(model_name)

##### initialize clock
dt = 5*us

##### define way of processing
backend = "serial"

##### define if plots should be generated
generate_plots = True

##### define which tests to run
all_tests = False
strength_duration_test = False
relative_spread_test = False
conduction_velocity_test = False
single_node_response_test = False
refractory_test = False
psth_test = False

if any([all_tests, single_node_response_test, refractory_test, psth_test]):
    # =============================================================================
    # Get thresholds for certain stimulation types and stimulus durations
    # =============================================================================
    
    ##### define phase durations to test (in us)
    phase_durations_mono = [40,50,100]
    phase_durations_bi = [20,40,50,200,400]
    
    ##### define test parameters
    phase_durations = [ii*1e-6 for ii in phase_durations_mono + phase_durations_bi]
    pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))
    
    ##### define varied parameters 
    params = [{"phase_duration" : phase_durations[ii],
               "pulse_form" : pulse_form[ii]} for ii in range(len(phase_durations))]
    
    ##### get thresholds
    threshold_table = th.util.map(func = test.get_threshold,
                                  space = params,
                                  backend = backend,
                                  cache = "no",
                                  kwargs = {"model_name" : model_name,
                                            "dt" : dt,
                                            "delta" : 0.0001*uA,
                                            "stimulation_type" : "extern",
                                            "amps_start_interval" : [0,30]*uA,
                                            "add_noise" : False,
                                            "run_number" : 1})
    
    ##### change index to column
    threshold_table.reset_index(inplace=True)
    
    ##### change column names
    threshold_table = threshold_table.rename(index = str, columns={"phase_duration" : "phase duration",
                                                                   "pulse_form" : "pulse form",
                                                                   0:"threshold"})
    
    ##### add unit to phase duration
    threshold_table["phase duration"] = [ii*second for ii in threshold_table["phase duration"]]
    
    ##### built subset of dataframe
    threshold_table = threshold_table[["phase duration", "pulse form", "threshold"]]
    
    ##### Save dataframe as csv    
    threshold_table.to_csv("test_battery_results/{}/Threshold_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)

if all_tests or strength_duration_test:
    # =============================================================================
    # Get chronaxie and rheobase
    # =============================================================================
    ##### get rheobase
    rheobase = test.get_threshold(model_name = model_name,
                                  dt = dt,
                                  phase_duration = 1*ms,
                                  delta = 0.0001*uA,
                                  amps_start_interval = [0,20]*uA,
                                  stimulation_type = "extern",
                                  pulse_form = "mono")
    
    ##### get chronaxie
    chronaxie = test.get_chronaxie(model = model,
                                   dt = dt,
                                   rheobase = rheobase,
                                   phase_duration_start_interval = [0,1000]*us,
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
    phase_durations = np.round(np.logspace(1, 9, num=50, base=2.0))*us
    
    ##### define varied parameter    
    params = {"phase_duration" : phase_durations}
    
    ##### get thresholds
    strength_duration_plot_table = th.util.map(func = test.get_threshold,
                                          space = params,
                                          backend = backend,
                                          cache = "no",
                                          kwargs = {"model_name" : model_name,
                                                    "dt" : 1*us,
                                                    "delta" : 0.01*uA,
                                                    "pulse_form" : "mono",
                                                    "stimulation_type" : "extern",
                                                    "amps_start_interval" : [0,20]*uA,
                                                    "add_noise" : False})
    
    ##### change index to column
    strength_duration_plot_table.reset_index(inplace=True)
    
    ##### change column names
    strength_duration_plot_table = strength_duration_plot_table.rename(index = str, columns={"phase_duration" : "phase duration (us)",
                                                                                             0 : "threshold (uA)"})
    
    ##### remove units of columns
    strength_duration_plot_table["phase duration (us)"] = [ii/us for ii in strength_duration_plot_table["phase duration (us)"]]
    
    ##### change unit of threshold column to uA
    strength_duration_plot_table["threshold (uA)"] = [ii*1e6 for ii in strength_duration_plot_table["threshold (uA)"]]
    
    ##### save strength duration table
    strength_duration_plot_table.to_csv("test_battery_results/{}/Strength_duration_plot_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)

    if generate_plots:
        ##### plot strength duration curve
        strength_duration_curve = plot.strength_duration_curve(plot_name = "Strength duration curve {}".format(model.display_name),
                                                               threshold_matrix = strength_duration_plot_table,
                                                               rheobase = rheobase,
                                                               chronaxie = chronaxie)
        
        ##### save strength duration curve and table
        strength_duration_curve.savefig("test_battery_results/{}/Strength_duration_curve {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    
if all_tests or relative_spread_test:
    # =============================================================================
    # Get the relative spread of thresholds
    # =============================================================================
    ##### define phase durations to test (in us)
    phase_durations_mono = [40,100]
    phase_durations_bi = [200,400]
    
    ##### define test parameters
    phase_durations = [ii*1e-6 for ii in phase_durations_mono + phase_durations_bi]
    pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))
    runs_per_stimulus_type = 60
    
    ##### define varied parameters 
    params = [{"phase_duration" : phase_durations[ii],
               "run_number" : jj,
               "pulse_form" : pulse_form[ii]} for ii in range(len(phase_durations)) for jj in range(runs_per_stimulus_type)]
    
    ##### get thresholds
    relative_spread_plot_table = th.util.map(func = test.get_threshold,
                                             space = params,
                                             backend = backend,
                                             cache = "no",
                                             kwargs = {"model_name" : model_name,
                                                       "dt" : dt,
                                                       "delta" : 0.0005*uA,
                                                       "stimulation_type" : "extern",
                                                       "amps_start_interval" : [0,20]*uA,
                                                       "time_before" : 2*ms,
                                                       "time_after" : 2*ms,
                                                       "add_noise" : True})
  
    ##### change index to column
    relative_spread_plot_table.reset_index(inplace=True)
    
    ##### change column names
    relative_spread_plot_table = relative_spread_plot_table.rename(index = str, columns={"phase_duration" : "phase duration",
                                                                                         "pulse_form" : "pulse form",
                                                                                         0:"threshold"})
    
    ##### exclude spontaneous APs
    relative_spread_plot_table = relative_spread_plot_table[relative_spread_plot_table["threshold"] > 1e-9]
    
    ##### adjust pulse form column
    relative_spread_plot_table["pulse form"] = ["monophasic" if relative_spread_plot_table["pulse form"][ii]=="mono" else "biphasic" for ii in range(np.shape(relative_spread_plot_table)[0])]
    
    ##### built subset of dataframe
    relative_spread_plot_table = relative_spread_plot_table[["phase duration", "pulse form", "threshold"]]
    
    if generate_plots:
        ##### plot relative spreads
        relative_spread_plot = plot.relative_spread(plot_name = "Relative spreads {}".format(model.display_name),
                                                    threshold_matrix = relative_spread_plot_table)
    
        ##### save relative spreads plot
        relative_spread_plot.savefig("test_battery_results/{}/Relative_spreads_plot {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    
    ##### save relative spreads plot table
    relative_spread_plot_table.to_csv("test_battery_results/{}/Relative_spread_plot_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)
    
    ##### calculate relative spread values
    thresholds = relative_spread_plot_table.groupby(["phase duration", "pulse form"])
    relative_spreads = round(thresholds.std()/thresholds.mean()*100, 2)
    relative_spreads.reset_index(inplace=True)
    relative_spreads = relative_spreads.rename(index = str, columns={"threshold" : "relative spread"})
    relative_spreads["relative spread"] = ["{}%".format(relative_spreads["relative spread"][ii]) for ii in range(np.shape(relative_spreads)[0])]
    
    ##### add unit to phase duration
    relative_spreads["phase duration"] = [ii*us for ii in relative_spreads["phase duration"]]
    
    ##### Save table as csv    
    relative_spreads.to_csv("test_battery_results/{}/Relative_spreads {}.csv".format(model.display_name,model.display_name), index=False, header=True)   

if all_tests or conduction_velocity_test:
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
        conduction_velocity_table = pd.DataFrame(np.array([[conduction_velocity_dendrite],[np.round(model.dendrite_outer_diameter/um,2)],[conduction_velocity_dendrite_ratio],
                                                           [conduction_velocity_axon],[np.round(model.axon_outer_diameter/um,2)],[conduction_velocity_axon_ratio]]).T,
                                                 columns = ["velocity dendrite (m/s)", "outer diameter dendrite (um)", "velocity/diameter dendrite",
                                                            "velocity axon (m/s)", "outer diameter axon (um)", "velocity/diameter axon"])
    
    ##### models without a soma 
    else:
       conduction_velocity = test.get_conduction_velocity(model = model,
                                                          dt = dt,
                                                          stimulated_compartment = node_indexes[1],
                                                          measurement_start_comp = node_indexes[2],
                                                          measurement_end_comp = node_indexes[-3])
       
       conduction_velocity_ratio = round((conduction_velocity/(meter/second))/(model.fiber_outer_diameter/um),2)
       
       ##### save values in dataframe
       conduction_velocity_table = pd.DataFrame(np.array([[conduction_velocity],[np.round(model.fiber_outer_diameter/um,2)],[conduction_velocity_ratio]]).T,
                                                columns = ["velocity (m/s)", "outer diameter (um)", "velocity/diameter"])
    
    ##### Save table as csv    
    conduction_velocity_table.to_csv("test_battery_results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)

if all_tests or single_node_response_test:
    # =============================================================================
    # Measure single node response properties
    # =============================================================================
    ##### define phase durations to test (in us)
    phase_durations_mono = [40,50,100]
    phase_durations_bi = [200]
    
    ##### generate lists with test parameters
    phase_durations = [ii*1e-6 for ii in phase_durations_mono + phase_durations_bi]
    pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))
    
    ##### look up thresholds
    thresholds = [threshold_table["threshold"][threshold_table["pulse form"] == pulse_form[ii]]\
                                             [threshold_table["phase duration"]/us == (phase_durations_mono + phase_durations_bi)[ii]].iloc[0]\
                                             for ii in range(len(phase_durations))]/amp
    
    ##### define stimulus durations to test
    stim_amp_levels = [1,2]
    stim_amps = [float(ii*jj) for ii in thresholds for jj in stim_amp_levels]
    
    ##### define number of stochastic runs
    nof_runs = 40
    
    ##### define varied parameters 
    params = [{"phase_duration" : phase_durations[ii],
               "pulse_form" : pulse_form[ii],
               "stim_amp" : stim_amps[2*ii+jj-1],
               "run_number" : kk}
                 for ii in range(len(phase_durations))\
                 for jj in stim_amp_levels\
                 for kk in range(nof_runs)]
    
    ##### get thresholds
    single_node_response_table = th.util.map(func = test.get_single_node_response,
                                             space = params,
                                             backend = backend,
                                             cache = "no",
                                             kwargs = {"model_name" : model_name,
                                                       "dt" : dt,
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
    
    ##### calculate means of AP characteristics and summarize them in a summary dataframe
    single_node_response_summary = single_node_response_summary.groupby(["phase duration (us)","stimulus amplitude (uA)", "amplitude level", "pulse form"])\
    ["AP height (mV)","rise time (us)","fall time (us)","AP duration (us)","latency (us)"].mean().reset_index()
    
    ##### add jitter to summary
    single_node_response_summary = pd.merge(single_node_response_summary, jitter, on=["phase duration (us)","stimulus amplitude (uA)","pulse form"])
    
    ##### round columns to 3 significant digits
    for ii in ["AP height (mV)","rise time (us)","fall time (us)","AP duration (us)","latency (us)","jitter (us)"]:
        single_node_response_summary[ii] = ["%.3g" %single_node_response_summary[ii][jj] for jj in range(single_node_response_summary.shape[0])]
    
    ##### Save table as csv    
    single_node_response_summary.to_csv("test_battery_results/{}/Single_node_response_summary {}.csv".format(model.display_name,model.display_name), index=False, header=True)
    
    ##### built dataset for voltage courses (to plot them)
    voltage_course_dataset = single_node_response_table[["phase duration (us)","stimulus amplitude (uA)", "amplitude level", "pulse form", "run", "membrane potential (mV)", "time (ms)"]]
    
    ##### split lists in membrane potential and time columns to multiple rows
    voltage_course_dataset = calc.explode(voltage_course_dataset, ["membrane potential (mV)", "time (ms)"])
    
    ##### convert membrane potential to mV and time to ms
    voltage_course_dataset["membrane potential (mV)"] = voltage_course_dataset["membrane potential (mV)"] *1e3
    voltage_course_dataset["time (ms)"] = voltage_course_dataset["time (ms)"] *1e3
    
    ##### start time values at zero
    voltage_course_dataset["time (ms)"] = voltage_course_dataset["time (ms)"] - min(voltage_course_dataset["time (ms)"])
    
    if generate_plots:
        ##### plot voltage courses of single node
        single_node_response = plot.single_node_response_voltage_course(plot_name = "Voltage courses {}".format(model.display_name),
                                                                        voltage_data = voltage_course_dataset)
        
        ##### save voltage courses plot
        single_node_response.savefig("test_battery_results/{}/Single_node_response {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    
    ###### save voltage courses table
    voltage_course_dataset.to_csv("test_battery_results/{}/Single_node_response_plot_data {}.csv".format(model.display_name,model.display_name), index=False, header=True)

if all_tests or refractory_test:
    # =============================================================================
    # Refractory periods
    # =============================================================================
    ##### define phase durations to test (in us)
    phase_durations_mono = [40,50,100]
    phase_durations_bi = [50,200]
    
    ##### generate lists with test parameters
    phase_durations = [ii*1e-6 for ii in phase_durations_mono + phase_durations_bi]
    pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))
    
    ##### look up thresholds
    thresholds = [threshold_table["threshold"][threshold_table["pulse form"] == pulse_form[ii]]\
                                             [threshold_table["phase duration"]/us == (phase_durations_mono + phase_durations_bi)[ii]].iloc[0]\
                                             for ii in range(len(phase_durations))]/amp
    
    ##### define varied parameters 
    params = [{"phase_duration" : phase_durations[ii],
               "pulse_form" : pulse_form[ii],
               "threshold" : float(thresholds[ii]),
               "amp_masker" : float(thresholds[ii]*1.5)} for ii in range(len(phase_durations))]
    
    ##### get refractory periods
    refractory_table = th.util.map(func = test.get_refractory_periods,
                                   space = params,
                                   backend = backend,
                                   cache = "no",
                                   kwargs = {"model_name" : model_name,
                                             "dt" : 1*us,
                                             "delta" : 1*us,
                                             "stimulation_type" : "extern"})
        
    ##### change index to column
    refractory_table.reset_index(inplace=True)
    
    ##### change column names
    refractory_table = refractory_table.rename(index = str, columns={"phase_duration" : "phase duration",
                                                                     "pulse_form" : "pulse form",
                                                                     0 : "absolute refractory period (us)",
                                                                     1 : "relative refractory period (ms)"})
    
    ##### add units to phase duration
    refractory_table["phase duration"] = [ii*second for ii in refractory_table["phase duration"]]
    
    ##### convert refractory periods to ms
    refractory_table["absolute refractory period (us)"] = refractory_table["absolute refractory period (us)"]*1e6
    refractory_table["relative refractory period (ms)"] = refractory_table["relative refractory period (ms)"]*1e3
    
    ##### adjust pulse form column
    refractory_table["pulse form"] = ["monophasic" if pulse_form[ii]=="mono" else "biphasic" for ii in range(len(phase_durations))]
    
    ##### built subset of dataframe
    refractory_table = refractory_table[["phase duration", "pulse form", "absolute refractory period (us)","relative refractory period (ms)"]]
    
    ##### round columns to 3 significant digits
    for ii in ["absolute refractory period (us)","relative refractory period (ms)"]:
        refractory_table[ii] = ["%.3g" %refractory_table[ii][jj] for jj in range(refractory_table.shape[0])]
    
    ##### Save dataframe as csv    
    refractory_table.to_csv("test_battery_results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)   
    
    # =============================================================================
    # Refractory curve
    # =============================================================================
    ##### define inter-pulse-intervals
    inter_pulse_intervals = model.inter_pulse_intervals
    
    ##### define stimulation parameters
    phase_duration = 100*us
    pulse_form = "mono"
    
    ##### look up threshold
    threshold = threshold_table["threshold"][threshold_table["pulse form"] == pulse_form]\
                                             [threshold_table["phase duration"]/us == phase_duration/us].iloc[0]*amp
    
    ##### define varied parameter    
    params = {"inter_pulse_interval" : inter_pulse_intervals}
    
    ##### get thresholds
    refractory_curve_table = th.util.map(func = test.get_refractory_curve,
                                         space = params,
                                         backend = backend,
                                         cache = "no",
                                         kwargs = {"model_name" : model_name,
                                                   "dt" : 1*us,
                                                   "delta" : 0.0001*uA,
                                                   "pulse_form" : pulse_form,
                                                   "stimulation_type" : "extern",
                                                   "phase_duration" : phase_duration,
                                                   "threshold" : threshold,
                                                   "amp_masker" : threshold*1.5})
    
    ##### change index to column
    refractory_curve_table.reset_index(inplace=True)
    
    ##### change column name
    refractory_curve_table = refractory_curve_table.rename(index = str, columns={"inter_pulse_interval" : "interpulse interval",
                                                                                 0 : "minimum required amplitude"})
    
    ##### built subset of dataframe
    refractory_curve_table = refractory_curve_table[["interpulse interval", "minimum required amplitude"]]
    
    ##### add threshold to dataframe
    refractory_curve_table["threshold"] = threshold/amp
    
    if generate_plots:
        ##### plot refractory curve
        refractory_curve = plot.refractory_curve(plot_name = "Refractory curve {}".format(model.display_name),
                                                 refractory_table = refractory_curve_table)
        
        ##### save refractory curve
        refractory_curve.savefig("test_battery_results/{}/Refractory_curve {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    
    ##### save refractory curve table
    refractory_curve_table.to_csv("test_battery_results/{}/Refractory_curve_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)   
    
if all_tests or psth_test:
    # =============================================================================
    # Post Stimulus Time Histogram
    # =============================================================================
    ##### pulse rates to test
    pulses_per_second = [250,1000,5000,10000]
    
    ##### phase durations for respective pulse rates
    phase_duration = [40,40,40,20]*us
    
    ##### stimulus levels (will be multiplied with the threshold for a certain stimulation) 
    stim_amp_level = [1,1.2,1.5]
    
    ##### number of runs
    nof_runs = 60
    
    ##### look up thresholds
    thresholds = [threshold_table["threshold"][threshold_table["pulse form"] == "bi"]\
                               [threshold_table["phase duration"]/us == phase_duration[ii]/us].iloc[0]\
                               for ii in range(len(phase_duration))]
    
    ##### define varied parameters 
    params = [{"pulses_per_second" : pulses_per_second[ii],
               "phase_duration" : phase_duration[ii]/second,
               "stim_amp" : stim_amp_level[jj]*thresholds[ii],
               "run_number" : kk}
                for ii in range(len(pulses_per_second))\
                for jj in range(len(stim_amp_level))\
                for kk in range(nof_runs)]
    
    ##### get thresholds
    psth_table = th.util.map(func = test.post_stimulus_time_histogram,
                             space = params,
                             backend = backend,
                             cache = "no",
                             kwargs = {"model_name" : model_name,
                                       "dt" : dt,
                                       "stim_duration" : 300*ms,
                                       "stimulation_type" : "extern",
                                       "pulse_form" : "bi"})
    
    ##### change index to column
    psth_table.reset_index(inplace=True)
    
    ##### change column names
    psth_table = psth_table.rename(index = str, columns={"phase_duration" : "phase duration (us)",
                                                         "pulses_per_second" : "pulse rate",
                                                         "stim_amp" : "stimulus amplitude (uA)",
                                                         "run_number" : "run",
                                                         0 : "spike times (us)"})
    
    ##### add row with stimulus amplitude information
    psth_table["amplitude"] = ["{}*threshold".format(stim_amp_level[jj])
                                for ii in range(len(pulses_per_second))
                                for jj in range(len(stim_amp_level))
                                for kk in range(nof_runs)]
    
    ##### built subset of dataframe
    psth_table = psth_table[["phase duration (us)", "pulse rate", "stimulus amplitude (uA)", "amplitude", "run", "spike times (us)"]]
    
    ##### split lists in spike times column to multiple rows
    psth_table = calc.explode(psth_table, ["spike times (us)"])
    
    ##### change units from second to us and form amp to uA
    psth_table["phase duration (us)"] = round(psth_table["phase duration (us)"]*1e6).astype(int)
    psth_table["stimulus amplitude (uA)"] = round(psth_table["stimulus amplitude (uA)"]*1e6,2)
    psth_table["spike times (us)"] = round(psth_table["spike times (us)"]*1e6).astype(int)
    
    if generate_plots:
        ##### plot post_stimulus_time_histogram
        post_stimulus_time_histogram = plot.post_stimulus_time_histogram(plot_name = "PSTH {}".format(model.display_name),
                                                                         psth_dataset = psth_table)
        
        ###### save post_stimulus_time_histogram
        post_stimulus_time_histogram.savefig("test_battery_results/{}/PSTH {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
        
    ###### save post_stimulus_time_histogram table
    psth_table.to_csv("test_battery_results/{}/PSTH_table {}.csv".format(model.display_name,model.display_name), index=False, header=True)   
