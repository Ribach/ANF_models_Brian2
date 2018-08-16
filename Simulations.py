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

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Definition of neuron and initialization of state monitor
# =============================================================================
##### choose model
model = smit_09

##### initialize clock
dt = 5*us

##### set up the neuron
neuron, param_string = model.set_up_model(dt = dt, model = model)

##### load the parameters of the differential equations in the workspace
exec(param_string)

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

##### save initialization of the monitor(s)
store('initialized')

# =============================================================================
# Simulations to be done / Plots to be shown
# =============================================================================
plot_voltage_course_lines = True
plot_voltage_course_colored = False
measure_single_node_response = False
measure_strength_duration_curve = False
measure_refractory_properties = False
post_stimulus_time_histogram = True

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines or plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = "extern",
                                                pulse_form = "bi",
                                                stimulated_compartment = 4,
                                                nof_pulses = 4,
                                                time_before = 0*ms,
                                                time_after = 1*ms,
                                                add_noise = True,
                                                ##### monophasic stimulation
                                                amp_mono = -1.5*uA,
                                                duration_mono = 200*us,
                                                ##### biphasic stimulation
                                                amps_bi = [-2,0.2]*uA,
                                                durations_bi = [100,0,100]*us,
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = 800*us)
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = dt)
    
    ##### reset state monitor
    restore('initialized')
            
    ##### run simulation
    run(runtime)
    
    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
        plot.voltage_course_lines(plot_name = f"Voltage course {model.display_name}",
                                  time_vector = M.t,
                                  voltage_matrix = M.v,
                                  comps_to_plot = model.comps_to_plot,
                                  distance_comps_middle = model.distance_comps_middle,
                                  length_neuron = model.length_neuron,
                                  V_res = model.V_res)
    
    if plot_voltage_course_colored:
        plot.voltage_course_colors(plot_name = f"Voltage course {model.display_name} (colored)",
                                   time_vector = M.t,
                                   voltage_matrix = M.v,
                                   distance_comps_middle = model.distance_comps_middle)

# =============================================================================
# Now a simulation will be run several times to calculate the
# following temporal characteristics of the model:
# - average AP amplitude
# - average AP rise time
# - average AP fall time
# - average latency 
# - jitter
# =============================================================================
if measure_single_node_response:

    ##### Possible parameter types are all model attributes, "model", "stim_amp", "phase_duration" and "stochastic_runs"
    voltage_data, node_response_data_summary, time_vector = \
    test.get_single_node_response(model = [rattay_01, smit_10, smit_09],
                                   dt = dt,
                                   param_1 = "k_noise",
                                   param_1_ratios = [0.6, 0.8, 1.0, 2.0, 3.0],
                                   param_2 = "stochastic_runs",
                                   param_2_ratios = [0.6, 0.8, 1.0, 2.0, 3.0],
                                   stimulation_type = "extern",
                                   pulse_form = "bi",
                                   time_after_stimulation = 1.5*ms,
                                   stim_amp = 2*uA,
                                   phase_duration = 200*us,
                                   nof_runs = 10)
    
    ##### plot results in bar plot
    plot.single_node_response_voltage_course(voltage_data = voltage_data)
    
    ##### plot results in bar plot
    plot.single_node_response_bar_plot(data = node_response_data_summary)

# =============================================================================
# Now a simulation will be run several times to calculate the strength-duration
#  curve. This allows to determine the following properties
# - Rheobase
# - chronaxie
# =============================================================================
if measure_strength_duration_curve:
    
    ##### define phase_durations to be tested
    phase_durations = np.round(np.logspace(1, 9, num=20, base=2.0))*us
    
    ##### calculated corresponding thresholds
    thresholds = test.get_strength_duration_curve(model = model,
                                                  dt = 1*us,
                                                  phase_durations = phase_durations,
                                                  start_intervall = [0.1,20]*uA,
                                                  delta = 0.01*uA,
                                                  stimulation_type = "extern",
                                                  pulse_form = "bi")
    
    ##### plot strength duration curve
    plot.strength_duration_curve(plot_name = f"Strength duration curve {model.display_name}",
                                 durations = phase_durations,
                                 stimulus_amps = thresholds)
    
# =============================================================================
# Now the relative and absolute refractory periods will be measured. This is
# done with two stimuli, the first one with an amplitude of 150% of threshold
# masks the second one. Threshold amplitudes of the second stimulus over inter-
# pulse interval are measured
# =============================================================================
if measure_refractory_properties:
    
    ##### define inter-pulse-intervalls
    inter_pulse_intervalls = np.logspace(-1, 2, num=20, base=2)*ms
    
    ##### Define stimulation
    stimulation_type = "extern"
    pulse_form = "mono"
    phase_duration = 100*us
    
    min_required_amps, threshold = test.get_refractory_curve(model = model,
                                                             dt = 5*us,
                                                             inter_pulse_intervalls = inter_pulse_intervalls,
                                                             delta = 0.02*uA,
                                                             stimulation_type = "extern",
                                                             pulse_form = "mono",
                                                             phase_duration = 100*us)
    
    ##### plot strength duration curve
    plot.refractory_curve(plot_name = f"Threshold curve {model.display_name}",
                          inter_pulse_intervalls = inter_pulse_intervalls,
                          stimulus_amps = min_required_amps,
                          threshold = threshold)
    
# =============================================================================
# Post Stimulus Time Histogram
# =============================================================================
if post_stimulus_time_histogram:
    
    ##### define bin_width
    bin_width = 2*ms
    
    ##### calculate bin heigths and edges
    bin_heigths, bin_edges = test.post_stimulus_time_histogram(model = model,
                                                        dt = dt,
                                                        nof_repeats = 15,
                                                        pulses_per_second = 2000,
                                                        stim_duration = 50*ms,
                                                        stim_amp = 10*uA,
                                                        stimulation_type = "extern",
                                                        pulse_form = "bi",
                                                        phase_duration = 20*us,
                                                        bin_width = bin_width)
    
    ##### plot post_stimulus_time_histogram
    plot.post_stimulus_time_histogram(plot_name = f"PSTH {model.display_name}",
                                      bin_edges = bin_edges,
                                      bin_heigths = bin_heigths,
                                      bin_width = bin_width/ms)

# =============================================================================
# Test map function
# =============================================================================
def multiply(a, b):
    """Multiply two numbers."""

    # The output should be wrapped in a dict
    y = {'y': a * b}

    return y


# Keys in the input dict MUST correspond to the key word
# arguments of the function (e.g. multiply)
xs = {
    'a': np.arange(50),
    'b': np.arange(50)
}

# Uncomment `backend` for parallel execution
ys = th.util.map(func = multiply,
                 space = xs,
                 backend='multiprocessing',
                 cache = 'no'
)
