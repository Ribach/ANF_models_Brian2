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
import models.Negm_2014_node_model as negm_14_node_model

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

Mm = StateMonitor(neuron, 'm', record=True)
Mn = StateMonitor(neuron, 'n', record=True)
Mh = StateMonitor(neuron, 'h', record=True)

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
post_stimulus_time_histogram = False

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines or plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = "extern",
                                                pulse_form = "mono",
                                                stimulated_compartment = 4,
                                                nof_pulses = 1,
                                                time_before = 2*ms,
                                                time_after = 2*ms,
                                                add_noise = False,
                                                ##### monophasic stimulation
                                                amp_mono = -2*uA,
                                                duration_mono = 100*us,
                                                ##### biphasic stimulation
                                                amps_bi = [-8,8]*uA,
                                                durations_bi = [40,0,40]*us,
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = 2*ms)
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = dt)
    
    ##### reset state monitor
    restore('initialized')
            
    ##### run simulation
    run(runtime)
    
    ##### generate figure
    comp_index = np.where(model.structure == 2)[0][10]
    fig = plt.figure(1)
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(Mm.t, Mm.m[comp_index,:], label = "m")
    axes.plot(Mh.t, Mh.h[comp_index,:], label = "h")
    axes.plot(Mn.t, Mn.n[comp_index,:], label = "n")
    plt.legend()

    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
        plot.voltage_course_lines(plot_name = "Voltage course {}".format(model.display_name),
                                  time_vector = M.t,
                                  voltage_matrix = M.v,
                                  comps_to_plot = model.comps_to_plot,
                                  distance_comps_middle = model.distance_comps_middle,
                                  length_neuron = model.length_neuron,
                                  V_res = model.V_res)
    
    if plot_voltage_course_colored:
        plot.voltage_course_colors(plot_name = "Voltage course {} (colored)".format(model.display_name),
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
    voltage_data, node_response_data_summary = \
    test.get_single_node_response(model = [rattay_01, negm_14, smit_10, frijns_05],
                                   dt = dt,
                                   param_1 = "length_internodes",
                                   param_1_ratios = [0.6, 0.8, 1, 2, 3],
                                   param_2 = "nof_segments_internode",
                                   param_2_ratios = [0.6, 0.8, 1, 2, 3],
                                   stimulation_type = "extern",
                                   pulse_form = "bi",
                                   time_after_stimulation = 1.5*ms,
                                   stim_amp = 2*uA,
                                   phase_duration = 200*us,
                                   nof_runs = 10)
    
    ##### plot voltage courses of single node
    plot.single_node_response_voltage_course(voltage_data = voltage_data)
    
    ##### plot results in bar plot
    plot.single_node_response_bar_plot(data = node_response_data_summary)
