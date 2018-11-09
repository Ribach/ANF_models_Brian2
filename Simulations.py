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
import models.Rattay_adap_2001 as rattay_adap_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Briaire_adap_2005 as briaire_adap_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14
import models.Negm2_2014 as negm2_14
import models.Negm3_2014 as negm3_14
import models.Rudnicki_2018 as rudnicki_18

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Definition of neuron and initialization of state monitor
# =============================================================================
##### choose model
model = briaire_adap_05

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
Mw = StateMonitor(neuron, 'w', record=True)
Mz = StateMonitor(neuron, 'z', record=True)
Mr = StateMonitor(neuron, 'r', record=True)

M_I_Na = StateMonitor(neuron, 'I_Na', record=True)
M_I_K = StateMonitor(neuron, 'I_K', record=True)
M_I_KLT = StateMonitor(neuron, 'I_KLT', record=True)
M_I_HCN = StateMonitor(neuron, 'I_HCN', record=True)
M_I_L = StateMonitor(neuron, 'I_L', record=True)

##### get compartment number of second node
stim_comp_index = np.where(model.structure == 2)[0][1]

##### save initialization of the monitor(s)
store('initialized')

# =============================================================================
# Simulations to be done / Plots to be shown
# =============================================================================
plot_voltage_course_lines = True
plot_voltage_course_colored = True
measure_single_node_response = False

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines or plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = "extern",
                                                pulse_form = "mono",
                                                stimulated_compartment = stim_comp_index,
                                                nof_pulses = 200,
                                                time_before = 1*ms,
                                                time_after = 2*ms,
                                                add_noise = False,
                                                ##### monophasic stimulation
                                                amp_mono = -4*uA,
                                                duration_mono = 100*us,
                                                ##### biphasic stimulation
                                                amps_bi = [-4,4]*uA,
                                                durations_bi = [50,0,50]*us,
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = 0.5*ms)
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = dt)
    
    ##### reset state monitor
    restore('initialized')
            
    ##### run simulation
    run(runtime)

    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
        plot.voltage_course_lines(plot_name = "Voltage course {} symetric pulses".format(model.display_name),
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

##### plot gating variables
comp_index = np.where(model.structure == 3)[0][3]
fig = plt.figure("gating variables")
axes = fig.add_subplot(1, 1, 1)
axes.plot(Mm.t/ms, Mm.m[comp_index,:], label = "m")
axes.plot(Mh.t/ms, Mh.h[comp_index,:], label = "h")
axes.plot(Mn.t/ms, Mn.n[comp_index,:], label = "n")
axes.plot(Mw.t/ms, Mw.w[comp_index,:], label = "w")
axes.plot(Mz.t/ms, Mz.z[comp_index,:], label = "z")
axes.plot(Mr.t/ms, Mr.r[comp_index,:], label = "r")
plt.legend()

##### plot currents
fig = plt.figure("ion currents")
axes = fig.add_subplot(1, 1, 1)
axes.plot(M.t/ms, M_I_Na.I_Na[comp_index,:], label = "I_Na")
axes.plot(M.t/ms, M_I_K.I_K[comp_index,:], label = "I_K")
axes.plot(M.t/ms, M_I_KLT.I_KLT[comp_index,:], label = "I_KLT")
axes.plot(M.t/ms, M_I_HCN.I_HCN[comp_index,:], label = "I_HCN")
axes.plot(M.t/ms, M_I_L.I_L[comp_index,:], label = "I_L")
plt.legend()
        
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
    test.get_single_node_response(model = [rattay_01, negm_14, smit_10, briaire_05],
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
