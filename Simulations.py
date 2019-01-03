# =============================================================================
# This script plots the voltage courses, ionic currents and gating variables
# for any stimulation of a certain model. This allows to investigate the impact
# of certain stimulation parameters.
# =============================================================================
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
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14
import models.Rudnicki_2018 as rudnicki_18
import models.trials.Rattay_adap_2001 as rattay_adap_01
import models.trials.Briaire_adap_2005 as briaire_adap_05
import models.trials.Imennov_adap_2009 as imennov_adap_09
import models.trials.Negm_ANF_2014 as negm_ANF_14
import models.trials.Woo_2010 as woo_10

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Simulations to be done / Plots to be shown
# =============================================================================
plot_voltage_course_lines = True
plot_voltage_course_colored = True
plot_gating_variables = False
plot_ion_currents = False

# =============================================================================
# Definition of neuron and initialization of state monitor
# =============================================================================
##### choose model
model = woo_10

##### initialize clock
dt = 5*us

##### set up the neuron
neuron, model = model.set_up_model(dt = dt, model = model)

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

##### record gating variables
if plot_gating_variables:
    if model in [rattay_01,rattay_adap_01,frijns_94,briaire_05,briaire_adap_05,negm_14,negm_ANF_14,rudnicki_18]:
        M_gate = StateMonitor(neuron, ['m','n','h'], record=True)
    if model == smit_09:
        M_gate = StateMonitor(neuron, ['m_t','m_p','n','h'], record=True)
    if model == smit_10:
        M_gate = StateMonitor(neuron, ['m_t_Smit','m_p_Smit','h_Smit','n_Smit','m_Rat','h_Rat','n_Rat'], record=True)
    if model in [imennov_09,imennov_adap_09]:
        M_gate = StateMonitor(neuron, ['m','h','ns','nf'], record=True)
    if model in [negm_14,rattay_adap_01,briaire_adap_05,negm_ANF_14,imennov_adap_09]:
        M_adap = StateMonitor(neuron, ['w','z','r'], record=True)

##### record currents
if plot_ion_currents:
    if model in [rattay_01,rattay_adap_01,frijns_94,briaire_05,briaire_adap_05,negm_14,negm_ANF_14,rudnicki_18]:
        I = StateMonitor(neuron, ['I_Na','I_K','I_L'], record=True)
    if model == smit_09:
        I = StateMonitor(neuron, ['I_Na','I_Na_p','I_K','I_L'], record=True)
    if model == smit_10:
        I = StateMonitor(neuron, ['I_Na_transient_Smit','I_Na_persistent_Smit','I_K_Smit','I_L_Smit','I_Na_Rat','I_K_Rat','I_L_Rat'], record=True)
    if model in [imennov_09,imennov_adap_09]:
        I = StateMonitor(neuron, ['I_Na','I_Ks','I_Kf','I_L'], record=True)
    if model in [negm_14,negm_ANF_14,rattay_adap_01,briaire_adap_05,imennov_adap_09]:
        I_adap = StateMonitor(neuron, ['I_KLT','I_HCN'], record=True)

##### get compartment number of second node
stim_comp_index = np.where(model.structure == 2)[0][1]

##### save initialization of the monitor(s)
store('initialized')

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines or plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = "intern",
                                                pulse_form = "mono",
                                                stimulated_compartment = stim_comp_index,
                                                nof_pulses = 1,
                                                time_before = 0.5*ms,
                                                time_after = 1*ms,
                                                add_noise = False,
                                                ##### monophasic stimulation
                                                amp_mono = 60*pA,
                                                duration_mono = 60*us,
                                                ##### biphasic stimulation
                                                amps_bi = [-150,150]*uA,
                                                durations_bi = [400,0,400]*us,
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = 1*ms)
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = dt)
    
    ##### reset state monitor
    restore('initialized')
            
    ##### run simulation
    run(runtime)

    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
       voltage_course_lines = plot.voltage_course_lines(plot_name = "Voltage course {}".format(model.display_name),
                                                        time_vector = M.t,
                                                        voltage_matrix = M.v,
                                                        comps_to_plot = model.comps_to_plot,
                                                        distance_comps_middle = model.distance_comps_middle,
                                                        length_neuron = model.length_neuron,
                                                        V_res = model.V_res)
    
    if plot_voltage_course_colored:
        voltage_course_colors = plot.voltage_course_colors(plot_name = "Voltage course {} (colored)".format(model.display_name),
                                                           time_vector = M.t,
                                                           voltage_matrix = M.v,
                                                           distance_comps_middle = model.distance_comps_middle)

if plot_gating_variables:
    ##### plot gating variables
    comp_index = np.where(model.structure == 2)[0][-2]
    comp_index = stim_comp_index
    fig = plt.figure("gating variables")
    axes = fig.add_subplot(1, 1, 1)
    if model in [rattay_01,rattay_adap_01,frijns_94,briaire_05,briaire_adap_05,negm_14,negm_ANF_14,rudnicki_18]:
        axes.plot(M.t/ms, M_gate.m[comp_index,:], label = "m")
        axes.plot(M.t/ms, M_gate.h[comp_index,:], label = "h")
        axes.plot(M.t/ms, M_gate.n[comp_index,:], label = "n")
    if model == smit_09:
        axes.plot(M.t/ms, M_gate.m_t[comp_index,:], label = "m_t")
        axes.plot(M.t/ms, M_gate.m_p[comp_index,:], label = "m_p")
        axes.plot(M.t/ms, M_gate.n[comp_index,:], label = "n")
        axes.plot(M.t/ms, M_gate.h[comp_index,:], label = "h")
    if model == smit_10:
        axes.plot(M.t/ms, M_gate.m_t_Smit[comp_index,:], label = "m_t_Smit")
        axes.plot(M.t/ms, M_gate.m_p_Smit[comp_index,:], label = "m_p_Smit")
        axes.plot(M.t/ms, M_gate.h_Smit[comp_index,:], label = "h_Smit")
        axes.plot(M.t/ms, M_gate.n_Smit[comp_index,:], label = "n_Smit")
        axes.plot(M.t/ms, M_gate.m_Rat[comp_index,:], label = "m_Rat")
        axes.plot(M.t/ms, M_gate.h_Rat[comp_index,:], label = "h_Rat")
        axes.plot(M.t/ms, M_gate.n_Rat[comp_index,:], label = "n_Rat")
    if model in [imennov_09, imennov_adap_09]:
        axes.plot(M.t/ms, M_gate.m[comp_index,:], label = "m")
        axes.plot(M.t/ms, M_gate.h[comp_index,:], label = "h")
        axes.plot(M.t/ms, M_gate.ns[comp_index,:], label = "ns")
        axes.plot(M.t/ms, M_gate.nf[comp_index,:], label = "nf")
    if model in [negm_14,negm_ANF_14,rattay_adap_01,briaire_adap_05,imennov_adap_09]:
        axes.plot(M.t/ms, M_adap.w[comp_index,:], label = "w")
        axes.plot(M.t/ms, M_adap.z[comp_index,:], label = "z")
        axes.plot(M.t/ms, M_adap.r[comp_index,:], label = "r")
    plt.legend()

if plot_ion_currents:
    ##### plot currents
    fig = plt.figure("ion currents")
    axes = fig.add_subplot(1, 1, 1)
    if model in [rattay_01,rattay_adap_01,frijns_94,briaire_05,briaire_adap_05,negm_14,negm_ANF_14,rudnicki_18]:
        axes.plot(M.t/ms, I.I_Na[comp_index,:], label = "I_Na")
        axes.plot(M.t/ms, I.I_K[comp_index,:], label = "I_K")
        axes.plot(M.t/ms, I.I_L[comp_index,:], label = "I_L")
    if model == smit_09:
        axes.plot(M.t/ms, I.I_Na[comp_index,:], label = "I_Na")
        axes.plot(M.t/ms, I.I_Na_p[comp_index,:], label = "I_Na_p")
        axes.plot(M.t/ms, I.I_K[comp_index,:], label = "I_K")
        axes.plot(M.t/ms, I.I_L[comp_index,:], label = "I_L")
    if model == smit_10:
        axes.plot(M.t/ms, I.I_Na_transient_Smit[comp_index,:], label = "I_Na_t")
        axes.plot(M.t/ms, I.I_Na_persistent_Smit[comp_index,:], label = "I_Na_p")
        axes.plot(M.t/ms, I.I_K_Smit[comp_index,:], label = "I_K_Smit")
        axes.plot(M.t/ms, I.I_L_Smit[comp_index,:], label = "I_L_Smit")
        axes.plot(M.t/ms, I.I_Na_Rat[comp_index,:], label = "I_Na_Rat")
        axes.plot(M.t/ms, I.I_K_Rat[comp_index,:], label = "I_K_Rat")
        axes.plot(M.t/ms, I.I_L_Rat[comp_index,:], label = "I_L_Rat")
    if model in [imennov_09, imennov_adap_09]:
        axes.plot(M.t/ms, I.I_Na[comp_index,:], label = "I_Na")
        axes.plot(M.t/ms, I.I_Ks[comp_index,:], label = "I_Ks")
        axes.plot(M.t/ms, I.I_Kf[comp_index,:], label = "I_Kf")
        axes.plot(M.t/ms, I.I_L[comp_index,:], label = "I_L")
    if model in [negm_14,negm_ANF_14,rattay_adap_01,briaire_adap_05,imennov_adap_09]:
        axes.plot(M.t/ms, I_adap.I_KLT[comp_index,:], label = "I_KLT")
        axes.plot(M.t/ms, I_adap.I_HCN[comp_index,:], label = "I_HCN")
    plt.legend()
