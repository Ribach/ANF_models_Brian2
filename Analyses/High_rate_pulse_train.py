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
# Definition of neuron and initialization of state monitor
# =============================================================================
##### choose model
model = negm_14

##### initialize clock
dt = 0.05*us

##### set up the neuron
neuron, param_string, model = model.set_up_model(dt = dt, model = model)

##### load the parameters of the differential equations in the workspace
exec(param_string)

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

##### record gating variables
if model in [rattay_01,frijns_94,briaire_05, negm_14]:
    Mm = StateMonitor(neuron, 'm', record=True)
    Mn = StateMonitor(neuron, 'n', record=True)
    Mh = StateMonitor(neuron, 'h', record=True)

if model == smit_09:
    Mm_t = StateMonitor(neuron, 'm_t', record=True)
    Mm_p = StateMonitor(neuron, 'm_p', record=True)
    Mn = StateMonitor(neuron, 'n', record=True)
    Mh = StateMonitor(neuron, 'h', record=True)

if model == smit_10:
    Mm_t = StateMonitor(neuron, 'm_t', record=True)
    Mm_p = StateMonitor(neuron, 'm_p', record=True)
    Mh_Smit = StateMonitor(neuron, 'h_Smit', record=True)
    Mn_Smit = StateMonitor(neuron, 'n_Smit', record=True)
    Mm_Rat = StateMonitor(neuron, 'm_Rat', record=True)
    Mh_Rat = StateMonitor(neuron, 'h_Rat', record=True)
    Mn_Rat = StateMonitor(neuron, 'n_Rat', record=True)
    
if model == imennov_09:
    Mm = StateMonitor(neuron, 'm', record=True)
    Mh = StateMonitor(neuron, 'h', record=True)
    Mns = StateMonitor(neuron, 'ns', record=True)
    Mnf = StateMonitor(neuron, 'nf', record=True)
    
if model == negm_14:
    Mw = StateMonitor(neuron, 'w', record=True)
    Mz = StateMonitor(neuron, 'z', record=True)
    Mr = StateMonitor(neuron, 'r', record=True)
    

##### save initialization of the monitor(s)
store('initialized')

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
##### define how the ANF is stimulated
I_stim, runtime = stim.get_stimulus_current(model = model,
                                            dt = dt,
                                            stimulation_type = "extern",
                                            pulse_form = "bi",
                                            stimulated_compartment = 4,
                                            nof_pulses = 100,
                                            time_before = 2*ms,
                                            time_after = 3*ms,
                                            add_noise = False,
                                            ##### monophasic stimulation
                                            amp_mono = -3*uA,
                                            duration_mono = 100*us,
                                            ##### biphasic stimulation
                                            amps_bi = [-1000,1000]*uA,
                                            durations_bi = [0.20,26,0.20]*us,
                                            ##### multiple pulses / pulse trains
                                            inter_pulse_gap = 29.3*us)

##### get TimedArray of stimulus currents
stimulus = TimedArray(np.transpose(I_stim), dt = dt)

##### reset state monitor
restore('initialized')
        
##### run simulation
run(runtime)

##### plot gating variables
comp_index = np.where(model.structure == 2)[0][10]
fig = plt.figure("gating variables")
axes = fig.add_subplot(1, 1, 1)
if model in [rattay_01,frijns_94,briaire_05,negm_14]:
    axes.plot(Mm.t/ms, Mm.m[comp_index,:], label = "m")
    axes.plot(Mh.t/ms, Mh.h[comp_index,:], label = "h")
    axes.plot(Mn.t/ms, Mn.n[comp_index,:], label = "n")
if model == smit_09:
    axes.plot(Mm_t.t/ms, Mm_t.m_t[comp_index,:], label = "m_t")
    axes.plot(Mm_p.t/ms, Mm_p.m_p[comp_index,:], label = "m_p")
    axes.plot(Mn.t/ms, Mn.n[comp_index,:], label = "n")
    axes.plot(Mh.t/ms, Mh.h[comp_index,:], label = "h")
if model == smit_10:
    axes.plot(Mm_t.t/ms, Mm_t.m_t[comp_index,:], label = "m_t")
    axes.plot(Mm_p.t/ms, Mm_p.m_p[comp_index,:], label = "m_p")
    axes.plot(Mh_Smit.t/ms, Mh_Smit.h_Smit[comp_index,:], label = "h_Smit")
    axes.plot(Mn_Smit.t/ms, Mn_Smit.n_Smit[comp_index,:], label = "n_Smit")
    axes.plot(Mm_Rat.t/ms, Mm_Rat.m_Rat[comp_index,:], label = "m_Rat")
    axes.plot(Mh_Rat.t/ms, Mh_Rat.h_Rat[comp_index,:], label = "h_Rat")
    axes.plot(Mn_Rat.t/ms, Mn_Rat.n_Rat[comp_index,:], label = "n_Rat")
if model == negm_14:
    axes.plot(Mw.t/ms, Mw.w[comp_index,:], label = "w")
    axes.plot(Mz.t/ms, Mz.z[comp_index,:], label = "z")
    axes.plot(Mr.t/ms, Mr.r[comp_index,:], label = "r")
if model == imennov_09:
    axes.plot(Mm.t/ms, Mm.m[comp_index,:], label = "m")
    axes.plot(Mh.t/ms, Mh.h[comp_index,:], label = "h")
    axes.plot(Mns.t/ms, Mns.ns[comp_index,:], label = "ns")
    axes.plot(Mnf.t/ms, Mnf.nf[comp_index,:], label = "nf")
plt.legend()

##### Plot membrane potential of all compartments over time (line plot)
if True:
    plot.voltage_course_lines(plot_name = "Voltage course {} symetric pulses".format(model.display_name),
                              time_vector = M.t,
                              voltage_matrix = M.v,
                              comps_to_plot = model.comps_to_plot,
                              distance_comps_middle = model.distance_comps_middle,
                              length_neuron = model.length_neuron,
                              V_res = model.V_res)

##### Plot membrane potential of all compartments over time (color plot)
if False:
    plot.voltage_course_colors(plot_name = "Voltage course {} (colored)".format(model.display_name),
                               time_vector = M.t,
                               voltage_matrix = M.v,
                               distance_comps_middle = model.distance_comps_middle)
        