##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import needed packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import my_modules.comp_data_smit_08 as data   # morphologic and physiologic data of the model by Smit et al.
import my_modules.stimulation as stim         # calculates currents for each compartment and timestep
import my_modules.create_plots as plot        # defines some plots

# =============================================================================
# Simulations to be done / Plots to be shown
# =============================================================================
plot_voltage_course_lines = True
plot_voltage_course_colored = True
measure_single_node_response = False

# =============================================================================
# Initialize parameters
# =============================================================================
start_scope()

##### define length of timesteps
defaultclock.dt = 5*us

##### Load parameters that are part of the equations in data.eqs
V_res = data.V_res
E_Na = data.E_Na
E_K = data.E_K
E_L = data.E_L
T_celsius = data.T_celsius

# =============================================================================
# Set up Neuron
# =============================================================================
##### define morphology
nof_comps = len(data.compartment_lengths)

morpho = Section(n = nof_comps,
                 length = data.compartment_lengths,
                 diameter = data.compartment_diameters)

##### define neuron
neuron = SpatialNeuron(morphology = morpho,
                       model = data.eqs,
                       Cm = data.c_m,
                       Ri = data.rho_in,
                       method="exponential_euler")

##### initial values
neuron.v = V_res
neuron.m_t = data.m_t_init
neuron.m_p = data.m_p_init
neuron.n = data.n_init
neuron.h = data.h_init

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

# =============================================================================
# Set parameter values (parameters that were initialised in the equations data.eqs
# and which are different for different compartment types)
# =============================================================================
##### conductances active compartments
neuron.g_Na = data.g_Na
neuron.g_K = data.g_K
neuron.g_L = data.g_L

##### conductances internodes
neuron.g_myelin = data.g_m
neuron.g_Na[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_K[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_L[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines | plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(dt = defaultclock.dt,
                                                stimulation_type = "extern",
                                                pulse_form = "mono",
                                                nof_pulses = 6,
                                                time_before = 0*ms,
                                                time_after = 1*ms,
                                                add_noise = True,
                                                ##### monophasic stimulation
                                                amp_mono = -3*uA,
                                                duration_mono = 200*us,
                                                ##### biphasic stimulation
                                                amps_bi = np.array([-0.75,0.75])*uA,
                                                durations_bi = np.array([100,0,100])*us,
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = 1600*us,
                                                ##### external stimulation
                                                compartment_lengths = data.compartment_lengths,
                                                stimulated_compartment = 4,
                                                electrode_distance = 300*um,
                                                rho_out = data.rho_out,
                                                axoplasmatic_resistances =  data.R_a,
                                                ##### noise
                                                k_noise = 0.0003*uA/np.sqrt(mS),
                                                noise_term = np.sqrt(data.A_surface*data.g_Na))
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt=defaultclock.dt)
    
    ##### save initializations of monitors
    store('initialized')
    
    ##### run simulation
    run(runtime)
    
    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
        plot.voltage_course_lines(plot_name = "Voltage course Smit 2008",
                                  time_vector = M.t,
                                  voltage_matrix = M.v,
                                  comps_to_plot = data.comps_to_plot,
                                  distance_comps_middle = data.distance_comps_middle,
                                  length_neuron = data.length_neuron,
                                  V_res = V_res)
    
    if plot_voltage_course_colored:
        plot.voltage_course_colors(plot_name = "Voltage course Smit 2008 (2)",
                                   time_vector = M.t,
                                   voltage_matrix = M.v,
                                   distance_comps_middle = data.distance_comps_middle)

# =============================================================================
# Now a second simulation with internal stimulation is done to calculate the
# following temporal characteristics of the model:
# - average AP amplitude
# - average AP rise time
# - average AP fall time
# - latency of the spike in the first compartment
# - conductance velocity (between first and last compartment)
# =============================================================================
if measure_single_node_response:
    ##### go back to initial values
    restore('initialized')
    
    ##### stimulus duration
    runtime = 3*ms
    
    ##### number of timesteps
    N = int(runtime/defaultclock.dt)
    
    ##### current vector for monophasic pulse
    I_elec_mono_ext = stim.single_monophasic_pulse_stimulus(nof_timesteps = N,
                                                        dt = defaultclock.dt,
                                                        current_amplitude = 0.8*uA,
                                                        time_before_pulse = 0*ms,
                                                        stimulus_duration = 310*us)
    
    ##### current at compartments
    I_ext = cur.get_currents_for_external_stimulation(compartment_lengths = data.compartment_lengths,
                                                       nof_timesteps = N,
                                                       stimulus_current_vector = I_elec_mono_ext,
                                                       stimulated_compartment = 5,
                                                       electrode_distance = 10*um,
                                                       rho_out = data.rho_out,
                                                       axoplasmatic_resistances = data.R_a)
    
    ##### Get TimedArray of stimulus currents and run simulation
    stimulus = TimedArray(np.transpose(I_ext), dt=defaultclock.dt)
    
    ##### run simulation
    run(runtime)
    
    ##### first compartment which is part of measurements
    first_comp = data.end_index_soma+5
    
    ##### AP amplitudes in all compartments
    AP_amps = [max(M.v[i,:]-V_res)
               for i in range(first_comp,nof_comps-1)]
    
    ##### AP peak time in all compartments
    AP_times = [float(M.t[M.v[i,:]-V_res == AP_amps[i-first_comp]])*second
                for i in range(first_comp,nof_comps-1)]
    
    ##### AP start point in all compartments (defined at 10% of peak value)
    AP_start_time = [M.t[np.argmin(abs(M.v[i,np.where(M.t<AP_times[i-first_comp])[0]]-V_res - 0.1*AP_amps[i-first_comp]))]
                     for i in range(first_comp,nof_comps-1)]
    
    ##### AP end point in all compartments (defined at 10% of peak value)
    AP_end_time = [M.t[np.where(M.t>AP_times[i-first_comp])[0]][np.argmin(abs(M.v[i,where(M.t>AP_times[i-first_comp])[0]]-V_res - 0.1*AP_amps[i-first_comp]))]
                     for i in range(first_comp,nof_comps-1)]
    
    ##### combine data in dataframe
    AP_data = pd.DataFrame([AP_amps, AP_times, AP_start_time, AP_end_time],
                           index=("amplitude", "peak_time", "start_time", "end_time")).T
    
    ##### Calculate rise and fall times
    AP_data["rise_time"] = AP_data.peak_time - AP_data.start_time
    AP_data["fall_time"] = AP_data.end_time - AP_data.peak_time
    
    ##### Calculate average values
    AP_average_data = AP_data[["amplitude", "rise_time", "fall_time"]].mean().multiply([volt,second,second])
    
    ##### Calculate latency (for spike in first compartment)
    latency = AP_data["peak_time"][0] - 0*ms
    
    ##### Calculate conductance velocity between first and last compartment
    conductance_velocity = sum(data.compartment_lengths[first_comp:nof_comps]) / (max(AP_times)-min(AP_times))
    
    print("The average AP amplitude (difference to resting potential at peak) is", AP_average_data.amplitude)
    print("The average AP rise time (period between 10% of peak value and peak value) is", AP_average_data.rise_time)
    print("The average AP fall time (period between peak value and 10% of peak value) is", AP_average_data.fall_time)
    print("The latency of the spike in the first compartment was", latency)
    print("The conductance velocity between the first and the last compartment was", conductance_velocity)
