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

import my_modules.comp_data_smit_10 as data   # morphologic and physiologic data of the model by Smit et al.
import my_modules.stimulation as stim         # calculates currents for each compartment and timestep
import my_modules.create_plots as plot        # defines some plots

# =============================================================================
# Simulations to be done / Plots to be shown
# =============================================================================
plot_voltage_course_lines = True
plot_voltage_course_colored = True
measure_single_node_response = True

# =============================================================================
# Initialize parameters
# =============================================================================
start_scope()

##### define length of timesteps
defaultclock.dt = 5*us

##### Load parameters that are part of the equations in data.eqs
V_res = data.V_res
E_Na_Smit = data.E_Na_Smit
E_K_Smit = data.E_K_Smit
E_L_Smit = data.E_L_Smit
E_Na_Rat = data.E_Na_Rat
E_K_Rat = data.E_K_Rat
E_L_Rat = data.E_L_Rat
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
neuron.m_t_Smit = data.m_t_init_Smit
neuron.m_p_Smit = data.m_p_init_Smit
neuron.n_Smit = data.n_init_Smit
neuron.h_Smit = data.h_init_Smit
neuron.m_Rat = data.m_init_Rat
neuron.n_Rat = data.n_init_Rat
neuron.h_Rat = data.h_init_Rat

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

# =============================================================================
# Set parameter values (parameters that were initialised in the equations data.eqs
# and which are different for different compartment types)
# =============================================================================
##### conductances dentritic nodes and peripheral terminal 
neuron.g_Na_Rat[0:data.start_index_soma] = data.g_Na_Rat
neuron.g_K_Rat[0:data.start_index_soma] = data.g_K_Rat
neuron.g_L_Rat[0:data.start_index_soma] = data.g_L_Rat

neuron.g_Na_Smit[0:data.start_index_soma] = 0*msiemens/cm**2
neuron.g_K_Smit[0:data.start_index_soma] = 0*msiemens/cm**2
neuron.g_L_Smit[0:data.start_index_soma] = 0*msiemens/cm**2

##### conductances axonal nodes
neuron.g_Na_Smit[data.end_index_soma+1:] = data.g_Na_Smit
neuron.g_K_Smit[data.end_index_soma+1:] = data.g_K_Smit
neuron.g_K_Smit[data.end_index_soma+1:] = data.g_L_Smit

neuron.g_Na_Rat[data.end_index_soma+1:] = 0*msiemens/cm**2
neuron.g_K_Rat[data.end_index_soma+1:] = 0*msiemens/cm**2
neuron.g_K_Rat[data.end_index_soma+1:] = 0*msiemens/cm**2

##### conductances soma
neuron.g_Na_Rat[data.index_soma] = data.g_Na_soma
neuron.g_K_Rat[data.index_soma] = data.g_K_soma
neuron.g_L_Rat[data.index_soma] = data.g_L_soma

neuron.g_Na_Smit[data.index_soma] = 0*msiemens/cm**2
neuron.g_K_Smit[data.index_soma] = 0*msiemens/cm**2
neuron.g_L_Smit[data.index_soma] = 0*msiemens/cm**2

##### conductances internodes
neuron.g_myelin = data.g_m

neuron.g_Na_Rat[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_K_Rat[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_L_Rat[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2

neuron.g_Na_Smit[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_K_Smit[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_L_Smit[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines | plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(dt = defaultclock.dt,
                                                stimulation_type = "extern",
                                                pulse_form = "bi",
                                                nof_pulses = 6,
                                                time_before = 0*ms,
                                                time_after = 1*ms,
                                                add_noise = True,
                                                ##### monophasic stimulation
                                                amp_mono = -0.6*uA,
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
                                                noise_term = np.sqrt(data.A_surface*data.g_Na_Rat))
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt=defaultclock.dt)
    
    ##### save initializations of monitors
    store('initialized')
    
    ##### run simulation
    run(runtime)
    
    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
        plot.voltage_course_lines(plot_name = "Voltage course Smit 2010",
                                  time_vector = M.t,
                                  voltage_matrix = M.v,
                                  comps_to_plot = data.comps_to_plot,
                                  distance_comps_middle = data.distance_comps_middle,
                                  length_neuron = data.length_neuron,
                                  V_res = V_res)
    
    if plot_voltage_course_colored:
        plot.voltage_course_colors(plot_name = "Voltage course Smit 2010 (2)",
                                   time_vector = M.t,
                                   voltage_matrix = M.v,
                                   distance_comps_middle = data.distance_comps_middle)

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

    ##### number of simulations to run
    nof_runs = 10
    
    ##### current amplitude
    current_amps = np.array([-0.3, -0.4, -0.6, -1.2, -5, -30])*uA
    
    ##### initialize dataset for measurements
    col_names = ["stimulation amplitude (uA)","AP height (mV)","AP peak time","AP start time","AP end time","rise time (ms)","fall time (ms)","latency (ms)","jitter"]
    node_response_data = pd.DataFrame(np.zeros((len(current_amps)*nof_runs, len(col_names))), columns = col_names)
    
    ##### compartments for measurements
    comp_index = 47
    
    ##### run several simulations and plot results
    plt.figure("Single node response Smit 2010")

    for ii in range(0, len(current_amps)):
        
        ##### sign stimulus amplitude
        unit = r'$\mu V$'
        plt.text(0.04, 100*ii+V_res/mV+10, f"{round(current_amps[ii]*10**6/amp,1)} {unit}")
        
        for jj in range(0,nof_runs):
            
            ##### go back to initial values
            restore('initialized')
        
            ##### define how the ANF is stimulated
            I_stim, runtime = stim.get_stimulus_current(dt = defaultclock.dt,
                                                        stimulation_type = "extern",
                                                        pulse_form = "mono",
                                                        nof_pulses = 1,
                                                        time_before = 0*ms,
                                                        time_after = 1.5*ms,
                                                        add_noise = True,
                                                        ##### monophasic stimulation
                                                        amp_mono = current_amps[ii],
                                                        duration_mono = 250*us,
                                                        ##### biphasic stimulation
                                                        amps_bi = np.array([-2,2])*uA,
                                                        durations_bi = np.array([100,0,100])*us,
                                                        ##### multiple pulses / pulse trains
                                                        inter_pulse_gap =800*us,
                                                        ##### external stimulation
                                                        compartment_lengths = data.compartment_lengths,
                                                        stimulated_compartment = 4,
                                                        electrode_distance = 300*um,
                                                        rho_out = data.rho_out,
                                                        axoplasmatic_resistances =  data.R_a,
                                                        ##### noise
                                                        k_noise = 0.0003*uA/np.sqrt(mS),
                                                        noise_term = np.sqrt(data.A_surface*data.g_Na_Rat))
        
            ##### Get TimedArray of stimulus currents and run simulation
            stimulus = TimedArray(np.transpose(I_stim), dt=defaultclock.dt)
            
            ##### run simulation
            run(runtime)
            
            ##### write results in table
            AP_amp = max(M.v[comp_index,:]-V_res)
            AP_time = M.t[M.v[comp_index,:]-V_res == AP_amp]
            AP_start_time = M.t[np.argmin(abs(M.v[comp_index,np.where(M.t<AP_time)[0]]-V_res - 0.1*AP_amp))]
            AP_end_time = M.t[np.where(M.t>AP_time)[0]][np.argmin(abs(M.v[comp_index,where(M.t>AP_time)[0]]-V_res - 0.1*AP_amp))]
            
            node_response_data["stimulation amplitude (uA)"][ii*nof_runs+jj] = current_amps[ii]/uA
            node_response_data["AP height (mV)"][ii*nof_runs+jj] = AP_amp/mV
            node_response_data["AP peak time"][ii*nof_runs+jj] = AP_time/ms
            node_response_data["AP start time"][ii*nof_runs+jj] = AP_start_time/ms
            node_response_data["AP end time"][ii*nof_runs+jj] = AP_end_time/ms

            ##### plot curve
            plt.plot(M.t/ms, 100*ii + M.v[comp_index, :]/mV, "#000000")
    
    ##### finish plot                 
    plt.xlabel('Time/ms', fontsize=16)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.show("Single node response Rattay 2001")
    
    ##### calculate remaining single node response data
    node_response_data["rise time (ms)"] = node_response_data["AP peak time"] - node_response_data["AP start time"]
    node_response_data["fall time (ms)"] = node_response_data["AP end time"] - node_response_data["AP peak time"]
    node_response_data["latency (ms)"] = node_response_data["AP peak time"]
    node_response_data["jitter"] = 0
    
    ##### exclude runs where no AP was elicited
    node_response_data = node_response_data[node_response_data["AP height (mV)"] > 60]
    
    ##### calculate average data and jitter for different stimulus amplitudes
    average_node_response_data = node_response_data.groupby(["stimulation amplitude (uA)"])["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"].mean()
    average_node_response_data["jitter (ms)"] = node_response_data.groupby(["stimulation amplitude (uA)"])["latency (ms)"].std()
    
    ##### plot results in bar plot
    average_node_response_data.iloc[:,1:].transpose().plot.bar(rot = 0)
    #average_node_response_data.plot.bar(rot = 0,secondary_y = ("AP height (mV)"))
    