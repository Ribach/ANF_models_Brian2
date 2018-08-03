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

import my_modules.comp_data_rattay_01 as data # morphologic and physiologic data of the model by Rattay et al.
import my_modules.stimulation as stim         # calculates currents for each compartment and timestep
import my_modules.create_plots as plot        # defines some plots

# =============================================================================
# Simulations to be done / Plots to be shown
# =============================================================================
plot_voltage_course_lines = True
plot_voltage_course_colored = False
measure_single_node_response = False
measure_strength_duration_curve = True

# =============================================================================
# Initialize parameters
# =============================================================================
start_scope()

##### define length of timesteps
defaultclock.dt = 5*us

##### load parameters that are part of the equations in data.eqs
V_res = data.V_res
E_Na = data.E_Na
E_K = data.E_K
E_L = data.E_L

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
neuron.m = data.m_init
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

##### conductances soma
neuron.g_Na[data.index_soma] = data.g_Na_soma
neuron.g_K[data.index_soma] = data.g_K_soma
neuron.g_L[data.index_soma] = data.g_L_soma

##### conductances internodes
neuron.g_myelin = data.g_m
neuron.g_Na[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_K[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2
neuron.g_L[np.asarray(np.where(data.structure == 1))] = 0*msiemens/cm**2

##### save initializations of monitors
store('initialized')

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines | plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(dt = defaultclock.dt,
                                                stimulation_type = "extern",
                                                pulse_form = "bi",
                                                nof_pulses = 4,
                                                time_before = 0*ms,
                                                time_after = 1*ms,
                                                add_noise = True,
                                                ##### monophasic stimulation
                                                amp_mono = -1*uA,
                                                duration_mono = 250*us,
                                                ##### biphasic stimulation
                                                amps_bi = np.array([-2,0.2])*uA,          # [-2,2]
                                                durations_bi = np.array([100,0,100])*us,    # [100,0,100]
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = 800*us,
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
    
    ##### run simulation
    run(runtime)
    
    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
        plot.voltage_course_lines(plot_name = "Voltage course Rattay 2001",
                                  time_vector = M.t,
                                  voltage_matrix = M.v,
                                  comps_to_plot = data.comps_to_plot,
                                  distance_comps_middle = data.distance_comps_middle,
                                  length_neuron = data.length_neuron,
                                  V_res = V_res)
    
    if plot_voltage_course_colored:
        plot.voltage_course_colors(plot_name = "Voltage course Rattay 2001 (2)",
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
    
    ##### current amplitudes
    current_amps = np.array([-0.5, -0.7, -1.2, -2, -10])*uA
    
    ##### initialize dataset for measurements
    col_names = ["stimulation amplitude (uA)","AP height (mV)","AP peak time","AP start time","AP end time","rise time (ms)","fall time (ms)","latency (ms)","jitter"]
    node_response_data = pd.DataFrame(np.zeros((len(current_amps)*nof_runs, len(col_names))), columns = col_names)
    
    ##### compartments for measurements
    comp_index = 47
    
    ##### run several simulations and plot results
    plt.figure("Single node response Rattay 2001")

    for ii in range(0, len(current_amps)):
        
        ##### sign stimulus amplitude in plot
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
                                                        noise_term = np.sqrt(data.A_surface*data.g_Na))
        
            ##### get TimedArray of stimulus currents and run simulation
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
            
            print(f"Stimulus amplitde: {current_amps[ii]}")

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
    
# =============================================================================
# Now a simulation will be run several times to calculate the strength-duration
#  curve. This allows to determine the following properties
# - Rheobase
# - chronaxie
# =============================================================================
if measure_strength_duration_curve:
      
    ##### phase durations
    phase_durations = np.round(np.logspace(1, 8, num=20, base=2.0),2)*us
    
    ##### initialize vector for amplitudes of stimulus currents
    stim_amps = np.zeros_like(phase_durations/second)*amp
    
    ##### minimum and maximum stimulus current amplitudes
    amps_min = 0.1*uA
    amps_max = 10*uA
    
    ##### start amplitde for measurements
    start_amp = (amps_max-amps_min)/2
    
    ##### number of runs per phase_duration
    nof_runs = 8
    
    ##### compartment for measurements
    comp_index = 47

    for ii in range(0, len(phase_durations)):
        
        ##### initializations of vectors for tested amplitudes and spike information
        amps = np.zeros(nof_runs)*uA
        spikes = np.zeros(nof_runs, dtype = bool)
              
        amps[0] = start_amp
        
        for jj in range(0,nof_runs):
            
            ##### go back to initial values
            restore('initialized')
            
            ##### define how the ANF is stimulated
            I_stim, runtime = stim.get_stimulus_current(dt = defaultclock.dt,
                                                        stimulation_type = "extern",
                                                        pulse_form = "bi",
                                                        nof_pulses = 1,
                                                        time_before = 0*ms,
                                                        time_after = 1*ms,
                                                        add_noise = False,
                                                        ##### monophasic stimulation
                                                        amp_mono = 0*uA,
                                                        duration_mono = 250*us,
                                                        ##### biphasic stimulation
                                                        amps_bi = np.array([-amps[jj],amps[jj]])*amp,
                                                        durations_bi = np.array([phase_durations[ii],0,phase_durations[ii]])*second,
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
                                                        noise_term = np.sqrt(data.A_surface*data.g_Na))
        
            ##### get TimedArray of stimulus currents and run simulation
            stimulus = TimedArray(np.transpose(I_stim), dt=defaultclock.dt)
            
            ##### run simulation
            run(runtime)
            
            ##### test if there was a spike
            if max(M.v[comp_index,:]-V_res) > 60*mV : spikes[jj] = True
            
            ##### calculate next test amplitude
            if jj < nof_runs-1:
                ##### There was a spike
                if spikes[jj]:
                    amps[jj+1] = (amps[jj] + max(np.array([amps_min] + list(amps[0:jj][amps[0:jj] < amps[jj]]))*amp))/2
                #### There was no spike
                else:
                    amps[jj+1] = (amps[jj] + min(np.array([amps_max] + list(amps[0:jj][amps[0:jj] > amps[jj]]))*amp))/2
                
            print(f"Duration: {phase_durations[ii]/us} us; Stimulus amplitde: {amps[jj]/uA} uA")
            
        ##### write the found minimum stimulus current in vector
        if any(spikes):
            stim_amps[ii] = min(amps[np.where(spikes)])
            start_amp = min(amps[np.where(spikes)])
    
    ##### plot strength duration curve
    plot.strength_duration_curve(plot_name = "Strength duration curve Rattay 2001",
                                 durations = phase_durations,
                                 stimulus_amps = stim_amps)

            
    
    
    
    
    
    
    
    
    