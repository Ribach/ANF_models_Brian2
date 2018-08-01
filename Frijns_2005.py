##### Don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### Import needed packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import my_modules.comp_data_frijns_05 as data # morphologic and physiologic data of the model by Rattay et al.
import my_modules.stimulation as stim         # calculates currents at current source for different types of stimuli
import my_modules.get_currents as cur         # calculates currents at each compartment over time

# =============================================================================
# Simulations to be done
# =============================================================================
plot_voltage_course_scheme = True
plot_voltage_course_colored = False
measure_single_node_response = True

# =============================================================================
# Initialize parameters
# =============================================================================
start_scope()

##### Load parameters that are part of the equations in data.eqs
V_res = data.V_res
T_celsius = data.T_celsius
T_kelvin = data.T_kelvin
Na_i = data.Na_i
Na_e = data.Na_e
K_i = data.K_i
K_e = data.K_e

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
neuron.v = V_res          # initial cell potential
neuron.m = data.m_init    # initial value for m
neuron.n = data.n_init    # initial value for n
neuron.h = data.h_init    # initial value for h

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

# =============================================================================
# Set parameter values (parameters that were initialised in the equations data.eqs
# and which are different for different compartment types)
# =============================================================================
##### permeabilities presomatic region and active compartments
neuron.P_Na[np.asarray(np.where(np.logical_or(data.structure == 0, data.structure == 2)))] = data.P_Na
neuron.P_K[np.asarray(np.where(np.logical_or(data.structure == 0, data.structure == 2)))] = data.P_K

##### permeabilities internodes
neuron.P_Na[np.asarray(np.where(data.structure == 1))] = 0*meter/second
neuron.P_K[np.asarray(np.where(data.structure == 1))] = 0*meter/second

##### permeabilities somatic region
neuron.P_Na[np.asarray(np.where(np.logical_or(data.structure == 3, data.structure == 4)))] = data.P_Na/data.dividing_factor
neuron.P_K[np.asarray(np.where(np.logical_or(data.structure == 3, data.structure == 4)))] = data.P_K/data.dividing_factor

##### conductances
neuron.g_myelin = data.g_m

# =============================================================================
# Initializations for simulation
# =============================================================================
##### duration of timesteps
defaultclock.dt = 5*us

##### duration of simulation
runtime = 5*ms

##### number of timesteps
N = int(runtime/defaultclock.dt)

# =============================================================================
# External stimulation
# =============================================================================
##### current vector for monophasic pulse
I_elec_mono_ext = stim.single_monophasic_pulse_stimulus(nof_timesteps = N,
                                                    dt = defaultclock.dt,
                                                    current_amplitude = -0.8*uA, #200 nA
                                                    time_before_pulse = 0*ms,
                                                    stimulus_duration = 250*us)
##### current vector for biphasic pulse
I_elec_bi_ext = stim.single_biphasic_pulse_stimulus(nof_timesteps = N,
                                                dt = defaultclock.dt,
                                                current_amplitude_first_phase = -2*uA,
                                                current_amplitude_second_phase= 2*uA,
                                                time_before_pulse = 0*us,
                                                duration_first_phase = 100*us,
                                                duration_second_phase = 100*us,
                                                duration_interphase_gap = 0*us)

##### current vector for pulse train
I_elec_pulse_train_ext = stim.pulse_train_stimulus(nof_timesteps = N,
                                                   dt = defaultclock.dt,
                                                   current_vector = I_elec_bi_ext, # leading and trailing zeros will be cut
                                                   time_before_pulse_train = 0*us,
                                                   nof_pulses = 4,
                                                   inter_pulse_gap = 800*us)

##### current at compartments
I_ext = cur.get_currents_for_external_stimulation(compartment_lengths = data.compartment_lengths,
                                                   nof_timesteps = N,
                                                   stimulus_current_vector = I_elec_pulse_train_ext,
                                                   stimulated_compartment = 4,
                                                   electrode_distance = 300*um,
                                                   rho_out = data.rho_out,
                                                   axoplasmatic_resistances = data.R_a)

# =============================================================================
# Internal stimulation
# =============================================================================
##### current vector for monophasic pulse
I_elec_mono_int = stim.single_monophasic_pulse_stimulus(nof_timesteps = N,
                                                    dt = defaultclock.dt,
                                                    current_amplitude = np.array([0.0])*nA, #0.5 nA
                                                    time_before_pulse = 0*ms,
                                                    stimulus_duration = 250*us)

##### current at compartments
I_int = cur.get_currents_for_internal_stimulation(nof_compartments = nof_comps,
                                                   nof_timesteps = N,
                                                   stimulus_current_vector = I_elec_mono_int,
                                                   stimulated_compartments = np.array([0]))

# =============================================================================
# Gaussian noise current term
# =============================================================================
k_noise = 0.0000001*uA*np.sqrt(second/um**3)
I_noise = np.transpose(np.transpose(np.random.normal(0, 1, np.shape(I_int)))*k_noise*np.sqrt(data.A_surface*data.P_Na))

# =============================================================================
# Get stimulus current for each compartment and timestep and run simulation
# =============================================================================
##### get TimedArray of stimulus currents (due to both intern and extern stimulation)
stimulus = TimedArray(np.transpose(I_ext + I_int + I_noise), dt=defaultclock.dt)

##### save initializations of monitors
store('initialized')

##### run simulation
run(runtime)

# =============================================================================
# Plot membrane potential course of all compartments over time (2 plots)
# =============================================================================
if plot_voltage_course_scheme:

    ##### factor to define voltage-amplitude heights
    v_amp_factor = 1/(50)
    
    ##### distances between lines and x-axis
    offset = np.cumsum(data.distance_comps_middle)/meter
    offset = (offset/max(offset))*10

    frijns_stimulation = plt.figure("Voltage course Frijns 2005")
    for ii in data.comps_to_plot:
        plt.plot(M.t/ms, offset[ii] - v_amp_factor*(M.v[ii, :]-V_res)/mV, "#000000")
    plt.yticks(np.linspace(0,10, int(data.length_neuron/mm)+1),range(0,int(data.length_neuron/mm)+1,1))
    plt.xlabel('Time/ms', fontsize=16)
    plt.ylabel('Position/mm [major] V/mV [minor]', fontsize=16)
    plt.gca().invert_yaxis()
    plt.show("Voltage course Frijns 2005")
    #frijns_stimulation.savefig('frijns_stimulation.png')

##### Here is a second plot, showing the same results a bit different
if plot_voltage_course_colored:
    plt.figure("Voltage course Frijns 2005 (2)")
    plt.set_cmap('jet')
    plt.pcolormesh(np.array(M.t/ms),np.cumsum(data.distance_comps_middle)/mm,np.array((M.v)/mV))
    clb = plt.colorbar()
    clb.set_label('V/mV')
    plt.xlabel('t/ms')
    plt.ylabel('Position/mm')
    plt.show("Voltage course Frijns 2005 (2)")

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
    
    ##### stimulus duration
    runtime = 1.5*ms
    
    ##### number of timesteps
    N = int(runtime/defaultclock.dt)
    
    ##### number of simulations to run
    nof_runs = 10
    
    ##### current amplitude
    current_amps = np.array([-0.6, -0.8, -1.2, -2, -10])*uA
    
    ##### initialize dataset for measurements
    col_names = ["stimulation amplitude (uA)","AP height (mV)","AP peak time","AP start time","AP end time","rise time (ms)","fall time (ms)","latency (ms)","jitter"]
    node_response_data = pd.DataFrame(np.zeros((len(current_amps)*nof_runs, len(col_names))), columns = col_names)
    
    ##### compartments for measurements
    comp_index = 47
    
    ##### run several simulations and plot results
    plt.figure("Single node response Frijns 2005")

    for ii in range(0, len(current_amps)):
        
        ##### sign stimulus amplitude
        unit = r'$\mu V$'
        plt.text(0.04, 100*ii+V_res/mV+10, f"{round(current_amps[ii]*10**6/amp,1)} {unit}")
        
        for jj in range(0,nof_runs):
            
            ##### go back to initial values
            restore('initialized')
        
            ##### current vector for monophasic pulse
            I_elec_mono_ext = stim.single_monophasic_pulse_stimulus(nof_timesteps = N,
                                                                    dt = defaultclock.dt,
                                                                    current_amplitude = current_amps[ii],
                                                                    time_before_pulse = 0*ms,
                                                                    stimulus_duration = 250*us)
            
            ##### current at compartments
            I_ext = cur.get_currents_for_external_stimulation(compartment_lengths = data.compartment_lengths,
                                                               nof_timesteps = N,
                                                               stimulus_current_vector = I_elec_mono_ext,
                                                               stimulated_compartment = 4,
                                                               electrode_distance = 300*um,
                                                               rho_out = data.rho_out,
                                                               axoplasmatic_resistances = data.R_a)
        
            ##### add Gaussian noise current term
            I_noise = np.transpose(np.transpose(np.random.normal(0, 1, np.shape(I_ext)))*k_noise*np.sqrt(data.A_surface*data.P_Na))  
            
            ##### Get TimedArray of stimulus currents and run simulation
            stimulus = TimedArray(np.transpose(I_ext + I_noise), dt=defaultclock.dt)
            
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
    