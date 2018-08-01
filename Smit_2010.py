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

import my_modules.comp_data_smit_10 as data   # morphologic and physiologic data of the model by Smit et al.
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
neuron.v = V_res                      # initial cell potential
neuron.m_t_Smit = data.m_t_init_Smit  # initial value for m_t
neuron.m_p_Smit = data.m_p_init_Smit  # initial value for m_p
neuron.n_Smit = data.n_init_Smit      # initial value for n
neuron.h_Smit = data.h_init_Smit      # initial value for h
neuron.m_Rat = data.m_init_Rat        # initial value for m_Rat
neuron.n_Rat = data.n_init_Rat        # initial value for n_Rat
neuron.h_Rat = data.h_init_Rat        # initial value for h_Rat

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
                                                        current_amplitude = -1.5*uA, #-1.5 uA
                                                        time_before_pulse = 0*ms,
                                                        stimulus_duration = 310*us)
##### current vector for biphasic pulse
I_elec_bi_ext = stim.single_biphasic_pulse_stimulus(nof_timesteps = N,
                                                    dt = defaultclock.dt,
                                                    current_amplitude_first_phase = -3*uA,
                                                    current_amplitude_second_phase= 3*uA,
                                                    time_before_pulse = 0*ms,
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
                                                  stimulated_compartment = 2,
                                                  electrode_distance = 300*um,
                                                  rho_out = data.rho_out,
                                                  axoplasmatic_resistances = data.R_a)

# =============================================================================
# Internal stimulation
# =============================================================================
##### current vector for monophasic pulse
I_elec_mono_int = stim.single_monophasic_pulse_stimulus(nof_timesteps = N,
                                                    dt = defaultclock.dt,
                                                    current_amplitude = np.array([0])*nA, #200 nA
                                                    time_before_pulse = 0*ms,
                                                    stimulus_duration = 300*us)

##### current at compartments
I_int = cur.get_currents_for_internal_stimulation(nof_compartments = nof_comps,
                                                   nof_timesteps = N,
                                                   stimulus_current_vector = I_elec_mono_int,
                                                   stimulated_compartments = np.array([0]))

# =============================================================================
# Gaussian noise current term
# =============================================================================
k_noise = 0.0003*uA/np.sqrt(mS)
I_noise = np.transpose(np.transpose(np.random.normal(0, 1, np.shape(I_int)))*k_noise*np.sqrt(data.A_surface*data.g_Na_Rat))

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
# Plot membrane potential of all compartments over time (2 plots)
# =============================================================================
if plot_voltage_course_scheme:

    ##### factor to define voltage-amplitude heights
    v_amp_factor = 1/(50)
    
    ##### distances between lines and x-axis
    offset = np.cumsum(data.distance_comps_middle)/meter
    offset = (offset/max(offset))*10

    smit_stimulation = plt.figure("Voltage course Smit 2010")
    for ii in data.comps_to_plot: 
        plt.plot(M.t/ms, offset[ii] - v_amp_factor*(M.v[ii, :]-V_res)/mV, "#000000")
    plt.yticks(np.linspace(0,10, int(data.length_neuron/mm)+1),range(0,int(data.length_neuron/mm)+1,1))
    plt.xlabel('Time/ms', fontsize=16)
    plt.ylabel('Position/mm [major] V/mV [minor]', fontsize=16)
    plt.gca().invert_yaxis()
    plt.show("Voltage course Smit 2010")
    #smit_stimulation.savefig('smit_stimulation.png')

##### Here is a second plot, showing the same results a bit different
if plot_voltage_course_colored:
    plt.figure("Voltage course Smit 2010 (2)")
    plt.set_cmap('jet')
    plt.pcolormesh(np.array(M.t/ms),np.cumsum(data.distance_comps_middle)/mm,np.array((M.v)/mV))
    clb = plt.colorbar()
    clb.set_label('V/mV')
    plt.xlabel('t/ms')
    plt.ylabel('Position/mm')
    plt.show("Voltage course Smit 2010 (2)")

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
    runtime = 2*ms
    
    ##### number of timesteps
    N = int(runtime/defaultclock.dt)
    
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
            I_noise = np.transpose(np.transpose(np.random.normal(0, 1, np.shape(I_ext)))*k_noise*np.sqrt(data.A_surface*data.g_Na_Rat))
            
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
    