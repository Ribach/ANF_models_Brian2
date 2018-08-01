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

import my_modules.comp_data_frijns_94 as data # morphologic and physiologic data of the model by Frijns et al.
import my_modules.stimulation as stim         # calculates currents at current source for different types of stimuli
import my_modules.get_currents as cur         # calculates currents at each compartment over time

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
neuron.v = V_res       # initial cell potential
neuron.h = data.h_init    # initial value for h
neuron.m = data.m_init    # initial value for m
neuron.n = data.n_init    # initial value for n

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

# =============================================================================
# Set parameter values (parameters that were initialised in the equations data.eqs
# and which are different for different compartment types)
# =============================================================================
##### permeabilities nodes
neuron.P_Na = data.P_Na
neuron.P_K = data.P_K

##### permeabilities internodes
neuron.P_Na[np.asarray(np.where(data.structure == 1))] = 0*meter/second
neuron.P_K[np.asarray(np.where(data.structure == 1))] = 0*meter/second

# =============================================================================
# Initializations for simulation
# =============================================================================
##### duration of timesteps
defaultclock.dt = 10*us

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
                                                    current_amplitude = -50*nA, #-50 nA
                                                    time_before_pulse = 0*ms,
                                                    stimulus_duration = 250*us)
##### current vector for biphasic pulse
I_elec_bi_ext = stim.single_biphasic_pulse_stimulus(nof_timesteps = N,
                                                dt = defaultclock.dt,
                                                current_amplitude_first_phase = -100*nA,
                                                current_amplitude_second_phase= 100*nA,
                                                time_before_pulse = 0*us,
                                                duration_first_phase = 250*us,
                                                duration_second_phase = 250*us,
                                                duration_interphase_gap = 0*us)

##### current vector for pulse train
I_elec_pulse_train_ext = stim.pulse_train_stimulus(nof_timesteps = N,
                                                   dt = defaultclock.dt,
                                                   current_vector = I_elec_mono_ext, # leading and trailing zeros will be cut
                                                   time_before_pulse_train = 0*us,
                                                   nof_pulses = 8,
                                                   inter_pulse_gap = 300*us)

##### current at compartments
I_ext = cur.get_currents_for_external_stimulation(compartment_lengths = data.compartment_lengths,
                                                   nof_timesteps = N,
                                                   stimulus_current_vector = I_elec_pulse_train_ext,
                                                   stimulated_compartment = 4,
                                                   electrode_distance = 1000*um,
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
# Get stimulus current for each compartment and timestep and run simulation
# =============================================================================
##### get TimedArray of stimulus currents (due to both intern and extern stimulation)
stimulus = TimedArray(np.transpose(I_ext + I_int), dt=defaultclock.dt)

##### save initializations of monitors
store('initialized')

##### run simulation
run(runtime)

# =============================================================================
# Plot membrane potential of all compartments over time (2 plots)
# =============================================================================
##### factor to define voltage-amplitude heights
v_amp_factor = 1/(50)

##### distances between lines and x-axis
offset = (np.sum(data.distance_comps_middle) - np.cumsum(data.distance_comps_middle))/meter
offset = (offset/max(offset))*10

frijns_stimulation = plt.figure("Voltage course Frijns 1994")
for ii in range(0, nof_comps): #data.comps_to_plot: 
    plt.plot(M.t/ms, offset[ii] + v_amp_factor*(M.v[ii, :]-V_res)/mV, data.plot_colors[ii])
plt.yticks(np.linspace(0,10, int(data.length_neuron/mm)+1),range(0,int(data.length_neuron/mm)+1,1))
plt.xlabel('Time/ms', fontsize=16)
plt.ylabel('Position/mm [major] V/mV [minor]', fontsize=16)
plt.show("Voltage course Frijns 1994")
#frijns_stimulation.savefig('frijns_stimulation.png')

##### Here is a second plot, showing the same results a bit different
plt.figure("Voltage course Frijns 1994 (2)")
plt.set_cmap('jet')
plt.pcolormesh(np.array(M.t/ms),np.cumsum(data.distance_comps_middle)/mm,np.array(M.v/mV))
clb = plt.colorbar()
clb.set_label('V/mV')
plt.xlabel('t/ms')
plt.ylabel('Position/mm')
plt.show("Voltage course Frijns 1994 (2)")

# =============================================================================
# Now a second simulation with internal stimulation is done to calculate the
# following temporal characteristics of the model:
# - average AP amplitude
# - average AP rise time
# - average AP fall time
# - latency of the spike in the first compartment
# - conductance velocity (between first and last compartment)
# =============================================================================
##### go back to initial values
restore('initialized')

##### stimulus duration
runtime = 1.2*ms

##### number of timesteps
N = int(runtime/defaultclock.dt)

##### current vector for monophasic pulse
I_elec_mono_ext = stim.single_monophasic_pulse_stimulus(nof_timesteps = N,
                                                    dt = defaultclock.dt,
                                                    current_amplitude = 100*nA, #100 nA
                                                    time_before_pulse = 0*ms,
                                                    stimulus_duration = 250*us)

##### current at compartments
I_ext = cur.get_currents_for_external_stimulation(compartment_lengths = data.compartment_lengths,
                                                   nof_timesteps = N,
                                                   stimulus_current_vector = I_elec_mono_ext,
                                                   stimulated_compartment = 0,
                                                   electrode_distance = 10*um,
                                                   rho_out = data.rho_out,
                                                   axoplasmatic_resistances = data.R_a)

##### Get TimedArray of stimulus currents and run simulation
stimulus = TimedArray(np.transpose(I_ext), dt=defaultclock.dt)

##### run simulation
run(runtime)

##### first compartment which is part of measurements
first_comp = 0

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