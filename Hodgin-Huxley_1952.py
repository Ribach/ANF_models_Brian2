##### Don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### Import needed packages
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import my_modules.elec_comp_data as elec
import my_modules.stimulation as stim

start_scope()

# =============================================================================
# Initialize parameters
# =============================================================================
defaultclock.dt = 10*us

##### conductances
g_Na = 120*msiemens/cm**2
g_K = 36*msiemens/cm**2
g_L = 0.3*msiemens/cm**2

#### Nernst potentials
V_res = 0*mV
E_Na = 115*mV
E_K = -12*mV
E_L = 10.6*mV

##### resistivities
rho_in = 50*ohm*cm
rho_out = 300*ohm*cm

##### capacitance cell membrane
c_m = 1*uF/cm**2

##### Morphological parameters
len_neuron = 10*cm
diameter_axon = 2*238*um

# =============================================================================
# Set up compartment morphology
# =============================================================================
nof_comps = 100
len_comps = np.repeat(len_neuron/nof_comps,nof_comps)
diameter_comps = np.repeat(diameter_axon,nof_comps)

morpho = Cylinder(n = nof_comps,
                  length = len_neuron,
                  diameter = diameter_axon,
                  type = 'axon')

# =============================================================================
# Calculate resistances and capacitances for each compartment
# =============================================================================
##### axoplasmatic resistance
R_a = elec.get_axoplasmatic_resistances(lengths = len_comps,
                                        diameters = diameter_comps,
                                        rho_in = rho_in)

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
Im = g_Na*m**3*h* (E_Na-(v-V_res)) + g_K*n**4*(E_K-(v-V_res)) + g_L*(E_L-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m = (0.1/mV) * (-(v-V_res)+25*mV) / (exp((-(v-V_res)+25*mV) / (10*mV)) - 1)/ms : Hz
beta_m = 4 * exp(-(v-V_res)/(18*mV))/ms : Hz
alpha_h = 0.07 * exp(-(v-V_res)/(20*mV))/ms : Hz
beta_h = 1/(exp((-(v-V_res)+30*mV) / (10*mV)) + 1)/ms : Hz
alpha_n = (0.01/mV) * (-(v-V_res)+10*mV) / (exp((-(v-V_res)+10*mV) / (10*mV)) - 1)/ms : Hz
beta_n = 0.125*exp(-(v-V_res)/(80*mV))/ms : Hz
'''

# =============================================================================
# Defining the neuron
# =============================================================================
neuron = SpatialNeuron(morphology = morpho,
                       model = eqs,
                       threshold = "v > V_res + 70*mV",
                       Cm = np.repeat(c_m,nof_comps),
                       Ri = rho_in,
                       method="exponential_euler")

##### Initial values
neuron.v = V_res   # initial cell potential
neuron.h = 0.6     # initial value for h
neuron.m = 0.05    # initial value for m
neuron.n = 0.3     # initial value for n

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

##### record spikes
spikes = SpikeMonitor(neuron)

# =============================================================================
# Get stimulus current for each compartment and timestep and run simulation
# =============================================================================
##### stimulus duration
runtime = 12*ms

##### number of timesteps
N = int(runtime/defaultclock.dt)

##### current at electrode
I_elec_mono = stim.single_monophasic_pulse_stimulus(nof_timesteps = N,
                                                    dt = defaultclock.dt,
                                                    current_amplitude = -5*mA,
                                                    time_before_puls = 1*ms,
                                                    stimulus_duration = 2*ms)

I_elec_bi = stim.single_biphasic_pulse_stimulus(nof_timesteps = N,
                                                dt = defaultclock.dt,
                                                current_amplitude_first_phase = -5*mA,
                                                current_amplitude_second_phase= 5*mA,
                                                time_before_puls = 1*ms,
                                                duration_first_phase = 1*ms,
                                                duration_second_phase = 1*ms,
                                                duration_interphase_gap = 0*ms)

##### current at compartments due to external stimulation wit electrode
I_ext = elec.get_currents_for_external_stimulation(compartment_lengths = len_comps,
                                                   nof_timesteps = N,
                                                   stimulus_current_vector = I_elec_mono,
                                                   stimulated_compartment = 50,
                                                   electrode_distance = 10*mm,
                                                   rho_out = rho_out,
                                                   axoplasmatic_resistances = R_a)

##### current at compartments due to internal stimulation direct at compartment
I_int = elec.get_currents_for_internal_stimulation(nof_compartments = nof_comps,
                                                   nof_timesteps = N,
                                                   dt = defaultclock.dt,
                                                   stimulated_compartments = np.array([20,80]),
                                                   current_amplitude = np.array([0,0])*uA,
                                                   time_before_puls = 1*ms,
                                                   stimulus_duration = 2*ms)

##### Get TimedArray of stimulus currents (due to both intern and extern stimulation)
stimulus = TimedArray(np.transpose(I_ext + I_int), dt=defaultclock.dt)

##### save initializations of monitors
store('initialized')

##### run simulation
run(runtime)

# =============================================================================
# Plot membrane potential of all compartments over time (2 plots)
# =============================================================================
##### every step-th compartment is plotted (to get about 50 lines independend of number of compartments)
step = int(np.ceil(nof_comps/50))

##### factor to define voltage-amplitude heights
v_amp_factor = 1/(60)

##### plot x-axis: time; y-axis: compartments
plt.figure(1)
for ii in range(0, nof_comps, step):
    plt.plot(M.t/ms, ii/5 + v_amp_factor*(M.v[ii, :]-V_res)/mV, 'k')
plt.yticks(np.linspace(0,int(nof_comps/5), int(len_neuron/cm)+1),range(0,int(len_neuron/cm)+1,1))
plt.xlabel('Time/ms')
plt.ylabel('Position/mm [major] V/mV [minor]')
plt.show()

##### Here is a second plot, showing the same results a bit different
plt.figure(2)
plt.set_cmap('jet')
plt.pcolormesh(np.array(M.t/ms),np.cumsum(neuron.length)/cm,np.array(M.v/mV))
clb = plt.colorbar()
clb.set_label('V/mV')
plt.xlabel('t/ms')
plt.ylabel('Position/cm')
plt.show(2)

# =============================================================================
# Run simulation to get conductance velocity
# =============================================================================
##### go back to initial values
restore('initialized')

##### stimulus duration
runtime = 14*ms

##### number of timesteps
N = int(runtime/defaultclock.dt)

##### current at compartments due to internal stimulation direct at compartment
I_int = elec.get_currents_for_internal_stimulation(nof_compartments = nof_comps,
                                                   nof_timesteps = N,
                                                   dt = defaultclock.dt,
                                                   stimulated_compartments = np.array([0]),
                                                   current_amplitude = np.array([2])*uA,
                                                   time_before_puls = 1*ms,
                                                   stimulus_duration = 2*ms)

stimulus = TimedArray(np.transpose(I_int), dt=defaultclock.dt)
run(runtime)

##### get conductance velocity
slope, intercept, r_value, p_value, std_err  = stats.linregress(spikes.t/second, neuron.distance[spikes.i]/meter)

###### Plot of the membrane potential of all compartments over time (1. Plot)
###### and the line regression of the spike-times (Second plot)
#plt.figure(3)
#plt.suptitle('Calculation of conductance velocity',fontsize=20)
#
#plt.subplot(211)
#for ii in range(0, nof_comps, step):
#    plot(M.t/ms, ii/5 + v_amp_factor*(M.v[ii, :]-V_res)/mV, 'k')
#plt.yticks(np.linspace(0,int(nof_comps/5), int(len_neuron/cm)+1),range(0,int(len_neuron/cm)+1,1))
#plt.xlabel('Time/ms')
#plt.ylabel('Position/mm [major] V/mV [minor]')
#
#plt.subplot(212)
#plt.plot(spikes.t/ms, neuron.distance[spikes.i]/cm, 'o', label='APs')
#plt.plot(spikes.t/ms, (intercept+slope*(spikes.t/second))/cm, 'r', label='fitted line')
#plt.xlabel('Time/ms')
#plt.ylabel('Position/cm')
#plt.xlim([min(M.t/ms),max(M.t/ms)])
#plt.legend()
#plt.show()

print("The conductance velocity is %.2f m/s" % slope)

# =============================================================================
# Now the following temporal characteristics of the action potentials are calculated:
# - average AP amplitude
# - average AP rise time
# - average AP fall time
# - latency of the spike in the first compartment
# =============================================================================
##### AP amplitudes in all compartments
AP_amps = [max(M.v[i,:]-V_res)
           for i in range(0,nof_comps)]

##### AP peak time in all compartments
AP_times = [float(M.t[M.v[i,:]-V_res == AP_amps[i]])*second
            for i in range(0,nof_comps)]

##### AP start point in all compartments (defined at 10% of peak value)
AP_start_time = [M.t[np.argmin(abs(M.v[i,np.where(M.t<AP_times[i])[0]]-V_res - 0.1*AP_amps[i]))]
                 for i in range(nof_comps)]

##### AP end point in all compartments (defined at 10% of peak value)
AP_end_time = [M.t[np.where(M.t>AP_times[i])[0]][np.argmin(abs(M.v[i,where(M.t>AP_times[i])[0]]-V_res - 0.1*AP_amps[i]))]
                 for i in range(0,nof_comps)]

##### combine data in dataframe
AP_data = pd.DataFrame([AP_amps, AP_times, AP_start_time, AP_end_time],
                       index=("amplitude", "peak_time", "start_time", "end_time")).T

##### Calculate rise and fall times
AP_data["rise_time"] = AP_data.peak_time - AP_data.start_time
AP_data["fall_time"] = AP_data.end_time - AP_data.peak_time

##### Calculate average values
AP_average_data = AP_data[["amplitude", "rise_time", "fall_time"]].mean().multiply([volt,second,second])

##### Calculate latency (for spike in first compartment)
latency = AP_data["peak_time"][0] - time_before

print("The average AP amplitude (difference to resting potential at peak) is:", AP_average_data.amplitude)
print("The average AP rise time (defined at 10% of peak value) is:", AP_average_data.rise_time)
print("The average AP fall time (defined at 10% of peak value) is:", AP_average_data.fall_time)
print("The latency of the spike in the first compartment was:", latency)
