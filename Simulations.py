##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### import functions
import functions.stimulation as stim
import functions.create_plots as plot

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Frijns_2005 as frijns_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Definition of neuron and initialization of state monitor
# =============================================================================
##### choose model
model = rattay_01

##### initialize clock
dt = 5*us

##### set up the neuron
neuron, param_string = model.set_up_model(dt = dt, model = model)

##### load the parameters of the differential equations in the workspace
exec(param_string)

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

##### save initialization of the monitor(s)
store('initialized')

# =============================================================================
# Simulations to be done / Plots to be shown
# =============================================================================
plot_voltage_course_lines = True
plot_voltage_course_colored = False
measure_single_node_response = False
measure_strength_duration_curve = True
measure_refractory_properties = False

# =============================================================================
# Run simulation and observe voltage courses for each compartment
# =============================================================================
if plot_voltage_course_lines or plot_voltage_course_colored:
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = "extern",
                                                pulse_form = "bi",
                                                stimulated_compartment = 4,
                                                nof_pulses = 4,
                                                time_before = 0*ms,
                                                time_after = 1*ms,
                                                add_noise = True,
                                                ##### monophasic stimulation
                                                amp_mono = -1.5*uA,
                                                duration_mono = 200*us,
                                                ##### biphasic stimulation
                                                amps_bi = [-2,0.2]*uA,
                                                durations_bi = [100,0,100]*us,
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = 800*us)
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = dt)
    
    ##### run simulation
    run(runtime)
    
    ##### Plot membrane potential of all compartments over time (2 plots)
    if plot_voltage_course_lines:
        plot.voltage_course_lines(plot_name = f"Voltage course {model.display_name}",
                                  time_vector = M.t,
                                  voltage_matrix = M.v,
                                  comps_to_plot = model.comps_to_plot,
                                  distance_comps_middle = model.distance_comps_middle,
                                  length_neuron = model.length_neuron,
                                  V_res = model.V_res)
    
    if plot_voltage_course_colored:
        plot.voltage_course_colors(plot_name = f"Voltage course {model.display_name} (colored)",
                                   time_vector = M.t,
                                   voltage_matrix = M.v,
                                   distance_comps_middle = model.distance_comps_middle)

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
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### loop over current ampltudes
    for ii in range(0, len(current_amps)):
        
        for jj in range(0,nof_runs):
            
            ##### go back to initial values
            restore('initialized')
            
            ##### define how the ANF is stimulated
            I_stim, runtime = stim.get_stimulus_current(model = model,
                                                        dt = dt,
                                                        stimulation_type = "extern",
                                                        pulse_form = "mono",
                                                        stimulated_compartment = 4,
                                                        nof_pulses = 1,
                                                        time_before = 0*ms,
                                                        time_after = 1.5*ms,
                                                        add_noise = True,
                                                        ##### monophasic stimulation
                                                        amp_mono = current_amps[ii],
                                                        duration_mono = 250*us,
                                                        ##### biphasic stimulation
                                                        amps_bi = [-2,2]*uA,
                                                        durations_bi = [100,0,100]*us)
        
            ##### get TimedArray of stimulus currents and run simulation
            stimulus = TimedArray(np.transpose(I_stim), dt=dt)
            
            ##### run simulation
            run(runtime)
            
            ##### write results in table
            AP_amp = max(M.v[comp_index,:]-model.V_res)
            AP_time = M.t[M.v[comp_index,:]-model.V_res == AP_amp]
            AP_start_time = M.t[np.argmin(abs(M.v[comp_index,np.where(M.t<AP_time)[0]]-model.V_res - 0.1*AP_amp))]
            AP_end_time = M.t[np.where(M.t>AP_time)[0]][np.argmin(abs(M.v[comp_index,np.where(M.t>AP_time)[0]]-model.V_res - 0.1*AP_amp))]
            
            node_response_data["stimulation amplitude (uA)"][ii*nof_runs+jj] = current_amps[ii]/uA
            node_response_data["AP height (mV)"][ii*nof_runs+jj] = AP_amp/mV
            node_response_data["AP peak time"][ii*nof_runs+jj] = AP_time/ms
            node_response_data["AP start time"][ii*nof_runs+jj] = AP_start_time/ms
            node_response_data["AP end time"][ii*nof_runs+jj] = AP_end_time/ms
            
            ##### print progress
            print(f"Stimulus amplitde: {np.round(current_amps[ii]/uA,3)} uA")

            ##### save voltage course of single compartment for plotting 
            if ii == jj == 0:
                voltage_data = np.zeros((len(current_amps)*nof_runs,np.shape(M.v)[1]))
            voltage_data[nof_runs*ii+jj,:] = M.v[comp_index, :]/mV
    
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
    
    ##### plot single node response curves
    plot.single_node_response(plot_name = f"Single node response {model.display_name}",
                              time_vector = M.t/ms,
                              voltage_matrix = voltage_data,
                              parameter_vector = current_amps*10**6/amp,
                              parameter_unit = r'$\mu A$',
                              V_res = model.V_res)
    
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
    
    ##### Initialize model with new defaultclock
    dt = 1*us
    neuron, param_string = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
          
    ##### phase durations
    phase_durations = np.round(np.logspace(1, 9, num=20, base=2.0))*us
    
    ##### initialize vector for minimum required stimulus current amplitudes
    min_required_amps = np.zeros_like(phase_durations/second)*amp
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    stim_amps_min = 0.1*uA
    stim_amps_max = 20*uA
    
    ##### start amplitde for first run
    start_amp = (stim_amps_max-stim_amps_min)/2
    
    ##### required accuracy
    delta = 0.05*uA
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]

    ##### loop over phase durations
    for ii in range(0, len(phase_durations)):
        
        ##### initializations
        min_amp_spiked = 0*amp
        lower_border = stim_amps_min
        upper_border = stim_amps_max
        stim_amp = start_amp
        amp_diff = upper_border - lower_border
        
        ##### adjust stimulus amplitude until required accuracy is obtained
        while amp_diff > delta:
            
            ##### print progress
            print(f"Duration: {phase_durations[ii]/us} us; Stimulus amplitde: {np.round(stim_amp/uA,4)} uA")
            
            ##### define how the ANF is stimulated
            I_stim, runtime = stim.get_stimulus_current(model = model,
                                                        dt = dt,
                                                        stimulation_type = "extern",
                                                        pulse_form = "mono",
                                                        stimulated_compartment = 4,
                                                        nof_pulses = 1,
                                                        time_before = 0*ms,
                                                        time_after = 1*ms,
                                                        add_noise = False,
                                                        ##### monophasic stimulation
                                                        amp_mono = -stim_amp,
                                                        duration_mono = phase_durations[ii],
                                                        ##### biphasic stimulation
                                                        amps_bi = [-stim_amp,stim_amp],
                                                        durations_bi = [phase_durations[ii],0*second,phase_durations[ii]],
                                                        ##### multiple pulses / pulse trains
                                                        inter_pulse_gap =800*us)
        
            ##### get TimedArray of stimulus currents and run simulation
            stimulus = TimedArray(np.transpose(I_stim), dt=dt)
            
            ##### reset state monitor
            restore('initialized')
            
            ##### run simulation
            run(runtime)
            
            ##### test if there was a spike
            if max(M.v[comp_index,:]-model.V_res) > 60*mV:
                min_amp_spiked = stim_amp
                upper_border = stim_amp
                stim_amp = (stim_amp + lower_border)/2
            else:
                lower_border = stim_amp
                stim_amp = (stim_amp + upper_border)/2
                
            amp_diff = upper_border - lower_border
                            
        ##### write the found minimum stimulus current in vector
        min_required_amps[ii] = min_amp_spiked
        start_amp[min_amp_spiked != 0*amp] = min_amp_spiked
        start_amp[min_amp_spiked == 0*amp] = stim_amps_max
    
    ##### plot strength duration curve
    plot.strength_duration_curve(plot_name = f"Strength duration curve {model.display_name}",
                                 durations = phase_durations,
                                 stimulus_amps = min_required_amps)

# =============================================================================
# Now the relative and absolute refractory periods will be measured. This is
# done with two stimuli, the first one with an amplitude of 150% of threshold
# masks the second one. Threshold amplitudes of the second stimulus over inter-
# pulse interval are measured
# =============================================================================
if measure_refractory_properties:
    
    ##### amplitude of masker stimulus (150% of threshold)
    amp_masker = 1.5 * -0.77*uA
          
    ##### phase durations
    phase_duration = 100*us
    
    ##### initialize vector for minimum required stimulus current amplitudes
    min_required_amps = np.zeros_like(phase_durations/second)*amp
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    stim_amps_min = amp_masker
    stim_amps_max = amp_masker * 8
    
    ##### start amplitde for first run
    start_amp = (stim_amps_max-stim_amps_min)/2
    
    ##### required accuracy
    delta = 0.05*uA
    
    ##### compartment for measurements
    comp_index = 47

    ##### loop over phase durations
    for ii in range(0, len(phase_durations)):
        
        ##### initializations
        min_amp_spiked = 0*amp
        lower_border = stim_amps_min
        upper_border = stim_amps_max
        stim_amp = start_amp
        amp_diff = upper_border - lower_border
        
        ##### adjust stimulus amplitude until required accuracy is obtained
        while amp_diff > delta:
            
            ##### print progress
            print(f"Inter pulse gap: {ipg[ii]/us} us; Stimulus amplitde: {np.round(stim_amp/uA,2)} uA")
            
            ##### go back to initial values
            restore('initialized')
            
            ##### define the first (masker) stimulus
            I_stim_masker, runtime_masker = stim.get_stimulus_current(dt = defaultclock.dt,
                                                                      stimulation_type = "extern",
                                                                      pulse_form = "mono",
                                                                      nof_pulses = 1,
                                                                      time_before = 0*ms,
                                                                      time_after = 2*ms,
                                                                      add_noise = False,
                                                                      ##### monophasic stimulation
                                                                      amp_mono = amp_masker,
                                                                      duration_mono = 100*ms,
                                                                      ##### biphasic stimulation
                                                                      amps_bi = np.array([-2,2])*uA,
                                                                      durations_bi = np.array([100,0,100])*us,
                                                                      ##### multiple pulses / pulse trains
                                                                      inter_pulse_gap = 1*ms,
                                                                      ##### external stimulation
                                                                      compartment_lengths = data.compartment_lengths,
                                                                      stimulated_compartment = 4,
                                                                      electrode_distance = 300*um,
                                                                      rho_out = data.rho_out,
                                                                      axoplasmatic_resistances =  data.R_a,
                                                                      ##### noise
                                                                      k_noise = data.k_noise,
                                                                      noise_term = data.noise_term)
            
            ##### combine both stimulations
            runtime = runtime_masker + runtime_masked
        
            ##### get TimedArray of stimulus currents and run simulation
            stimulus = TimedArray(np.transpose(I_stim), dt=defaultclock.dt)
            
            ##### run simulation
            run(runtime)
            
            ##### test if there was a spike
            if max(M.v[comp_index,:]-V_res) > 60*mV:
                min_amp_spiked = stim_amp
                upper_border = stim_amp
                stim_amp = (stim_amp + lower_border)/2
            else:
                lower_border = stim_amp
                stim_amp = (stim_amp + upper_border)/2
                
            amp_diff = upper_border - lower_border
                            
        ##### write the found minimum stimulus current in vector
        min_required_amps[ii] = min_amp_spiked
        start_amp[min_amp_spiked != 0*amp] = min_amp_spiked
        start_amp[min_amp_spiked == 0*amp] = stim_amps_max
    
    ##### plot strength duration curve
    plot.strength_duration_curve(plot_name = "Strength duration curve Rattay 2001",
                                 durations = phase_durations,
                                 stimulus_amps = min_required_amps)
    
    
    