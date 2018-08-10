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
import functions.model_tests as test

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
measure_single_node_response = True
measure_strength_duration_curve = False
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

    ##### Possible parameter types are all model attributes, "model", "stim_amp", "phase_duration" and "stochastic_runs"
    voltage_matrix, node_response_data_summary, time_vector = \
    test.get_single_node_response(model = [rattay_01, smit_10, smit_09],
                                   dt = dt,
                                   param_1 = "model",
                                   param_1_ratios = [0.8, 1.0, 1.2],
                                   param_2 = "compartment_diameters",
                                   param_2_ratios = [0.6, 0.8, 1.0, 2.0, 3.0],
                                   stimulation_type = "extern",
                                   pulse_form = "bi",
                                   time_after_stimulation = 1.5*ms,
                                   stim_amp = 2*uA,
                                   phase_duration = 200*us,
                                   nof_runs = 10)
    
#    ##### plot single node response curves
#    plot.single_node_response(plot_name = f"Single node response {model.display_name}",
#                              time_vector = time_vector,
#                              voltage_matrix = voltage_matrix,
#                              parameter_vector = [0.8, 0.9, 1.0, 1.1, 1.2],
#                              parameter_unit = "of standard compartment length",
#                              V_res = model.V_res)
    

    ##### plot results in bar plot
    plot.single_node_response_bar_plot(data = node_response_data_summary)

# =============================================================================
# Now a simulation will be run several times to calculate the strength-duration
#  curve. This allows to determine the following properties
# - Rheobase
# - chronaxie
# =============================================================================
if measure_strength_duration_curve:
    
    ##### define phase_durations to be tested
    phase_durations = np.round(np.logspace(1, 9, num=20, base=2.0))*us
    
    ##### calculated corresponding thresholds
    thresholds = test.get_strength_duration_curve(model = model,
                                                  dt = 1*us,
                                                  phase_durations = phase_durations,
                                                  start_intervall = [0.1,20]*uA,
                                                  delta = 0.01*uA,
                                                  stimulation_type = "extern",
                                                  pulse_form = "bi")
    
    ##### plot strength duration curve
    plot.strength_duration_curve(plot_name = f"Strength duration curve {model.display_name}",
                                 durations = phase_durations,
                                 stimulus_amps = thresholds)
    
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
    
    
    