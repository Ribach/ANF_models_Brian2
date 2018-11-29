# =============================================================================
# This script contains functions that are needed for model analyses and
# comparisons, that go further than the test in the "Run_test_battery" scipt.
# The functions here are used in the "Model_analyses" script.
# =============================================================================
##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import thorns as th
from scipy.signal import savgol_filter
from pytictoc import TicToc
import peakutils as peak
from importlib import reload
import pandas as pd
pd.options.mode.chained_assignment = None

##### import functions
import functions.stimulation as stim
import functions.calculations as calc
import functions.model_tests as test

##### import models
import models.Rattay_2001 as rattay_01
import models.Rattay_adap_2001 as rattay_adap_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Briaire_adap_2005 as briaire_adap_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Imennov_adap_2009 as imennov_adap_09
import models.Negm_2014 as negm_14
import models.Negm_ANF_2014 as negm_ANF_14
import models.Rudnicki_2018 as rudnicki_18

# =============================================================================
# Calculate the number of the node where a certain latency can be measured 
# =============================================================================
def get_node_number_for_latency(model_name,
                                dt,
                                latency_desired,
                                stim_amp,
                                phase_duration,
                                delta,
                                numbers_start_interval,
                                inter_phase_gap = 0*us,
                                pulse_form = "mono",
                                stimulation_type = "extern",
                                time_before = 2*ms,
                                time_after = 3*ms,
                                print_progress = True):
    """This function calculates the node number (starting at peripheral end),
    where the latency has to be measured to obtain a certain latency value
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    latency_desired : time or numeric value (numeric values are interpreted as time in second)
        Defines latency that should be measured.
    stim_amp : current or numeric (numeric values are interpreted as current in ampere)
        Current amplitude of stimulus pulse; Using biphasic pulses, the second phase
        has the inverted stimulus amplitude
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    delta : integer
        Maximum error for the number of the measurement node
    numbers_start_interval : list of integers of length two
        First value gives lower border of expected node number, second value gives upper border
    inter_phase_gap : time or numeric value (numeric values are interpreted as time in second)
        Length of the gap between the two phases of a biphasic stimulation
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    time_before : time
        Time until stimulation starts.
    time_after : time
        Time which is still simulated after the end of the stimulation
    print_progress : boolean
        Defines, if information about the progress are printed on the console
    
    Returns
    -------
    node number
        Gives back the number of the node where the desired latency can be measured
    """
    
    ##### add quantity to phase_duration, inter_phase_gap, latency and stim_amp
    phase_duration = float(phase_duration)*second
    inter_phase_gap = float(inter_phase_gap)*second
    latency_desired = float(latency_desired)*second
    stim_amp = float(stim_amp)*amp
    
    ##### get model
    model = eval(model_name)
    
    ##### initializations
    node_number = 0
    lower_border = numbers_start_interval[0]
    upper_border = numbers_start_interval[1]
    node = round((upper_border - lower_border)/2)
    node_diff = upper_border - lower_border
    
    ##### adjust stimulus amplitude until required accuracy is obtained
    while node_diff > delta:
        
        ##### print progress
        if print_progress: print("Model: {}; Node Number: {}".format(model_name, node))
        
        ##### extend model
        if hasattr(model, "nof_internodes"):
            model.nof_internodes = upper_border
        else:
            model.nof_axonal_internodes = upper_border
        
        ##### initialize model
        neuron, model = model.set_up_model(dt = dt, model = model, update = True)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')

        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    stimulation_type = stimulation_type,
                                                    pulse_form = pulse_form,
                                                    add_noise = False,
                                                    time_before = time_before,
                                                    time_after = time_after,
                                                    ##### monophasic stimulation
                                                    amp_mono = -stim_amp,
                                                    duration_mono = phase_duration,
                                                    ##### biphasic stimulation
                                                    amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                    durations_bi = [phase_duration/second,inter_phase_gap/second,phase_duration/second]*second)
    
        ##### get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim), dt=dt)
        
        ##### reset state monitor
        restore('initialized')
        
        ##### run simulation
        run(runtime)
        
        ##### calculate latency
        node_index = np.where(model.structure == 2)[0][node-1]
        AP_amp = max(M.v[node_index,:]-model.V_res)
        AP_time = M.t[M.v[node_index,:]-model.V_res == AP_amp]
        latency = (AP_time - time_before)[0]
        
        ##### set latency to zero if no AP could be measured
        if max(M.v[node_index,:]-model.V_res) < 60*mV:
            latency = 0
        
        ##### test if there was a spike
        if latency > latency_desired or latency == 0:
            node_number = node
            upper_border = node
            node = round((node + lower_border)/2)
        else:
            lower_border = node
            node = round((node + upper_border)/2)
            
        node_diff = upper_border - lower_border
    
    ##### reload module
    model = reload(model)

    return node_number

# =============================================================================
#  Meassure latency
# =============================================================================
def get_latency(model_name,
                dt,
                stim_amp,
                stimulus_node,
                measurement_node,
                phase_duration,
                inter_phase_gap = 0*us,
                pulse_form = "bi",
                stimulation_type = "extern",
                electrode_distance = 300*um,
                time_before = 2*ms,
                time_after = 2*ms,
                add_noise = False,
                print_progress = True):
    """This function calculates the the latency at a certain node for a certain
    electrode distance.
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    stim_amp : current or numeric value (numeric values are interpreted as a current in ampere)
        Current amplitude of stimulus pulse; Using biphasic pulses, the second phase
        has the inverted stimulus amplitude
    stimulus_node : integer
        Compartment number where the the model is stimulated
    measurement_node : integer
        Compartment number where the the latency is measured
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    inter_phase_gap : time or numeric value (numeric values are interpreted as time in second)
        Length of the gap between the two phases of a biphasic stimulation
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    electrode_distance : length or numeric value (numeric values are interpreted as length in meter)
        Defines the distance of the electrode to the measurement node
    time_before : time
        Time until stimulation starts.
    time_after : time
        Time which is still simulated after the end of the stimulation
    add_noise : boolean
        Defines, if Gaussian noise is added to the stimulus current.
    print_progress : boolean
        Defines, if information about the progress are printed on the console
    
    Returns
    -------
    latency (numeric)
        Gives back the latency as a numeric value which can be interpreted as time in second.
    """
    
    ##### add quantity to phase_duration, stim_amp and electrode_distance
    phase_duration = float(phase_duration)*second
    stim_amp = float(stim_amp)*amp
    electrode_distance = float(electrode_distance)*meter
    
    ##### get model
    model = eval(model_name)
    
    ##### get stimulus node location
    stimulus_node_index = np.where(model.structure == 2)[0][stimulus_node-1]
    
    ##### extend model if needed
    if len(np.where(model.structure == 2)[0]) < measurement_node:
        if hasattr(model, "nof_internodes"):
            model.nof_internodes = measurement_node + 5
        else:
            model.nof_axonal_internodes = measurement_node + 2
    
    ##### adjust electrode distance
    model.electrode_distance = electrode_distance
     
    ##### initialize model
    neuron, model = model.set_up_model(dt = dt, model = model, update = True)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### get measurement node location
    measurement_node_index = np.where(model.structure == 2)[0][measurement_node-1]
    
    ##### print progress
    if print_progress: print("Model: {}; Stimulus amplitde: {} uA;  electrode distance: {}; measurement location: {}".format(model_name,
                             np.round(stim_amp/uA,4),electrode_distance, measurement_node))
            
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = stimulation_type,
                                                pulse_form = pulse_form,
                                                stimulated_compartment = stimulus_node_index,
                                                time_before = time_before,
                                                time_after = time_after,
                                                add_noise = add_noise,
                                                ##### monophasic stimulation
                                                amp_mono = -stim_amp,
                                                duration_mono = phase_duration,
                                                ##### biphasic stimulation
                                                amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                durations_bi = [phase_duration/second,inter_phase_gap/second,phase_duration/second]*second)

    ##### get TimedArray of stimulus currents and run simulation
    stimulus = TimedArray(np.transpose(I_stim), dt=dt)
    
    ##### reset state monitor
    restore('initialized')
    
    ##### run simulation
    run(runtime)
    
    ##### AP amplitude
    AP_amp = max(M.v[measurement_node_index,:]-model.V_res)
    
    ##### AP time
    AP_time = M.t[M.v[measurement_node_index,:]-model.V_res == AP_amp]
    
    ##### calculate latency
    latency = (AP_time - time_before)[0]
    
    ##### set latency to zero if no AP could be measured
    if max(M.v[measurement_node_index,:]-model.V_res) < 60*mV:
        latency = 0
    
    ##### reload module
    model = reload(model)
    
    return latency/second

# =============================================================================
# Calculate electrode distance to obtain a certain threshold
# =============================================================================
def get_electrode_distance(model_name,
                           dt,
                           threshold,
                           phase_duration,
                           delta,
                           distances_start_interval,
                           inter_phase_gap = 0*us,
                           pulse_form = "mono",
                           stimulation_type = "extern",
                           time_before = 3*ms,
                           time_after = 3*ms,
                           print_progress = True):
    """This function calculates the the required electrode distance to obtain a
     certain threshold current.
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    threshold : current
        Defines current that should be the threshold of the model.
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    delta : length
        Maximum error for the required electrode distance
    distances_start_interval : list of length-values of length two
        First value gives lower border of expected electrode distances, second value gives upper border
    inter_phase_gap : time or numeric value (numeric values are interpreted as time in second)
        Length of the gap between the two phases of a biphasic stimulation
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    time_before : time
        Time until stimulation starts.
    time_after : time
        Time which is still simulated after the end of the stimulation
    print_progress : boolean
        Defines, if information about the progress are printed on the console
    
    Returns
    -------
    electrode distance
        Gives back the the required electrode distance
    """
    
    ##### add quantity to phase_duration and inter_phase_gap
    phase_duration = float(phase_duration)*second
    inter_phase_gap = float(inter_phase_gap)*second
    
    ##### get model
    model = eval(model_name)
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### initializations
    electrode_distance = 0*meter
    lower_border = distances_start_interval[0]
    upper_border = distances_start_interval[1]
    distance = (upper_border - lower_border)/2
    distance_diff = upper_border - lower_border
    
    ##### initialize model
    model.electrode_distance = distance
    neuron, model = model.set_up_model(dt = dt, model = model, update = True)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### adjust stimulus amplitude until required accuracy is obtained
    while distance_diff > delta:
        
        ##### print progress
        if print_progress: print("Model: {}; Threshold: {} uA; Distance: {} mm".format(model_name,
                                 np.round(threshold/uA),np.round(distance/mm,3)))
        
        ##### initialize model with new distance
        model.electrode_distance = distance
        neuron, model = model.set_up_model(dt = dt, model = model, update = True)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
        
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    stimulation_type = stimulation_type,
                                                    pulse_form = pulse_form,
                                                    add_noise = False,
                                                    time_before = time_before,
                                                    time_after = time_after,
                                                    ##### monophasic stimulation
                                                    amp_mono = -threshold,
                                                    duration_mono = phase_duration,
                                                    ##### biphasic stimulation
                                                    amps_bi = [-threshold/amp,threshold/amp]*amp,
                                                    durations_bi = [phase_duration/second,inter_phase_gap/second,phase_duration/second]*second)
    
        ##### get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim), dt=dt)
        
        ##### reset state monitor
        restore('initialized')
        
        ##### run simulation
        run(runtime)
        
        ##### test if there was a spike
        if max(M.v[comp_index,:]-model.V_res) < 60*mV:
            electrode_distance = distance
            upper_border = distance
            distance = (distance + lower_border)/2
        else:
            lower_border = distance
            distance = (distance + upper_border)/2
            
        distance_diff = upper_border - lower_border
    
    ##### reload module
    model = reload(model)
    
    return electrode_distance

# =============================================================================
#  Computational efficiency test
# =============================================================================
def computational_efficiency_test(model_names,
                                  dt,
                                  stimulus_duration,
                                  nof_runs):
    """This function calculates the required computation time for a certain
    stimulation of the models.

    Parameters
    ----------
    model_names : list of strings
        List with strings with the model names in the format of the imported
        modules on top of the script.
    dt : time
        Sampling timestep.
    stimulus_duration : time
        Defines how long the models are stimulated.
    nof_runs : integer
        Defines how often the simulations are repeated.

    Returns
    -------
    pandas dataframe
        Dataframe includes the computation times and the model names
    """
    
    ##### get models
    models = [eval(model) for model in model_names]
    
    ##### initialize dataframe
    computation_times = pd.DataFrame(np.zeros((nof_runs, len(models))), columns  = [model.display_name for model in models])
    
    ##### loop over models
    for model in models:
        
        ##### loop over runs
        for ii in range(nof_runs):
                
            ##### set up the neuron
            neuron, model = model.set_up_model(dt = dt, model = model)
            
            ##### define how the ANF is stimulated
            I_stim, runtime = stim.get_stimulus_current(model = model,
                                                        dt = dt,
                                                        nof_pulses = 0,
                                                        time_before = 0*ms,
                                                        time_after = stimulus_duration)
            
            ##### get TimedArray of stimulus currents
            stimulus = TimedArray(np.transpose(I_stim), dt = dt)
            
            ##### start timer
            t = TicToc()
            t.tic()
            
            ##### run simulation
            run(runtime)
            
            ##### end timer
            t.toc()
            
            ##### write result in dataframe
            computation_times[model.display_name][ii] = t.tocvalue()
    
    return computation_times
