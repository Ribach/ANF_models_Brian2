# =============================================================================
# This script includes all functions for the tests that are part of the test
# battery, which is implemented in the "Run_test_battery" script.
# =============================================================================
##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import thorns as th
from scipy.signal import savgol_filter
import peakutils as peak
from importlib import reload
import pandas as pd
pd.options.mode.chained_assignment = None

##### import functions
import functions.stimulation as stim
import functions.calculations as calc

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14
import models.Rudnicki_2018 as rudnicki_18
import models.trials.Rattay_adap_2001 as rattay_adap_01
import models.trials.Briaire_adap_2005 as briaire_adap_05
import models.trials.Imennov_adap_2009 as imennov_adap_09
import models.trials.Negm_ANF_2014 as negm_ANF_14

# =============================================================================
# Calculate threshold
# =============================================================================
def get_threshold(model_name,
                  dt,
                  phase_duration,
                  delta,
                  upper_border,
                  polarity = "cathodic",
                  inter_phase_gap = 0*us,
                  parameter = None,
                  parameter_ratio = None,
                  pulse_form = "mono",
                  stimulation_type = "extern",
                  time_before = 2*ms,
                  time_after = 3*ms,
                  stimulated_compartment = None,
                  nof_pulses = 1,
                  inter_pulse_gap = 1*ms,
                  add_noise = False,
                  print_progress = True,
                  run_number = 0):
    """This function calculates the spiking threshold of a model for mono- and
    biphasic pulses, intern and extern stimulation, single pulses and pulse trains.
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    delta : current
        Maximum error for the measured spiking threshold
    upper_border : current
        Upper border of considered currents. Threshold is expected to be below it
    polarity : string
        Describes polarity of stimulus current; either cathodic or anodic is possible
    inter_phase_gap : time or numeric value (numeric values are interpreted as time in second)
        Length of the gap between the two phases of a biphasic stimulation
    parameter : string
        String with the name of a model parameter, which will be adjusted
    parameter_ratio : numeric
        Numeric value used in cobination with a given parameter. Original value is multiplied with the parameter ratio
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    time_before : time
        Time until (first) pulse starts.
    time_after : time
        Time which is still simulated after the end of the last pulse
    nof_pulses : integer
        Number of pulses.
    inter_pulse_gap : time or numeric value (numeric values are interpreted as time in second)
        Time interval between pulses for nof_pulses > 1.
    add_noise : boolean
        Defines, if Gaussian noise is added to the stimulus current.
    print_progress : boolean
        Defines, if information about the progress are printed on the console
    run_number : integer
        Usefull for multiple threshold measurements using multiprocessing with
        the thorns package.
    
    Returns
    -------
    threshold current
        Gives back the spiking threshold.
    """
    
    ##### add quantity to phase_duration and inter_phase_gap
    phase_duration = float(phase_duration)*second
    inter_phase_gap = float(inter_phase_gap)*second
    inter_pulse_gap = float(inter_pulse_gap)*second
    
    ##### get model
    model = eval(model_name)
    
    ##### initialize model
    if parameter is not None:
        
        ##### adjust model parameter
        exec("model.{} = parameter_ratio*model.{}".format(parameter,parameter))
            
        ##### initialize model with changed parameter
        neuron, model = model.set_up_model(dt = dt, model = model, update = True)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
        
    else:
        
        ##### initialize model (no parameter was changed)
        neuron, model = model.set_up_model(dt = dt, model = model)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][-5]
    
    ##### calculate runtime
    if pulse_form == "mono":
        runtime = time_before + nof_pulses*phase_duration + (nof_pulses-1)*inter_pulse_gap + time_after
    else:
        runtime = time_before + nof_pulses*(phase_duration*2 + inter_phase_gap) + (nof_pulses-1)*inter_pulse_gap + time_after
    
    ##### calculate number of timesteps
    nof_timesteps = int(np.ceil(runtime/dt))
    
    ##### include noise
    if add_noise:
        np.random.seed()
        I_noise = np.transpose(np.transpose(np.random.normal(0, 1, (model.nof_comps,nof_timesteps)))*model.k_noise*model.noise_term)
    else:
        I_noise = np.zeros((model.nof_comps,nof_timesteps))
    
    ##### consider polarity
    if polarity == "cathodic":
        pol = -1
    else:
        pol = 1
    
    ##### initializations
    threshold = 0*amp
    lower_border = 0*amp
    stim_amp = upper_border*0.2
    amp_diff = upper_border - lower_border
    
    ##### adjust stimulus amplitude until required accuracy is obtained
    while amp_diff > delta:
        
        ##### print progress
        if print_progress: print("Model: {}; {}: {}; Duration: {} us; Run: {}; Stimulus amplitde: {} uA".format(model_name,
                                 parameter, parameter_ratio, np.round(phase_duration/us),run_number+1,np.round(stim_amp/uA,4)))
        
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    stimulation_type = stimulation_type,
                                                    pulse_form = pulse_form,
                                                    add_noise = False,
                                                    time_before = time_before,
                                                    time_after = time_after,
                                                    stimulated_compartment = stimulated_compartment,
                                                    nof_pulses = nof_pulses,
                                                    ##### monophasic stimulation
                                                    amp_mono = stim_amp * pol,
                                                    duration_mono = phase_duration,
                                                    ##### biphasic stimulation
                                                    amps_bi = [stim_amp/amp,-stim_amp/amp]*amp * pol,
                                                    durations_bi = [phase_duration/second,inter_phase_gap/second,phase_duration/second]*second,
                                                    ##### multiple pulses / pulse trains
                                                    inter_pulse_gap = inter_pulse_gap)
        
        ##### get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim + I_noise), dt=dt)
        
        ##### reset state monitor
        restore('initialized')
        
        ##### run simulation
        run(runtime)
        
        ##### test if there was a spike
        if max(M.v[comp_index,:]-model.V_res) > 60*mV:
            threshold = stim_amp
            upper_border = stim_amp
            stim_amp = (stim_amp + lower_border)/2
        else:
            lower_border = stim_amp
            stim_amp = (stim_amp + upper_border)/2
            
        amp_diff = upper_border - lower_border
    
    if parameter is not None:
        ##### reload module
        model = reload(model)

    return threshold

# =============================================================================
#  Calculate conduction velocity
# =============================================================================
def get_conduction_velocity(model_name,
                            dt,
                            measurement_start_comp = 2,
                            measurement_end_comp = 6,
                            pulse_form = "mono",
                            stimulation_type = "extern",
                            time_before = 2*ms,
                            time_after = 1.5*ms,
                            stimulated_compartment = 2,
                            stim_amp = 2*uA,
                            phase_duration = 200*us,
                            add_noise = False,
                            run_number = 0):
    """This function calculates the conduction velocity of a model between two
    compartments.
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    measurement_start_comp : integer
        Compartment number where the first time-point of the AP is measured
    measurement_end_comp : integer
        Compartment number where the last time-point of the AP is measured
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    time_before : time
        Time until (first) pulse starts.
    time_after : time
        Time which is still simulated after the end of the last pulse
    stimulated_compartment : integer
        Compartment number where the the model is stimulated
    stim_amp : current
        Current amplitude of stimulus pulse; Using biphasic pulses, the second phase
        has the inverted stimulus amplitude
    phase_duration : time
        Duration of one phase of the stimulus current
    add_noise : boolean
        Defines, if Gaussian noise is added to the stimulus current.
    run_number : integer
        Usefull for multiple threshold measurements using multiprocessing with
        the thorns package.
    
    Returns
    -------
    conduction velocity
        Gives back the conduction velocity.
    """
    
    ##### get model
    model = eval(model_name)
    
    ##### calculate length of neuron part for measurement
    conduction_length = sum(model.compartment_lengths[measurement_start_comp:measurement_end_comp+1])
    
    ##### initialize neuron and state monitor
    neuron, model = model.set_up_model(dt = dt, model = model)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = stimulation_type,
                                                pulse_form = pulse_form,
                                                time_before = time_before,
                                                time_after = time_after,
                                                add_noise = add_noise,
                                                stimulated_compartment = stimulated_compartment,
                                                ##### monophasic stimulation
                                                amp_mono = -stim_amp,
                                                duration_mono = phase_duration,
                                                ##### biphasic stimulation
                                                amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                durations_bi = [phase_duration/second,0,phase_duration/second]*second)

    ##### get TimedArray of stimulus currents and run simulation
    stimulus = TimedArray(np.transpose(I_stim), dt=dt)
    
    ##### reset state monitor
    restore('initialized')
    
    ##### run simulation
    run(runtime)
    
    ##### calculate point in time at AP start
    AP_amp_start_comp = max(M.v[measurement_start_comp,:]-model.V_res)
    AP_time_start_comp = M.t[M.v[measurement_start_comp,:]-model.V_res == AP_amp_start_comp][0]
    
    ##### calculate point in time at AP end
    AP_amp_end_comp = max(M.v[measurement_end_comp,:]-model.V_res)
    AP_time_end_comp = M.t[M.v[measurement_end_comp,:]-model.V_res == AP_amp_end_comp][0]
    
    ##### calculate conduction velocity
    conduction_time = AP_time_end_comp - AP_time_start_comp
    conduction_velocity = conduction_length/conduction_time
    
    return conduction_velocity

# =============================================================================
#  Calculate single node respones
# =============================================================================
def get_single_node_response(model_name,
                             dt,
                             stim_amp,
                             phase_duration,
                             parameter = None,
                             parameter_ratio = None,
                             pulse_form = "bi",
                             stimulation_type = "extern",
                             time_before = 3*ms,
                             time_after = 2*ms,
                             stimulated_compartment = 4,
                             add_noise = True,
                             print_progress = True,
                             run_number = 0):
    """This function calculates the spiking threshold of a model for mono- and
    biphasic pulses, intern and extern stimulation, single pulses and pulse trains.
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    stim_amp : current or numeric value (numeric values are interpreted as a current in ampere)
        Current amplitude of stimulus pulse; Using biphasic pulses, the second phase
        has the inverted stimulus amplitude
    parameter : string
        String with the name of a model parameter, which will be adjusted
    parameter_ratio : numeric
        Numeric value used in cobination with a given parameter. Original value is multiplied with the parameter ratio
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    time_before : time
        Time until (first) pulse starts.
    time_after : time
        Time which is still simulated after the end of the last pulse
    stimulated_compartment : integer
        Compartment number where the the model is stimulated
    add_noise : boolean
        Defines, if Gaussian noise is added to the stimulus current.
    print_progress : boolean
        Defines, if information about the progress are printed on the console
    run_number : integer
        Usefull for multiple threshold measurements using multiprocessing with
        the thorns package.
    
    Returns
    -------
    AP height (numeric)
        Gives back the AP amplitude as a numeric value which can be interpreted as voltage in volt.
    AP rise time (numeric)
        Gives back the AP rise time as a numeric value which can be interpreted as a time in second.
    AP fall time (numeric)
        Gives back the AP fall time as a numeric value which can be interpreted as a time in second.
    latency (numeric)
        Gives back the AP latency as a numeric value which can be interpreted as a time in second.
    membrane potential (list of numerics)
        Gives back a list with the membrane potentials of the measurement node;
        Can be used to plot the voltage course
    time vector (list of numerics)
        Gives back a list of time points, that correspond to the voltage values
    """
    
    ##### add quantity to phase_duration
    phase_duration = float(phase_duration)*second
    stim_amp = float(stim_amp)*amp
    
    ##### get model
    model = eval(model_name)
    
    ##### initialize model
    if parameter is not None:
        
        ##### get parameter value of model
        original_param_value = eval("model.{}".format(parameter))
        
        ##### adjust model parameter
        exec("model.{} = parameter_ratio*original_param_value".format(parameter))
            
        ##### initialize model with changed parameter
        neuron, model = model.set_up_model(dt = dt, model = model, update = True)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
        
    else:
        
        ##### initialize model (no parameter was changed)
        neuron, model = model.set_up_model(dt = dt, model = model)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][-5]
    
    ##### print progress
    if print_progress: print("Model: {}; {}: {}; Duration: {} us; Run: {}; Stimulus amplitde: {} uA".format(model_name,
                             parameter, parameter_ratio, np.round(phase_duration/us),run_number+1,np.round(stim_amp/uA,4)))
            
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = stimulation_type,
                                                pulse_form = pulse_form,
                                                time_before = time_before,
                                                time_after = time_after,
                                                stimulated_compartment = stimulated_compartment,
                                                add_noise = add_noise,
                                                ##### monophasic stimulation
                                                amp_mono = -stim_amp,
                                                duration_mono = phase_duration,
                                                ##### biphasic stimulation
                                                amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                durations_bi = [phase_duration/second,0,phase_duration/second]*second)

    ##### get TimedArray of stimulus currents and run simulation
    stimulus = TimedArray(np.transpose(I_stim), dt=dt)
    
    ##### reset state monitor
    restore('initialized')
    
    ##### run simulation
    run(runtime)
    
    ##### AP amplitude
    AP_amp = max(M.v[comp_index,:]-model.V_res)
    
    ##### AP time
    AP_time = M.t[M.v[comp_index,:]-model.V_res == AP_amp]
    
    ##### time of AP start (10% of AP height before AP)
    if any(M.t<AP_time):
        AP_start_time = M.t[np.argmin(abs(M.v[comp_index,np.where(M.t<AP_time)[0]]-model.V_res - 0.1*AP_amp))]
    else:
        AP_start_time = 0*ms
        
    ##### time of AP end (10% of AP height after AP))
    if any(M.t>AP_time):
        AP_end_time = M.t[np.where(M.t>AP_time)[0]][np.argmin(abs(M.v[comp_index,np.where(M.t>AP_time)[0]]-model.V_res - 0.1*AP_amp))]
    else:
        AP_end_time = 0*ms
        
    ##### set AP amplitude to 0 if no start or end time could be measured
    if AP_start_time == 0*ms or AP_end_time == 0*ms:
        AP_amp = 0*mV

    ##### calculate the AP time at the stimulated compartment (for latency measurement)
    AP_amp_stim_comp = max(M.v[stimulated_compartment,:]-model.V_res)
    AP_time_stim_comp = M.t[M.v[stimulated_compartment,:]-model.V_res == AP_amp_stim_comp]
    
    ##### calculate AP properties
    AP_rise_time = (AP_time - AP_start_time)[0]
    AP_fall_time = (AP_end_time - AP_time)[0]
    latency = (AP_time_stim_comp - time_before)[0]
        
    ##### save voltage course of single compartment and corresponding time vector
    voltage_course = (M.v[comp_index, int(np.floor(time_before/dt)):]/volt).tolist()
    time_vector = (M.t[int(np.floor(time_before/dt)):]/second).tolist()
    
    if parameter is not None:
        ##### reload module
        model = reload(model)
    
    return {"AP height (mV)" : AP_amp/volt,
            "rise time (us)" : AP_rise_time/second,
            "fall time (us)" : AP_fall_time/second,
            "latency (us)" : latency/second,
            "membrane potential (mV)" : voltage_course,
            "time (ms)" : time_vector}

# =============================================================================
#  Calculate cronaxie for a given rheobase
# =============================================================================
def get_chronaxie(model_name,
                  dt,
                  rheobase,
                  phase_duration_start_interval,
                  delta,
                  polarity = "cathodic",                  
                  pulse_form = "mono",
                  stimulation_type = "extern",
                  time_before = 1*ms,
                  time_after = 1.5*ms,
                  stimulated_compartment = None,
                  print_progress = True):
    """This function calculates the chronaxie, which is defined as the time a
    model has stimulated with twice the Rheobase to elicit an AP.
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    rheobase : current
        Current value, which gives the spiking threshold for a very long stimulus
    phase_duration_start_interval : list of times of length two
        First value gives lower border of expected chronaxie; second value gives upper border
    delta : time
        Maximum error for the measured chronaxie
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    time_before : time
        Time until (first) pulse starts.
    time_after : time
        Time which is still simulated after the end of the last pulse
    print_progress : boolean
        Defines, if information about the progress are printed on the console
    
    Returns
    -------
    chronaxie time
        Gives back the chronaxie
    """
    
    ##### get model
    model = eval(model_name)
        
    ##### initialize model with given defaultclock dt
    neuron, model = model.set_up_model(dt = dt, model = model)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    phase_duration_min = phase_duration_start_interval[0]
    phase_duration_max = phase_duration_start_interval[1]
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### consider polarity
    if polarity == "cathodic":
        pol = -1
    else:
        pol = 1
        
    ##### initializations
    chronaxie = 0*second
    lower_border = phase_duration_min
    upper_border = phase_duration_max
    phase_duration = (phase_duration_max-phase_duration_min)/2
    duration_diff = upper_border - lower_border
    
    ##### adjust stimulus amplitude until required accuracy is obtained
    while duration_diff > delta:
        
        ##### print progress
        if print_progress: print("Duration: {} us".format(np.round(phase_duration/us)))
        
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    stimulation_type = stimulation_type,
                                                    pulse_form = pulse_form,
                                                    time_before = time_before,
                                                    time_after = time_after,
                                                    stimulated_compartment = stimulated_compartment,
                                                    ##### monophasic stimulation
                                                    amp_mono = 2*rheobase * pol,
                                                    duration_mono = phase_duration,
                                                    ##### biphasic stimulation
                                                    amps_bi = [2*rheobase/amp,-2*rheobase/amp]*amp * pol,
                                                    durations_bi = [phase_duration/second,0,phase_duration/second]*second)
    
        ##### get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim), dt=dt)
        
        ##### reset state monitor
        restore('initialized')
        
        ##### run simulation
        run(runtime)
        
        ##### test if there was a spike
        if max(M.v[comp_index,:]-model.V_res) > 60*mV:
            chronaxie = phase_duration
            upper_border = phase_duration
            phase_duration = (phase_duration + lower_border)/2
        else:
            lower_border = phase_duration
            phase_duration = (phase_duration + upper_border)/2
            
        duration_diff = upper_border - lower_border
        
    return chronaxie

# =============================================================================
#  Calculate refractory periods
# =============================================================================
def get_refractory_periods(model_name,
                           dt,
                           delta = 1*us,
                           threshold = None,
                           amp_masker = None,
                           pulse_form = "mono",
                           stimulation_type = "extern",
                           phase_duration = 100*us,
                           time_before = 2*ms,
                           stimulated_compartment = None,
                           print_progress = True):
    """This function measures both the absolute and relative refractory period.

    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    delta : time
        Maximum error for the measured refractory periods
    threshold : current or numeric value (numeric values are interpreted as current in ampere)
        Spiking threshold of the model for given stimulus; If not defined it will
        be calculated
    amp_masker : current or numeric value (numeric values are interpreted as current in ampere)
        Current amplitude of the masker stimulus; If not defined it will be set to 1.5 the threshold
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    time_before : time
        Time until (first) pulse starts.
    print_progress : boolean
        Defines, if information about the progress are printed on the console
                
    Returns
    -------
    absolute refractory period (numeric)
        Value can be interpreted as time in second
    relative refractory period (numeric)
        Value can be interpreted as time in second
    """
    
    ##### add quantity to phase_duration, threshold and amp_masker
    phase_duration = float(phase_duration)*second
    if threshold is not None:
        threshold = float(threshold)*amp
    if amp_masker is not None:
        amp_masker = float(amp_masker)*amp
        
    ##### get model
    model = eval(model_name)
    
    ##### calculate theshold of model
    if threshold is None:
        threshold = get_threshold(model_name = model_name,
                           dt = dt,
                           phase_duration = phase_duration,
                           upper_border = 800*uA,
                           delta = 0.0001*uA,
                           stimulation_type = stimulation_type,
                           pulse_form = pulse_form,
                           time_after = 3*ms,
                           print_progress = False)
    
    ##### amplitude of masker stimulus
    if amp_masker is None:
        amp_masker = 1.5 * threshold
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    inter_pulse_gap_min = 0*ms
    inter_pulse_gap_max = 10*ms
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### thresholds for second spike that define the refractory periods
    stim_amp_arp = 4*threshold
    stim_amp_rrp = 1.01*threshold    

    ##### get absolute refractory period
    # initializations
    arp = 0*second
    lower_border = inter_pulse_gap_min.copy()
    upper_border = inter_pulse_gap_max.copy()
    inter_pulse_gap = (inter_pulse_gap_max-inter_pulse_gap_min)/2
    inter_pulse_gap_diff = upper_border - lower_border
    
    # initialize model with given defaultclock dt
    neuron, model = model.set_up_model(dt = dt, model = model)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    # adjust stimulus amplitude until required accuracy is obtained
    while inter_pulse_gap_diff > delta:
        
        # print progress
        if print_progress: print("ARP: Phase duration: {} us, Inter pulse gap: {} us".format(np.round(phase_duration/us), np.round(inter_pulse_gap/us)))
        
        # define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = time_before,
                                                                  time_after = 0*ms,
                                                                  stimulated_compartment = stimulated_compartment,
                                                                  # monophasic stimulation
                                                                  amp_mono = -amp_masker,
                                                                  duration_mono = phase_duration,
                                                                  # biphasic stimulation
                                                                  amps_bi = [-amp_masker/amp,amp_masker/amp]*amp,
                                                                  durations_bi = [phase_duration/second,0,phase_duration/second]*second)
 
        I_stim_2nd, runtime_2nd = stim.get_stimulus_current(model = model,
                                                            dt = dt,
                                                            stimulation_type = stimulation_type,
                                                            pulse_form = pulse_form,
                                                            time_before = inter_pulse_gap,
                                                            time_after = 3*ms,
                                                            stimulated_compartment = stimulated_compartment,
                                                            # monophasic stimulation
                                                            amp_mono = -stim_amp_arp,
                                                            duration_mono = phase_duration,
                                                            # biphasic stimulation
                                                            amps_bi = [-stim_amp_arp/amp,stim_amp_arp/amp]*amp,
                                                            durations_bi = [phase_duration/second,0,phase_duration/second]*second)
        
        # combine stimuli
        I_stim = np.concatenate((I_stim_masker, I_stim_2nd), axis = 1)*amp
        runtime = runtime_masker + runtime_2nd
        
        # get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim), dt=dt)
        
        # reset state monitor
        restore('initialized')
        
        # run simulation
        run(runtime)
        
        # test if there were two spikes (one for masker and one for 2. stimulus)
        nof_spikes = len(peak.indexes(M.v[comp_index,:], thres = model.V_res + 60*mV, thres_abs=True))
        
        if nof_spikes > 1:
            arp = inter_pulse_gap
            upper_border = inter_pulse_gap
            inter_pulse_gap = (inter_pulse_gap + lower_border)/2
        else:
            lower_border = inter_pulse_gap
            inter_pulse_gap = (inter_pulse_gap + upper_border)/2
            
        inter_pulse_gap_diff = upper_border - lower_border
                
    ##### get relative refractory period
    # initializations
    rrp = 0*second
    lower_border = arp.copy()
    upper_border = inter_pulse_gap_max.copy()
    inter_pulse_gap = (inter_pulse_gap_max-inter_pulse_gap_min)/2
    inter_pulse_gap_diff = upper_border - lower_border
    
    # adjust stimulus amplitude until required accuracy is obtained
    while inter_pulse_gap_diff > delta:
                
        # print progress
        if print_progress: print("RRP: Phase duration: {} us, Inter pulse gap: {} us".format(np.round(phase_duration/us), np.round(inter_pulse_gap/us)))
        
        # define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = time_before,
                                                                  time_after = 0*ms,
                                                                  stimulated_compartment = stimulated_compartment,
                                                                  # monophasic stimulation
                                                                  amp_mono = -amp_masker,
                                                                  duration_mono = phase_duration,
                                                                  # biphasic stimulation
                                                                  amps_bi = [-amp_masker/amp,amp_masker/amp]*amp,
                                                                  durations_bi = [phase_duration/second,0,phase_duration/second]*second)
 
        I_stim_2nd, runtime_2nd = stim.get_stimulus_current(model = model,
                                                            dt = dt,
                                                            stimulation_type = stimulation_type,
                                                            pulse_form = pulse_form,
                                                            time_before = inter_pulse_gap,
                                                            time_after = 3*ms,
                                                            stimulated_compartment = stimulated_compartment,
                                                            # monophasic stimulation
                                                            amp_mono = -stim_amp_rrp,
                                                            duration_mono = phase_duration,
                                                            # biphasic stimulation
                                                            amps_bi = [-stim_amp_rrp/amp,stim_amp_rrp/amp]*amp,
                                                            durations_bi = [phase_duration/second,0,phase_duration/second]*second)
        
        # combine stimuli
        I_stim = np.concatenate((I_stim_masker, I_stim_2nd), axis = 1)*amp
        runtime = runtime_masker + runtime_2nd
        
        # get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim), dt=dt)
        
        # reset state monitor
        restore('initialized')
        
        # run simulation
        run(runtime)
        
        # test if there were two spikes (one for masker and one for 2. stimulus)
        nof_spikes = len(peak.indexes(M.v[comp_index,:], thres = model.V_res + 60*mV, thres_abs=True))
        
        if nof_spikes > 1:
            rrp = inter_pulse_gap
            upper_border = inter_pulse_gap
            inter_pulse_gap = (inter_pulse_gap + lower_border)/2
        else:
            lower_border = inter_pulse_gap
            inter_pulse_gap = (inter_pulse_gap + upper_border)/2
            
        inter_pulse_gap_diff = upper_border - lower_border
                            
    return float(arp), float(rrp)

# =============================================================================
#  Calculate refractory curve
# =============================================================================
def get_refractory_curve(model_name,
                         dt,
                         inter_pulse_interval,
                         delta,
                         threshold = None,
                         amp_masker = None,
                         pulse_form = "mono",
                         stimulation_type = "extern",
                         phase_duration = 100*us,
                         time_before = 3*ms,
                         stimulated_compartment = None,
                         print_progress = True):
    """This function measures the minimum required amplitude (threshold) to elicit a second
    spike with a stimulus that is masked by a first spike for a given inter pulse interval

    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    inter_pulse_interval : time or numeric value (numeric values are interpreted as time in second)
        time period between first stimulus (masker) and second stimulus
    delta : amplitude
        Maximum error for the measured threshold of the second stimulus
    threshold : current or numeric value (numeric values are interpreted as current in ampere)
        Spiking threshold of the model for given stimulus; If not defined it will
        be calculated
    amp_masker : current or numeric value (numeric values are interpreted as current in ampere)
        Current amplitude of the masker stimulus; If not defined it will be set to 1.5 the threshold
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    time_before : time
        Time until (first) pulse starts.
    print_progress : boolean
        Defines, if information about the progress are printed on the console
                
    Returns
    -------
    threshold current for second stimulus (float)
        Value can be interpreted as current in ampere
    threshold for first stimulus (float)
        Value can be interpreted as current in ampere
    """
    
    ##### add quantity to phase_duration, inter_pulse_interval, threshold and amp_masker
    phase_duration = float(phase_duration)*second
    inter_pulse_interval = float(inter_pulse_interval)*second
    if threshold is not None:
        threshold = float(threshold)*amp
    if amp_masker is not None:
        amp_masker = float(amp_masker)*amp
        
    ##### get model
    model = eval(model_name)
    
    ##### calculate theshold of model
    if threshold is None:
        threshold = get_threshold(model_name = model_name,
                                  dt = dt,
                                  phase_duration = phase_duration,
                                  upper_border = 800*uA,
                                  delta = 0.0001*uA,
                                  stimulation_type = stimulation_type,
                                  pulse_form = pulse_form,
                                  time_before = 2*ms,
                                  time_after = 3*ms,
                                  stimulated_compartment = stimulated_compartment,
                                  print_progress = False)
    
    ##### amplitude of masker stimulus (150% of threshold)
    if amp_masker is None:
        amp_masker = 1.5 * threshold
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### initialize model and monitors
    neuron, model = model.set_up_model(dt = dt, model = model)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
                
    ##### initializations
    min_amp_spiked = 0*amp
    lower_border = 0*amp
    upper_border = threshold * 20
    stim_amp = upper_border
    amp_diff = upper_border - lower_border
            
    ##### adjust stimulus amplitude until required accuracy is obtained
    while amp_diff > delta:
        
        ##### print progress
        if print_progress: print("Inter pulse interval: {} us; Amplitude of second stimulus: {} uA".format(round(inter_pulse_interval/us),np.round(stim_amp/uA,2)))
        
        ##### define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = time_before,
                                                                  time_after = 0*ms,
                                                                  stimulated_compartment = stimulated_compartment,
                                                                  ##### monophasic stimulation
                                                                  amp_mono = -amp_masker,
                                                                  duration_mono = phase_duration,
                                                                  ##### biphasic stimulation
                                                                  amps_bi = [-amp_masker/amp,amp_masker/amp]*amp,
                                                                  durations_bi = [phase_duration/second,0,phase_duration/second]*second)
                
        I_stim_2nd, runtime_2nd = stim.get_stimulus_current(model = model,
                                                            dt = dt,
                                                            stimulation_type = stimulation_type,
                                                            pulse_form = pulse_form,
                                                            time_before = inter_pulse_interval,
                                                            time_after = 3*ms,
                                                            stimulated_compartment = stimulated_compartment,
                                                            ##### monophasic stimulation
                                                            amp_mono = -stim_amp,
                                                            duration_mono = phase_duration,
                                                            ##### biphasic stimulation
                                                            amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                            durations_bi = [phase_duration/second,0,phase_duration/second]*second)
        
        ##### combine stimuli
        I_stim = np.concatenate((I_stim_masker, I_stim_2nd), axis = 1)*amp
        runtime = runtime_masker + runtime_2nd
        
        ##### get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim), dt=dt)
        
        ##### reset state monitor
        restore('initialized')
        
        ##### run simulation
        run(runtime)
                
        ##### test if there were two spikes (one for masker and one for 2. stimulus)
        nof_spikes = len(peak.indexes(M.v[comp_index,:], thres = model.V_res + 60*mV, thres_abs=True))
        
        if nof_spikes > 1:
            min_amp_spiked = stim_amp
            upper_border = stim_amp
            stim_amp = (stim_amp + lower_border)/2
        else:
            lower_border = stim_amp
            stim_amp = (stim_amp + upper_border)/2
            
        amp_diff = upper_border - lower_border
                
    return float(min_amp_spiked), float(threshold)

# =============================================================================
#  Calculate poststimulus time histogram (PSTH)
# =============================================================================
def post_stimulus_time_histogram(model_name,
                                 dt,
                                 pulses_per_second,
                                 stim_duration,
                                 stim_amp,
                                 pulse_form = "bi",
                                 stimulation_type = "extern",
                                 phase_duration = 50*us,
                                 stimulated_compartment = None,
                                 print_progress = True,
                                 run_number = 0):
    """This function calculates the spiking threshold of a model for mono- and
    biphasic pulses, intern and extern stimulation, single pulses and pulse trains.
    
    Parameters
    ----------
    model_name : string
        String with the model name in the format of the imported modules on top of the script
    dt : time
        Sampling timestep.
    pulses_per_second : integer
        Defines pulse rate
    stim_duration : time
        Defines length of the pulse train
    stim_amp : current or numeric value (numeric values are interpreted as a current in ampere)
        Current amplitude of stimulus pulse; Using biphasic pulses, the second phase
        has the inverted stimulus amplitude
    pulse_form : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    stimulation_type : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_duration : time or numeric value (numeric values are interpreted as time in second)
        Duration of one phase of the stimulus current
    print_progress : boolean
        Defines, if information about the progress are printed on the console
    run_number : integer
        Usefull for multiple threshold measurements using multiprocessing with
        the thorns package.
    
    Returns
    -------
    list of spike times (numeric values)
        Spike times can be interpreted as time values in second
    a random second argument
        this one is needed, as the map function of the thorn package which
        allows multiprocessing works different for multiple outputs. For only
        one output all spike times would be in additional columns in the resulting
        data frame. By using a second argument, this behaviour is avoided and the
        spike times are saved in a column of lists, which is desired.
    """
    
    ##### add quantity to stim_amp and phase_duration
    stim_amp = float(stim_amp)*amp
    phase_duration = float(phase_duration)*second
    
    ##### get model
    model = eval(model_name)
    
    ##### set up the neuron
    neuron, model = model.set_up_model(dt = dt, model = model)
    
    ##### initialize monitors
    M = StateMonitor(neuron, 'v', record=True)
    
    ##### save initialization of the monitor(s)
    store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]

    ##### calculate nof_pulses
    nof_pulses = np.floor(pulses_per_second*stim_duration/second).astype(int)
        
    ##### calculate inter_pulse_gap
    if pulse_form == "mono":
        inter_pulse_gap = (1e6/pulses_per_second - phase_duration/us)*us
    else:
        inter_pulse_gap = (1e6/pulses_per_second - phase_duration*2/us)*us
        
    ##### initialize pulse train dataframe
    spike_times = np.zeros(nof_pulses*3)
     
    ##### print progress
    if print_progress: print("Pulse rate: {} pps; Stimulus Amplitude: {} uA; Run: {}".format(pulses_per_second,np.round(stim_amp/uA,2),run_number))
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = stimulation_type,
                                                pulse_form = pulse_form,
                                                nof_pulses = nof_pulses,
                                                time_before = 0*ms,
                                                time_after = 0*ms,
                                                add_noise = True,
                                                stimulated_compartment = stimulated_compartment,
                                                ##### monophasic stimulation
                                                amp_mono = -stim_amp*uA,
                                                duration_mono = phase_duration,
                                                ##### biphasic stimulation
                                                amps_bi = [-stim_amp/uA,0,stim_amp/uA]*uA,
                                                durations_bi = [phase_duration/us,0,phase_duration/us]*us,
                                                ##### multiple pulses / pulse trains
                                                inter_pulse_gap = inter_pulse_gap)

    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = dt)
    
    ##### reset state monitor
    restore('initialized')
            
    ##### run simulation
    run(runtime)
    
    ##### get spike times
    spikes = M.t[peak.indexes(savgol_filter(M.v[comp_index,:], 51,3)*volt, thres = model.V_res + 60*mV, thres_abs=True)]/second
    spike_times[0:len(spikes)] = spikes
    
    ##### trim zeros
    spike_times = spike_times[spike_times != 0].tolist()
    
    ##### return Na, when no spike was measured
    if len(spike_times) == 0:
        spike_times = [None]
    
    return spike_times, "needed for map function"
