##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import thorns as th
from scipy.signal import savgol_filter
import peakutils as peak
import pandas as pd
pd.options.mode.chained_assignment = None

##### import functions
import functions.stimulation as stim
import functions.calculations as calc

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Frijns_2005 as frijns_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

# =============================================================================
# Calculate threshold
# =============================================================================
def get_threshold(model_name,
                  dt,
                  phase_duration,
                  delta,
                  amps_start_interval,
                  pulse_form = "mono",
                  stimulation_type = "extern",
                  time_before = 3*ms,
                  time_after = 3*ms,
                  add_noise = False,
                  print_progress = True,
                  run_number = 1):
    
    ##### add quantity to phase_duration
    phase_duration = float(phase_duration)*second
    
    ##### get model
    model = eval(model_name)
        
    ##### initialize model with given defaultclock dt
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### calculate runtime
    if pulse_form == "mono":
        runtime = time_before + phase_duration + time_after
    else:
        runtime = time_before + phase_duration*2 + time_after
    
    ##### calculate number of timesteps
    nof_timesteps = int(np.ceil(runtime/dt))
    
    ##### include noise
    if add_noise:
        np.random.seed()
        I_noise = np.transpose(np.transpose(np.random.normal(0, 1, (model.nof_comps,nof_timesteps)))*model.k_noise*model.noise_term)
    else:
        I_noise = np.zeros((model.nof_comps,nof_timesteps))
    
    ##### initializations
    threshold = 0*amp
    lower_border = amps_start_interval[0]
    upper_border = amps_start_interval[1]
    stim_amp = (upper_border-lower_border)/2
    amp_diff = upper_border - lower_border
    
    ##### adjust stimulus amplitude until required accuracy is obtained
    while amp_diff > delta:
        
        ##### print progress
        if print_progress: print("Duration: {} us; Run: {}; Stimulus amplitde: {} uA".format(np.round(phase_duration/us),run_number,np.round(stim_amp/uA,4)))
        
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
                                                    durations_bi = [phase_duration/second,0,phase_duration/second]*second)
    
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

    return threshold

# =============================================================================
#  Calculate conduction velocity
# =============================================================================
def get_conduction_velocity(model,
                            dt,
                            measurement_start_comp = 2,
                            measurement_end_comp = 6,
                            stimulation_type = "extern",
                            pulse_form = "bi",
                            time_after_stimulation = 1.5*ms,
                            stimulated_compartment = 2,
                            stim_amp = 2*uA,
                            phase_duration = 200*us,
                            nof_runs = 1):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time
    
    Parameters
    ----------
    model : time
        Lenght of one time step.
    dt : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_durations : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    start_interval : time
        Time until (first) pulse starts.
    delta : time
        Time which is still simulated after the end of the last pulse
    stimulation_type : amp/[k_noise]
        Is multiplied with k_noise.
    pulse_form : amp/[k_noise]
        Is multiplied with k_noise.
    
    Returns
    -------
    current matrix
        Gives back a vector of currents for each timestep
    runtime
        Gives back the duration of the simulation
    """
    
    ##### stochastic runs in case nof_runs > 1
    add_noise = False
    if nof_runs > 1: add_noise = True
    
    ##### calculate length of neuron part for measurement
    conduction_length = sum(model.compartment_lengths[measurement_start_comp:measurement_end_comp+1])
    
    ##### initialize neuron and state monitor
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### initialize vector to save conduction velocities
    conduction_velocity = [0]*nof_runs
    
    ##### stochastic runs
    for ii in range(0, nof_runs):
        
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    stimulation_type = stimulation_type,
                                                    pulse_form = pulse_form,
                                                    time_before = 2*ms,
                                                    time_after = time_after_stimulation,
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
        AP_time_start_comp = M.t[M.v[measurement_start_comp,:]-model.V_res == AP_amp_start_comp]
        
        ##### calculate point in time at AP end
        AP_amp_end_comp = max(M.v[measurement_end_comp,:]-model.V_res)
        AP_time_end_comp = M.t[M.v[measurement_end_comp,:]-model.V_res == AP_amp_end_comp]
        
        ##### calculate conduction velocity
        conduction_time = AP_time_end_comp - AP_time_start_comp
        conduction_velocity[ii] = conduction_length/conduction_time
        
    conduction_velocity = round(np.mean(conduction_velocity),3)*meter/second
        
    return conduction_velocity

# =============================================================================
#  Calculate single node respones
# =============================================================================
def get_single_node_response(model_name,
                             dt,
                             stim_amp,
                             phase_duration,
                             stimulation_type = "extern",
                             pulse_form = "bi",
                             time_before = 3*ms,
                             time_after = 2*ms,
                             stimulated_compartment = 4,
                             add_noise = True,
                             print_progress = True,
                             run_number = 0):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time
    
    Parameters
    ----------
    model : time
        Lenght of one time step.
    dt : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_durations : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    start_interval : time
        Time until (first) pulse starts.
    delta : time
        Time which is still simulated after the end of the last pulse
    stimulation_type : amp/[k_noise]
        Is multiplied with k_noise.
    pulse_form : amp/[k_noise]
        Is multiplied with k_noise.
    
    Returns
    -------
    current matrix
        Gives back a vector of currents for each timestep
    runtime
        Gives back the duration of the simulation
    """
    
    ##### add quantity to phase_duration
    phase_duration = float(phase_duration)*second
    stim_amp = float(stim_amp)*amp
    
    ##### get model
    model = eval(model_name)
        
    ##### initialize model with given defaultclock dt
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### print progress
    if print_progress: print("Duration: {} us; Run: {}; Stimulus amplitde: {} uA".format(np.round(phase_duration/us),run_number+1,np.round(stim_amp/uA,4)))
            
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
    
    return AP_amp/volt, AP_rise_time/second, AP_fall_time/second, latency/second, voltage_course, time_vector

# =============================================================================
#  Calculate cronaxie for a given rheobase
# =============================================================================
def get_chronaxie(model,
                  dt,
                  rheobase,
                  phase_duration_start_interval,
                  delta,
                  stimulation_type = "extern",
                  pulse_form = "mono",
                  time_before = 1*ms,
                  time_after = 1.5*ms,
                  print_progress = True):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    model : time
        Lenght of one time step.
    dt : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_durations : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    amps_start_interval : time
        Time until (first) pulse starts.
    delta : time
        Time which is still simulated after the end of the last pulse
    stimulation_type : amp/[k_noise]
        Is multiplied with k_noise.
    pulse_form : amp/[k_noise]
        Is multiplied with k_noise.
                
    Returns
    -------
    thresholds matrix
        Gives back a vector of currents for each timestep
    """
        
    ##### initialize model with given defaultclock dt
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    phase_duration_min = phase_duration_start_interval[0]
    phase_duration_max = phase_duration_start_interval[1]
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
        
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
                                                    ##### monophasic stimulation
                                                    amp_mono = -2*rheobase,
                                                    duration_mono = phase_duration,
                                                    ##### biphasic stimulation
                                                    amps_bi = [-2*rheobase/amp,2*rheobase/amp]*amp,
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
                           threshold = 0,
                           amp_masker = 0,
                           stimulation_type = "extern",
                           pulse_form = "mono",
                           phase_duration = 100*us,
                           time_before = 2*ms,
                           print_progress = True):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    model : time
        Lenght of one time step.
    dt : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_durations : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    amps_start_interval : time
        Time until (first) pulse starts.
    delta : time
        Time which is still simulated after the end of the last pulse
    stimulation_type : amp/[k_noise]
        Is multiplied with k_noise.
    pulse_form : amp/[k_noise]
        Is multiplied with k_noise.
                
    Returns
    -------
    thresholds matrix
        Gives back a vector of currents for each timestep
    """
    
    ##### add quantity to phase_duration, threshold and amp_masker
    phase_duration = float(phase_duration)*second
    if threshold != 0:
        threshold = float(threshold)*amp
    if amp_masker != 0:
        amp_masker = float(amp_masker)*amp
        
    ##### get model
    model = eval(model_name)
    
    ##### calculate theshold of model
    if threshold == 0:
        threshold = get_threshold(model_name = model_name,
                           dt = dt,
                           phase_duration = phase_duration,
                           amps_start_interval = [0,100]*uA,
                           delta = 0.0001*uA,
                           stimulation_type = stimulation_type,
                           pulse_form = pulse_form,
                           time_after = 3*ms,
                           print_progress = False)
    
    ##### amplitude of masker stimulus
    if amp_masker == 0:
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
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
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
                         threshold = 0,
                         amp_masker = 0,
                         stimulation_type = "extern",
                         pulse_form = "mono",
                         phase_duration = 100*us,
                         time_before = 3*ms,
                         print_progress = True):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    model : time
        Lenght of one time step.
    dt : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_durations : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    start_interval : time
        Time until (first) pulse starts.
    delta : time
        Time which is still simulated after the end of the last pulse
    stimulation_type : amp/[k_noise]
        Is multiplied with k_noise.
    pulse_form : amp/[k_noise]
        Is multiplied with k_noise.
                
    Returns
    -------
    min_required_amps matrix
        Gives back a vector of currents for each timestep
    """
    
    ##### add quantity to phase_duration, threshold and amp_masker
    inter_pulse_interval = float(inter_pulse_interval)*second
        
    ##### get model
    model = eval(model_name)
    
    ##### calculate theshold of model
    if threshold == 0:
        threshold = get_threshold(model_name = model_name,
                           dt = dt,
                           phase_duration = phase_duration,
                           amps_start_interval = [0,20]*uA,
                           delta = 0.0001*uA,
                           stimulation_type = stimulation_type,
                           pulse_form = pulse_form,
                           time_before = 2*ms,
                           time_after = 3*ms,
                           print_progress = False)
    
    ##### amplitude of masker stimulus (150% of threshold)
    if amp_masker == 0:
        amp_masker = 1.5 * threshold
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### initialize model and monitors
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
                
    ##### initializations
    min_amp_spiked = 0*amp
    lower_border = 0*amp
    upper_border = threshold * 10
    stim_amp = upper_border
    amp_diff = upper_border - lower_border
            
    ##### adjust stimulus amplitude until required accuracy is obtained
    while amp_diff > delta:
        
        ##### print progress
        if print_progress : print("1. Inter pulse interval: {} us; Amplitude of second stimulus: {} uA".format(round(inter_pulse_interval/us),np.round(stim_amp/uA,2)))
        
        ##### define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = time_before,
                                                                  time_after = 0*ms,
                                                                  ##### monophasic stimulation
                                                                  amp_mono = -amp_masker,
                                                                  duration_mono = phase_duration,
                                                                  ##### biphasic stimulation
                                                                  amps_bi = [-amp_masker/amp,amp_masker/amp]*amp,
                                                                  durations_bi = [phase_duration/second,0,phase_duration/second]*second)
        
        if print_progress : print("2. Inter pulse interval: {} us; Amplitude of second stimulus: {} uA".format(round(inter_pulse_interval/us),np.round(stim_amp/uA,2)))
        
        I_stim_2nd, runtime_2nd = stim.get_stimulus_current(model = model,
                                                            dt = dt,
                                                            stimulation_type = stimulation_type,
                                                            pulse_form = pulse_form,
                                                            time_before = inter_pulse_interval,
                                                            time_after = 3*ms,
                                                            ##### monophasic stimulation
                                                            amp_mono = -stim_amp,
                                                            duration_mono = phase_duration,
                                                            ##### biphasic stimulation
                                                            amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                            durations_bi = [phase_duration/second,0,phase_duration/second]*second)

        if print_progress : print("3. Inter pulse interval: {} us; Amplitude of second stimulus: {} uA".format(round(inter_pulse_interval/us),np.round(stim_amp/uA,2)))
        
        ##### combine stimuli
        I_stim = np.concatenate((I_stim_masker, I_stim_2nd), axis = 1)*amp
        runtime = runtime_masker + runtime_2nd
        
        ##### get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim), dt=dt)
        
        ##### reset state monitor
        restore('initialized')
        
        ##### run simulation
        run(runtime)
        
        if print_progress : print("4. Inter pulse interval: {} us; Amplitude of second stimulus: {} uA".format(round(inter_pulse_interval/us),np.round(stim_amp/uA,2)))
        
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

        if print_progress : print("5. Inter pulse interval: {} us; Amplitude of second stimulus: {} uA".format(round(inter_pulse_interval/us),np.round(stim_amp/uA,2)))
                
    return float(min_amp_spiked), float(threshold)

# =============================================================================
#  Calculate poststimulus time histogram (PSTH)
# =============================================================================
def post_stimulus_time_histogram(model_name,
                                 dt,
                                 pulses_per_second,
                                 stim_duration,
                                 stim_amp,
                                 stimulation_type,
                                 pulse_form,
                                 phase_duration,
                                 run_number):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    model : time
        Lenght of one time step.
    dt : string
        Describes, how the ANF is stimulated; either "internal" or "external" is possible
    phase_durations : string
        Describes, which pulses are used; either "mono" or "bi" is possible
    start_interval : time
        Time until (first) pulse starts.
    delta : time
        Time which is still simulated after the end of the last pulse
    stimulation_type : amp/[k_noise]
        Is multiplied with k_noise.
    pulse_form : amp/[k_noise]
        Is multiplied with k_noise.
                
    Returns
    -------
    min_required_amps matrix
        Gives back a vector of currents for each timestep
    """
    
    ##### add quantity to stim_amp and phase_duration
    stim_amp = float(stim_amp)*amp
    phase_duration = float(phase_duration)*second
    
    ##### get model
    model = eval(model_name)
    
    ##### set up the neuron
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    
    ##### load the parameters of the differential equations in the workspace
    exec(param_string)
    
    ##### initialize monitors
    M = StateMonitor(neuron, 'v', record=True)
    
    ##### save initialization of the monitor(s)
    store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]

    ##### calculate nof_pulses
    nof_pulses = round(pulses_per_second*stim_duration/second)
        
    ##### calculate inter_pulse_gap
    if pulse_form == "mono":
        inter_pulse_gap = (1e6/pulses_per_second - phase_duration/us)*us
    else:
        inter_pulse_gap = (1e6/pulses_per_second - phase_duration*2/us)*us
        
    ##### initialize pulse train dataframe
    spike_times = np.zeros(nof_pulses*2)
    
    ##### print progress
    print("Pulse rate: {} pps; Stimulus Amplitude: {} uA; Run: {}".format(pulses_per_second,np.round(stim_amp/uA,2),run_number))
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = dt,
                                                stimulation_type = stimulation_type,
                                                pulse_form = pulse_form,
                                                nof_pulses = nof_pulses,
                                                time_before = 0*ms,
                                                time_after = 0*ms,
                                                add_noise = True,
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
    
    return spike_times, "Hallo"
