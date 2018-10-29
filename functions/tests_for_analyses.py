##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import thorns as th
from scipy.signal import savgol_filter
from pytictoc import TicToc
import peakutils as peak
import pandas as pd
pd.options.mode.chained_assignment = None

##### import functions
import functions.stimulation as stim
import functions.calculations as calc
import functions.model_tests as test

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

# =============================================================================
#  Computational efficiency test
# =============================================================================
def computational_efficiency_test(model_names,
                                  dt,
                                  stimulus_duration,
                                  nof_runs):
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
    
    ##### get models
    models = [eval(model) for model in model_names]
    
    ##### initialize dataframe
    computation_times = pd.DataFrame(np.zeros((nof_runs, len(models))), columns  = [model.display_name for model in models])
    
    ##### loop over models
    for model in models:
        
        ##### loop over runs
        for ii in range(nof_runs):
                
            ##### set up the neuron
            neuron, param_string, model = model.set_up_model(dt = dt, model = model)
            
            ##### load the parameters of the differential equations in the workspace
            exec(param_string)
            
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


# =============================================================================
#  Calculate refractory periods for pulse trains
# =============================================================================
def get_refractory_periods_for_pulse_trains(model_name,
                                            dt,
                                            pulse_rate,
                                            phase_durations = 100*us,
                                            pulse_train_duration = 5*ms,
                                            delta = 1*us,
                                            threshold = 0,
                                            amp_masker = 0,
                                            stimulation_type = "extern",
                                            pulse_form = "mono",
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
        
    ##### get model
    model = eval(model_name)
    
    ##### calculate number of pulses
    nof_pulses = int(pulse_train_duration * pulse_rate/second)
    
    ##### calculate inter pulse gap
    inter_pulse_gap = pulse_train_duration/nof_pulses - sum(phase_durations)
    
    ##### calculate theshold of model
    if threshold == 0:
        threshold = test.get_threshold(model_name = model_name,
                                       dt = dt,
                                       phase_duration = phase_durations[0],
                                       amps_start_interval = [0,1000]*uA,
                                       delta = 0.01*uA,
                                       stimulation_type = stimulation_type,
                                       pulse_form = pulse_form,
                                       time_after = 3*ms,
                                       print_progress = print_progress)
    
    ##### amplitude of masker stimulus
    if amp_masker == 0:
        amp_masker = 1.5 * threshold
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    inter_train_gap_min = 0*ms
    inter_train_gap_max = 15*ms
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### thresholds for second spike that define the refractory periods
    stim_amp_arp = 4*threshold
    stim_amp_rrp = 1.01*threshold    

    ##### get absolute refractory period
    # initializations
    arp = 0*second
    lower_border = inter_train_gap_min.copy()
    upper_border = inter_train_gap_max.copy()
    inter_train_gap = (inter_train_gap_max-inter_train_gap_min)/2
    inter_train_gap_diff = upper_border - lower_border
    
    # initialize model with given defaultclock dt
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    # adjust stimulus amplitude until required accuracy is obtained
    while inter_train_gap_diff > delta:
        
        # print progress
        if print_progress: print("ARP: Model: {} us, Pulse rate: {}, Inter pulse gap: {} us".format(model_name, pulse_rate, np.round(inter_train_gap/us)))
        
        # define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = time_before,
                                                                  time_after = 0*ms,
                                                                  nof_pulses = nof_pulses,
                                                                  # monophasic stimulation
                                                                  amp_mono = -amp_masker,
                                                                  duration_mono = phase_durations[0],
                                                                  # biphasic stimulation
                                                                  amps_bi = [-amp_masker/amp,amp_masker/amp]*amp,
                                                                  durations_bi = phase_durations,
                                                                  inter_pulse_gap = inter_pulse_gap)
 
        I_stim_2nd, runtime_2nd = stim.get_stimulus_current(model = model,
                                                            dt = dt,
                                                            stimulation_type = stimulation_type,
                                                            pulse_form = pulse_form,
                                                            time_before = inter_train_gap,
                                                            time_after = 3*ms,
                                                            nof_pulses = nof_pulses,
                                                            # monophasic stimulation
                                                            amp_mono = -stim_amp_arp,
                                                            duration_mono = phase_durations[0],
                                                            # biphasic stimulation
                                                            amps_bi = [-stim_amp_arp/amp,stim_amp_arp/amp]*amp,
                                                            durations_bi = phase_durations,
                                                            inter_pulse_gap = inter_pulse_gap)
        
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
        nof_spikes = len(peak.indexes(M.v[comp_index,M.t > runtime_masker+inter_train_gap], thres = model.V_res + 60*mV, thres_abs=True))
        
        if nof_spikes > 0:
            arp = inter_train_gap
            upper_border = inter_train_gap
            inter_train_gap = (inter_train_gap + lower_border)/2
        else:
            lower_border = inter_train_gap
            inter_train_gap = (inter_train_gap + upper_border)/2
            
        inter_train_gap_diff = upper_border - lower_border
                
    ##### get relative refractory period
    # initializations
    rrp = 0*second
    lower_border = arp.copy()
    upper_border = inter_train_gap_max.copy()
    inter_train_gap = (inter_train_gap_max-inter_train_gap_min)/2
    inter_train_gap_diff = upper_border - lower_border
    
    # adjust stimulus amplitude until required accuracy is obtained
    while inter_train_gap_diff > delta:
                
        # print progress
        if print_progress: print("RRP: Phase duration: {} us, Inter pulse gap: {} us".format(np.round(phase_durations[0]/us), np.round(inter_train_gap/us)))
        
        # define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = time_before,
                                                                  time_after = 0*ms,
                                                                  nof_pulses = nof_pulses,
                                                                  # monophasic stimulation
                                                                  amp_mono = -amp_masker,
                                                                  duration_mono = phase_durations[0],
                                                                  # biphasic stimulation
                                                                  amps_bi = [-amp_masker/amp,amp_masker/amp]*amp,
                                                                  durations_bi = phase_durations,
                                                                  inter_pulse_gap = inter_pulse_gap)
 
        I_stim_2nd, runtime_2nd = stim.get_stimulus_current(model = model,
                                                            dt = dt,
                                                            stimulation_type = stimulation_type,
                                                            pulse_form = pulse_form,
                                                            time_before = inter_train_gap,
                                                            time_after = 3*ms,
                                                            nof_pulses = nof_pulses,
                                                            # monophasic stimulation
                                                            amp_mono = -stim_amp_rrp,
                                                            duration_mono = phase_durations[0],
                                                            # biphasic stimulation
                                                            amps_bi = [-stim_amp_rrp/amp,stim_amp_rrp/amp]*amp,
                                                            durations_bi = phase_durations,
                                                            inter_pulse_gap = inter_pulse_gap)
        
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
        nof_spikes = len(peak.indexes(M.v[comp_index,M.t > runtime_masker+inter_train_gap], thres = model.V_res + 60*mV, thres_abs=True))
        
        if nof_spikes > 0:
            rrp = inter_train_gap
            upper_border = inter_train_gap
            inter_train_gap = (inter_train_gap + lower_border)/2
        else:
            lower_border = inter_train_gap
            inter_train_gap = (inter_train_gap + upper_border)/2
            
        inter_train_gap_diff = upper_border - lower_border
                            
    return float(arp), float(rrp)
