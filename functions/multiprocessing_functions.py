##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import thorns as th
import numpy as np
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
# Calculate thresholds: multiprocessing function
# =============================================================================
def get_threshold(model_name,
                  dt,
                  phase_duration,
                  run_number,
                  delta,
                  pulse_form,
                  stimulation_type,
                  amps_start_interval,
                  time_before,
                  time_after,
                  add_noise,
                  print_progress = True):
    
    ##### add quantity to phase_duration
    phase_duration = phase_duration*us
    
    ##### get model
    model = eval(model_name)
        
    ##### initialize model with given defaultclock dt
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    stim_amps_min = amps_start_interval[0]
    stim_amps_max = amps_start_interval[1]
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### calculate runtime
    if pulse_form == "mono":
        runtime = time_before + phase_duration + time_after
    else:
        runtime = time_before + phase_durations*2 + time_after
    
    ##### calculate number of timesteps
    nof_timesteps = int(np.ceil(runtime/dt))
    
    ##### include noise
    if add_noise:
        I_noise = np.transpose(np.transpose(np.random.normal(0, 1, (model.nof_comps,nof_timesteps)))*model.k_noise*model.noise_term)
    else:
        I_noise = np.zeros((model.nof_comps,nof_timesteps))
    
    ##### initializations
    threshold = 0*amp
    lower_border = stim_amps_min
    upper_border = stim_amps_max
    stim_amp = (stim_amps_max-stim_amps_min)/2
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

