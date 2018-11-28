##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import thorns as th
from scipy.signal import savgol_filter
import peakutils as peak
import matplotlib.pyplot as plt
import h5py
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

# =============================================================================
# Calculate threshold for given potential distribution
# =============================================================================
def get_threshold_for_pot_dist(model_name,
                               dt,
                               h5py_path,
                               elec_nr,
                               phase_duration,
                               delta,
                               start_amp,
                               amps_start_interval,
                               inter_phase_gap = 0*us,
                               pulse_form = "mono",
                               time_before = 2*ms,
                               time_after = 3*ms,
                               nof_pulses = 1,
                               inter_pulse_gap = 1*ms,
                               add_noise = False,
                               print_progress = True,
                               neuron_number = 0,
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
    amps_start_interval : list of currents of length two
        First value gives lower border of expected threshold; second value gives upper border
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
    
    ##### add quantity to phase_duration, inter_phase_gap and inter_pulse_gap
    phase_duration = float(phase_duration)*second
    inter_phase_gap = float(inter_phase_gap)*second
    inter_pulse_gap = float(inter_pulse_gap)*second
    
    ##### get model
    model = eval(model_name)
    
    ##### open h5py file
    with h5py.File(h5py_path, 'r') as potential_data:

        ##### break down 3D coordinates to 1D
        distances = calc.coordinates_to_1D(x = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,0],
                                           y = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,1],
                                           z = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,2])
        
        ##### get potential distribution
        potentials = potential_data['neuron{}'.format(neuron_number)]["potentials"][:,elec_nr]*1e-3
    
    ##### get potentials at compartment middle points by intepolation
    start_potentials = calc.interpolate_potentials(potentials = potentials,
                                                   pot_distances = distances,
                                                   comp_distances = np.cumsum(model.distance_comps_middle)/meter,
                                                   comp_lenghts = model.compartment_lengths/meter,
                                                   method = "linear")
    
    ##### initialize model (no parameter was changed)
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][-3]
    
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
    
    ##### initializations
    threshold = 0*amp
    lower_border = amps_start_interval[0]
    upper_border = amps_start_interval[1]
    stim_amp = start_amp
    potentials_at_comps = start_potentials
    amp_diff = upper_border - lower_border
    
    ##### adjust stimulus amplitude until required accuracy is obtained
    while amp_diff > delta:
        
        ##### print progress
        if print_progress: print("Model: {}; Duration: {} us; Run: {}; Stimulus amplitde: {} uA".format(model_name,
                                 np.round(phase_duration/us),run_number+1,np.round(stim_amp/uA,4)))
        
        ##### adjust potentials according to the stimulation current amplitude
        potentials_at_comps = start_potentials * stim_amp/start_amp
        
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stim_current_for_given_potentials(model = model,
                                                                     dt = dt,
                                                                     V  = potentials_at_comps*volt,
                                                                     pulse_form = pulse_form,
                                                                     add_noise = add_noise,
                                                                     time_before = time_before,
                                                                     time_after = time_after,
                                                                     nof_pulses = nof_pulses,
                                                                     ##### monophasic stimulation
                                                                     duration_mono = phase_duration,
                                                                     ##### biphasic stimulation
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

    return {"threshold" : threshold}

# =============================================================================
# Calculate required stimulus amplitude for certain firing efficiency and for a
# given potential distribution
# =============================================================================
def get_threshold_for_fire_eff(model_name,
                               dt,
                               h5py_path,
                               elec_nr,
                               fire_eff_desired,
                               phase_duration,
                               delta,
                               start_amp,
                               amps_start_interval,
                               inter_phase_gap = 0*us,
                               pulse_form = "mono",
                               time_before = 2*ms,
                               time_after = 3*ms,
                               nof_pulses = 1,
                               inter_pulse_gap = 1*ms,
                               add_noise = False,
                               print_progress = True,
                               neuron_number = 0,
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
    amps_start_interval : list of currents of length two
        First value gives lower border of expected threshold; second value gives upper border
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
        
    ##### add quantity to phase_duration, inter_phase_gap and inter_pulse_gap
    phase_duration = float(phase_duration)*second
    inter_phase_gap = float(inter_phase_gap)*second
    inter_pulse_gap = float(inter_pulse_gap)*second
    
    ##### get model
    model = eval(model_name)
    
    ##### open h5py file
    with h5py.File(h5py_path, 'r') as potential_data:

        ##### break down 3D coordinates to 1D
        distances = calc.coordinates_to_1D(x = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,0],
                                           y = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,1],
                                           z = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,2])
        
        ##### get potential distribution
        potentials = potential_data['neuron{}'.format(neuron_number)]["potentials"][:,elec_nr]*1e-3
    
    ##### get potentials at compartment middle points by intepolation
    start_potentials = calc.interpolate_potentials(potentials = potentials,
                                                   pot_distances = distances,
                                                   comp_distances = np.cumsum(model.distance_comps_middle)/meter,
                                                   comp_lenghts = model.compartment_lengths/meter,                                                   
                                                   method = "linear")
    
    ##### initialize model (no parameter was changed)
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][-3]
    
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
    
    ##### initializations
    fire_eff = 0
    threshold = 0*amp
    lower_border = amps_start_interval[0]
    upper_border = amps_start_interval[1]
    stim_amp = start_amp
    potentials_at_comps = start_potentials
    amp_diff = upper_border - lower_border
    
    ##### adjust stimulus amplitude until required accuracy is obtained
    while amp_diff > delta:
        
        ##### print progress
        if print_progress: print("Model: {}; Duration: {} us; Run: {}; Firing efficiency: {} Stimulus amplitde: {} uA".format(model_name,
                                 np.round(phase_duration/us), run_number+1, fire_eff, np.round(stim_amp/uA,4)))
        
        ##### adjust potentials according to the stimulation current amplitude
        potentials_at_comps = start_potentials * stim_amp/start_amp
        
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stim_current_for_given_potentials(model = model,
                                                                     dt = dt,
                                                                     V  = potentials_at_comps*volt,
                                                                     pulse_form = pulse_form,
                                                                     add_noise = add_noise,
                                                                     time_before = time_before,
                                                                     time_after = time_after,
                                                                     nof_pulses = nof_pulses,
                                                                     ##### monophasic stimulation
                                                                     duration_mono = phase_duration,
                                                                     ##### biphasic stimulation
                                                                     durations_bi = [phase_duration/second,inter_phase_gap/second,phase_duration/second]*second,
                                                                     ##### multiple pulses / pulse trains
                                                                     inter_pulse_gap = inter_pulse_gap)
        
        ##### get TimedArray of stimulus currents and run simulation
        stimulus = TimedArray(np.transpose(I_stim + I_noise), dt=dt)
        
        ##### reset state monitor
        restore('initialized')
        
        ##### run simulation
        run(runtime)
        
        ##### calculate firing efficiency
        spikes = peak.indexes(savgol_filter(M.v[comp_index,:], 51,3), thres = (model.V_res + 60*mV)/volt, thres_abs=True, min_dist=0.2*1e3)
        fire_eff = len(spikes)/nof_pulses
        
        ##### test if there was a spike
        if fire_eff >= fire_eff_desired:
            threshold = stim_amp
            upper_border = stim_amp
            stim_amp = (stim_amp + lower_border)/2
        else:
            lower_border = stim_amp
            stim_amp = (stim_amp + upper_border)/2
            
        amp_diff = upper_border - lower_border

    return {"threshold" : threshold}

# =============================================================================
#  Calculate spike times for pulse trains and given potential distribution
# =============================================================================
def get_spike_trains(model_name,
                     dt,
                     h5py_path,
                     potential_multiplier,
                     elec_nr,
                     nof_pulses,
                     inter_pulse_gap,
                     pulse_form = "bi",
                     phase_duration = 50*us,
                     inter_phase_gap = 0*us,
                     time_before = 2*ms,
                     time_after = 10*ms,
                     print_progress = True,
                     add_noise = True,
                     neuron_number = 0,
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
    
    ##### add quantity to phase_duration, inter_phase_gap and inter_pulse_gap
    phase_duration = float(phase_duration)*second
    inter_phase_gap = float(inter_phase_gap)*second
    inter_pulse_gap = float(inter_pulse_gap)*second
    
    ##### get model
    model = eval(model_name)
    
    ##### open h5py file
    with h5py.File(h5py_path, 'r') as potential_data:

        ##### break down 3D coordinates to 1D
        distances = calc.coordinates_to_1D(x = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,0],
                                           y = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,1],
                                           z = potential_data['neuron{}'.format(neuron_number)]["coordinates"][:,2])
        
        ##### get potential distribution
        potentials = potential_data['neuron{}'.format(neuron_number)]["potentials"][:,elec_nr]*1e-3 * potential_multiplier
    
    ##### get potentials at compartment middle points by intepolation
    potentials_at_comps = calc.interpolate_potentials(potentials = potentials,
                                                      pot_distances = distances,
                                                      comp_distances = np.cumsum(model.distance_comps_middle)/meter,
                                                      comp_lenghts = model.compartment_lengths/meter,
                                                      method = "linear")
    
    ##### set up the neuron
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    
    ##### load the parameters of the differential equations in the workspace
    exec(param_string)
    
    ##### initialize monitors
    M = StateMonitor(neuron, 'v', record=True)
    
    ##### save initialization of the monitor(s)
    store('initialized')
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][-3]
        
    ##### print progress
    if print_progress: print("Model: {}; Fiber: {};".format(model_name,neuron_number))
    
    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stim_current_for_given_potentials(model = model,
                                                                 dt = dt,
                                                                 V  = potentials_at_comps*volt,
                                                                 pulse_form = pulse_form,
                                                                 add_noise = add_noise,
                                                                 time_before = time_before,
                                                                 time_after = time_after,
                                                                 nof_pulses = nof_pulses,
                                                                 ##### monophasic stimulation
                                                                 duration_mono = phase_duration,
                                                                 ##### biphasic stimulation
                                                                 durations_bi = [phase_duration/second,inter_phase_gap/second,phase_duration/second]*second,
                                                                 ##### multiple pulses / pulse trains
                                                                 inter_pulse_gap = inter_pulse_gap)

    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = dt)
    
    ##### reset state monitor
    restore('initialized')
            
    ##### run simulation
    run(runtime)
    
    ##### get spike times
    spike_times = M.t[peak.indexes(savgol_filter(M.v[comp_index,:], 51,3), thres = (model.V_res + 60*mV)/volt, thres_abs=True, min_dist=0.2*1e3)]/second
    spike_times = spike_times.tolist()
    
#    ##### plot
#    fig = plt.figure("Model: {}, Fiber: {}, Spikes: {}".format(model_name, neuron_number, len(spike_times)))
#    axes = fig.add_subplot(1, 1, 1)
#    axes.plot(M.t/ms, M.v[comp_index,:]/mV)
#    axes.plot(M.t/ms, savgol_filter(M.v[comp_index,:]/mV, 51,3))
#    axes.scatter([spike_times[ii]*1e3 for ii in range(len(spike_times))], [model.V_res/mV + 100 for ii in range(len(spike_times))], color = "red")
    
    return {"spikes" : spike_times,
            "duration" : runtime/second}
