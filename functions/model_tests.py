##### import packages
from brian2 import *
import numpy as np
import pandas as pd

##### import functions
import functions.stimulation as stim

# =============================================================================
#  Calculate single node respones
# =============================================================================
def get_single_node_response(model,
                             dt,
                             test_param_type,
                             test_param_display_name,
                             test_param_values,
                             stimulation_type,
                             pulse_form,
                             stim_amp,
                             phase_duration,
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
    start_intervall : time
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
    
    ##### current amplitudes
    current_amps = np.array([-0.5, -0.7, -1.2, -2, -10])*uA
    
    ##### initialize dataset for measurements
    col_names = [test_param_display_name,"AP height (mV)","AP peak time","AP start time","AP end time","rise time (ms)","fall time (ms)","latency (ms)","jitter"]
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

# =============================================================================
#  Calculate phase duration curve / thresholds for certain phase_durations
# =============================================================================
def get_phase_duration_curve(model,
                             dt,
                             phase_durations,
                             start_intervall,
                             delta,
                             stimulation_type = "extern",
                             pulse_form = "mono"):
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
    start_intervall : time
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
    
    ##### test if phase_durations is a list, if not convert it to a list
    if not isinstance(phase_durations, (list,)):
        phase_durations = [phase_durations]
        
    ##### initialize model with given defaultclock dt
    neuron, param_string = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### initialize vector for minimum required stimulus current amplitudes
    min_required_amps = np.zeros_like(phase_durations/second)*amp
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    stim_amps_min = start_intervall[0]
    stim_amps_max = start_intervall[1]
    
    ##### start amplitde for first run
    start_amp = (stim_amps_max-stim_amps_min)/2
    
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
                                                        stimulation_type = stimulation_type,
                                                        pulse_form = pulse_form,
                                                        ##### monophasic stimulation
                                                        amp_mono = -stim_amp,
                                                        duration_mono = phase_durations[ii],
                                                        ##### biphasic stimulation
                                                        amps_bi = [-stim_amp,stim_amp],
                                                        durations_bi = [phase_durations[ii],0*second,phase_durations[ii]])
        
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
        
    return min_required_amps
        