##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

##### import functions
import functions.stimulation as stim
import functions.calculations as calc

# =============================================================================
#  Calculate single node respones
# =============================================================================
def get_single_node_response(model,
                             dt,
                             param_1,
                             param_1_ratios,
                             param_2,
                             param_2_ratios,
                             stimulation_type,
                             pulse_form,
                             time_after_stimulation,
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
    
    model = [rattay_01, smit_10, smit_09]
    dt = dt
    param_1 = "model"
    param_1_ratios = [0.8, 1.0, 1.2]
    param_2 = "compartment_diameters"
    param_2_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    stimulation_type = "extern"
    pulse_form = "bi"
    time_after_stimulation = 1.5*ms
    stim_amp = 2*uA
    phase_duration = 200*us
    nof_runs = 10
    
    ##### initializations
    add_noise = False
    max_nof_timesteps = 0
    param_info = {'param_1_display_name': 0,
                  'param_1_values': 0,
                  'param_2_display_name': 0,
                  'param_2_values': 0,
                  'V_res': [0]}
    
    ##### Test if model entry is a list
    if param_1 == "model" or param_2 == "model":
        if not isinstance(model, (list,)):
            print("model must be a list, if one of the parameters is set to 'model'")
            # return
        # initialize V_res of param_info dictionary
        param_info["V_res"] = np.zeros_like(model)
    else:
        ##### define neuron and state monitor
        if not isinstance(model, (list,)):
            model = [model]
        neuron, param_string = model[0].set_up_model(dt = dt, model = model[0], model_name = "model[0]")
        exec(param_string)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
         
    ##### parameter 1
    if hasattr(model[0], param_1):
        # get display name for plots and dataframe
        param_1_display_name = param_1
        # save the number of observations for parameter 1
        length_param_1 = len(param_1_ratios)
        # save the original value of the parameter
        param_1_original_value = np.zeros_like(model)
        for ii in range(0,len(model)):
            param_1_original_value[ii] = eval(f"model[ii].{param_1}")

    elif param_1 == "model":
        # get display name for plots and dataframe
        param_1_display_name = "model name"
        # save the number of observations for parameter 1
        length_param_1 = len(model)

    elif param_1 == "stochastic_runs":
        # include noise
        add_noise = True
        # get display name for plots and dataframe
        param_1_display_name = "run"
        # save the number of observations for parameter 1
        length_param_1 = nof_runs
    
    elif param_1 == "stim_amp":
        # save the original stim_amp value
        stim_amp_original_value = stim_amp
        # get display name for plots and dataframe
        param_1_display_name = "stimulus amplitude (uA)"
        # save the number of observations for parameter 1
        length_param_1 = len(param_1_ratios)

    elif param_1 == "phase_duration":
        # save the original phase duration value
        phase_duration_original_value = phase_duration
        # get display name for plots and dataframe
        param_1_display_name = "phase duration (us)"
        # save the number of observations for parameter 1
        length_param_1 = len(param_1_ratios)
        # calculate maximum timesteps needed
        if pulse_form == "mono":
            max_runtime = max(param_1_ratios)*phase_duration_original_value + time_after_stimulation
        else:
            max_runtime = max(param_1_ratios)*phase_duration_original_value*2 + time_after_stimulation
        max_nof_timesteps = int(np.ceil(max_runtime/dt))

    else:
        # print error message for wrong entry
        print("param_1 has to be either a model attribute or one of: 'model', 'stochastic_runs', 'stim_amp' and 'phase_duration")
        #return
    
    ##### parameter 2
    if hasattr(model[0], param_2):
        # get display name for plots and dataframe
        param_2_display_name = param_2
        # save the number of observations for parameter 1
        length_param_2 = len(param_2_ratios)
        # save the original value of the parameter
        param_2_original_value = np.zeros_like(model)
        for ii in range(0,len(model)):
            param_2_original_value[ii] = eval(f"model[ii].{param_2}")

    elif param_2 == "model":
        # get display name for plots and dataframe
        param_2_display_name = "model name"
        # save the number of observations for parameter 1
        length_param_2 = len(model)

    elif param_2 == "stochastic_runs":
        # include noise
        add_noise = True
        # get display name for plots and dataframe
        param_2_display_name = "run"
        # save the number of observations for parameter 1
        length_param_2 = nof_runs
        
    elif param_2 == "stim_amp":
        # save the original stim_amp value
        stim_amp_original_value = stim_amp
        # get display name for plots and dataframe
        param_2_display_name = "stimulus amplitude (uA)"
        # save the number of observations for parameter 1
        length_param_2 = len(param_2_ratios)

    elif param_2 == "phase_duration":
        # save the original phase duration value
        phase_duration_original_value = phase_duration
        # get display name for plots and dataframe
        param_2_display_name = "phase duration (us)"
        # save the number of observations for parameter 1
        length_param_2 = len(param_2_ratios)
        if pulse_form == "mono":
            max_runtime = max(param_1_ratios)*phase_duration_original_value + time_after_stimulation
        else:
            max_runtime = max(param_2_ratios)*phase_duration_original_value*2 + time_after_stimulation
        max_nof_timesteps = int(np.ceil(max_runtime/dt))

    else:
        # print error message for wrong entry
        print("param_2 has to be either a model parameter, or one of: 'model', 'stochastic_runs', 'stim_amp' and 'phase_duration")
        #return
    
    ##### initialize dataframe for measurements
    col_names = [param_1_display_name, param_2_display_name, "AP height (mV)","AP peak time","AP start time","AP end time","rise time (ms)","fall time (ms)","latency (ms)"]
    node_response_data = pd.DataFrame(np.zeros((length_param_1*length_param_2, len(col_names))), columns = col_names)
    
    ##### compartments for measurements
    comp_index = np.where(model[0].structure == 2)[0][10]
    
    # initialize length of dict entry parameter values
    param_info["param_1_values"] = np.zeros(length_param_1)
    param_info["param_2_values"] = np.zeros(length_param_2)
    
    ##### loop over parameter 1
    for ii in range(0, length_param_1):
        
        if hasattr(model[0], param_1):
            if not param_2 == "model":
                ##### adjust model parameter
                exec(f"model[0].{param_1} = param_1_ratios[ii]*param_1_original_value[0]")
                ##### set up neuron with adjusted model parameter
                neuron, param_string = model[0].set_up_model(dt = dt, model = model[0], model_name = "model[0]")
                exec(param_string)
                M = StateMonitor(neuron, 'v', record=True)
                store('initialized')
        
        elif param_1 == "model":
            ##### set up neuron for actual model
            neuron, param_string = model[ii].set_up_model(dt = dt, model = model[ii], model_name = "model[ii]")
            exec(param_string)
            M = StateMonitor(neuron, 'v', record=True)
            store('initialized')
            ##### compartments for measurements
            comp_index = np.where(model[ii].structure == 2)[0][10]
                    
        elif param_1 == "stim_amp":
            ##### set stimulus amplitude for actual iteration
            stim_amp = stim_amp_original_value*param_1_ratios[ii]
        
        elif param_1 == "phase_duration":
            ##### set phase duration for actual iteration
            phase_duration = phase_duration_original_value*param_1_ratios[ii]
        
        ##### loop over parameter 2
        for jj in range(0,length_param_2):
            
            if hasattr(model[0], param_2):
               if not param_1 == "model":
                    ##### adjust model parameter
                    exec(f"model[0].{param_2} = param_2_ratios[jj]*param_2_original_value[0]")
                    ##### set up neuron with adjusted model parameter
                    neuron, param_string = model[0].set_up_model(dt = dt, model = model[0], model_name = "model[0]")
                    exec(param_string)
                    M = StateMonitor(neuron, 'v', record=True)
                    store('initialized')
               else:
                    exec(f"model[ii].{param_2} = param_2_ratios[jj]*param_2_original_value[ii]")
                    ##### set up neuron again with adjusted model parameter
                    neuron, param_string = model[ii].set_up_model(dt = dt, model = model[ii], model_name = "model[ii]")
                    exec(param_string)
                    M = StateMonitor(neuron, 'v', record=True)
                    store('initialized')
            
            elif param_2 == "model":
                if hasattr(model[0], param_1):
                    ##### adjust model parameter
                    exec(f"model[jj].{param_1} = param_1_ratios[ii]*param_1_original_value[jj]")
                ##### set up neuron for actual model
                neuron, param_string = model[jj].set_up_model(dt = dt, model = model[jj], model_name = "model[jj]")
                exec(param_string)
                M = StateMonitor(neuron, 'v', record=True)
                store('initialized')
                ##### compartments for measurements
                comp_index = np.where(model[jj].structure == 2)[0][10]
                            
            elif param_2 == "stim_amp":
                ##### set stimulus amplitude for actual iteration
                stim_amp = stim_amp_original_value*param_2_ratios[jj]
            
            elif param_2 == "phase_duration":
                ##### set phase duration for actual iteration
                phase_duration = phase_duration_original_value*param_2_ratios[jj]
                
            ##### save actual model type in model_type
            if param_1 == "model":
                model_type = model[ii]
            elif param_2 == "model":
                model_type = model[jj]
            else:
                model_type = model[0]
            
            ##### define how the ANF is stimulated
            I_stim, runtime = stim.get_stimulus_current(model = model_type,
                                                        dt = dt,
                                                        stimulation_type = stimulation_type,
                                                        pulse_form = pulse_form,
                                                        time_after = time_after_stimulation,
                                                        add_noise = add_noise,
                                                        ##### monophasic stimulation
                                                        amp_mono = stim_amp,
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
            
            ##### write results in table
            AP_amp = max(M.v[comp_index,:]-model_type.V_res)
            AP_time = M.t[M.v[comp_index,:]-model_type.V_res == AP_amp]
            if any(M.t<AP_time):
                AP_start_time = M.t[np.argmin(abs(M.v[comp_index,np.where(M.t<AP_time)[0]]-model_type.V_res - 0.1*AP_amp))]
            else:
                AP_start_time = 0*ms
            if any(M.t>AP_time):
                AP_end_time = M.t[np.where(M.t>AP_time)[0]][np.argmin(abs(M.v[comp_index,np.where(M.t>AP_time)[0]]-model_type.V_res - 0.1*AP_amp))]
            else:
                AP_end_time = 0*ms
            
            if hasattr(model_type, param_1): node_response_data[param_1_display_name][ii*length_param_2+jj] = param_1_ratios[ii]
            elif param_1 == "model":
                node_response_data[param_1_display_name][ii*length_param_2+jj] = model_type.display_name
                param_info["V_res"][ii] = model_type.V_res
            elif param_1 == "stochastic_runs": node_response_data[param_1_display_name][ii*length_param_2+jj] = ii
            elif param_1 == "stim_amp": node_response_data[param_1_display_name][ii*length_param_2+jj] = round(stim_amp/uA,2)
            elif param_1 == "phase_duration": node_response_data[param_1_display_name][ii*length_param_2+jj] = round(phase_duration/us,2)
            
            if hasattr(model_type, param_2): node_response_data[param_2_display_name][ii*length_param_2+jj] = param_2_ratios[jj]
            elif param_2 == "model":
                node_response_data[param_2_display_name][ii*length_param_2+jj] = model_type.display_name
                param_info["V_res"][jj] = model_type.V_res
            elif param_2 == "stochastic_runs": node_response_data[param_2_display_name][ii*length_param_2+jj] = jj
            elif param_2 == "stim_amp": node_response_data[param_2_display_name][ii*length_param_2+jj] = round(stim_amp/uA,2)
            elif param_2 == "phase_duration": node_response_data[param_2_display_name][ii*length_param_2+jj] = round(phase_duration/us,2)
            
            node_response_data["AP height (mV)"][ii*length_param_2+jj] = AP_amp/mV
            node_response_data["AP peak time"][ii*length_param_2+jj] = AP_time/ms
            node_response_data["AP start time"][ii*length_param_2+jj] = AP_start_time/ms
            node_response_data["AP end time"][ii*length_param_2+jj] = AP_end_time/ms
            
            ##### Write current parametersin info dictionary
            param_info["param_1_values"][ii] = node_response_data[param_1_display_name][ii*length_param_2+jj]
            param_info["param_2_values"][jj] = node_response_data[param_2_display_name][ii*length_param_2+jj]

            ##### print progress
            print(f"{param_1_display_name}: {ii+1}/{length_param_1}; {param_2_display_name}: {jj+1}/{length_param_2}")

            ##### save voltage course of single compartment for plotting
            if ii == jj == 0:
                voltage_data = np.zeros((length_param_1*length_param_2,max(max_nof_timesteps,np.shape(M.v)[1])))
            voltage_data[length_param_2*ii+jj,0:np.shape(M.v)[1]] = M.v[comp_index, :]/mV
    
    ##### calculate remaining single node response data
    node_response_data["rise time (ms)"] = node_response_data["AP peak time"] - node_response_data["AP start time"]
    node_response_data["fall time (ms)"] = node_response_data["AP end time"] - node_response_data["AP peak time"]
    node_response_data["latency (ms)"] = node_response_data["AP peak time"]
    
    ##### exclude runs where no AP was elicited or where no start or end time could be calculated
    node_response_data = node_response_data[node_response_data["AP height (mV)"] > 60]
    node_response_data = node_response_data[node_response_data["AP start time"] > 0]
    node_response_data = node_response_data[node_response_data["AP end time"] > 0]
    
    ##### sum up relevant single node response properties in dataframe
    if param_1 == "stochastic_runs":
        node_response_data[param_1_display_name] = node_response_data[param_1_display_name].astype(int)
        node_response_data_summary = node_response_data.groupby([param_2_display_name])["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"].mean().reset_index()
        node_response_data_summary["jitter (ms)"] = node_response_data.groupby([param_2_display_name])["latency (ms)"].std()

    elif param_2 == "stochastic_runs":
        node_response_data[param_2_display_name] = node_response_data[param_2_display_name].astype(int)
        node_response_data_summary = node_response_data.groupby([param_1_display_name])["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"].mean().reset_index()
        node_response_data_summary["jitter (ms)"] = node_response_data.groupby([param_1_display_name])["latency (ms)"].std()
        
    else:
        node_response_data_summary = node_response_data[[param_1_display_name, param_2_display_name, "AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]]

    ##### Write remaining information in param_info dictionary
    param_info["param_1_display_name"] = param_1_display_name
    param_info["param_2_display_name"] = param_2_display_name

    ##### reset the adjusted parameter
    if hasattr(model[0], param_1):
        for ii in range(0, len(model)):
            exec(f"model[ii].{param_1} = param_1_original_value[ii]")
    if hasattr(model[0], param_2):
        for ii in range(0, len(model)):
            exec(f"model[ii].{param_2} = param_2_original_value[ii]")
            
    ##### save time_vector
    time_vector = M.t
        
    ##### reset neuron
    for ii in range(0, len(model)):
        neuron, param_string = model[ii].set_up_model(dt = dt, model = model[ii], model_name = "model[ii]")
        exec(param_string)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
    
    return voltage_data, node_response_data_summary, time_vector, param_info

# =============================================================================
#  Calculate phase duration curve / thresholds for certain phase_durations
# =============================================================================
def get_strength_duration_curve(model,
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
    
    ##### test if phase_durations is a list or array, if not convert it to a list
    if not isinstance(phase_durations, (list,)) and not isinstance(phase_durations, (np.ndarray,)):
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
                                                        amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                        durations_bi = [phase_durations[ii]/second,0,phase_durations[ii]/second]*second)
        
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
        