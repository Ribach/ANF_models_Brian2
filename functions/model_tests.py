##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
from scipy.signal import savgol_filter
import peakutils as peak
import pandas as pd
pd.options.mode.chained_assignment = None

##### import functions
import functions.stimulation as stim
import functions.calculations as calc

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
def get_single_node_response(model,
                             dt,
                             param_1,
                             param_1_ratios = [0.6, 0.8, 1, 2, 3],
                             param_2 = "stochastic_runs",
                             param_2_ratios = [0.6, 0.8, 1, 2, 3],
                             stimulation_type = "extern",
                             pulse_form = "bi",
                             time_after_stimulation = 1.5*ms,
                             stimulated_compartment = 4,
                             stim_amp = 2*uA,
                             phase_duration = 100*us,
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
    
    ##### initializations
    add_noise = False
    if pulse_form == "mono": max_runtime = phase_duration + time_after_stimulation
    else: max_runtime = phase_duration*2 + time_after_stimulation
    max_nof_timesteps = int(np.ceil(max_runtime/dt))
    
    ##### Test if model entry is a list
    if param_1 == "model" or param_2 == "model":
        if not isinstance(model, (list,)):
            print("model must be a list, if one of the parameters is set to 'model'")
            return
    else:
        ##### define neuron and state monitor
        if not isinstance(model, (list,)):
            model = [model]
        neuron, param_string, model[0] = model[0].set_up_model(dt = dt, model = model[0], model_name = "model[0]")
        exec(param_string)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
         
    ##### parameter 1
    if hasattr(model[0], param_1):
        # get display name for plots and dataframe
        param_1_display_name = param_1.replace("_", " ") + " ratios"
        # save the number of observations for parameter 1
        length_param_1 = len(param_1_ratios)
        # save the original value of the parameter
        param_1_original_value = np.zeros_like(model)
        if param_2 == "model":
            for ii in range(0,len(model)):
                param_1_original_value[ii] = eval(f"model[ii].{param_1}")
        else:
            param_1_original_value = [eval(f"model[0].{param_1}")]
        
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
        return
    
    ##### parameter 2
    if hasattr(model[0], param_2):
        # get display name for plots and dataframe
        param_2_display_name = param_2.replace("_", " ") + " ratios"
        # save the number of observations for parameter 1
        length_param_2 = len(param_2_ratios)
        # save the original value of the parameter
        param_2_original_value = np.zeros_like(model)
        if param_1 == "model":
            for ii in range(0,len(model)):
                param_2_original_value[ii] = eval(f"model[ii].{param_2}")
        else:
            param_2_original_value = [eval(f"model[0].{param_2}")]

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
        return
    
    ##### initialize dataframe for measurements
    col_names = [param_1_display_name, param_2_display_name, "AP height (mV)","AP peak time","AP peak time (stimulated compartment)",
                 "AP start time","AP end time","rise time (ms)","fall time (ms)","latency (ms)"]
    node_response_data = pd.DataFrame(np.zeros((length_param_1*length_param_2, len(col_names))), columns = col_names)
    
    ##### initialize matrix for voltage courses
    voltage_data = np.zeros((length_param_1*length_param_2,max_nof_timesteps))
    
    ##### compartments for measurements
    comp_index = np.where(model[0].structure == 2)[0][10]
            
    ##### loop over parameter 1
    for ii in range(0, length_param_1):
        
        if hasattr(model[0], param_1):
            if not param_2 == "model":
                ##### adjust model parameter
                exec(f"model[0].{param_1} = param_1_ratios[ii]*param_1_original_value[0]")
                ##### set up neuron with adjusted model parameter
                neuron, param_string, model[0] = model[0].set_up_model(dt = dt, model = model[0], update = True, model_name = "model[0]")
                exec(param_string)
                M = StateMonitor(neuron, 'v', record=True)
                store('initialized')
        
        elif param_1 == "model":
            ##### set up neuron for actual model
            neuron, param_string, model[ii] = model[ii].set_up_model(dt = dt, model = model[ii], model_name = "model[ii]")
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
                    neuron, param_string, model[0] = model[0].set_up_model(dt = dt, model = model[0], update = True, model_name = "model[0]")
                    exec(param_string)
                    M = StateMonitor(neuron, 'v', record=True)
                    store('initialized')
               else:
                    ##### adjust model parameter
                    exec(f"model[ii].{param_2} = param_2_ratios[jj]*param_2_original_value[ii]")
                    ##### set up neuron again with adjusted model parameter
                    neuron, param_string, model[ii] = model[ii].set_up_model(dt = dt, model = model[ii], update = True, model_name = "model[ii]")
                    exec(param_string)
                    M = StateMonitor(neuron, 'v', record=True)
                    store('initialized')
            
            elif param_2 == "model":
                if hasattr(model[0], param_1):
                    ##### adjust model parameter
                    exec(f"model[jj].{param_1} = param_1_ratios[ii]*param_1_original_value[jj]")
                ##### set up neuron for actual model
                neuron, param_string, model[jj] = model[jj].set_up_model(dt = dt, model = model[jj], update = True, model_name = "model[jj]")
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
            
            ##### write results in table
            # calculated AP properties at the compartment specified with comp_index
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
                
            # calculate the AP time at the stimulated compartment (for latency measurement)
            AP_amp_stim_comp = max(M.v[stimulated_compartment,:]-model_type.V_res)
            AP_time_stim_comp = M.t[M.v[stimulated_compartment,:]-model_type.V_res == AP_amp_stim_comp]
            
            # fill in the value for parameter 1
            if hasattr(model_type, param_1): node_response_data[param_1_display_name][ii*length_param_2+jj] = param_1_ratios[ii]
            elif param_1 == "model": node_response_data[param_1_display_name][ii*length_param_2+jj] = model_type.display_name
            elif param_1 == "stochastic_runs": node_response_data[param_1_display_name][ii*length_param_2+jj] = ii
            elif param_1 == "stim_amp": node_response_data[param_1_display_name][ii*length_param_2+jj] = round(stim_amp/uA,2)
            elif param_1 == "phase_duration": node_response_data[param_1_display_name][ii*length_param_2+jj] = round(phase_duration/us,2)
            
            # fill in the value for parameter 2
            if hasattr(model_type, param_2): node_response_data[param_2_display_name][ii*length_param_2+jj] = param_2_ratios[jj]
            elif param_2 == "model": node_response_data[param_2_display_name][ii*length_param_2+jj] = model_type.display_name
            elif param_2 == "stochastic_runs": node_response_data[param_2_display_name][ii*length_param_2+jj] = jj
            elif param_2 == "stim_amp": node_response_data[param_2_display_name][ii*length_param_2+jj] = round(stim_amp/uA,2)
            elif param_2 == "phase_duration": node_response_data[param_2_display_name][ii*length_param_2+jj] = round(phase_duration/us,2)
            
            # fill in remaining values
            node_response_data["AP height (mV)"][ii*length_param_2+jj] = AP_amp/mV
            node_response_data["AP peak time"][ii*length_param_2+jj] = AP_time/ms
            node_response_data["AP peak time (stimulated compartment)"][ii*length_param_2+jj] = AP_time_stim_comp/ms
            node_response_data["AP start time"][ii*length_param_2+jj] = AP_start_time/ms
            node_response_data["AP end time"][ii*length_param_2+jj] = AP_end_time/ms

            ##### print progress
            print(f"{param_1_display_name}: {ii+1}/{length_param_1}; {param_2_display_name}: {jj+1}/{length_param_2}")

            ##### save voltage course of single compartment for plotting
            voltage_data[length_param_2*ii+jj,0:np.shape(M.v)[1]] = M.v[comp_index, :]/mV
    
    ##### Change voltage_data to dataframe and add parameter information
    voltage_data = pd.DataFrame(voltage_data)
    voltage_data[param_1_display_name] = node_response_data[param_1_display_name]
    voltage_data[param_2_display_name] = node_response_data[param_2_display_name]
    
    ##### change structure for seaborn plot
    voltage_data = voltage_data.melt(id_vars = [param_1_display_name, param_2_display_name], var_name = "time")
    voltage_data["time"] = voltage_data["time"]*dt/ms
    voltage_data = voltage_data[voltage_data["value"]!=0]
    
    ##### rename columns of voltage data
    voltage_data = voltage_data.rename(index=str, columns={"time": "time / ms", "value": "membrane potential / mV"})

    ##### calculate remaining single node response data
    node_response_data["rise time (ms)"] = node_response_data["AP peak time"] - node_response_data["AP start time"]
    node_response_data["fall time (ms)"] = node_response_data["AP end time"] - node_response_data["AP peak time"]
    node_response_data["latency (ms)"] = node_response_data["AP peak time (stimulated compartment)"]
    
    ##### exclude runs where no AP was elicited or where no start or end time could be calculated
    node_response_data = node_response_data[node_response_data["AP height (mV)"] > 60]
    node_response_data = node_response_data[node_response_data["AP start time"] > 0]
    node_response_data = node_response_data[node_response_data["AP end time"] > 0]
    
    ##### sum up relevant single node response properties in dataframe
    if param_1 == "stochastic_runs":
        node_response_data[param_1_display_name] = node_response_data[param_1_display_name].astype(int)
        #node_response_data_summary = node_response_data.groupby([param_2_display_name])["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"].mean().reset_index()
        #node_response_data_summary["jitter (ms)"] = node_response_data.groupby([param_2_display_name])["latency (ms)"].std()

    elif param_2 == "stochastic_runs":
        node_response_data[param_2_display_name] = node_response_data[param_2_display_name].astype(int)
        #node_response_data_summary = node_response_data.groupby([param_1_display_name])["AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"].mean().reset_index()
        #node_response_data_summary["jitter (ms)"] = node_response_data.groupby([param_1_display_name])["latency (ms)"].std()
        
    else:
        node_response_data_summary = node_response_data[[param_1_display_name, param_2_display_name, "AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]]

    node_response_data_summary = node_response_data[[param_1_display_name, param_2_display_name, "AP height (mV)", "rise time (ms)", "fall time (ms)", "latency (ms)"]]

    ##### reset the adjusted parameter
    if hasattr(model[0], param_1):
        if param_2 == "model":
            for ii in range(0, len(model)):
                exec(f"model[ii].{param_1} = param_1_original_value[ii]")
        else:
            exec(f"model[0].{param_1} = param_1_original_value[0]")                        
    if hasattr(model[0], param_2):
        if param_1 == "model":
            for ii in range(0, len(model)):
                exec(f"model[ii].{param_2} = param_2_original_value[ii]")
        else:
            exec(f"model[0].{param_2} = param_2_original_value[0]")
            
    ##### save time_vector
    time_vector = M.t
        
    ##### reset neuron
    for ii in range(0, len(model)):
        neuron, param_string, model[ii] = model[ii].set_up_model(dt = dt, model = model[ii], update = True, model_name = "model[ii]")
        exec(param_string)
        M = StateMonitor(neuron, 'v', record=True)
        store('initialized')
    
    return voltage_data, node_response_data_summary

# =============================================================================
#  Calculate thresholds for certain phase durations
# =============================================================================
def get_thresholds(model,
                   dt,
                   phase_durations,
                   amps_start_intervall,
                   delta,
                   nof_runs = 1,
                   stimulation_type = "extern",
                   pulse_form = "mono",
                   time_after = 2*ms,
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
    amps_start_intervall : time
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
    
    ##### test if phase_durations is a list or array, if not convert it to a list
    if not isinstance(phase_durations, (list,)) and len(np.shape(phase_durations))==0:
        phase_durations = [phase_durations]
        
    ##### stochastic runs in case nof_runs > 1
    add_noise = False
    if nof_runs > 1: add_noise = True
        
    ##### initialize model with given defaultclock dt
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')
    
    ##### initialize vector for minimum required stimulus current amplitudes
    thresholds = pd.DataFrame(np.zeros((len(phase_durations)*nof_runs, 3)), columns = ["phase duration","run","threshold"])
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    stim_amps_min = amps_start_intervall[0]
    stim_amps_max = amps_start_intervall[1]
    
    ##### start amplitde for first run
    start_amp = (stim_amps_max-stim_amps_min)/2
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]

    ##### loop over phase durations
    for ii in range(0, len(phase_durations)):
        
        ##### calculate runtime
        if pulse_form == "mono":
            runtime = phase_durations[ii] + time_after
        else:
            runtime = phase_durations[ii]*2 + time_after
        
        ##### calculate number of timesteps
        nof_timesteps = int(np.ceil(runtime/dt))
        
        ##### loop over stochastic runs
        for jj in range(0, nof_runs):
            
            ##### initializations
            min_amp_spiked = 0*amp
            lower_border = stim_amps_min
            upper_border = stim_amps_max
            stim_amp = start_amp
            amp_diff = upper_border - lower_border
            
            ##### include noise
            if add_noise:
                I_noise = np.transpose(np.transpose(np.random.normal(0, 1, (model.nof_comps,nof_timesteps)))*model.k_noise*model.noise_term)
            else:
                I_noise = np.zeros((model.nof_comps,nof_timesteps))
            
            ##### adjust stimulus amplitude until required accuracy is obtained
            while amp_diff > delta:
                
                ##### print progress
                if print_progress: print(f"Duration: {np.round(phase_durations[ii]/us)} us; Run: {jj+1}; Stimulus amplitde: {np.round(stim_amp/uA,4)} uA")
                
                ##### define how the ANF is stimulated
                I_stim, runtime = stim.get_stimulus_current(model = model,
                                                            dt = dt,
                                                            stimulation_type = stimulation_type,
                                                            pulse_form = pulse_form,
                                                            add_noise = add_noise,
                                                            time_after = time_after,
                                                            ##### monophasic stimulation
                                                            amp_mono = -stim_amp,
                                                            duration_mono = phase_durations[ii],
                                                            ##### biphasic stimulation
                                                            amps_bi = [-stim_amp/amp,stim_amp/amp]*amp,
                                                            durations_bi = [phase_durations[ii]/second,0,phase_durations[ii]/second]*second)
            
                ##### get TimedArray of stimulus currents and run simulation
                stimulus = TimedArray(np.transpose(I_stim + I_noise), dt=dt)
                
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
                                            
            ##### write values in threshold matrix
            thresholds["phase duration"][ii*nof_runs + jj] = phase_durations[ii]
            thresholds["run"][ii*nof_runs + jj] = jj+1
            thresholds["threshold"][ii*nof_runs + jj] = min_amp_spiked
            
            ##### get start amplitude for next run/phase-duration
            start_amp[min_amp_spiked != 0*amp] = min_amp_spiked
            start_amp[min_amp_spiked == 0*amp] = stim_amps_max
    
    if add_noise == True:
        thresholds["run"] = thresholds["run"].astype(int)
    else:
        thresholds = thresholds.drop(columns = "run")
        
    thresholds["threshold"][thresholds["threshold"] == 0] = None
        
    return thresholds

# =============================================================================
#  Calculate cronaxie for a given rheobase
# =============================================================================
def get_chronaxie(model,
                  dt,
                  rheobase,
                  phase_duration_start_intervall,
                  delta,
                  stimulation_type = "extern",
                  pulse_form = "mono",
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
    amps_start_intervall : time
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
    phase_duration_min = phase_duration_start_intervall[0]
    phase_duration_max = phase_duration_start_intervall[1]
    
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
        if print_progress: print(f"Duration: {np.round(phase_duration/us)} us")
        
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    stimulation_type = stimulation_type,
                                                    pulse_form = pulse_form,
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
def get_refractory_periods(model,
                           dt,
                           delta = 1*us,
                           threshold = 0,
                           amp_masker = 0,
                           stimulation_type = "extern",
                           pulse_form = "mono",
                           phase_duration = 100*us,
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
    amps_start_intervall : time
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
    
    ##### calculate theshold of model
    if threshold == 0:
        threshold = get_thresholds(model = model,
                           dt = dt,
                           phase_durations = phase_duration,
                           amps_start_intervall = [0,100]*uA,
                           delta = 0.0001*uA,
                           stimulation_type = stimulation_type,
                           pulse_form = pulse_form,
                           time_after = 3*ms,
                           print_progress = False)["threshold"][0]*amp
    
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
    stim_amp_rrp = 1.2*threshold    

    ##### get absolute refractory period
    # initializations
    arp = 0*second
    lower_border = inter_pulse_gap_min
    upper_border = inter_pulse_gap_max
    inter_pulse_gap = (inter_pulse_gap_max-inter_pulse_gap_min)/2
    inter_pulse_gap_diff = upper_border - lower_border
    
    # adjust stimulus amplitude until required accuracy is obtained
    while inter_pulse_gap_diff > delta:
        
        # print progress
        if print_progress: print(f"Inter pulse gap: {np.round(inter_pulse_gap/us)} us")
        
        # define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = 0*ms,
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
    lower_border = arp
    upper_border = inter_pulse_gap_max
    inter_pulse_gap = (inter_pulse_gap_max-inter_pulse_gap_min)/2
    inter_pulse_gap_diff = upper_border - lower_border
    
    # adjust stimulus amplitude until required accuracy is obtained
    while inter_pulse_gap_diff > delta:
        
        # print progress
        if print_progress: print(f"Inter pulse gap: {np.round(inter_pulse_gap/us)} us")
        
        # define how the ANF is stimulated
        I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                  dt = dt,
                                                                  stimulation_type = stimulation_type,
                                                                  pulse_form = pulse_form,
                                                                  time_before = 0*ms,
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
                    
    return arp, rrp

# =============================================================================
#  Calculate refractory curve
# =============================================================================
def get_refractory_curve(model,
                         dt,
                         inter_pulse_intervalls,
                         delta,
                         threshold = 0,
                         amp_masker = 0,
                         stimulation_type = "extern",
                         pulse_form = "mono",
                         phase_duration = 100*us,
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
    
    ##### calculate theshold of model
    if threshold == 0:
        threshold = get_thresholds(model = model,
                           dt = dt,
                           phase_durations = phase_duration,
                           amps_start_intervall = [0,100]*uA,
                           delta = 0.0001*uA,
                           stimulation_type = stimulation_type,
                           pulse_form = pulse_form,
                           time_after = 1.5*ms,
                           print_progress = False)["threshold"][0]*amp
    
    ##### amplitude of masker stimulus (150% of threshold)
    if amp_masker == 0:
        amp_masker = 1.5 * threshold

    ##### initialize vector for minimum required stimulus current amplitudes
    min_required_amps = np.zeros_like(inter_pulse_intervalls/second)*amp
    
    ##### minimum and maximum stimulus current amplitudes that are tested
    stim_amps_min = 0*amp
    stim_amps_max = threshold * 10
    
    ##### start amplitde for first run
    start_amp = (stim_amps_max-stim_amps_min)/2
    
    ##### compartment for measurements
    comp_index = np.where(model.structure == 2)[0][10]
    
    ##### initialize model and monitors
    neuron, param_string, model = model.set_up_model(dt = dt, model = model)
    exec(param_string)
    M = StateMonitor(neuron, 'v', record=True)
    store('initialized')

    ##### loop over phase durations
    for ii in range(0, len(inter_pulse_intervalls)):
        
        ##### initializations
        min_amp_spiked = 0*amp
        lower_border = stim_amps_min
        upper_border = stim_amps_max
        stim_amp = start_amp
        amp_diff = upper_border - lower_border
        
        ##### adjust stimulus amplitude until required accuracy is obtained
        while amp_diff > delta:
            
            ##### print progress
            if print_progress : print(f"Inter pulse intervall: {round(inter_pulse_intervalls[ii]/us)} us; Amplitude of second stimulus: {np.round(stim_amp/uA,2)} uA")
            
            ##### define how the ANF is stimulated
            I_stim_masker, runtime_masker = stim.get_stimulus_current(model = model,
                                                                      dt = dt,
                                                                      stimulation_type = stimulation_type,
                                                                      pulse_form = pulse_form,
                                                                      time_before = 0*ms,
                                                                      time_after = 0*ms,
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
                                                                time_before = inter_pulse_intervalls[ii],
                                                                time_after = 3*ms,
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
                            
        ##### write the found minimum stimulus current in vector
        min_required_amps[ii] = min_amp_spiked
        start_amp[min_amp_spiked != 0*amp] = min_amp_spiked
        start_amp[min_amp_spiked == 0*amp] = stim_amps_max
        
    return min_required_amps, threshold

# =============================================================================
#  Calculate poststimulus time histogram (PSTH)
# =============================================================================
def post_stimulus_time_histogram(model,
                                 dt,
                                 nof_repeats,
                                 pulses_per_second,
                                 stim_duration,
                                 stim_amp,
                                 stimulation_type,
                                 pulse_form,
                                 phase_duration):
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
    
#     nof_repeats = 3
#     pulses_per_second = [250,1000,5000,10000]
#     stim_duration = 50*ms
#     stim_amp = [2,3,5]*uA
#     stimulation_type = "extern"
#     pulse_form = "bi"
#     phase_duration = [40,40,40,20]*us
#     further_bin_widths = [4,8,12,12,12,52,100,100]*ms
    
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
    
    ##### initialize dataset to save histogram and stimulus information
    psth = pd.DataFrame()

    ##### calculate nof_pulses
    nof_pulses = round(pulses_per_second*stim_duration/second)
        
    ##### calculate inter_pulse_gap
    if pulse_form == "mono":
        inter_pulse_gap = (1e6/pulses_per_second - phase_duration/us)*us
    else:
        inter_pulse_gap = (1e6/pulses_per_second - phase_duration*2/us)*us
        
    ##### initialize pulse train dataframe
    spike_times = np.zeros((nof_repeats, nof_pulses*2))
            
    ##### loop over number of repeats
    for ii in range(0, nof_repeats):
        
        ##### print progress
        print(f"Pulse rate: {pulses_per_second} pps; Stimulus Amplitude: {np.round(stim_amp/us,2)} us; Run: {ii+1}/{nof_repeats}")
        
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
        spike_times[ii, 0:len(spikes)] = spikes
            
    ##### connect all spike times to one vector
    spike_times = np.concatenate(spike_times)
    
    ##### trim zeros
    spike_times = spike_times[spike_times != 0]
    
    ##### save spike information in dataset
    psth["spike times"] = spike_times
    psth["pulse rate"] = pulses_per_second
    psth["phase duration"] = phase_duration
    psth["stimulus amplitude"] = stim_amp
    
    return psth
    