##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import itertools as itl
from string import ascii_uppercase as letters

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Frijns_2005 as frijns_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

##### import functions
import functions.create_plots_for_model_comparison as plot

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### list of all models
models = [rattay_01, frijns_94, frijns_05, smit_09, smit_10, imennov_09, negm_14]

##### distinguish models with and without soma
models_with_soma = list(itl.compress(models, [hasattr(model, "index_soma") for model in models]))
models_without_soma = list(itl.compress(models, [not hasattr(model, "index_soma") for model in models]))

##### save plots
save_plots = True
save_tables = True
interim_report_image_path = "C:/Users/Richard/Documents/Studium/Master Elektrotechnik/Semester 4/Masterarbeit/Zwischenbericht Masterarbeit/images"
interim_report_table_path = "C:/Users/Richard/Documents/Studium/Master Elektrotechnik/Semester 4/Masterarbeit/Zwischenbericht Masterarbeit/tables"

# =============================================================================
# Conduction velocity table
# =============================================================================
##### table for models with soma
for ii,model in enumerate(models_with_soma):
    
    ##### get strength duration data
    data = pd.read_csv("test_battery_results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name)).transpose()
    
    ##### round for three significant digits
    data[0] = ["%.3g" %data[0][jj] for jj in range(data.shape[0])]
    
    if ii == 0:
        ##### use model name as column header
        conduction_velocity_table_with_soma = data.rename(index = str, columns={0:model.display_name_short})
        
    else:
        ##### add column with AP shape data of current model
        conduction_velocity_table_with_soma[model.display_name_short] = data[0]

##### table for models without soma
for ii,model in enumerate(models_without_soma):
    
    ##### get strength duration data
    data = pd.read_csv("test_battery_results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name)).transpose()
    
    ##### round for three significant digits
    data[0] = ["%.3g" %data[0][jj] for jj in range(data.shape[0])]
    
    if ii == 0:
        ##### use model name as column header
        conduction_velocity_table_without_soma = data.rename(index = str, columns={0:model.display_name_short})
        
    else:
        ##### add column with AP shape data of current model
        conduction_velocity_table_without_soma[model.display_name_short] = data[0]

##### save tables as tex
if save_tables:
    with open("{}/conduction_velocity_table_with_soma.tex".format(interim_report_table_path), "w") as tf:
        tf.write(conduction_velocity_table_with_soma.to_latex(column_format ="lccc"))
    
    with open("{}/conduction_velocity_table_without_soma.tex".format(interim_report_table_path), "w") as tf:
        tf.write(conduction_velocity_table_without_soma.to_latex(column_format ="lcccc"))

# =============================================================================
# Single node response plot
# =============================================================================
##### initialize list of dataframes to save voltage courses
voltage_courses = [pd.DataFrame()]*len(models)

##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"
amplitude_level = "1*threshold"

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    voltage_course_dataset = pd.read_csv("test_battery_results/{}/Single_node_response_plot_data {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    voltage_course_dataset["model"] = model.display_name
    
    ##### write subset of dataframe in voltage courses list
    voltage_courses[ii] = voltage_course_dataset[["model","run","membrane potential (mV)","time (ms)"]]\
                                                [voltage_course_dataset["pulse form"] == pulse_form]\
                                                [voltage_course_dataset["phase duration (us)"] == phase_duration/us]\
                                                [voltage_course_dataset["amplitude level"] == amplitude_level]

##### connect dataframes to one dataframe
voltage_courses = pd.concat(voltage_courses,ignore_index = True)

##### plot voltage courses
single_node_response = plot.single_node_response_comparison(plot_name = "Voltage courses model comparison",
                                                            voltage_data = voltage_courses)

##### save plot
if save_plots:
    single_node_response.savefig("{}/single_node_response comparison.png".format(interim_report_image_path), bbox_inches='tight')

# =============================================================================
# AP shape table
# =============================================================================
##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"
amplitude_level = "2*threshold"

##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    node_response_data_summary = pd.read_csv("test_battery_results/{}/Single_node_response_summary {}.csv".format(model.display_name,model.display_name))
    
    ##### built subset of relevant rows and columns and transpose dataframe
    node_response_data_summary = node_response_data_summary[["AP height (mV)", "rise time (us)", "fall time (us)", "AP duration (us)"]]\
                                                           [node_response_data_summary["pulse form"] == pulse_form]\
                                                           [node_response_data_summary["phase duration (us)"] == phase_duration/us]\
                                                           [node_response_data_summary["amplitude level"] == amplitude_level].transpose()
    
    if ii == 0:
        ##### use model name as column header
        AP_shape = node_response_data_summary.rename(index = str, columns={node_response_data_summary.columns.values[0]:model.display_name_short})
        
    else:
        ##### add column with AP shape data of current model
        AP_shape[model.display_name_short] = node_response_data_summary[node_response_data_summary.columns.values[0]]
    
##### transpose dataframe
AP_shape = AP_shape.transpose()

##### round columns to 3 significant digits
for ii in ["AP height (mV)","rise time (us)","fall time (us)","AP duration (us)"]:
    AP_shape[ii] = ["%.4g" %AP_shape[ii][jj] for jj in range(AP_shape.shape[0])]

##### save table as tex
if save_tables:
    with open("{}/AP_shape_models.tex".format(interim_report_table_path), "w") as tf:
         tf.write(AP_shape.to_latex(column_format ="lcccc"))
         #tf.write(AP_shape.to_latex(column_format ="p{0.4\linewidth}p{0.146\linewidth}p{0.146\linewidth}p{0.146\linewidth}p{0.16\linewidth}"))

# =============================================================================
# Plot experimental results for rise and fall time
# =============================================================================
##### connect conduction velocities with AP shape values
AP_shape_cond_vel_table = AP_shape[["rise time (us)","fall time (us)","AP duration (us)"]]
AP_shape_cond_vel_table["velocity dendrite (m/s)"] = 0.0
AP_shape_cond_vel_table["velocity axon (m/s)"] = 0.0
AP_shape_cond_vel_table["velocity fiber (m/s)"] = 0.0

##### Fill in conduction velocities
for ii,model in enumerate(models):
    
    ##### models with soma
    if model in models_with_soma:
        AP_shape_cond_vel_table["velocity dendrite (m/s)"][model.display_name_short] = conduction_velocity_table_with_soma[model.display_name_short]["velocity dendrite (m/s)"]
        AP_shape_cond_vel_table["velocity axon (m/s)"][model.display_name_short] = conduction_velocity_table_with_soma[model.display_name_short]["velocity axon (m/s)"]
    
    ##### models without soma
    else:
        AP_shape_cond_vel_table["velocity fiber (m/s)"][model.display_name_short] = conduction_velocity_table_without_soma[model.display_name_short]["velocity (m/s)"]

##### change index to column
AP_shape_cond_vel_table.reset_index(inplace=True)
AP_shape_cond_vel_table = AP_shape_cond_vel_table.rename(index = str, columns={"index" : "model_name"})

##### change rise time column type to float
AP_shape_cond_vel_table["rise time (us)"] = AP_shape_cond_vel_table["rise time (us)"].astype(float)
AP_shape_cond_vel_table["fall time (us)"] = AP_shape_cond_vel_table["fall time (us)"].astype(float)
AP_shape_cond_vel_table["AP duration (us)"] = AP_shape_cond_vel_table["AP duration (us)"].astype(float)

##### Plot rise time comparison
rise_time_comparison_paintal = plot.paintal_rise_time_curve(plot_name = "Comparison of rise times with data from Paintal 1965",
                                                            model_data = AP_shape_cond_vel_table)

##### Plot fall time comparison
fall_time_comparison_paintal = plot.paintal_fall_time_curve(plot_name = "Comparison of fall times with data from Paintal 1965",
                                                            model_data = AP_shape_cond_vel_table)

##### save plots
if save_plots:
    rise_time_comparison_paintal.savefig("{}/rise_time_comparison_paintal.png".format(interim_report_image_path), bbox_inches='tight')
    fall_time_comparison_paintal.savefig("{}/fall_time_comparison_paintal.png".format(interim_report_image_path), bbox_inches='tight')

# =============================================================================
# Latency table
# =============================================================================
##### define which data to show
phase_durations = [40, 50, 50, 100]
amplitude_level = ["1*threshold", "1*threshold", "2*threshold", "1*threshold"]
pulse_forms = ["monophasic", "monophasic", "monophasic", "monophasic"]

##### create dataframe, that defines which data to show
stimulations = pd.DataFrame([phase_durations,amplitude_level,pulse_forms]).transpose()
stimulations = stimulations.rename(index = str, columns={0:"phase duration (us)",
                                                         1:"amplitude level",
                                                         2:"pulse form"})

##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data = pd.read_csv("test_battery_results/{}/Single_node_response_summary {}.csv".format(model.display_name,model.display_name))
    
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","amplitude level", "pulse form"])["latency (us)"].astype(int))
    
    if ii == 0:
        ##### use model name as column header
        latency_table = data.rename(index = str, columns={"latency (us)":model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        latency_table[model.display_name] = data["latency (us)"].tolist()

##### Add experimental data
latency_table["Miller et al. 1999"] = ["-", "-", "-", "650"]
latency_table["Van den Honert and Stypulkowski 1984"] = ["-", "685", "352", "-"]
latency_table["Cartee et al. 2000 (threshold)"] = ["440", "-", "-", "-"]

##### Transpose dataframe
latency_table = latency_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(latency_table.columns)]):
    latency_table = latency_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### save table as tex
if save_tables:
    with open("{}/latency_table.tex".format(interim_report_table_path), "w") as tf:
        tf.write(latency_table.to_latex(column_format ="lcccc"))

# =============================================================================
# Jitter table
# =============================================================================
##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data = pd.read_csv("test_battery_results/{}/Single_node_response_summary {}.csv".format(model.display_name,model.display_name))
    
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","amplitude level", "pulse form"])["jitter (us)"].astype(int))
    
    if ii == 0:
        ##### use model name as column header
        jitter_table = data.rename(index = str, columns={"jitter (us)":model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        jitter_table[model.display_name] = data["jitter (us)"].tolist()

##### Add experimental data
jitter_table["Miller et al. 1999"] = ["-", "-", "-", "100"]
jitter_table["Van den Honert and Stypulkowski 1984"] = ["-", "352", "8", "-"]
jitter_table["Cartee et al. 2000 (threshold)"] = ["80", "-", "-", "-"]

##### Transpose dataframe
jitter_table = jitter_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(latency_table.columns)]):
    jitter_table = jitter_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### save table as tex
if save_tables:
    with open("{}/jitter_table.tex".format(interim_report_table_path), "w") as tf:
        tf.write(jitter_table.to_latex(column_format ="lcccc"))
    
# =============================================================================
# Strength duration table
# =============================================================================
##### loop over models
for ii,model in enumerate(models):
    
    ##### get strength duration data
    data = pd.read_csv("test_battery_results/{}/Strength_duration_data {}.csv".format(model.display_name,model.display_name)).transpose()
    
    if ii == 0:
        ##### use model name as column header
        strength_duration_table = data.rename(index = str, columns={0:model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        strength_duration_table[model.display_name] = data[0]

##### add experimental data
strength_duration_table["Van den Honert and Stypulkowski 1984"] = ["95.8", "247"]

##### transpose dataframe
strength_duration_table = strength_duration_table.transpose()

##### save table as tex
if save_tables:
    with open("{}/strength_duration_table.tex".format(interim_report_table_path), "w") as tf:
        tf.write(strength_duration_table.to_latex(column_format ="lcc"))

# =============================================================================
# Strength duration curve
# =============================================================================
##### initialize list of dataframes to save strength duration curves
stength_duration_curves = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    stength_duration_curves[ii] = pd.read_csv("test_battery_results/{}/Strength_duration_plot_table {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    stength_duration_curves[ii]["model"] = model.display_name

##### connect dataframes to one dataframe
stength_duration_curves = pd.concat(stength_duration_curves,ignore_index = True)

##### plot strength duration curve
strength_duration_curve = plot.strength_duration_curve_comparison(plot_name = "Strength duration curve model comparison",
                                                                  threshold_matrix = stength_duration_curves,
                                                                  strength_duration_table = strength_duration_table)

##### save plot
if save_plots:
    strength_duration_curve.savefig("{}/strength_duration_curve comparison.png".format(interim_report_image_path), bbox_inches='tight')

# =============================================================================
# Refractory curves
# =============================================================================
##### initialize list of dataframes to save voltage courses
refractory_curves = [pd.DataFrame()]*len(models)

##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"
amplitude_level = "1*threshold"

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    refractory_curves[ii] = pd.read_csv("test_battery_results/{}/Refractory_curve_table {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    refractory_curves[ii]["model"] = model.display_name

##### connect dataframes to one dataframe
refractory_curves = pd.concat(refractory_curves,ignore_index = True)

##### plot voltage courses
refractory_curves_plot = plot.refractory_curves_comparison(plot_name = "Refractory curves model comparison",
                                                           refractory_curves = refractory_curves)

##### save plot
if save_plots:
    refractory_curves_plot.savefig("{}/refractory_curves_plot comparison.png".format(interim_report_image_path), bbox_inches='tight')

# =============================================================================
# Absolute refractory table model comparison
# =============================================================================
##### define which data to show
phase_durations = [40, 50, 100, 50]
pulse_forms = ["monophasic", "monophasic", "monophasic", "biphasic"]

##### create dataframe, that defines which data to show
stimulations = pd.DataFrame([phase_durations,pulse_forms]).transpose()
stimulations = stimulations.rename(index = str, columns={0:"phase duration (us)",
                                                         1:"pulse form"})

##### loop over models
for ii,model in enumerate(models):
    
    ##### get data
    data = pd.read_csv("test_battery_results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name))
        
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","pulse form"])["absolute refractory period (us)"].astype(int))

    if ii == 0:
        ##### use model name as column header
        ARP_comparison_table = data.rename(index = str, columns={"absolute refractory period (us)":model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        ARP_comparison_table[model.display_name] = data["absolute refractory period (us)"].tolist()
    
##### Add experimental data
ARP_comparison_table["Miller et al. 2001"] = ["334", "-", "-", "-"]
ARP_comparison_table["Stypulkowski and Van den Honert 1984"] = ["-", "300", "-", "-"]
ARP_comparison_table["Dynes 1996"] = ["-", "-", "500-700", "-"]
ARP_comparison_table["Brown and Abbas 1990"] = ["-", "-", "-", "400-500"]

##### Transpose dataframe
ARP_comparison_table = ARP_comparison_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(ARP_comparison_table.columns)]):
    ARP_comparison_table = ARP_comparison_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### save table as tex
if save_tables:
    with open("{}/ARP_comparison_table.tex".format(interim_report_table_path), "w") as tf:
        tf.write(ARP_comparison_table.to_latex(column_format ="lcccc"))

# =============================================================================
# Relative refractory table model comparison
# =============================================================================
##### define which data to show
phase_durations = [50, 100, 200]
pulse_forms = ["monophasic", "monophasic", "biphasic"]

##### create dataframe, that defines which data to show
stimulations = pd.DataFrame([phase_durations,pulse_forms]).transpose()
stimulations = stimulations.rename(index = str, columns={0:"phase duration (us)",
                                                         1:"pulse form"})

##### loop over models
for ii,model in enumerate(models):
    
    ##### get data
    data = pd.read_csv("test_battery_results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name))
        
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","pulse form"])["relative refractory period (ms)"])

    if ii == 0:
        ##### use model name as column header
        RRP_comparison_table = data.rename(index = str, columns={"relative refractory period (ms)":model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        RRP_comparison_table[model.display_name] = data["relative refractory period (ms)"].tolist()
    
##### Add experimental data
RRP_comparison_table["Stypulkowski and Van den Honert 1984"] = ["3-4", "-", "-"]
RRP_comparison_table["Cartee et al. 2000"] = ["4-5", "-", "-"]
RRP_comparison_table["Dynes 1996"] = ["-", "5", "-"]
RRP_comparison_table["Hartmann et al. 1984"] = ["-", "-", "5"]

##### Transpose dataframe
RRP_comparison_table = RRP_comparison_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(RRP_comparison_table.columns)]):
    RRP_comparison_table = RRP_comparison_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### save table as tex
if save_tables:
    with open("{}/RRP_comparison_table.tex".format(interim_report_table_path), "w") as tf:
        tf.write(RRP_comparison_table.to_latex(column_format ="lccc"))

# =============================================================================
# relative spread plots
# =============================================================================
##### define model to show how the noise factor affects the relative spread values
model = rattay_01

##### get data for plots
relative_spread_plot_table_1k = pd.read_csv("test_battery_results/{}/Relative_spread_plot_table {}.csv".format(model.display_name,model.display_name))
relative_spread_plot_table_2k = pd.read_csv("test_battery_results/{}/2_knoise/Relative_spread_plot_table {}.csv".format(model.display_name,model.display_name))
relative_spread_plot_table_4k = pd.read_csv("test_battery_results/{}/4_knoise/Relative_spread_plot_table {}.csv".format(model.display_name,model.display_name))

##### add noise levels to dataframes
relative_spread_plot_table_1k["noise level"] = "1*k_noise"
relative_spread_plot_table_2k["noise level"] = "2*k_noise"
relative_spread_plot_table_4k["noise level"] = "4*k_noise"

##### connect dataframes
relative_spread_plot_table = pd.concat([relative_spread_plot_table_1k,relative_spread_plot_table_2k,relative_spread_plot_table_4k], ignore_index = True)

##### relative spreads plot
relative_spread_plot = plot.relative_spread_comparison(plot_name = "Relative spreads {}".format(model.display_name),
                                                       threshold_matrix = relative_spread_plot_table)

##### save plot
if save_plots:
    relative_spread_plot.savefig("{}/relative_spread_plot comparison.png".format(interim_report_image_path), bbox_inches='tight')

# =============================================================================
# relative spread table (Rattay used as example)
# =============================================================================
##### define model to show how the noise factor affects the relative spread values
model = rattay_01

##### get tables
relative_spreads_1k = pd.read_csv("test_battery_results/{}/Relative_spreads {}.csv".format(model.display_name,model.display_name))
relative_spreads_2k = pd.read_csv("test_battery_results/{}/2_knoise/Relative_spreads {}.csv".format(model.display_name,model.display_name))
relative_spreads_4k = pd.read_csv("test_battery_results/{}/4_knoise/Relative_spreads {}.csv".format(model.display_name,model.display_name))

##### Relative spread of thresholds
relative_spreads = relative_spreads_1k.rename(index = str, columns={"relative spread":"{} 1*k_noise".format(model.display_name_short)})
relative_spreads["{} 2*k_noise".format(model.display_name_short)] = relative_spreads_2k["relative spread"].tolist()
relative_spreads["{} 4*k_noise".format(model.display_name_short)] = relative_spreads_4k["relative spread"].tolist()
relative_spreads["Miller et al. 1999"] = ["6.3%","-","-","-"]
relative_spreads["Dynes 1996"] = ["-","5-10%","-","-"]
relative_spreads["Javel et al. 1987"] = ["-","-","12%","11%"]

##### save stimulus information, build subset and transpose dataframe
stimulation = relative_spreads[["phase duration (us)", "pulse form"]]
relative_spreads = relative_spreads.drop(columns = ["phase duration (us)", "pulse form"]).transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(relative_spreads.columns)]):
    relative_spreads = relative_spreads.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### save table as tex
if save_tables:
    with open("{}/relative_spreads_comparison.tex".format(interim_report_table_path), "w") as tf:
        tf.write(relative_spreads.to_latex(column_format ="lcccc"))






