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
    
    if ii == 0:
        ##### use model name as column header
        conduction_velocity_table_with_soma = data.rename(index = str, columns={0:model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        conduction_velocity_table_with_soma[model.display_name] = data[0]

##### table for models without soma
for ii,model in enumerate(models_without_soma):
    
    ##### get strength duration data
    data = pd.read_csv("test_battery_results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name)).transpose()
    
    if ii == 0:
        ##### use model name as column header
        conduction_velocity_table_without_soma = data.rename(index = str, columns={0:model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        conduction_velocity_table_without_soma[model.display_name] = data[0]

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
        AP_shape = node_response_data_summary.rename(index = str, columns={node_response_data_summary.columns.values[0]:model.display_name})
        
    else:
        ##### add column with AP shape data of current model
        AP_shape[model.display_name] = node_response_data_summary[node_response_data_summary.columns.values[0]]
    
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
        AP_shape_cond_vel_table["velocity dendrite (m/s)"][model.display_name] = conduction_velocity_table_with_soma[model.display_name]["velocity dendrite (m/s)"]
        AP_shape_cond_vel_table["velocity axon (m/s)"][model.display_name] = conduction_velocity_table_with_soma[model.display_name]["velocity axon (m/s)"]
    
    ##### models without soma
    else:
        AP_shape_cond_vel_table["velocity fiber (m/s)"][model.display_name] = conduction_velocity_table_without_soma[model.display_name]["velocity (m/s)"]

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
models = [rattay_01, frijns_05, smit_09, smit_10, imennov_09, negm_14]

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



















