# =============================================================================
# This script collects the test battery results of all (!) models and generates
# and saves plots that compare the results among each other and with experimental
# data. Furthermore dataframes are generated and saved in a latex-compatibel
# format, which contain both model and experimental data.
# =============================================================================
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

##### import functions
import functions.create_plots_for_model_comparison as plot
import functions.pandas_to_latex as ptol

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### list of all models
models = [rattay_01, frijns_94, briaire_05, smit_09, smit_10, imennov_09, negm_14]

##### distinguish models with and without soma
models_with_soma = list(itl.compress(models, [hasattr(model, "index_soma") for model in models]))
models_without_soma = list(itl.compress(models, [not hasattr(model, "index_soma") for model in models]))

##### save plots
save_plots = True
save_tables = True
theses_image_path = "C:/Users/Richard/Documents/Studium/Master Elektrotechnik/Semester 4/Masterarbeit/Abschlussbericht/images/single_fiber_characteristics"
theses_table_path = "C:/Users/Richard/Documents/Studium/Master Elektrotechnik/Semester 4/Masterarbeit/Abschlussbericht/tables/single_fiber_characteristics"

# =============================================================================
# Conduction velocity tables
# =============================================================================
##### table for models with soma
for ii,model in enumerate(models_with_soma):
    
    ##### get strength duration data
    data = pd.read_csv("results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name)).transpose()
    
    ##### round for three significant digits
    data[0] = ["%.3g" %data[0][jj] for jj in range(data.shape[0])]
    
    if ii == 0:
        ##### use model name as column header
        conduction_velocity_table_with_soma = data.rename(index = str, columns={0:model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        conduction_velocity_table_with_soma[model.display_name_plots] = data[0]

##### table for models without soma
for ii,model in enumerate(models_without_soma):
    
    ##### get strength duration data
    data = pd.read_csv("results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name)).transpose()
    
    ##### round for three significant digits
    data[0] = ["%.3g" %data[0][jj] for jj in range(data.shape[0])]
    
    if ii == 0:
        ##### use model name as column header
        conduction_velocity_table_without_soma = data.rename(index = str, columns={0:model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        conduction_velocity_table_without_soma[model.display_name_plots] = data[0]

# =============================================================================
# Conduction velocity plot
# =============================================================================
##### built subset with relevant columns of dataset of models without soma
cond_vel_without_soma = conduction_velocity_table_without_soma.transpose()[["velocity (m/s)","outer diameter (um)"]]
cond_vel_without_soma["section"] = "fiber"

##### built subset with relevant columns of dataset of models with soma for dendritic part
cond_vel_with_soma = conduction_velocity_table_with_soma.transpose()[["velocity dendrite (m/s)","outer diameter dendrite (um)"]]
cond_vel_with_soma["section"] = "dendrite"
cond_vel_with_soma = cond_vel_with_soma.rename(index = str, columns={"velocity dendrite (m/s)":"velocity (m/s)",
                                                                     "outer diameter dendrite (um)":"outer diameter (um)"})

##### connect dataframes
conduction_velocity_table = pd.concat((cond_vel_without_soma, cond_vel_with_soma), axis=0)

##### built subset with relevant columns of dataset of models with soma for axonal part
cond_vel_with_soma = conduction_velocity_table_with_soma.transpose()[["velocity axon (m/s)","outer diameter axon (um)"]]
cond_vel_with_soma["section"] = "axon"
cond_vel_with_soma = cond_vel_with_soma.rename(index = str, columns={"velocity axon (m/s)":"velocity (m/s)",
                                                                     "outer diameter axon (um)":"outer diameter (um)"})

##### connect dataframes
conduction_velocity_table = pd.concat((conduction_velocity_table, cond_vel_with_soma), axis=0)

##### index to column
conduction_velocity_table.reset_index(inplace=True)
conduction_velocity_table = conduction_velocity_table.rename(index = str, columns={"index":"model_name"})

##### order dataframe
conduction_velocity_table = conduction_velocity_table.sort_values("model_name")

##### Plot conduction velocity comparison
conduction_velocity_plot = plot.conduction_velocity_comparison(plot_name = "Comparison of conduction velocities with experimental data",
                                                               model_data = conduction_velocity_table)

##### save plots
if save_plots:
    conduction_velocity_plot.savefig("{}/conduction_velocity_plot.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# Connect tables for models with and without soma and bring them in an appropriate format for latex
# =============================================================================
##### transpose tables
conduction_velocity_table_with_soma = conduction_velocity_table_with_soma.transpose()
conduction_velocity_table_without_soma = conduction_velocity_table_without_soma.transpose()

##### change column names
conduction_velocity_table_without_soma = conduction_velocity_table_without_soma.rename(index = str, columns={"velocity (m/s)":"velocity axon (m/s)",
                                                                                                             "outer diameter (um)":"outer diameter axon (um)",
                                                                                                             "velocity/diameter":"velocity/diameter axon"})

##### connect tables
conduction_velocity_table = pd.concat([conduction_velocity_table_with_soma,conduction_velocity_table_without_soma])

##### order columns
conduction_velocity_table = conduction_velocity_table[["velocity dendrite (m/s)", "outer diameter dendrite (um)","velocity/diameter dendrite",
                                                       "velocity axon (m/s)","outer diameter axon (um)","velocity/diameter axon"]]

##### change column names again
conduction_velocity_table = conduction_velocity_table.rename(index = str, columns={"velocity dendrite (m/s)":"$v_{\T{c}}$/$ms^{-1}$",
                                                                                   "outer diameter dendrite (um)":"$D$/\SI{}{\micro\meter}",
                                                                                   "velocity/diameter dendrite":"$k$",
                                                                                   "velocity axon (m/s)":"$v_{\T{c}}$/$ms^{-1}$",
                                                                                   "outer diameter axon (um)":"$D$/\SI{}{\micro\meter}",
                                                                                   "velocity/diameter axon":"$k$"})
    
##### fill NA values with ""
conduction_velocity_table = conduction_velocity_table.fillna("-")

##### define captions and save tables as tex
if save_tables:
    caption_top = "Comparison of conduction velocities, outer diameters and scaling factors predicted by the ANF models."    
    with open("{}/conduction_velocity_table.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(conduction_velocity_table, label = "tbl:con_vel_table",
                                         caption_top = caption_top, vert_line = [2], upper_col_names = ["dendrite","axon"]))

# =============================================================================
# Single node response plot
# =============================================================================
##### initialize list of dataframes to save voltage courses
voltage_courses = [pd.DataFrame()]*len(models)

##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    voltage_course_dataset = pd.read_csv("results/{}/Single_node_response_plot_data_deterministic {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    voltage_course_dataset["model"] = model.display_name_plots
    
    ##### write subset of dataframe in voltage courses list
    voltage_courses[ii] = voltage_course_dataset[["model", "membrane potential (mV)","time (ms)", "amplitude level"]]\
                                                [voltage_course_dataset["pulse form"] == pulse_form]\
                                                [voltage_course_dataset["phase duration (us)"] == phase_duration/us]

##### connect dataframes to one dataframe
voltage_courses = pd.concat(voltage_courses,ignore_index = True)

##### plot voltage courses
single_node_response = plot.single_node_response_comparison(plot_name = "Voltage courses model comparison",
                                                            voltage_data = voltage_courses)

##### save plot
if save_plots:
    single_node_response.savefig("{}/single_node_response comparison.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# AP shape table axons
# =============================================================================
##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"
amplitude_level = "2*threshold"

##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data  = pd.read_csv("results/{}/Single_node_response_deterministic {}.csv".format(model.display_name,model.display_name))
    
    ##### built subset of relevant rows and columns and transpose dataframe
    data = data[["AP height (mV)", "rise time (us)", "fall time (us)"]]\
               [data["pulse form"] == pulse_form]\
               [data["phase duration (us)"] == phase_duration/us]\
               [data["amplitude level"] == amplitude_level].transpose()
    
    if ii == 0:
        ##### use model name as column header
        AP_shape_axon = data.rename(index = str, columns={data.columns.values[0]:model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        AP_shape_axon[model.display_name_plots] = data[data.columns.values[0]]
    
##### transpose dataframe
AP_shape_axon = AP_shape_axon.transpose()

##### round columns to 3 significant digits
for ii in ["AP height (mV)","rise time (us)","fall time (us)"]:
    AP_shape_axon[ii] = ["%.4g" %AP_shape_axon[ii][jj] for jj in range(AP_shape_axon.shape[0])]

##### change column names for latex export
AP_shape_latex = AP_shape_axon.rename(index = str, columns={"AP height (mV)":"AP height/mV",
                                                 "rise time (us)":"rise time/\SI{}{\micro\second}",
                                                 "fall time (us)":"fall time/\SI{}{\micro\second}"})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of properties, describing the AP shape, measured with the ANF models due to a stimulation with a monophasic \SI{100}{\micro\second}\
                   cathodic current pulse with amplitude $2I_{\T{th}}$."
    with open("{}/AP_shape_models.tex".format(theses_table_path), "w") as tf:
         tf.write(ptol.dataframe_to_latex(AP_shape_latex, label = "tbl:AP_shape_comparison", caption_top = caption_top))

# =============================================================================
# AP shapes dendrite
# =============================================================================
##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"
amplitude_level = "2*threshold"

##### loop over models with a soma
for ii,model in enumerate(models_with_soma):
    
    ##### get node response summery table
    data  = pd.read_csv("results/{}/Single_node_response_deterministic dendrite {}.csv".format(model.display_name,model.display_name))
    
    ##### built subset of relevant rows and columns and transpose dataframe
    data = data[["AP height (mV)", "rise time (us)", "fall time (us)"]]\
               [data["pulse form"] == pulse_form]\
               [data["phase duration (us)"] == phase_duration/us]\
               [data["amplitude level"] == amplitude_level].transpose()
    
    if ii == 0:
        ##### use model name as column header
        AP_shape_dendrite = data.rename(index = str, columns={data.columns.values[0]:"{} dendrite".format(model.display_name_plots)})
        
    else:
        ##### add column with AP shape data of current model
        AP_shape_dendrite["{} dendrite".format(model.display_name_plots)] = data[data.columns.values[0]]
    
##### transpose dataframe
AP_shape_dendrite = AP_shape_dendrite.transpose()

##### round columns to 3 significant digits
for ii in ["AP height (mV)","rise time (us)","fall time (us)"]:
    AP_shape_dendrite[ii] = ["%.4g" %AP_shape_dendrite[ii][jj] for jj in range(AP_shape_dendrite.shape[0])]

# =============================================================================
# Plot experimental results for rise and fall time
# =============================================================================
##### add axon label in row names for human ANF models
AP_shape_axon = AP_shape_axon.rename(index={"Rattay et al. (2001)":"Rattay et al. (2001) axon",
                                             "Briaire and Frijns (2005)":"Briaire and Frijns (2005) axon",
                                             "Smit et al. (2010)":"Smit et al. (2010) axon"})

##### connect conduction velocities with AP shape values
AP_shape_cond_vel_table = pd.concat([AP_shape_axon[["rise time (us)","fall time (us)"]], AP_shape_dendrite[["rise time (us)","fall time (us)"]]])
AP_shape_cond_vel_table["conduction velocity axon (m/s)"] = 0.0
AP_shape_cond_vel_table["conduction velocity dendrite (m/s)"] = 0.0
AP_shape_cond_vel_table["section"] = ""

##### Fill in conduction velocities
for ii,model in enumerate(models):
    
    ##### models with soma
    if model in models_with_soma:
        #### write velocity in table
        AP_shape_cond_vel_table["conduction velocity dendrite (m/s)"]["{} dendrite".format(model.display_name_plots)] = conduction_velocity_table_with_soma["velocity dendrite (m/s)"][model.display_name_plots]
        AP_shape_cond_vel_table["conduction velocity axon (m/s)"]["{} axon".format(model.display_name_plots)] = conduction_velocity_table_with_soma["velocity axon (m/s)"][model.display_name_plots]
        #### write section in table
        AP_shape_cond_vel_table["section"]["{} dendrite".format(model.display_name_plots)] = "dendrite"
        AP_shape_cond_vel_table["section"]["{} axon".format(model.display_name_plots)] = "axon"
                
    ##### models without soma
    else:
        #### write velocity in table
        AP_shape_cond_vel_table["conduction velocity axon (m/s)"][model.display_name_plots] = conduction_velocity_table_without_soma["velocity axon (m/s)"][model.display_name_plots]
        #### write section in table
        AP_shape_cond_vel_table["section"][model.display_name_plots] = ""
        
##### change index to column
AP_shape_cond_vel_table.reset_index(inplace=True)
AP_shape_cond_vel_table = AP_shape_cond_vel_table.rename(index = str, columns={"index" : "model_name"})

##### change rise time column type to float
AP_shape_cond_vel_table["rise time (us)"] = AP_shape_cond_vel_table["rise time (us)"].astype(float)
AP_shape_cond_vel_table["fall time (us)"] = AP_shape_cond_vel_table["fall time (us)"].astype(float)

##### order dataframe
AP_shape_cond_vel_table = AP_shape_cond_vel_table.sort_values("model_name")

##### Plot rise and fall time comparison
rise_and_fall_time_comparison_paintal = plot.rise_and_fall_time_comparison(plot_name = "Comparison of rise times with data from Paintal 1966",
                                                                           model_data = AP_shape_cond_vel_table)

##### save plots
if save_plots:
    rise_and_fall_time_comparison_paintal.savefig("{}/rise_and_fall_time_comparison_paintal.pdf".format(theses_image_path), bbox_inches='tight')

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
    data = pd.read_csv("results/{}/Single_node_response_deterministic {}.csv".format(model.display_name,model.display_name))
    
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","amplitude level", "pulse form"])["latency (us)"].astype(int))
    
    if ii == 0:
        ##### use model name as column header
        latency_table = data.rename(index = str, columns={"latency (us)":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        latency_table[model.display_name_plots] = data["latency (us)"].tolist()

##### Add experimental data
latency_table["\cite{Miller1999}"] = ["-", "-", "-", "650"]
latency_table["\cite{VandenHonert1984}"] = ["-", "685", "352", "-"]
latency_table["\cite{Cartee2000}"] = ["440", "-", "-", "-"]

##### Transpose dataframe
latency_table = latency_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(latency_table.columns)]):
    latency_table = latency_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of latencies, measured with the ANF models, to experimental data (italiced). Four different stimuli were applied. Latencies are given in \SI{}{\micro\second}"
    caption_bottom = ""
    for ii,letter in enumerate(letters[:len(stimulations)]):
        if stimulations["amplitude level"][ii][0] == "1": stim_amp = ""
        else: stim_amp = stimulations["amplitude level"][ii][0]
        caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                         + "}{\micro\second}" + " cathodic current pulse with amplitude {}".format(stim_amp) + "$I_{\T{th}}$\\\\\n"
    italic_range = range(len(models),len(latency_table))
    with open("{}/latency_table.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(latency_table, label = "tbl:latency_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))

# =============================================================================
# Jitter table
# =============================================================================
##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data = pd.read_csv("results/{}/Single_node_response_stochastic {}.csv".format(model.display_name,model.display_name))
    
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","amplitude level", "pulse form"])["jitter (us)"].astype(int))
    
    if ii == 0:
        ##### use model name as column header
        jitter_table = data.rename(index = str, columns={"jitter (us)":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        jitter_table[model.display_name_plots] = data["jitter (us)"].tolist()

##### Add experimental data
jitter_table["\cite{Miller1999}"] = ["-", "-", "-", "100"]
jitter_table["\cite{VandenHonert1984}"] = ["-", "352", "8", "-"]
jitter_table["\cite{Cartee2000}"] = ["80", "-", "-", "-"]

##### Transpose dataframe
jitter_table = jitter_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(jitter_table.columns)]):
    jitter_table = jitter_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of jitters"
    caption_bottom = "All jitters are given in $\mu$s. Four different stimuli were applied and compared with experimental data (italicised)\\\\\n"
    for ii,letter in enumerate(letters[:len(stimulations)]):
        caption_bottom = caption_bottom + "{}: {}, phase duration: {} $\mu$s, stimulus amplitude: {}\\\\\n".format(letter,stimulations["pulse form"][ii],
                             stimulations["phase duration (us)"][ii],stimulations["amplitude level"][ii])
    italic_range = range(len(models),len(jitter_table))
    with open("{}/jitter_table.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(jitter_table, label = "tbl:jitter_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))
    
# =============================================================================
# Strength duration table
# =============================================================================
##### loop over models
for ii,model in enumerate(models):
    
    ##### get strength duration data
    data = pd.read_csv("results/{}/Strength_duration_data {}.csv".format(model.display_name,model.display_name)).transpose()
    
    if ii == 0:
        ##### use model name as column header
        strength_duration_table = data.rename(index = str, columns={0:model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        strength_duration_table[model.display_name_plots] = data[0]

##### round for three significant digits
for ii in strength_duration_table.columns.values.tolist():
    strength_duration_table[ii] = ["%.3g" %strength_duration_table[ii][jj] for jj in range(strength_duration_table.shape[0])]

##### add experimental data
strength_duration_table["\cite{VandenHonert1984}"] = ["95.8", "247"]
strength_duration_table["\cite{Bostock1983}"] = ["-", "64.9"]

##### transpose dataframe
strength_duration_table = strength_duration_table.transpose()

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of the rheobase and chronaxie for all models with experimental data (italicised)."
    italic_range = range(len(models),len(strength_duration_table))
    with open("{}/strength_duration_table.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(strength_duration_table, label = "tbl:strength_duration_comparison", caption_top = caption_top, italic = italic_range))

# =============================================================================
# Strength duration curve
# =============================================================================
##### initialize list of dataframes to save strength duration curves
stength_duration_curves = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    stength_duration_curves[ii] = pd.read_csv("results/{}/Strength_duration_plot_table {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    stength_duration_curves[ii]["model"] = model.display_name_plots

##### connect dataframes to one dataframe
stength_duration_curves = pd.concat(stength_duration_curves,ignore_index = True)

##### plot strength duration curve
strength_duration_curve = plot.strength_duration_curve_comparison(plot_name = "Strength duration curve model comparison",
                                                                  threshold_data = stength_duration_curves,
                                                                  strength_duration_table = strength_duration_table)

##### save plot
if save_plots:
    strength_duration_curve.savefig("{}/strength_duration_curve comparison.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# Refractory curves
# =============================================================================
##### initialize list of dataframes to save voltage courses
refractory_curves = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    refractory_curves[ii] = pd.read_csv("results/{}/Refractory_curve_table {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    refractory_curves[ii]["model"] = model.display_name_plots

##### connect dataframes to one dataframe
refractory_curves = pd.concat(refractory_curves,ignore_index = True)

##### remove rows where no second spikes were obtained
refractory_curves = refractory_curves[refractory_curves["minimum required amplitude"] != 0]
    
##### calculate the ratio of the threshold of the second spike and the masker
refractory_curves["threshold ratio"] = refractory_curves["minimum required amplitude"]/refractory_curves["threshold"]

##### convert interpulse intervals to ms
refractory_curves["interpulse interval"] = refractory_curves["interpulse interval"]*1e3

##### plot voltage courses
refractory_curves_plot = plot.refractory_curves_comparison(plot_name = "Refractory curves model comparison",
                                                           refractory_curves = refractory_curves)

##### save plot
if save_plots:
    refractory_curves_plot.savefig("{}/refractory_curves_plot comparison.pdf".format(theses_image_path), bbox_inches='tight')

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
    data = pd.read_csv("results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name))
        
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","pulse form"])["absolute refractory period (us)"].astype(int))

    if ii == 0:
        ##### use model name as column header
        ARP_comparison_table = data.rename(index = str, columns={"absolute refractory period (us)":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        ARP_comparison_table[model.display_name_plots] = data["absolute refractory period (us)"].tolist()

##### round for four significant digits
for ii in ARP_comparison_table.columns.values.tolist():
    ARP_comparison_table[ii] = ["%.4g" %ARP_comparison_table[ii][jj] for jj in range(ARP_comparison_table.shape[0])]
        
##### Add experimental data
ARP_comparison_table["\cite{Miller2001}"] = ["334", "-", "-", "-"]
ARP_comparison_table["\cite{Stypulkowski1984}"] = ["-", "300", "-", "-"]
ARP_comparison_table["\cite{Dynes1996}"] = ["-", "-", "500-700", "-"]
ARP_comparison_table["\cite{Brown1990}"] = ["-", "-", "-", "400-500"]

##### Transpose dataframe
ARP_comparison_table = ARP_comparison_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(ARP_comparison_table.columns)]):
    ARP_comparison_table = ARP_comparison_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of ARPs, measured with the ANF models, to experimental data (italiced). Four different stimuli were used. ARPs are given in \SI{}{\micro\second}"
    caption_bottom = ""
    for ii,letter in enumerate(letters[:len(stimulations)]):
        if stimulations["pulse form"][ii] == "monophasic":
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                            + "}{\micro\second} cathodic current pulses\\\\\n"
        else:
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                + "}{\micro\second} cathodic first current pulses\\\\\n"
    italic_range = range(len(models),len(ARP_comparison_table))
    with open("{}/ARP_comparison_table.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(ARP_comparison_table, label = "tbl:ARP_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))

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
    data = pd.read_csv("results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name))
        
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","pulse form"])["relative refractory period (ms)"])

    if ii == 0:
        ##### use model name as column header
        RRP_comparison_table = data.rename(index = str, columns={"relative refractory period (ms)":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        RRP_comparison_table[model.display_name_plots] = data["relative refractory period (ms)"].tolist()

##### round for three significant digits
for ii in RRP_comparison_table.columns.values.tolist():
    RRP_comparison_table[ii] = ["%.3g" %RRP_comparison_table[ii][jj] for jj in range(RRP_comparison_table.shape[0])]
    
##### Add experimental data
RRP_comparison_table["\cite{Stypulkowski1984}"] = ["3-4", "-", "-"]
RRP_comparison_table["\cite{Cartee2000}"] = ["4-5", "-", "-"]
RRP_comparison_table["\cite{Dynes1996}"] = ["-", "5", "-"]
RRP_comparison_table["\cite{Hartmann1984a}"] = ["-", "-", "5"]

##### Transpose dataframe
RRP_comparison_table = RRP_comparison_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(RRP_comparison_table.columns)]):
    RRP_comparison_table = RRP_comparison_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of RRPs, measured with the ANF models, to experimental data (italiced). Three different stimuli were used. RRPs are given in \SI{}{\milli\second}"
    caption_bottom = ""
    for ii,letter in enumerate(letters[:len(stimulations)]):
        if stimulations["pulse form"][ii] == "monophasic":
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                            + "}{\micro\second} cathodic current pulses\\\\\n"
        else:
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                + "}{\micro\second} cathodic first current pulses\\\\\n"
    italic_range = range(len(models),len(RRP_comparison_table))
    with open("{}/RRP_comparison_table.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(RRP_comparison_table, label = "tbl:RRP_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))

# =============================================================================
# relative spread plots
# =============================================================================
##### define model to show how the noise factor affects the relative spread values
model = rattay_01

##### get data for plots
relative_spread_plot_table_1k = pd.read_csv("results/{}/Relative_spread_plot_table {}.csv".format(model.display_name_plots,model.display_name_plots))
relative_spread_plot_table_2k = pd.read_csv("results/{}/2_knoise/Relative_spread_plot_table {}.csv".format(model.display_name_plots,model.display_name_plots))
relative_spread_plot_table_4k = pd.read_csv("results/{}/4_knoise/Relative_spread_plot_table {}.csv".format(model.display_name_plots,model.display_name_plots))

##### add noise levels to dataframes
relative_spread_plot_table_1k["noise level"] = "1 $k_{noise}$"
relative_spread_plot_table_2k["noise level"] = "2 $k_{noise}$"
relative_spread_plot_table_4k["noise level"] = "4 $k_{noise}$"

##### connect dataframes
relative_spread_plot_table = pd.concat([relative_spread_plot_table_1k,relative_spread_plot_table_2k,relative_spread_plot_table_4k], ignore_index = True)

##### relative spreads plot
relative_spread_plot = plot.relative_spread_comparison(plot_name = "Relative spreads {}".format(model.display_name_plots),
                                                       threshold_data = relative_spread_plot_table)

##### save plot
if save_plots:
    relative_spread_plot.savefig("{}/relative_spread_plot comparison.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# relative spread table (Rattay used as example)
# =============================================================================
##### define model to show how the noise factor affects the relative spread values
model = rattay_01

##### get tables
relative_spreads_1k = pd.read_csv("results/{}/Relative_spreads {}.csv".format(model.display_name_plots,model.display_name_plots))
relative_spreads_2k = pd.read_csv("results/{}/2_knoise/Relative_spreads {}.csv".format(model.display_name_plots,model.display_name_plots))
relative_spreads_4k = pd.read_csv("results/{}/4_knoise/Relative_spreads {}.csv".format(model.display_name_plots,model.display_name_plots))

##### Relative spread of thresholds
relative_spreads = relative_spreads_1k.rename(index = str, columns={"relative spread":"{} 1*knoise".format(model.display_name_plots_short)})
relative_spreads["{} 2*knoise".format(model.display_name_plots_short)] = relative_spreads_2k["relative spread"].tolist()
relative_spreads["{} 4*knoise".format(model.display_name_plots_short)] = relative_spreads_4k["relative spread"].tolist()

##### save stimulus information, build subset and transpose dataframe
stimulation = relative_spreads[["phase duration (us)", "pulse form"]]
relative_spreads = relative_spreads.drop(columns = ["phase duration (us)", "pulse form"]).transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(relative_spreads.columns)]):
    relative_spreads = relative_spreads.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    stimulations = relative_spreads_1k[["phase duration (us)","pulse form"]]
    caption_top = "Comparison of relative spread values for different noise levels."
    caption_bottom = "Relative spread values were calculated with the model of rattay. Four different stimuli were applied and compared with experimental data (italicised)\\\\\n"
    for ii,letter in enumerate(letters[:len(stimulations)]):
        caption_bottom = caption_bottom + "{}: {}, phase duration: {} $\mu$s\\\\\n".format(letter,stimulations["pulse form"][ii],stimulations["phase duration (us)"][ii])
    with open("{}/relative_spread_rattay.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(relative_spreads, label = "tbl:relative_spread_comparison_rattay",
                                         caption_top = caption_top, caption_bottom = caption_bottom))

# =============================================================================
# Relative spread table all models
# =============================================================================
##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data = pd.read_csv("results/{}/Relative_spreads {}.csv".format(model.display_name_plots,model.display_name_plots))
    
    if ii == 0:
        ##### use model name as column header
        relative_spread_table = data.rename(index = str, columns={"relative spread":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        relative_spread_table[model.display_name_plots] = data["relative spread"].tolist()
    
    if model == rattay_01:
        relative_spreads_2k = pd.read_csv("results/{}/2_knoise/Relative_spreads {}.csv".format(model.display_name_plots,model.display_name_plots))
        relative_spreads_4k = pd.read_csv("results/{}/4_knoise/Relative_spreads {}.csv".format(model.display_name_plots,model.display_name_plots))
        relative_spread_table["{} 2*knoise".format(model.display_name_plots)] = relative_spreads_2k["relative spread"].tolist()
        relative_spread_table["{} 4*knoise".format(model.display_name_plots)] = relative_spreads_4k["relative spread"].tolist()

##### Add experimental data
relative_spread_table["Miller et al. 1999"] = ["6.3%","-","-","-"]
relative_spread_table["Dynes 1996"] = ["-","5-10%","-","-"]
relative_spread_table["Javel et al. 1987"] = ["-","-","12%","11%"]

##### Transpose dataframe
relative_spread_table = relative_spread_table.drop(columns = ["phase duration (us)", "pulse form"]).transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(relative_spread_table.columns)]):
    relative_spread_table = relative_spread_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of relative spreads"
    caption_bottom = "Four different stimuli were applied and compared with experimental data (italicised). For the model of Rattay et al. 2001, three different noise levels are compared.\\\\\n"
    for ii,letter in enumerate(letters[:len(stimulations)]):
        caption_bottom = caption_bottom + "{}: {}, phase duration: {} $\mu$s\\\\\n".format(letter,stimulations["pulse form"][ii], stimulations["phase duration (us)"][ii])
    italic_range = range(len(models),len(relative_spread_table))
    with open("{}/relative_spread_comparison.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(relative_spread_table, label = "tbl:relative_spread_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))
    
# =============================================================================
# Computational efficiency comparison
# =============================================================================
##### load table with computation times
computation_times_table = pd.read_csv("results/Analyses/computational_efficiency.csv")

##### calculate means, transpose dataframe and order it ascendingly
computation_times_table = computation_times_table.mean().round(2).to_frame().sort_values(0).rename(index = str, columns={0:"calculation time (sec)"})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of computational efficiency."
    caption_bottom = "Table shows average calculation times of 10 runs of 50 ms"    
    with open("{}/computation_times_table.tex".format(theses_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(computation_times_table, label = "tbl:comp_efficiency_table",
                                         caption_top = caption_top, caption_bottom = caption_bottom))




















