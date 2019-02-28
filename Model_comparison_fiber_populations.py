# =============================================================================
# This script collects the fiber population results of all (!) models and generates
# and saves plots that compare the results among each other.
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
import h5py

##### import models
import models.Rattay_2001 as rattay_01
import models.Briaire_2005 as briaire_05
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09

##### import functions
import functions.create_plots_for_fiber_populations as plot

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### list of all models
models = ["rattay_01", "briaire_05", "smit_10", "imennov_09"]

##### save plots
save_plots = True
theses_image_path = "C:/Users/Richard/Documents/Studium/Master Elektrotechnik/Semester 4/Masterarbeit/Abschlussbericht/images/fiber_populations"

##### define path of potential distribution data
h5py_path = "Measurements/Potential_distributions/original_mdl.h5"

##### define range of simulated neurons
neuron_range = range(0,400)

##### define range of considered electrodes
electrode_range = range(0,12)

# =============================================================================
# Calculate the distances of the peripheral terminal of the neurons from the origin
# =============================================================================
##### open h5py file
with h5py.File(h5py_path, 'r') as potential_data:
    
    ##### initialize dataframe
    distances = pd.DataFrame(np.zeros((len(neuron_range),2)), columns = ["neuron_number", "dist_along_sl"])
    distances["neuron_number"] = neuron_range
    
    ##### calculate distances along the spiral lamina
    for ii in range(1,len(neuron_range)):
        
        coord = potential_data['neuron{}'.format(ii)]["coordinates"][0,:]
        if ii == 1: last_coord = potential_data['neuron{}'.format(ii-1)]["coordinates"][0,:]
        
        distances["dist_along_sl"].iloc[ii] = distances["dist_along_sl"].iloc[ii-1] + \
            np.sqrt((coord[0] - last_coord[0])**2 + (coord[1] - last_coord[1])**2 + (coord[2] - last_coord[2])**2)*1e3
        
        last_coord = coord
    
    ##### set minimum distance to zero
    distances["dist_along_sl"] = distances["dist_along_sl"] - min(distances["dist_along_sl"])

# =============================================================================
# Dynamic range comparison (among electrodes and models)
# =============================================================================
##### get dynamic range plot comparison for all models and electrodes
for ii, electrode in enumerate(electrode_range):
    
    ##### get strength duration data
    data = pd.read_csv("results/Fiber_population/dynamic_ranges/181206_spikes_per_stim_amp_elec{}.csv".format(electrode))
    
    ##### add electrode number
    data["elec_nr"] = electrode
    
    ##### connect data of all electrodes in one dataframe
    if ii == 0:
        dyn_range_table = data
    else:
        ##### add column with AP shape data of current model
        dyn_range_table = pd.concat([dyn_range_table, data])
    
##### calculate spikes per fiber
dyn_range_table = dyn_range_table.groupby(["model_name","stim_amp", "elec_nr"])["spike"].sum().reset_index()
dyn_range_table = dyn_range_table.rename(index = str, columns={"spike" : "nof_spikes"})

##### plot comparison
dyn_range_comparison = plot.nof_spikes_over_stim_amp_comparison(plot_name = "Comparison of dynamic ranges",
                                                                spike_table = dyn_range_table)

##### save plot
if save_plots:
    dyn_range_comparison.savefig("{}/dyn_range_comparison.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# Raster plot comparison
# =============================================================================
##### define electrode number
elec_nr = 5

##### get data
for ii, pulse_rate in enumerate([100,1000]):
    
    ##### load table with spike times
    data = pd.read_csv("results/Fiber_population/spike_trains/190204_spike_trains_elec{}_{}pps.csv".format(elec_nr, pulse_rate))
    
    ##### add electrode number
    data["pulse_rate"] = pulse_rate
    
    ##### connect data of all pulse rates in one dataframe
    if ii == 0:
        spike_table = data
    else:
        ##### add column with AP shape data of current model
        spike_table = pd.concat([spike_table, data])

##### delete rows whith no spike
spike_table = spike_table[pd.to_numeric(spike_table['spikes'], errors='coerce').notnull()]

##### only consider first 50 ms
spike_table["duration"] = 0.05
spike_table = spike_table[spike_table["spikes"] <= spike_table["duration"].iloc[0]]

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])
spike_table["max_dist_along_sl"] = max(distances["dist_along_sl"])

##### add electrode number
spike_table["elec_nr"] = elec_nr

##### plot comparison
raster_plot_comparison = plot.raster_plot_comparison(plot_name = "Raster plot comparison",
                                                         spike_table = spike_table)
        
##### save plot
raster_plot_comparison.savefig("{}/raster_plot_comparison.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# Comparison of excitation differences of cathodic, anodic and biphasic pulses
# =============================================================================
##### define which electrodes to show
electrodes = [1,5,11]

##### get data
for ii, electrode in enumerate(electrodes):
    
    ##### load tables with spike times
    spike_table_bi = pd.read_csv("results/Fiber_population/dynamic_ranges/190201_spikes_per_stim_amp_elec{}.csv".format(electrode))
    spike_table_cat = pd.read_csv("results/Fiber_population/dynamic_ranges/190131_spikes_per_stim_amp_elec{}_cat.csv".format(electrode))
    spike_table_ano = pd.read_csv("results/Fiber_population/dynamic_ranges/190204_spikes_per_stim_amp_elec{}_ano.csv".format(electrode))
    
    ##### add column with pulse form information
    spike_table_bi["pulse_form"] = "biphasic"
    spike_table_cat["pulse_form"] = "cathodic"
    spike_table_ano["pulse_form"] = "anodic"
    
    ##### connect dataframes
    data = pd.concat([spike_table_bi, spike_table_cat], ignore_index = True)
    data = pd.concat([data, spike_table_ano], ignore_index = True)
    
    ##### add electrode information
    data["elec_nr"] = electrode
    
    ##### connect data of all electrodes in one dataframe
    if ii == 0:
        spike_table = data
    else:
        ##### add column with AP shape data of current model
        spike_table = pd.concat([spike_table, data])

##### remove rows without spike
spike_table = spike_table[spike_table["spike"] == 1]
spike_table = spike_table[spike_table["spike_at_last_comp"] == False]

##### get absolute values of stimulus amplitudes
spike_table["stim_amp"] = abs(spike_table["stim_amp"])

##### calculate dynamic range values
dyn_range = spike_table.groupby(["model_name", "elec_nr"])["stim_amp"].min().reset_index()
dyn_range = dyn_range.rename(index = str, columns={"stim_amp" : "min_amp_spike"})
spike_table = pd.merge(spike_table, dyn_range, on=["model_name", "elec_nr"])
spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/spike_table["min_amp_spike"])

##### get minimum amplitudes with spikes for each fiber
spike_table = spike_table.groupby(["model_name", "elec_nr", "neuron_number", "pulse_form"])["dynamic_range"].min().reset_index()

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])

##### generate plot
pulse_form_comparison_plot = plot.compare_pulse_forms_for_multiple_electrodes(plot_name = "Compare excation of anodic, cathodic and biphasic pulses for different electrodes",
                                                                              spike_table = spike_table)

##### save plot
if save_plots:
    pulse_form_comparison_plot.savefig("{}/pulse_form_comparison_plot.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked location of spike initiation for one electrode
# =============================================================================
##### define which electrodes to show
electrode = 11

##### load tables with spike times
data = pd.read_csv("results/Fiber_population/dynamic_ranges/190131_spikes_per_stim_amp_elec{}.csv".format(electrode))

##### add electrode information
data["elec_nr"] = electrode

##### loop over models
for ii, model_name in enumerate(models):
    
    ##### get model module
    model = eval(model_name)
    
    ##### build subset for current model
    data_model = data[data["model_name"] == model_name].copy()
    
    ##### remove rows without spike
    data_model = data_model[(data_model["spike"] == 1) & (data_model["spike_at_last_comp"] == False)]
    
    ##### add distance of compartments to terminal
    data_model["first_spike_dist"] = [np.cumsum(model.compartment_lengths)[int(data_model["first_spike_comp"].iloc[jj])]/mm for jj in range(data_model.shape[0])]

    ##### connect data of all electrodes and models in one dataframe
    if ii == 0:
        spike_table = data_model
    else:
        ##### add column with AP shape data of current model
        spike_table = pd.concat([spike_table, data_model])

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])

##### calculate dynamic range values
dyn_range = spike_table.groupby(["model_name"])["stim_amp"].min().reset_index()
dyn_range = dyn_range.rename(index = str, columns={"stim_amp" : "min_amp_spike"})
spike_table = pd.merge(spike_table, dyn_range, on=["model_name"])
spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/spike_table["min_amp_spike"])

##### generate plot
first_spike_color_plot_comparison = plot.spikes_color_plot_comparison(plot_name = "dB above threshold over distance along spiral lamina plot comparison for electrode {}".format(electrode),
                                                                      spike_table = spike_table)

##### save plot
if save_plots:
    first_spike_color_plot_comparison.savefig("{}/first_spike_color_plot_comparison.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked location of spike initiation for multiple electrodes
# =============================================================================
##### define which electrodes to show
electrodes = [0,5,11]

##### get data
for ii, electrode in enumerate(electrodes):
    
    ##### load tables with spike times
    data = pd.read_csv("results/Fiber_population/dynamic_ranges/190201_spikes_per_stim_amp_elec{}.csv".format(electrode))
    
    ##### add electrode information
    data["elec_nr"] = electrode
    
    ##### loop over models
    for jj, model_name in enumerate(models):
        
        ##### get model module
        model = eval(model_name)
        
        ##### build subset for current model
        data_model = data[data["model_name"] == model_name].copy()
        
        ##### remove rows without spike
        data_model = data_model[(data_model["spike"] == 1) & (data_model["spike_at_last_comp"] == False)]
        
        ##### add distance of compartments to terminal
        data_model["first_spike_dist"] = [np.cumsum(model.compartment_lengths)[int(data_model["first_spike_comp"].iloc[kk])]/mm for kk in range(data_model.shape[0])]
        
        ##### connect data of all electrodes and models in one dataframe
        if (ii == 0) & (jj==0):
            spike_table = data_model
        else:
            ##### add column with AP shape data of current model
            spike_table = pd.concat([spike_table, data_model])

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])

##### calculate dynamic range values
dyn_range = spike_table.groupby(["model_name", "elec_nr"])["stim_amp"].min().reset_index()
dyn_range = dyn_range.rename(index = str, columns={"stim_amp" : "min_amp_spike"})
spike_table = pd.merge(spike_table, dyn_range, on=["model_name", "elec_nr"])
spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/spike_table["min_amp_spike"])

##### generate plot
first_spike_color_plot_comparison_multiple_electrodes = plot.spikes_color_plot_comparison_multiple_electrodes(plot_name = "dB above threshold over distance along spiral lamina plot comparison",
                                                                                                              spike_table = spike_table)

##### save plot
if save_plots:
    first_spike_color_plot_comparison_multiple_electrodes.savefig("{}/first_spike_color_plot_comparison_multiple_electrodes.pdf".format(theses_image_path), bbox_inches='tight')

# =============================================================================
# Comparison of plots showing dB above threshold over distance along spiral lamina
# and marked latency for one electrode
# =============================================================================
##### define which electrodes to show
electrode = 5

##### load tables with spike times
spike_table = pd.read_csv("results/Fiber_population/dynamic_ranges/190131_spikes_per_stim_amp_elec{}.csv".format(electrode))

##### add electrode information
spike_table["elec_nr"] = electrode

##### remove rows without spike
spike_table = spike_table[spike_table["spike"] == 1]
spike_table = spike_table[spike_table["spike_at_last_comp"] == False]

##### add distances of fibers from base
spike_table = pd.merge(spike_table, distances, on=["neuron_number"])

##### calculate dynamic range values
dyn_range = spike_table.groupby(["model_name"])["stim_amp"].min().reset_index()
dyn_range = dyn_range.rename(index = str, columns={"stim_amp" : "min_amp_spike"})
spike_table = pd.merge(spike_table, dyn_range, on=["model_name"])
spike_table["dynamic_range"] = 20*np.log10(spike_table["stim_amp"]/spike_table["min_amp_spike"])

##### convert latencies to ms
spike_table["latency"] = [ii*1e-3 for ii in spike_table["latency"]]

##### generate plot
latency_color_plot_comparison = plot.latencies_color_plot_comparions(plot_name = "dB above threshold over distance along spiral lamina plot with marked latency1",
                                                                     spike_table = spike_table)

##### save plot
if save_plots:
    latency_color_plot_comparison.savefig("{}/latency_color_plot_comparison.pdf".format(theses_image_path), bbox_inches='tight')
