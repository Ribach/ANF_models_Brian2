##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

##### sets working directory
os.chdir("C:/Users/Richard/Documents/Studium/Master Elektrotechnik/Semester 4/Python/Models Brian2")

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Frijns_2005 as frijns_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Load data
# =============================================================================
##### choose model
model = rattay_01

##### load datasets
strength_duration_data = pd.read_csv(f'test_battery_results/{model.display_name}/Strength_duration_data {model.display_name}.csv')
threshold_table = pd.read_csv(f'test_battery_results/{model.display_name}/Threshold_table {model.display_name}.csv')
relative_spreads = pd.read_csv(f'test_battery_results/{model.display_name}/Relative_spreads {model.display_name}.csv')
conduction_velocity_table = pd.read_csv(f'test_battery_results/{model.display_name}/Conduction_velocity_table {model.display_name}.csv')
node_response_data_summary = pd.read_csv(f'test_battery_results/{model.display_name}/Node_response_data_summary {model.display_name}.csv')
refractory_table = pd.read_csv(f'test_battery_results/{model.display_name}/Refractory_table {model.display_name}.csv')

# =============================================================================
# Add experimental results to tables
# =============================================================================
##### Strength duration data
strength_duration_data = strength_duration_data.transpose()
strength_duration_data = strength_duration_data.rename(index = str, columns={0:"model"})
strength_duration_data["model"] = ["%.3g" %strength_duration_data["model"][i] for i in range(0,2)]
strength_duration_data["Van den Honert and Stypulkowski 1984"] = ["95.8", "247"]

##### Relative spread of thresholds
relative_spreads = relative_spreads.rename(index = str, columns={"relative spread":"model"})
relative_spreads["experiments"] = ["6.3%","5-10%","12%","11%"]
relative_spreads["reference"] = ["Miller et al. 1999","Dynes 1996","Javel et al. 1987","Javel et al. 1987"]
relative_spreads = relative_spreads.set_index(["phase duration","pulse form"])

##### Conduction velocity
conduction_velocity_table = conduction_velocity_table.transpose()
conduction_velocity_table = conduction_velocity_table.rename(index = str, columns={0:"model"})
if hasattr(model, "index_soma"):
    conduction_velocity_table["Hursh 1939"] = ["-","6","-","-"]
    if model.diameter_dendrite < 12*um:
        dendrite_ratio = 4.6
    else:
        dendrite_ratio = 5.66
    conduction_velocity_table["Boyd and Kalu 1979"] = ["-",f"{dendrite_ratio}","-","-"]
    conduction_velocity_table["Czèh et al 1976"] = ["-","-","v_ax = 0.9*v_den-6.9*m/s","-"]

else:
    conduction_velocity_table["Hursh 1939"] = ["-","6"]
    if model.diameter_fiber < 12*um:
        dendrite_ratio = 4.6
    else:
        dendrite_ratio = 5.66
    conduction_velocity_table["Boyd and Kalu 1979"] = ["-",f"{dendrite_ratio}"]

##### Latency and jitter
latency_jitter = node_response_data_summary[["phase duration (us)", "pulse form", "stimulus amplitude level", "latency (ms)",  "jitter (ms)"]]
latency_jitter = latency_jitter.rename(index = str, columns={"latency (ms)":"latency (us)", "jitter (ms)":"jitter (us)"})
latency_jitter = pd.melt(latency_jitter, id_vars=["phase duration (us)", "pulse form", "stimulus amplitude level"], value_vars=["latency (us)",  "jitter (us)"])
latency_jitter["value"] = latency_jitter["value"]*1000
latency_jitter["value"] = ["%.3g" %latency_jitter["value"][i] for i in range(0,latency_jitter.shape[0])]
latency_jitter["phase duration (us)"] = ["%g us" %latency_jitter["phase duration (us)"][i] for i in range(0,latency_jitter.shape[0])]
latency_jitter = latency_jitter.rename(index = str, columns={"phase duration (us)":"phase duration"})

latency_jitter_th = latency_jitter[latency_jitter["stimulus amplitude level"] == "threshold"]
latency_jitter_th = latency_jitter_th.rename(index = str, columns={"value":"model (threshold)"})

latency_jitter_2th = latency_jitter[latency_jitter["stimulus amplitude level"] == "2*threshold"]
latency_jitter_2th = latency_jitter_2th.rename(index = str, columns={"value":"model (2*threshold)"})

latency_jitter = latency_jitter.drop(columns = ["stimulus amplitude level", "value"]).drop_duplicates()
latency_jitter["model (threshold)"] = latency_jitter_th["model (threshold)"].tolist()
latency_jitter["model (2*threshold)"] = latency_jitter_2th["model (2*threshold)"].tolist()
latency_jitter = latency_jitter.sort_values(by=["pulse form", "phase duration"], ascending = [False, True])

latency_jitter["Miller et al. 1999"] = ["650", "100", "-", "-", "-", "-", "-", "-"]
latency_jitter["Van den Honert and Stypulkowski 1984 (threshold)"] = ["-", "-", "-", "-", "685", "352", "-", "-"]
latency_jitter["Van den Honert and Stypulkowski 1984 (2*threshold)"] = ["-", "-", "-", "-", "352", "8", "-", "-"]
latency_jitter["Hartmann and al. 1984"] = ["-", "-", "-", "-", "-", "-", "300-400", "-"]
latency_jitter["Cartee et al. 2000 (threshold)"] = ["-", "-", "440", "80", "-", "-", "-", "-"]

latency_jitter = latency_jitter.rename(index = str, columns={"variable":"property"})
latency_jitter = latency_jitter.set_index(["phase duration","pulse form", "property"])

##### AP shape
AP_shape = node_response_data_summary[["AP height (mV)", "rise time (ms)", "fall time (ms)", "AP duration (ms)"]].iloc[[5]].transpose()
AP_shape = AP_shape.rename(index = str, columns={5:"model"})

t_rise = round(-0.000625*conduction_velocity_table["model"].iloc[[0]] + 0.14, 3).tolist()[0]
t_fall = round(-0.002083*conduction_velocity_table["model"].iloc[[0]] + 0.3933, 3).tolist()[0]
AP_duration = t_rise + t_fall
AP_shape["Paintal 1965"] = ["-", t_rise, t_fall, AP_duration]

##### Refractory periods
absolute_refractory_periods = refractory_table.drop(columns = ["relative refractory period (ms)"])
absolute_refractory_periods = absolute_refractory_periods.rename(index = str, columns={"absolute refractory period (ms)":"ARP model (us)"})
absolute_refractory_periods["ARP model (us)"] = absolute_refractory_periods["ARP model (us)"]*1000
absolute_refractory_periods["ARP Experiments (us)"] = ["334","300","500-700","400-500","-"]
absolute_refractory_periods["reference"] = ["Miller et al. 2001","Stypulkowski and Van den Honert 1984","Dynes 1996","Brown and Abbas 1990", "-"]
absolute_refractory_periods = absolute_refractory_periods[absolute_refractory_periods["ARP Experiments (us)"] != "-"]
absolute_refractory_periods = absolute_refractory_periods.set_index(["phase duration (us)","pulse form"])

relative_refractory_periods = refractory_table.drop(columns = ["absolute refractory period (ms)"])
relative_refractory_periods = relative_refractory_periods.rename(index = str, columns={"relative refractory period (ms)":"RRP model (ms)"})
relative_refractory_periods["RRP Experiments (ms)"] = ["-","3-4; 4-5","5","-","5"]
relative_refractory_periods["reference"] = ["-","Stypulkowski and Van den Honert 1984; Cartee et al. 2000","Dynes 1996","-", "Hartmann et al. 1984"]
relative_refractory_periods = relative_refractory_periods[relative_refractory_periods["RRP Experiments (ms)"] != "-"]
relative_refractory_periods = relative_refractory_periods.set_index(["phase duration (us)","pulse form"])

