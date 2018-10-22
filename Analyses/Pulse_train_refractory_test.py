# =============================================================================
# This script tests, how the pulse rate of a masker stimulus influences the
# following refractory periods.
# =============================================================================
##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
import thorns as th

##### import functions
import functions.stimulation as stim
import functions.create_plots as plot
import functions.tests_for_analyses as test
import functions.calculations as calc

##### import models
import models.Rattay_2001 as rattay_01
import models.Frijns_1994 as frijns_94
import models.Briaire_2005 as briaire_05
import models.Smit_2009 as smit_09
import models.Smit_2010 as smit_10
import models.Imennov_2009 as imennov_09
import models.Negm_2014 as negm_14

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### initialize clock
dt = 1*us

##### define backend
backend = "serial"

##### define pulse rate of masker and second stimulus (in pulses per second)
pulse_rate = 1200/second

##### define phase durations and inter_pulse_gap
t_phase = 23*us
t_ipg = 2*us

##### define pulse train duration
t_pulse_train = 100*ms

##### calculate number of pulses
nof_pulses = int(t_pulse_train * pulse_rate)

##### calculate inter pulse gap
inter_pulse_gap = t_pulse_train/nof_pulses - 2*t_phase - t_ipg

##### define varied parameters
models = ["rattay_01","frijns_94","briaire_05","smit_09","smit_10","imennov_09","negm_14"]
pulse_rates = [1200,1500,18000,25000]

params = {"model_name" : models,
           "pulse_rate" : pulse_rates}

##### get thresholds
refractory_table = th.util.map(func = test.get_refractory_periods_for_pulse_trains,
                               space = params,
                               backend = backend,
                               cache = "no",
                               kwargs = {"dt" : dt,
                                         "delta" : 1*us,
                                         "stimulation_type" : "extern",
                                         "pulse_form" : "bi",
                                         "phase_durations" : [t_phase/us,t_ipg/us,t_phase/us]*us,
                                         "pulse_train_duration" : t_pulse_train})

##### change index to column
refractory_table.reset_index(inplace=True)

##### change column names
refractory_table = refractory_table.rename(index = str, columns={"model_name" : "model name",
                                                                 "pulse_rate" : "pulse rate",
                                                                 0 : "absolute refractory period (us)",
                                                                 1 : "relative refractory period (ms)"})

##### convert refractory periods to ms
refractory_table["absolute refractory period (us)"] = refractory_table["absolute refractory period (us)"]*1e6
refractory_table["relative refractory period (ms)"] = refractory_table["relative refractory period (ms)"]*1e3

##### round columns to 4 significant digits
for ii in ["absolute refractory period (us)","relative refractory period (ms)"]:
    refractory_table[ii] = ["%.4g" %refractory_table[ii][jj] for jj in range(refractory_table.shape[0])]

##### Save dataframe as csv    
refractory_table.to_csv("Analyses/analyses_results/Refractory_table_pulse_trains.csv", index=False, header=True)   


#model_name = "rattay_01"
#pulse_rate = 1200
#delta = 1*us
#stimulation_type = "extern"
#pulse_form = "bi"
#phase_durations = [t_phase/us,t_ipg/us,t_phase/us]*us
#pulse_train_duration = t_pulse_train
#threshold = 0
#amp_masker = 0
#print_progress = True
#time_before = 2*ms
