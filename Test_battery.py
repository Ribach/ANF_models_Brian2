##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import thorns as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### import functions
import functions.stimulation as stim
import functions.create_plots as plot
import functions.model_tests as test

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
# Definition of neuron and initialization of state monitor
# =============================================================================
##### choose model
model = rattay_01

##### initialize clock
dt = 5*us

##### set up the neuron
neuron, param_string, model = model.set_up_model(dt = dt, model = model)

##### load the parameters of the differential equations in the workspace
exec(param_string)

##### record the membrane voltage
M = StateMonitor(neuron, 'v', record=True)

##### save initialization of the monitor(s)
store('initialized')

# =============================================================================
# Get chronaxie and rheobase
# =============================================================================
rheobase_matrix = test.get_thresholds(model = model,
                                      dt = 5*us,
                                      phase_durations = 1*ms,
                                      amps_start_intervall = [0,20]*uA,
                                      delta = 0.0001*uA,
                                      stimulation_type = "extern",
                                      pulse_form = "mono")

rheobase = rheobase_matrix["threshold"][0]*amp

chronaxie = test.get_chronaxie(model = model,
                               dt = dt,
                               rheobase = rheobase,
                               phase_duration_start_intervall = [0,1000]*us,
                               delta = 1*us,
                               stimulation_type = "extern",
                               pulse_form = "mono")

# =============================================================================
# Get strength-duration curve
# =============================================================================
##### define phase durations
phase_durations = np.round(np.logspace(1, 9, num=20, base=2.0))*us

##### calculate deterministic thresholds for the given phase durations and monophasic stimulation
strength_duration_matrix = test.get_thresholds(model = model,
                                               dt = 1*us,
                                               phase_durations = phase_durations,
                                               amps_start_intervall = [0,20]*uA,
                                               delta = 0.01*uA,
                                               nof_runs = 1,
                                               stimulation_type = "extern",
                                               pulse_form = "mono")

##### plot strength duration curve
strength_duration_curve = plot.strength_duration_curve(plot_name = f"Strength duration curve {model.display_name}",
                                                       threshold_matrix = strength_duration_matrix,
                                                       rheobase = rheobase,
                                                       chronaxie = chronaxie)



# =============================================================================
# Get thresholds for certain stimulations stimulus durations
# =============================================================================
phase_durations_mono = [40,50,100]*us
phase_durations_bi = [20,40,50,200,400]*us

thresholds = pd.DataFrame(np.zeros((len(phase_durations)*nof_runs, 3)), columns = ["phase duration","run","threshold"])

















