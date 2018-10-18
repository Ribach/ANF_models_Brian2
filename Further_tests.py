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
import itertools as itl
import os

##### import functions
import functions.stimulation as stim
import functions.create_plots as plot
import functions.model_tests as test
import functions.calculations as calc

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
# Initializations
# =============================================================================
##### choose model
model_name = "rattay_01"
model = eval(model_name)

##### initialize clock
dt = 5*us

##### define way of processing
backend = "serial"

##### define if plots should be generated
generate_plots = True

##### define which tests to run
all_tests = False
strength_duration_test = False
relative_spread_test = False
conduction_velocity_test = True
single_node_response_test = False
refractory_periods = False
refractory_curve = False
psth_test = False

