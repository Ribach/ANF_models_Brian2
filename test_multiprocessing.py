##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import thorns as th
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

##### import functions
import functions.multiprocessing_functions as mp

# =============================================================================
# Test multiprocessing
# =============================================================================
##### initialize clock
dt = 5*us

##### define phase durations to test (in us)
phase_durations_mono = [100]
phase_durations_bi = [200]

##### define test parameters
phase_durations = phase_durations_mono + phase_durations_bi
pulse_form = np.repeat(["mono","bi"], (len(phase_durations_mono),len(phase_durations_bi)))
runs_per_stimulus_type = 2

##### include noise    
xs = {  'phase_duration' : phase_durations,
        'run_number' : range(0,runs_per_stimulus_type)
}

ys = th.util.map(func = mp.get_threshold,
             space = xs,
             backend='serial',
             cache = 'no',
             kwargs = {'model_name' : 'rattay_01',
                       'dt' : dt,
                       'delta' : 0.0005*uA,
                       'pulse_form' : 'mono',
                       'stimulation_type' : 'extern',
                       'amps_start_intervall' : [0,20]*uA,
                       'time_before' : 2*ms,
                       'time_after' : 2*ms,
                       'add_noise' : True}
             )
