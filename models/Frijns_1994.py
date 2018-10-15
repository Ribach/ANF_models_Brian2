##### import needed packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np

##### import functions
import functions.calculations as calc

# =============================================================================
# Temperature
# =============================================================================
T_celsius = 37
T_kelvin = zero_celsius + T_celsius*kelvin

# =============================================================================
# Permeabilities
# =============================================================================
P_Na = 51.5*um/second
P_K = 2.0*um/second

# =============================================================================
# Conductivities
# =============================================================================
# conductivity of leakage channels; not needed, as the leakage channels were excluded
g_L = 728*siemens/meter**2

# =============================================================================
# Ion concentrations
# =============================================================================
Na_i = 10*mole/meter**3
Na_e = 142*mole/meter**3
K_i = 141*mole/meter**3
K_e = 4.2*mole/meter**3

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 70*ohm*cm
##### external resistivity
rho_out = 300*ohm*cm

# =============================================================================
# Initial values for gating variables (steady state values at resting potential)
# =============================================================================
m_init = 0.00775
n_init = 0.0268
h_init = 0.7469

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
I_Na = P_Na*m**3*h*(v*F**2)/(R*T_kelvin) * (Na_e-Na_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) : amp/meter**2
I_K = P_K*n**2*(v*F**2)/(R*T_kelvin) * (K_e-K_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) : amp/meter**2
Im = I_Na + I_K: amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m = 0.49/mV*((v-V_res)-25.41*mV)/(1-exp((25.41*mV-(v-V_res))/(6.06*mV))) * 2.2**(0.1*(T_celsius-20))/ms : Hz
alpha_n = 0.02/mV*((v-V_res)-35*mV)/(1-exp((35*mV-(v-V_res))/(10*mV))) * 3**(0.1*(T_celsius-20))/ms : Hz
alpha_h = 0.09/mV*(-27.74*mV-(v-V_res))/(1-exp(((v-V_res)+27.74*mV)/(9.06*mV))) * 2.9**(0.1*(T_celsius-20))/ms : Hz
beta_m = 1.04/mV*(21*mV-(v-V_res))/(1-exp(((v-V_res)-21*mV)/(9.41*mV))) * 2.2**(0.1*(T_celsius-20))/ms : Hz
beta_n = 0.05/mV*(10*mV-(v-V_res))/(1-exp(((v-V_res)-10*mV)/(10*mV))) * 3**(0.1*(T_celsius-20))/ms : Hz
beta_h = 3.7/(1+exp((56*mV-(v-V_res))/(12.5*mV))) * 2.9**(0.1*(T_celsius-20))/ms : Hz
P_Na : meter/second
P_K : meter/second
'''

# =============================================================================
#  Morphologic data
# ============================================================================= 
##### structure
nof_internodes = 15
##### lengths
length_internodes = 1500*um
length_nodes = 1*um
##### diameters
fiber_outer_diameter = 15*um

# =============================================================================
# Capacitivites
# =============================================================================
##### membrane capacitivity
c_m_layer = 2*uF/cm**2

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.0000005*uA*np.sqrt(second/um**3)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 3*mm

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Frijns et al. 1994"
display_name_short = "Frijns 94"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.append(np.linspace(0.5, 0.6, num=50, endpoint = False),
                                            np.linspace(0.6, 1.2, num=10, endpoint = False)),
                                            np.linspace(1.2, 8, num=15))*1e-3

# =============================================================================
# Calculations
# =============================================================================
###### Temperature in Kelvin
T_kelvin = zero_celsius + T_celsius*kelvin

##### Potentials
# Resting potential (calculated with Goldman equation)
V_res = (R*T_kelvin)/F * np.log((P_K*n_init**2*K_e + P_Na*h_init*m_init**3*Na_e)/
         (P_K*n_init**2*K_i + P_Na*h_init*m_init**3*Na_i))

# Nerst potential for leakage current; leakage chanels were excluded but could be added by using: g_L*(E_L-(v-V_res))  
E_L = (-1/g_L)*(P_Na*m_init**3*h_init*(V_res*F**2)/(R*T_kelvin) * 
       (Na_e-Na_i*exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))) + 
       P_K*n_init**2*(V_res*F**2)/(R*T_kelvin) *
       (K_e-K_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))))


##### structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5)
structure = np.array(list(np.tile([2,1],nof_internodes)) + [2])
nof_comps = len(structure)

##### Compartment lengths
# initialize
compartment_lengths = np.zeros_like(structure)*um
# length internodes
compartment_lengths[structure == 1] = length_internodes
# length nodes
compartment_lengths[structure == 2] = length_nodes
# total length neuron
length_neuron = sum(compartment_lengths)

##### Compartment diameters
compartment_diameters = np.zeros(nof_comps+1)*um
fiber_inner_diameter = 0.7* fiber_outer_diameter
compartment_diameters[:] = fiber_inner_diameter

#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]

##### Capacitivites
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# internodes
c_m[np.where(structure == 1)] = 0*uF/cm**2
# nodes
c_m[np.where(structure == 2)] = c_m_layer

##### Condactivities internodes
# membrane condactivity is zero for internodes
g_m = np.zeros_like(structure)*msiemens/cm**2

##### Axoplasmatic resistances
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2                                
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

##### Surface arias
# lateral surfaces
m = [np.sqrt(abs(compartment_diameters[i+1] - compartment_diameters[i])**2 + compartment_lengths[i]**2)
           for i in range(0,nof_comps)]
# total surfaces
A_surface = [(compartment_diameters[i+1] + compartment_diameters[i])*np.pi*m[i]*0.5
           for i in range(0,nof_comps)]

##### Noise term
P_Na_vector = np.zeros(nof_comps)*um/second
P_Na_vector[structure == 2] = P_Na
noise_term = np.sqrt(A_surface*P_Na_vector)

##### Compartments to plot
comps_to_plot = range(1,nof_comps)

# =============================================================================
# Set up the model
# =============================================================================
def set_up_model(dt, model, update = False, model_name = "model"):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    dt : time
        Sets the defaultclock.
    model : module
        Contains all morphologic and physiologic data of a model
    model_name : string
        Sting with the variable name, in which the module is saved
                
    Returns
    -------
    neuron
        Gives back a brian2 neuron
    param_string
        Gives back a string of parameter assignments
    """
    
    start_scope()
    
    ##### Update model parameters (should be done, if original parameters have been changed)
    if update:
        ###### Temperature in Kelvin
        model.T_kelvin = model.zero_celsius + model.T_celsius*kelvin
        
        ##### Potentials
        # Resting potential (calculated with Goldman equation)
        model.V_res = (model.R*model.T_kelvin)/model.F * np.log((model.P_K*model.n_init**2*model.K_e + model.P_Na*model.h_init*model.m_init**3*model.Na_e)/\
                 (model.P_K*model.n_init**2*model.K_i + model.P_Na*model.h_init*model.m_init**3*model.Na_i))
        
        # Nerst potential for leakage current; leakage chanels were excluded but could be added by using: g_L*(E_L-(v-V_res))  
        model.E_L = (-1/model.g_L)*(model.P_Na*model.m_init**3*model.h_init*(model.V_res*model.F**2)/(model.R*model.T_kelvin) * \
               (model.Na_e-model.Na_i*exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) + \
               model.P_K*model.n_init**2*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *\
               (model.K_e-model.K_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))))
        
        
        ##### structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        # postsomatic region = 5)
        model.structure = np.array(list(np.tile([2,1],model.nof_internodes)) + [2])
        model.nof_comps = len(model.structure)
        
        ##### Compartment lengths
        # initialize
        model.compartment_lengths = np.zeros_like(model.structure)*um
        # length internodes
        model.compartment_lengths[model.structure == 1] = model.length_internodes
        # length nodes
        model.compartment_lengths[model.structure == 2] = model.length_nodes
        # total length neuron
        model.length_neuron = sum(model.compartment_lengths)
        
        ##### Compartment diameters
        # initialize
        model.compartment_diameters = np.zeros(model.nof_comps+1)*um
        # dendrite
        model.fiber_inner_diameter = 0.7* model.fiber_outer_diameter
        model.compartment_diameters[:] = model.fiber_inner_diameter
        
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5* model.compartment_lengths[ii] + 0.5* model.compartment_lengths[ii+1]
        
        ##### Capacitivites
        # initialize
        model.c_m = np.zeros_like(model.structure)*uF/cm**2
        # internodes
        model.c_m[np.where(model.structure == 1)] = 0*uF/cm**2
        # nodes
        model.c_m[np.where(model.structure == 2)] = model.c_m_layer
        
        ##### Condactivities internodes
        # membrane condactivity is zero for internodes
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        
        ##### Axoplasmatic resistances
        model.compartment_center_diameters = np.zeros(model.nof_comps)*um
        model.compartment_center_diameters = (model.compartment_diameters[0:-1] + model.compartment_diameters[1:]) / 2                                
        model.R_a = (model.compartment_lengths*model.rho_in) / ((model.compartment_center_diameters*0.5)**2*np.pi)
        
        ##### Surface arias
        # lateral surfaces
        m = [np.sqrt(abs(model.compartment_diameters[i+1] - model.compartment_diameters[i])**2 + model.compartment_lengths[i]**2)
                   for i in range(0,model.nof_comps)]
        # total surfaces
        model.A_surface = [(model.compartment_diameters[i+1] + model.compartment_diameters[i])*np.pi*m[i]*0.5
                   for i in range(0,model.nof_comps)]
        
        ##### Noise term
        model.P_Na_vector = np.zeros(model.nof_comps)*um/second
        model.P_Na_vector[model.structure == 2] = model.P_Na
        model.noise_term = np.sqrt(model.A_surface*model.P_Na_vector)
        
        ##### Compartments to plot
        model.comps_to_plot = range(1,model.nof_comps)
    
    ##### initialize defaultclock
    defaultclock.dt = dt

    ##### define morphology
    morpho = Section(n = model.nof_comps,
                     length = model.compartment_lengths,
                     diameter = model.compartment_diameters)
    
    ##### define neuron
    neuron = SpatialNeuron(morphology = morpho,
                           model = model.eqs,
                           Cm = model.c_m,
                           Ri = model.rho_in,
                           method="rk4")
    
    ##### initial values
    neuron.v = V_res
    neuron.m = model.m_init
    neuron.n = model.n_init
    neuron.h = model.h_init
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
    # permeabilities nodes
    neuron.P_Na = model.P_Na
    neuron.P_K = model.P_K
    
    # permeabilities internodes
    neuron.P_Na[np.asarray(np.where(model.structure == 1))] = 0*meter/second
    neuron.P_K[np.asarray(np.where(model.structure == 1))] = 0*meter/second
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = '''
    V_res = {}.V_res
    T_celsius = {}.T_celsius
    T_kelvin = {}.T_kelvin
    Na_i = {}.Na_i
    Na_e = {}.Na_e
    K_i = {}.K_i
    K_e = {}.K_e
    '''.format(model_name,model_name,model_name,model_name,model_name,model_name,model_name)
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string, model
