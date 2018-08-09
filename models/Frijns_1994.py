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
# Potentials
# =============================================================================
##### Resting potential (calculated with Goldman equation)
V_res = (R*T_kelvin)/F * np.log((P_K*n_init**2*K_e + P_Na*h_init*m_init**3*Na_e)/(P_K*n_init**2*K_i + P_Na*h_init*m_init**3*Na_i))

##### Nerst potential for leakage current; leakage chanels were excluded but could be added by using: g_L*(E_L-(v-V_res))  
E_L = (-1/g_L)*(P_Na*m_init**3*h_init*(V_res*F**2)/(R*T_kelvin) * (Na_e-Na_i*exp(V_res*F/(R*T_kelvin)))/(1-exp(V_res*F/(R*T_kelvin))) + P_K*n_init**2*(V_res*F**2)/(R*T_kelvin) * (K_e-K_i*exp(V_res*F/(R*T_kelvin)))/(1-exp(V_res*F/(R*T_kelvin))))

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
Im = P_Na*m**3*h*(v*F**2)/(R*T_kelvin) * (Na_e-Na_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) + P_K*n**2*(v*F**2)/(R*T_kelvin) * (K_e-K_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1): amp/meter**2
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
#  Structure
# =============================================================================
##### structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5)

##### build structure
structure = np.array(list(np.tile([2,1],15)) + [2])
nof_comps = len(structure)

# =============================================================================
#  Compartment lengths
# ============================================================================= 
##### initialize
compartment_lengths = np.zeros_like(structure)*um
##### length internodes
compartment_lengths[structure == 1] = 1500*um
##### length nodes
compartment_lengths[structure == 2] = 1*um

# =============================================================================
#  Compartment middle point distances (needed for plots)
# ============================================================================= 
distance_comps_middle = np.zeros_like(compartment_lengths)

for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]
    
# =============================================================================
#  Total length neuron
# ============================================================================= 
length_neuron = sum(compartment_lengths)

# =============================================================================
# Compartment diameters
# =============================================================================
##### define values
dendrite_diameter = 1*um
##### initialize
compartment_diameters = np.zeros(nof_comps+1)*um
##### dendrite
compartment_diameters[:] = 10.5*um

# =============================================================================
# Capacitivites
# =============================================================================
##### membrane capacitivity
c_m_layer = 2*uF/cm**2
##### initialize
c_m = np.zeros_like(structure)*uF/cm**2
##### internodes
c_m[np.where(structure == 1)] = 0*uF/cm**2
##### nodes
c_m[np.where(structure == 2)] = c_m_layer

# =============================================================================
# Condactivities internodes
# =============================================================================
##### membrane condactivity is zero for internodes
g_m = np.zeros_like(structure)*msiemens/cm**2

# =============================================================================
# Axoplasmatic resistances
# =============================================================================
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2
                                
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

# =============================================================================
# Surface arias
# =============================================================================
##### lateral surfaces
m = [np.sqrt(abs(compartment_diameters[i+1] - compartment_diameters[i])**2 + compartment_lengths[i]**2)
           for i in range(0,nof_comps)]

##### total surfaces
A_surface = [(compartment_diameters[i+1] + compartment_diameters[i])*np.pi*m[i]
           for i in range(0,nof_comps)]

# =============================================================================
# Noise term
# =============================================================================
k_noise = 0.0000001*uA*np.sqrt(second/um**3)
noise_term = np.sqrt(A_surface*P_Na)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 3*mm

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Frijns et al. 1994"

# =============================================================================
# Compartments to plot
# =============================================================================
comps_to_plot = range(1,nof_comps)

# =============================================================================
# Set up the model
# =============================================================================
def set_up_model(dt, model, model_name = "model"):
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
                           method="exponential_euler")
    
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
    param_string = f'''
    V_res = {model_name}.V_res
    T_celsius = {model_name}.T_celsius
    T_kelvin = {model_name}.T_kelvin
    Na_i = {model_name}.Na_i
    Na_e = {model_name}.Na_e
    K_i = {model_name}.K_i
    K_e = {model_name}.K_e
    '''
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string
