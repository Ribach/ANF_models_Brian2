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
# Ionic concentrations
# =============================================================================
##### Na_e / Na_i
Na_ratio = 7.210
K_ratio = 0.036
Leak_ratio = 0.0367

# =============================================================================
# Nernst potentials
# =============================================================================
##### Resting potential of cell
V_res = -79.4*mV *1.035**((T_celsius-6.3)/10)
##### Nernst potential sodium
E_Na = R*T_kelvin/F * log(Na_ratio) - V_res
##### Nernst potential potassium
E_K = R*T_kelvin/F * log(K_ratio) - V_res
##### Nerst potential for leakage current
E_L = R*T_kelvin/F * log(Leak_ratio) - V_res

# =============================================================================
# Conductivities
# =============================================================================
##### conductivities active compartments
g_Na = 640*msiemens/cm**2 * 1.02**((T_celsius-24)/10)
g_K = 60*msiemens/cm**2 * 1.16**((T_celsius-20)/10)
g_L = 57.5*msiemens/cm**2 * 1.418**((T_celsius-24)/10)

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 25*ohm*cm * (1/1.35)**((T_celsius-37)/10)
##### external resistivity
rho_out = 300*ohm*cm

# =============================================================================
# Initial values for gating variables
# =============================================================================
m_t_init = 0.05
m_p_init = 0.05
n_init = 0.32
h_init = 0.6

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
Im = 0.975*g_Na*m_t**3*h*(E_Na-(v-V_res)) + 0.025*g_Na*m_p**3*h*(E_Na-(v-V_res)) + g_K*n**4*(E_K-(v-V_res)) + g_L*(E_L-(v-V_res)) + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm_t/dt = alpha_m_t * (1-m_t) - beta_m_t * m_t : 1
dm_p/dt = alpha_m_p * (1-m_p) - beta_m_p * m_p : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m_t = 4.42*(2.5-0.1*(v-V_res)/mV)/(1*(exp(2.5-0.1*(v-V_res)/mV))-1) * 2.23**(0.1*(T_celsius-20))/ms : Hz
alpha_m_p = 2.06*(2.5-0.1*((v-V_res)/mV-20))/(1*(exp(2.5-0.1*((v-V_res)/mV-20)))-1) * 1.99**(0.1*(T_celsius-20))/ms : Hz
alpha_n = 0.2*(1.0-0.1*(v-V_res)/mV)/(10*(exp(1-0.1*(v-V_res)/mV)-1)) * 1.5**(0.1*(T_celsius-20))/ms : Hz
alpha_h = 1.47*0.07*exp(-(v-V_res)/mV/20) * 1.5**(0.1*(T_celsius-20))/ms : Hz
beta_m_t = 4.42*4.0*exp(-(v-V_res)/mV/18) * 2.23**(0.1*(T_celsius-20))/ms : Hz
beta_m_p = 2.06*4.0*exp(-((v-V_res)/mV-20)/18) * 1.99**(0.1*(T_celsius-20))/ms : Hz
beta_n = 0.2*0.125*exp(-(v-V_res)/mV/80) * 1.5**(0.1*(T_celsius-20))/ms : Hz
beta_h = 1.47/(1+exp(3.0-0.1*(v-V_res)/mV)) * 1.5**(0.1*(T_celsius-20))/ms : Hz
g_Na : siemens/meter**2
g_K : siemens/meter**2
g_L : siemens/meter**2
g_myelin : siemens/meter**2
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

nof_internodes = 22

##### build structure
structure = np.array(list(np.tile([2,1],22)) + [2])

nof_comps = len(structure)

# =============================================================================
# Compartment diameters
# =============================================================================
##### define values
internode_outer_diameter = 15*um
internode_inner_diameter = 0.63*internode_outer_diameter - 3.4*10**-5*cm
node_diameter =  internode_inner_diameter #(8.502*10**5*(internode_outer_diameter/cm)**3 - 1.376*10**3*(internode_outer_diameter/cm)**2 + 8.202*10**-1*(internode_outer_diameter/cm) - 3.622*10**-5)*cm
##### initialize
compartment_diameters = np.zeros(nof_comps+1)*um
##### internodes
compartment_diameters[:] = internode_inner_diameter
##### nodes
#compartment_diameters[np.where(structure == 2)] = node_diameter

# =============================================================================
#  Compartment lengths
# ============================================================================= 
##### initialize
compartment_lengths = np.zeros_like(structure)*um
##### internodes
compartment_lengths[np.where(structure == 1)] = 7.9*10**-2*np.log((internode_outer_diameter/cm)/(3.4*10**-4))*cm
##### nodes
compartment_lengths[np.where(structure == 2)] = 1.061*um

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
# Myelin data
# =============================================================================
myelin_layer_thicknes = 16*nmeter
myelin_layers = np.floor(0.5*(internode_outer_diameter-internode_inner_diameter)/myelin_layer_thicknes)

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacities one layer
c_mem = 2.8*uF/cm**2
##### myelin layer capacity
c_my = 0.6*uF/cm**2

##### initialize
c_m = np.zeros_like(structure)*uF/cm**2
##### nodes
c_m[np.where(structure == 2)] = c_mem
##### internodes
c_m[structure == 1] = 1/(1/c_mem + myelin_layers/c_my)

# =============================================================================
# Condactivities internodes
# =============================================================================
##### cell membrane conductivity internodes
r_mem = 4.871*10**4*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)
##### cell membrane conductivity internodes
r_my = 104*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)

##### initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
##### calculate values
g_m[structure == 1] = 1/(r_mem + myelin_layers*r_my)

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
k_noise = 0.0006*uA/np.sqrt(mS)
noise_term = np.sqrt(A_surface*g_Na)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 2*mm

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Smit et al. 2009"

# =============================================================================
# Compartments to plot
# =============================================================================
comps_to_plot = range(1,nof_comps)

# =============================================================================
# Set up the model
# =============================================================================
def set_up_model(dt, model):
    """This function calculates the stimulus current at the current source for
    a single monophasic pulse stimulus at each point of time

    Parameters
    ----------
    dt : time
        Sets the defaultclock.
    model : module
        Contains all morphologic and physiologic data of a model
                
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
    neuron.m_t = model.m_t_init
    neuron.m_p = model.m_p_init
    neuron.n = model.n_init
    neuron.h = model.h_init
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
    # conductances active compartments
    neuron.g_Na = model.g_Na
    neuron.g_K = model.g_K
    neuron.g_L = model.g_L
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.g_Na[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_K[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_L[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = '''
    V_res = model.V_res
    E_Na = model.E_Na
    E_K = model.E_K
    E_L = model.E_L
    T_celsius = model.T_celsius
    '''
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string