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

# =============================================================================
# Nernst potentials
# =============================================================================
##### Resting potential of cell
V_res = -78*mV
##### Nernst potential sodium
E_Na = 66*mV - V_res
##### Nernst potential potassium (normal and low threshold potassium (KLT) channels)
E_K = -88*mV - V_res
##### Reversal potential hyperpolarization-activated cation (HCN) channels
E_HCN = -43*mV - V_res

# =============================================================================
# Conductivities
# =============================================================================
##### conductivities active compartments
gamma_Na = 25.69*psiemens
gamma_K = 50*psiemens
gamma_KLT = 13*psiemens
gamma_HCN = 13*psiemens

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

##### number of segments per internode
nof_segments_internode = 1

##### build structure
structure = np.array(list(np.tile([2] + np.tile([1],nof_segments_internode).tolist(),30)) + [2])
nof_comps = len(structure)

##### number of myelin layers
myelin_layers = 35

# =============================================================================
#  Compartment lengths
# ============================================================================= 
##### initialize
compartment_lengths = np.zeros_like(structure)*um
##### length internodes
compartment_lengths[structure == 1] = 300*um / nof_segments_internode
##### length nodes
compartment_lengths[structure == 2] = 1.5*um

# =============================================================================
# Compartment diameters
# =============================================================================
##### initialize
compartment_diameters = np.zeros(nof_comps+1)*um
##### dendrite
compartment_diameters[:] = 1.0*um

# =============================================================================
# Surface arias
# =============================================================================
##### lateral surfaces
m = [np.sqrt(abs(compartment_diameters[i+1] - compartment_diameters[i])**2 + compartment_lengths[i]**2)
           for i in range(0,nof_comps)]

##### total surfaces
A_surface = [(compartment_diameters[i+1] + compartment_diameters[i])*np.pi*m[i]
           for i in range(0,nof_comps)]

##### nodal surface
node_surface_aria = np.array(A_surface)[structure == 2][0]*meter**2

# =============================================================================
# Conductivities
# =============================================================================
# conductivity of leakage channels
g_L = (1953.49*Mohm)**-1/node_surface_aria

# =============================================================================
# Numbers of channels per aria
# =============================================================================
rho_Na = 1000/node_surface_aria
rho_K = 166/node_surface_aria
rho_KLT = 166/node_surface_aria
rho_HCN = 100/node_surface_aria

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 50*ohm*cm
##### external resistivity
rho_out = 300*ohm*cm

# =============================================================================
# Initial values for gating variables
# =============================================================================
m_init = 0.00775
n_init = 0.0119
h_init = 0.747
w_init = 0.5127
z_init = 0.6615
r_init = 0.1453

# =============================================================================
# Reverse potential of leakage channels
# =============================================================================
# The reverse potential of the leakage channels is calculated by using
# I_Na + I_K + I_KLT + I_HCN * I_L = 0 with v = V_res and the initial values for
# the gating variables.
E_L = -(gamma_Na*rho_Na*m_init**3*h_init*E_Na +  gamma_K*rho_K*n_init**4*(E_K+V_res) +\
        gamma_KLT*rho_KLT*w_init**4*z_init*E_K + gamma_HCN*rho_HCN*r_init*E_HCN) / g_L

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
I_Na = gamma_Na*rho_Na*m**3*h* (E_Na-(v-V_res)) : amp/meter**2
I_K = gamma_K*rho_K*n**4*(E_K-(v-V_res)) : amp/meter**2
I_KLT = gamma_KLT*rho_KLT*w**4*z*(E_K-(v-V_res)) : amp/meter**2
I_HCN = gamma_HCN*rho_HCN*r*(E_HCN-(v-V_res)) : amp/meter**2
I_L = g_L*(E_L-(v-V_res)) : amp/meter**2
Im = I_Na + I_K + I_KLT + I_HCN + I_L + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
dw/dt = alpha_w * (1-w) - beta_w * w : 1
dz/dt = alpha_z * (1-z) - beta_z * z : 1
dr/dt = alpha_r * (1-r) - beta_r * r : 1
alpha_m = 1.875/mV*((v-V_res)-25.41*mV)/(1-exp((25.41*mV-(v-V_res))/(6.6*mV)))/ms : Hz
beta_m = 3.973/mV*(21.001*mV-(v-V_res))/(1-exp(((v-V_res)-21.001*mV)/(9.41*mV)))/ms : Hz
alpha_h = -0.549/mV*(27.74*mV + (v-V_res))/(1-exp(((v-V_res)+27.74*mV)/(9.06*mV)))/ms : Hz
beta_h = 22.57/(1+exp((56.0*mV-(v-V_res))/(12.5*mV)))/ms : Hz
alpha_n = 0.129/mV*((v-V_res)-35*mV)/(1-exp((35*mV-(v-V_res))/(10*mV)))/ms : Hz
beta_n = 0.3236/mV*(35*mV-(v-V_res))/(1-exp(((v-V_res)-35*mV)/(10*mV)))/ms : Hz
w_inf = 1/(exp(13/5-(v-V_res)/(6*mV))+1)**(1/4) : 1
tau_w = 0.2887 + (17.53*exp((v-V_res)/(45*mV)))/(3*exp(17*(v-V_res)/(90*mV))+15.791) : 1
alpha_w = w_inf/tau_w * 3**(0.1*(T_celsius-37))/ms : Hz
beta_w = (1-w_inf)/tau_w * 3**(0.1*(T_celsius-37))/ms : Hz
z_inf = 1/(2*(exp((v-V_res)/(10*mV)+0.74)+1))+0.5 : 1
tau_z = 9.6225 + (2073.6*exp((v-V_res)/(8*mV)))/(9*(exp(7*(v-V_res)/(40*mV))+1.8776)) : 1
alpha_z = z_inf/tau_z * 3**(0.1*(T_celsius-37))/ms : Hz
beta_z = (1-z_inf)/tau_z * 3**(0.1*(T_celsius-37))/ms : Hz
r_inf = 1/(exp((v-V_res)/(7*mV)+62/35)+1) : 1
tau_r = 50000/(711*exp((v-V_res)/(12*mV)-3/10)+51*exp(9/35-(v-V_res)/(14*mV)))+25/6 : 1
alpha_r = r_inf/tau_r * 3.3**(0.1*(T_celsius-37))/ms : Hz
beta_r = (1-r_inf)/tau_r * 3.3**(0.1*(T_celsius-37))/ms : Hz
gamma_Na : siemens
gamma_K : siemens
gamma_KLT : siemens
gamma_HCN : siemens
g_L : siemens/meter**2
g_myelin : siemens/meter**2
'''

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
# Capacities
# =============================================================================
##### cell membrane capacitiy one layer
c_mem = 0.1*pF/node_surface_aria
##### myelin layer capacity
c_my = 0.02*pF/node_surface_aria

##### initialize
c_m = np.zeros_like(structure)*uF/cm**2
##### all but internodes axon
c_m[structure == 2] = c_mem
##### axonal internodes
c_m[structure == 1] = 1/(1/c_mem + myelin_layers/c_my)

# =============================================================================
# Condactivities internodes
# =============================================================================
##### cell membrane conductivity internodes
r_mem = 10*kohm*cm**2
##### cell membrane conductivity internodes
r_my = 0.1*kohm*cm**2

##### initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
##### axonal internodes
g_m[structure == 1] = 1/(r_mem + myelin_layers*r_my)

# =============================================================================
# Axoplasmatic resistances
# =============================================================================
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2
                                
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

# =============================================================================
# Noise term
# =============================================================================
k_noise = 0.0005*uA/np.sqrt(mS)
noise_term = np.sqrt(A_surface*gamma_Na*rho_Na)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Negm and Bruce 2014"

# =============================================================================
# Compartments to plot
# =============================================================================
##### get indexes of all compartments that are not segmented
indexes_comps = np.where(structure == 2)[0]
##### calculate middle compartments of internodes
middle_comps_internodes = np.ceil(indexes_comps[:-1] + nof_segments_internode/2).astype(int)
##### create array with all compartments to plot
comps_to_plot = np.sort(np.append(indexes_comps, middle_comps_internodes))

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
    neuron.v = model.V_res
    neuron.m = model.m_init
    neuron.h = model.h_init
    neuron.n = model.n_init
    neuron.w = model.w_init
    neuron.z = model.z_init
    neuron.r = model.r_init
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
    # conductances active compartments
    neuron.gamma_Na = model.gamma_Na
    neuron.gamma_K = model.gamma_K
    neuron.gamma_KLT = model.gamma_KLT
    neuron.gamma_HCN = model.gamma_HCN
    neuron.g_L = model.g_L
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.gamma_Na[np.where(model.structure == 1)[0]] = 0*psiemens
    neuron.gamma_K[np.where(model.structure == 1)[0]] = 0*psiemens
    neuron.gamma_KLT[np.where(model.structure == 1)[0]] = 0*psiemens
    neuron.gamma_HCN[np.where(model.structure == 1)[0]] = 0*psiemens
    neuron.g_L[np.where(model.structure == 1)[0]] = 0*msiemens/cm**2
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = f'''
    V_res = {model_name}.V_res
    T_celsius = {model_name}.T_celsius
    E_Na = {model_name}.E_Na
    E_K = {model_name}.E_K
    E_HCN = {model_name}.E_HCN
    E_L = {model_name}.E_L
    rho_Na = {model_name}.rho_Na
    rho_K = {model_name}.rho_K
    rho_KLT = {model_name}.rho_KLT
    rho_HCN = {model_name}.rho_HCN
    '''
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string
