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
#  Morphologic data
# ============================================================================= 
##### structure
nof_internodes = 25
nof_segments_internode = 1
nof_myelin_layers = 35
thicknes_myelin_layer = 8.5*nmeter # value form Rattay2001
##### lengths
length_internodes = 300*um
length_nodes = 1.5*um
##### diameters
diameter_fiber = 1.0*um

# =============================================================================
# Conductivity
# =============================================================================
##### conductivity of leakage channels
g_L = (1953.49*Mohm)**-1/(1*um*np.pi*1.5*um)

# =============================================================================
# Total ion channel numbers per node
# =============================================================================
max_Na = 1000
max_K = 166
max_KLT = 166
max_HCN = 100

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
# Capacities
# =============================================================================
##### cell membrane capacity
c_mem = 0.1*pF/(1*um*np.pi*1.5*um)
##### myelin layer capacity
c_my = 0.02*pF/(1*um*np.pi*1.5*um)

# =============================================================================
# Condactivities internodes
# =============================================================================
##### cell membrane conductivity internodes
r_mem = 10*kohm*cm**2
##### cell membrane conductivity internodes
r_my = 0.1*kohm*cm**2

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.01*uA/np.sqrt(mS)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Negm and Bruce 2014"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(1.28, 1.29, num=20, endpoint = False),
                                  np.linspace(1.29, 1.5, num=20, endpoint = False),
                                  np.linspace(1.5, 4, num=20))*1e-3

# =============================================================================
# Calculations
# =============================================================================
#####  Structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5
structure = np.array(list(np.tile([2] + np.tile([1],nof_segments_internode).tolist(),nof_internodes)) + [2])
nof_comps = len(structure)

##### compartment lengths
# initialize
compartment_lengths = np.zeros_like(structure)*um
# length internodes
compartment_lengths[structure == 1] = length_internodes / nof_segments_internode
# length nodes
compartment_lengths[structure == 2] = length_nodes
# total length neuron
length_neuron = sum(compartment_lengths)

##### Compartment diameters
# initialize
compartment_diameters = np.zeros(nof_comps+1)*um
# same diameter for whole fiber
compartment_diameters[:] = diameter_fiber
fiber_outer_diameter = diameter_fiber + nof_myelin_layers*thicknes_myelin_layer*2

##### nodal surface aria
surface_aria_node = diameter_fiber*np.pi*length_nodes

##### ion channels per aria
rho_Na = max_Na/surface_aria_node
rho_K = max_K/surface_aria_node
rho_KLT = max_KLT/surface_aria_node
rho_HCN = max_HCN/surface_aria_node

##### Surface arias
# lateral surfaces
m = [np.sqrt(abs(compartment_diameters[i+1] - compartment_diameters[i])**2 + compartment_lengths[i]**2)
           for i in range(0,nof_comps)]
# total surfaces
A_surface = [(compartment_diameters[i+1] + compartment_diameters[i])*np.pi*m[i]*0.5
           for i in range(0,nof_comps)]

##### Reverse potential of leakage channels
# The reverse potential of the leakage channels is calculated by using
# I_Na + I_K + I_KLT + I_HCN * I_L = 0 with v = V_res and the initial values for
# the gating variables.
E_L = -(gamma_Na*rho_Na*m_init**3*h_init*E_Na +  gamma_K*rho_K*n_init**4*(E_K+V_res) +
        gamma_KLT*rho_KLT*w_init**4*z_init*E_K + gamma_HCN*rho_HCN*r_init*E_HCN) / g_L
        
#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]

##### Capacities
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# all but internodes axon
c_m[structure == 2] = c_mem
# axonal internodes
c_m[structure == 1] = 1/(1/c_mem + nof_myelin_layers/c_my)

##### Condactivities internodes
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# axonal internodes
g_m[structure == 1] = 1/(r_mem + nof_myelin_layers*r_my)

##### Axoplasmatic resistances
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

##### Noise term
gamma_Na_vector = np.zeros(nof_comps)*psiemens
gamma_Na_vector[structure == 2] = gamma_Na
noise_term = np.sqrt(A_surface*gamma_Na_vector*rho_Na)

##### Compartments to plot
# get indexes of all compartments that are not segmented
indexes_comps = np.where(structure == 2)[0]
# calculate middle compartments of internodes
middle_comps_internodes = np.ceil(indexes_comps[:-1] + nof_segments_internode/2).astype(int)
# create array with all compartments to plot
comps_to_plot = np.sort(np.append(indexes_comps, middle_comps_internodes))

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
        #####  Structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        # postsomatic region = 5
        model.structure = np.array(list(np.tile([2] + np.tile([1],model.nof_segments_internode).tolist(),model.nof_internodes)) + [2])
        model.nof_comps = len(model.structure)
        
        ##### compartment lengths
        # initialize
        model.compartment_lengths = np.zeros_like(structure)*um
        # length internodes
        model.compartment_lengths[model.structure == 1] = model.length_internodes / model.nof_segments_internode
        # length nodes
        model.compartment_lengths[model.structure == 2] = model.length_nodes
        # total length neuron
        model.length_neuron = sum(model.compartment_lengths)
        
        ##### Compartment diameters
        # initialize
        model.compartment_diameters = np.zeros(model.nof_comps+1)*um
        # same diameter for whole fiber
        model.compartment_diameters[:] = model.diameter_fiber
        
        ##### Surface arias
        # lateral surfaces
        m = [np.sqrt(abs(model.compartment_diameters[i+1] - model.compartment_diameters[i])**2 + model.compartment_lengths[i]**2)
                   for i in range(0,model.nof_comps)]
        # total surfaces
        model.A_surface = [(model.compartment_diameters[i+1] + model.compartment_diameters[i])*np.pi*m[i]*0.5
                   for i in range(0,model.nof_comps)]
    
        ##### Reverse potential of leakage channels
        # The reverse potential of the leakage channels is calculated by using
        # I_Na + I_K + I_KLT + I_HCN * I_L = 0 with v = V_res and the initial values for
        # the gating variables.
        model.E_L = -(model.gamma_Na*rho_Na*model.m_init**3*model.h_init*model.E_Na +  model.gamma_K*rho_K*model.n_init**4*(model.E_K+model.V_res) +\
                      model.gamma_KLT*rho_KLT*model.w_init**4*model.z_init*model.E_K + model.gamma_HCN*rho_HCN*model.r_init*model.E_HCN) / g_L
                
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5* model.compartment_lengths[ii] + 0.5* model.compartment_lengths[ii+1]
        
        ##### Capacities
        # initialize
        model.c_m = np.zeros_like(model.structure)*uF/cm**2
        # all but internodes axon
        model.c_m[model.structure == 2] = model.c_mem
        # axonal internodes
        model.c_m[model.structure == 1] = 1/(1/model.c_mem + model.nof_myelin_layers/c_my)
        
        ##### Condactivities internodes
        # initialize
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        # axonal internodes
        model.g_m[model.structure == 1] = 1/(model.r_mem + model.nof_myelin_layers*model.r_my)
        
        ##### Axoplasmatic resistances
        model.compartment_center_diameters = np.zeros(model.nof_comps)*um
        model.compartment_center_diameters = (model.compartment_diameters[0:-1] + model.compartment_diameters[1:]) / 2
        model.R_a = (model.compartment_lengths*model.rho_in) / ((model.compartment_center_diameters*0.5)**2*np.pi)
        
        ##### Noise term
        model.gamma_Na_vector = np.zeros(model.nof_comps)*psiemens
        model.gamma_Na_vector[model.structure == 2] = model.gamma_Na
        model.noise_term = np.sqrt(model.A_surface*model.gamma_Na_vector*model.rho_Na)
        
        ##### Compartments to plot
        # get indexes of all compartments that are not segmented
        model.indexes_comps = np.where(model.structure == 2)[0]
        # calculate middle compartments of internodes
        model.middle_comps_internodes = np.ceil(model.indexes_comps[:-1] + model.nof_segments_internode/2).astype(int)
        # create array with all compartments to plot
        model.comps_to_plot = np.sort(np.append(model.indexes_comps, model.middle_comps_internodes))
            
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
    param_string = '''
    V_res = {}.V_res
    T_celsius = {}.T_celsius
    E_Na = {}.E_Na
    E_K = {}.E_K
    E_HCN = {}.E_HCN
    E_L = {}.E_L
    rho_Na = {}.rho_Na
    rho_K = {}.rho_K
    rho_KLT = {}.rho_KLT
    rho_HCN = {}.rho_HCN
    '''.format(model_name,model_name,model_name,model_name,model_name,model_name,model_name,model_name,model_name,model_name)
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string, model
