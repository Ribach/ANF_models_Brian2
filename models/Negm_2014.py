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
nof_internodes = 30
nof_segments_internode = 1
##### lengths
length_internodes = 350*um
length_nodes = 2.5*um
##### diameters
diameter_fiber = 1.0*um

# =============================================================================
# Myelin data
# =============================================================================
nof_myelin_layers = 40
thicknes_myelin_layer = 8.5*nmeter

# =============================================================================
# Conductance
# =============================================================================
##### conductance of leakage channels per node
g_L_node = (1953.49*Mohm)**-1

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
V_res : volt
T_celsius : 1
E_Na : volt
E_K : volt
E_L : volt
E_HCN : volt
rho_Na : 1/meter**2
rho_K : 1/meter**2
rho_KLT : 1/meter**2
rho_HCN : 1/meter**2
'''

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacitance node
c_m_node = 0.0714*pF
##### myelin layer capacity internodes
c_m_layer = 1*uF/cm**2

# =============================================================================
# Condactivity internodes
# =============================================================================
##### membrane conductivity internodes one layer
g_m_layer = 1*msiemens/cm**2

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.006*uA/np.sqrt(mS)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 500*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Negm and Bruce 2014"
display_name_plots = "Negm and Bruce (2014)"
display_name_short = "Negm 14"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(1.7, 1.8, num=30, endpoint = False),
                                  np.linspace(1.8, 5, num=20))*1e-3

# =============================================================================
# Calculations
# =============================================================================
##### nodal surface aria
surface_aria_node = diameter_fiber*np.pi*length_nodes

##### conductivity of leakage channels
g_L = g_L_node/surface_aria_node

##### ion channels per aria
rho_Na = max_Na/surface_aria_node
rho_K = max_K/surface_aria_node
rho_KLT = max_KLT/surface_aria_node
rho_HCN = max_HCN/surface_aria_node
                                  
##### rates for resting potential
alpha_m_0 = 1.872*(-25.41)/(1-np.exp(25.41/6.06))
beta_m_0 = 3.793*(21.001)/(1-np.exp(-21.001/9.41))
alpha_h_0 = -0.549*27.74/(1-np.exp(27.74/9.06))
beta_h_0 = 22.57/(1+np.exp(56.0/12.5))
alpha_n_0 = 0.129*(-35)/(1-np.exp((35)/10))
beta_n_0 = 0.324*35/(1-np.exp(-35/10))

##### initial values for gating variables
m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  
w_init = 1/(np.exp(13/5)+1)**(1/4)
z_init = 1/(2*(np.exp(0.74)+1))+0.5
r_init = 1/(np.exp(+62/35)+1)

##### calculate Nerst potential for leakage current
E_L = -(1/g_L)* (gamma_Na*rho_Na*m_init**3*h_init* E_Na + gamma_K*rho_K*n_init**4*E_K + gamma_KLT*rho_KLT*w_init**4*z_init*E_K + gamma_HCN*rho_HCN*r_init*E_HCN)

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
distance_comps_middle[0] = 0.5*compartment_lengths[0]
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]

##### Capacities
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# nodes
c_m[structure == 2] = c_m_node/surface_aria_node
# internodes
c_m[structure == 1] = c_m_layer/(1+nof_myelin_layers)

##### Condactivities internodes
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# internodes
g_m[structure == 1] = g_m_layer/(1+nof_myelin_layers)

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
def set_up_model(dt, model, update = False):
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
    model
        Gives back the whole module
    """
    
    start_scope()
    
    ##### Update model parameters (should be done, if original parameters have been changed)
    if update:
        ##### rates for resting potential
        alpha_m_0 = 1.872*(-25.41)/(1-np.exp(25.41/6.06))
        beta_m_0 = 3.793*(21.001)/(1-np.exp(-21.001/9.41))
        alpha_h_0 = -0.549*27.74/(1-np.exp(27.74/9.06))
        beta_h_0 = 22.57/(1+np.exp(56.0/12.5))
        alpha_n_0 = 0.129*(-35)/(1-np.exp((35)/10))
        beta_n_0 = 0.324*35/(1-np.exp(-35/10))
        
        ##### initial values for gating variables
        model.m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
        model.n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
        model.h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  
        model.w_init = 1/(np.exp(13/5)+1)**(1/4)
        model.z_init = 1/(2*(np.exp(0.74)+1))+0.5
        model.r_init = 1/(np.exp(+62/35)+1)
        
        ##### calculate Nerst potential for leakage current
        model.E_L = -(1/model.g_L)* (model.gamma_Na*model.rho_Na*model.m_init**3*model.h_init* model.E_Na + model.gamma_K*model.rho_K*model.n_init**4*model.E_K \
                     + model.gamma_KLT*model.rho_KLT*model.w_init**4*model.z_init*model.E_K + model.gamma_HCN*model.rho_HCN*model.r_init*model.E_HCN)
        
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
        
        ##### conductivity of leakage channels
        model.g_L = model.g_L_node/model.surface_aria_node

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
        model.distance_comps_middle[0] = 0.5*model.compartment_lengths[0]
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5* model.compartment_lengths[ii] + 0.5* model.compartment_lengths[ii+1]
        
        ##### Capacities
        # initialize
        model.c_m = np.zeros_like(model.structure)*uF/cm**2
        # nodes
        model.c_m[model.structure == 2] = model.c_m_node/model.surface_aria_node
        # internodes
        model.c_m[structure == 1] = model.c_m_layer/(1+model.nof_myelin_layers)
        
        ##### Condactivities internodes
        # initialize
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        # internodes
        model.g_m[model.structure == 1] = model.g_m_layer/(1+model.nof_myelin_layers)
        
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
    
    ##### Set parameter values of differential equations
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
    
    # other parameters
    neuron.V_res = model.V_res
    neuron.T_celsius = model.T_celsius
    neuron.E_Na = model.E_Na
    neuron.E_K = model.E_K
    neuron.E_HCN = model.E_HCN
    neuron.E_L = model.E_L
    neuron.rho_Na = model.rho_Na
    neuron.rho_K = model.rho_K
    neuron.rho_KLT = model.rho_KLT
    neuron.rho_HCN = model.rho_HCN
    
    return neuron, model
