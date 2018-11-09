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
##### Nernst potential sodium
E_Na = 115*mV
##### Nernst potential potassium
E_K = -12*mV
##### Reversal potential hyperpolarization-activated cation (HCN) channels
E_HCN = 35*mV

# =============================================================================
# Conductivities
# =============================================================================
##### conductivities active compartments
g_Na = 1200*msiemens/cm**2
g_K = 360*msiemens/cm**2
g_KLT = 0*msiemens/cm**2
g_HCN = 0*msiemens/cm**2
g_L = 3*msiemens/cm**2
##### conductivities soma
g_Na_soma = 120*msiemens/cm**2
g_K_soma = 36*msiemens/cm**2
g_KLT_soma = 5*msiemens/cm**2
g_HCN_soma = 3*msiemens/cm**2
g_L_soma = 0.3*msiemens/cm**2
##### conductivities somatic region
g_KLT_somatic_region = 10*msiemens/cm**2
g_HCN_somatic_region = 6*msiemens/cm**2

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 50*ohm*cm
##### external resistivity
rho_out = 300*ohm*cm

# =============================================================================
# Myelin data
# =============================================================================
nof_myelin_layers_dendrite = 40
nof_myelin_layers_soma = 3
nof_myelin_layers_axon = 80
thicknes_myelin_layer = 8.5*nmeter

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
I_Na = g_Na*m**3*h* (E_Na-(v-V_res)) : amp/meter**2
I_K = g_K*n**4*(E_K-(v-V_res)) : amp/meter**2
I_KLT = g_KLT*w**4*z*(E_K-(v-V_res)) : amp/meter**2
I_HCN = g_HCN*r*(E_HCN-(v-V_res)) : amp/meter**2
I_L = g_L*(E_Leak-(v-V_res)) : amp/meter**2
Im = I_Na + I_K + I_KLT + I_HCN + I_L + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = 12 * (alpha_m * (1-m) - beta_m * m) : 1
dn/dt = 12 * (alpha_n * (1-n) - beta_n * n) : 1
dh/dt = 12 * (alpha_h * (1-h) - beta_h * h) : 1
dw/dt = alpha_w * (1-w) - beta_w * w : 1
dz/dt = alpha_z * (1-z) - beta_z * z : 1
dr/dt = alpha_r * (1-r) - beta_r * r : 1
alpha_m = (0.1) * (-(v-V_res)/mV+25) / (exp((-(v-V_res)/mV+25) / 10) - 1)/ms : Hz
beta_m = 4 * exp(-(v-V_res)/mV/18)/ms : Hz
alpha_h = 0.07 * exp(-(v-V_res)/mV/20)/ms : Hz
beta_h = 1/(exp((-(v-V_res)/mV+30) / 10) + 1)/ms : Hz
alpha_n = (0.01) * (-(v-V_res)/mV+10) / (exp((-(v-V_res)/mV+10) / 10) - 1)/ms : Hz
beta_n = 0.125*exp(-(v-V_res)/mV/80)/ms : Hz
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
g_Na : siemens/meter**2
g_K : siemens/meter**2
g_KLT : siemens/meter**2
g_HCN : siemens/meter**2
g_L : siemens/meter**2
g_myelin : siemens/meter**2
E_Leak : volt
'''

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_segments_presomatic_region = 3
nof_segments_soma = 20
nof_axonal_internodes = 10
##### lengths
length_peripheral_terminal = 1*um
length_internodes_dendrite = 350*um
length_internodes_axon = 500*um
length_nodes_dendrite = 2.5*um
length_nodes_axon = 2.5*um
length_presomatic_region = 100*um
length_postsomatic_region = 5*um
##### diameters
diameter_dendrite = 1*um
diameter_soma = 30*um
diameter_axon = 2*um

# =============================================================================
# Capacity
# =============================================================================
##### membrane capacity one layer
c_m_layer = 1*uF/cm**2

# =============================================================================
# Condactivity internodes
# =============================================================================
##### membrane conductivity internodes one layer
g_m_layer = 1*msiemens/cm**2

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.002*uA/np.sqrt(mS)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name
# =============================================================================
display_name = "Rattay et al. 2001 a."
display_name_short = "Rattay 01 a."

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(1.2, 1.3, num=40, endpoint = False),
                                  np.linspace(1.35, 5, num=15))*1e-3

# =============================================================================
# Calculations
# =============================================================================
##### rates for resting potential
alpha_m_0 = 0.1 * 25 / (np.exp(25 / 10) - 1)
beta_m_0 = 4
alpha_h_0 = 0.07
beta_h_0 = 1/(np.exp(3) + 1)
alpha_n_0 = 0.01 * 10 / (np.exp(1) - 1)
beta_n_0 = 0.125

##### initial values for gating variables
m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  
w_init = 1/(np.exp(13/5)+1)**(1/4)
z_init = 1/(2*(np.exp(0.74)+1))+0.5
r_init = 1/(np.exp(+62/35)+1)

##### calculate resting potential
g_total = g_Na + g_K + g_KLT + g_HCN
V_res = -(g_Na/g_total)*E_Na - (g_K/g_total)*E_K - (g_KLT/g_total)*E_K - (g_HCN/g_total)*E_HCN
                                  
##### calculate Nerst potential for leakage current
E_L = -(1/g_L)* (g_Na*m_init**3*h_init* E_Na + g_K*n_init**4*E_K + g_KLT*w_init**4*z_init*E_K + g_HCN*r_init*E_HCN)
E_L_presomatic_region = -(1/g_L)* (g_Na*m_init**3*h_init* E_Na + g_K*n_init**4*E_K + g_KLT_soma*w_init**4*z_init*E_K + g_HCN_soma*r_init*E_HCN)
E_L_soma = -(1/g_L_soma)* (g_Na_soma*m_init**3*h_init* E_Na + g_K_soma*n_init**4*E_K + g_KLT_soma*w_init**4*z_init*E_K + g_HCN_soma*r_init*E_HCN)

##### structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5)
structure = np.array([0] + list(np.tile([1,2],5)) + [1] + list(np.tile([3],nof_segments_presomatic_region)) + 
                     list(np.tile([4],nof_segments_soma)) + [5] + list(np.tile([1,2],nof_axonal_internodes-1)) + [1])
# indexes presomatic region
index_presomatic_region = np.argwhere(structure == 3)
start_index_presomatic_region = int(index_presomatic_region[0])
# indexes of soma
index_soma = np.argwhere(structure == 4)
start_index_soma = int(index_soma[0])
end_index_soma = int(index_soma[-1])
# further structural data
nof_comps = len(structure)
nof_comps_dendrite = len(structure[:start_index_soma])
nof_comps_axon = len(structure[end_index_soma+1:])

##### compartment lengths
# initialize
compartment_lengths = np.zeros_like(structure)*um
# peripheral terminal
compartment_lengths[np.where(structure == 0)] = length_peripheral_terminal
# internodes dendrite
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 1] = length_internodes_dendrite
# internodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 1] = length_internodes_axon
# nodes dendrite
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 2] = length_nodes_dendrite
# nodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 2] = length_nodes_axon
# presomatic region
compartment_lengths[structure == 3] = length_presomatic_region/nof_segments_presomatic_region
# soma
compartment_lengths[structure == 4] = diameter_soma/nof_segments_soma
# postsomatic region
compartment_lengths[structure == 5] = length_postsomatic_region
# total length neuron
length_neuron = sum(compartment_lengths)

##### compartment diameters
# initialize
compartment_diameters = np.zeros(nof_comps+1)*um
# dendrite
compartment_diameters[0:start_index_soma] = diameter_dendrite
dendrite_outer_diameter = diameter_dendrite + nof_myelin_layers_dendrite*thicknes_myelin_layer*2
# soma
soma_comp_diameters = calc.get_soma_diameters(nof_segments_soma,
                                              diameter_dendrite,
                                              diameter_soma,
                                              diameter_axon)
compartment_diameters[start_index_soma:end_index_soma+2] = soma_comp_diameters
# axon
compartment_diameters[end_index_soma+2:] = diameter_axon
axon_outer_diameter = diameter_axon + nof_myelin_layers_axon*thicknes_myelin_layer*2

#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5*compartment_lengths[ii] + 0.5*compartment_lengths[ii+1]
    
##### Capacities
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# all but internodes
c_m[np.where(structure != 1)] = c_m_layer
# dendrite internodes
c_m[0:start_index_soma][structure[0:start_index_soma] == 1] = c_m_layer/(1+nof_myelin_layers_dendrite)
# soma
c_m[np.where(structure == 4)] = c_m_layer/(1+nof_myelin_layers_soma)
# axon internodes
c_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = c_m_layer/(1+nof_myelin_layers_axon)

##### Condactivities internodes
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# dendritic internodes
g_m[0:start_index_soma][structure[0:start_index_soma] == 1] = g_m_layer/(1+nof_myelin_layers_dendrite)
# axonal internodes
g_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = g_m_layer/(1+nof_myelin_layers_axon)

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
g_Na_vector = np.zeros(nof_comps)*msiemens/cm**2
g_Na_vector[:] = g_Na
g_Na_vector[structure == 1] = 0*msiemens/cm**2
g_Na_vector[structure == 4] = g_Na_soma
noise_term = np.sqrt(A_surface*g_Na_vector)

##### Compartments to plot
# get indexes of all compartments that are not segmented
indexes_comps = np.where(np.logical_or(np.logical_or(np.logical_or(structure == 0, structure == 1), structure == 2), structure == 5))
# calculate middle compartments of presomatic region and soma
middle_comp_presomatic_region = int(start_index_presomatic_region + np.floor((nof_segments_presomatic_region)/2))
middle_comp_soma = int(start_index_soma + np.floor((nof_segments_soma)/2))
# create array with all compartments to plot
comps_to_plot = np.sort(np.append(indexes_comps, [middle_comp_presomatic_region, middle_comp_soma]))

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
        ##### rates for resting potential
        alpha_m_0 = 0.1 * 25 / (np.exp(25 / 10) - 1)
        beta_m_0 = 4
        alpha_h_0 = 0.07
        beta_h_0 = 1/(np.exp(3) + 1)
        alpha_n_0 = 0.01 * 10 / (np.exp(1) - 1)
        beta_n_0 = 0.125
        
        ##### initial values for gating variables
        model.m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
        model.n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
        model.h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  
        model.w_init = 1/(np.exp(13/5)+1)**(1/4)
        model.z_init = 1/(2*(np.exp(0.74)+1))+0.5
        model.r_init = 1/(np.exp(+62/35)+1)
        
        ##### calculate resting potential
        model.g_total = model.g_Na + model.g_K + model.g_KLT + model.g_HCN
        model.V_res = -(model.g_Na/model.g_total)*model.E_Na - (model.g_K/model.g_total)*model.E_K - (model.g_KLT/model.g_total)*model.E_K - (model.g_HCN/model.g_total)*model.E_HCN
                                          
        ##### calculate Nerst potential for leakage current
        model.E_L = -(1/model.g_L)* (model.g_Na*model.m_init**3*model.h_init* model.E_Na + model.g_K*model.n_init**4*model.E_K + model.g_KLT*model.w_init**4*model.z_init*model.E_K + model.g_HCN*model.r_init*model.E_HCN)
        model.E_L_presomatic_region = -(1/model.g_L)* (model.g_Na*model.m_init**3*model.h_init* model.E_Na + model.g_K*model.n_init**4*model.E_K + model.g_KLT_soma*model.w_init**4*model.z_init*model.E_K + model.g_HCN_soma*model.r_init*model.E_HCN)
        model.E_L_soma = -(1/model.g_L_soma)* (model.g_Na_soma*model.m_init**3*model.h_init* model.E_Na + model.g_K_soma*model.n_init**4*model.E_K + model.g_KLT_soma*model.w_init**4*model.z_init*model.E_K + model.g_HCN_soma*model.r_init*model.E_HCN)
        
        ##### structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        # postsomatic region = 5)
        model.structure = np.array([0] + list(np.tile([1,2],5)) + [1] + list(np.tile([3],model.nof_segments_presomatic_region)) + \
                             list(np.tile([4],model.nof_segments_soma)) + [5] + list(np.tile([1,2],model.nof_axonal_internodes-1)) + [1])
        ##### indexes presomatic region
        model.index_presomatic_region = np.argwhere(model.structure == 3)
        model.start_index_presomatic_region = int(model.index_presomatic_region[0])
        ##### indexes of soma
        model.index_soma = np.argwhere(model.structure == 4)
        model.start_index_soma = int(model.index_soma[0])
        model.end_index_soma = int(model.index_soma[-1])
        ##### further structural data
        model.nof_comps = len(model.structure)
        model.nof_comps_dendrite = len(model.structure[:model.start_index_soma])
        model.nof_comps_axon = len(model.structure[model.end_index_soma+1:])
        
        ##### compartment lengths
        # initialize
        model.compartment_lengths = np.zeros_like(model.structure)*um
        # peripheral terminal
        model.compartment_lengths[np.where(model.structure == 0)] = model.length_peripheral_terminal
        # internodes dendrite
        model.compartment_lengths[0:model.start_index_soma][model.structure[0:model.start_index_soma] == 1] = model.length_internodes_dendrite
        # internodes axon
        model.compartment_lengths[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] == 1] = model.length_internodes_axon
        # nodes dendrite
        model.compartment_lengths[0:model.start_index_soma][model.structure[0:model.start_index_soma] == 2] = model.length_nodes_dendrite
        # nodes axon
        model.compartment_lengths[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] == 2] = model.length_nodes_axon
        # presomatic region
        model.compartment_lengths[model.structure == 3] = model.length_presomatic_region/model.nof_segments_presomatic_region
        # soma
        model.compartment_lengths[model.structure == 4] = model.diameter_soma/model.nof_segments_soma
        # postsomatic region
        model.compartment_lengths[model.structure == 5] = model.length_postsomatic_region
        # total length neuron
        model.length_neuron = sum(model.compartment_lengths)
        
        ##### compartment diameters
        # initialize
        model.compartment_diameters = np.zeros(model.nof_comps+1)*um
        # dendrite
        model.compartment_diameters[0:model.start_index_soma] = model.diameter_dendrite
        # soma
        soma_comp_diameters = calc.get_soma_diameters(model.nof_segments_soma,
                                                      model.diameter_dendrite,
                                                      model.diameter_soma,
                                                      model.diameter_axon)
        model.compartment_diameters[model.start_index_soma:model.end_index_soma+2] = soma_comp_diameters
        # axon
        model.compartment_diameters[model.end_index_soma+2:] = model.diameter_axon
        
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5*model.compartment_lengths[ii] + 0.5*model.compartment_lengths[ii+1]
            
        ##### Capacities
        # initialize
        model.c_m = np.zeros_like(model.structure)*uF/cm**2
        # all but internodes
        model.c_m[np.where(model.structure != 1)] = model.c_m_layer
        # dendrite internodes
        model.c_m[0:model.start_index_soma][model.structure[0:model.start_index_soma] == 1] = model.c_m_layer/(1+model.nof_myelin_layers_dendrite)
        # soma
        model.c_m[np.where(model.structure == 4)] = model.c_m_layer/(1+model.nof_myelin_layers_soma)
        # axon internodes
        model.c_m[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] == 1] = model.c_m_layer/(1+model.nof_myelin_layers_axon)
        
        ##### Condactivities internodes
        # initialize
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        # dendritic internodes
        model.g_m[0:model.start_index_soma][model.structure[0:model.start_index_soma] == 1] = model.g_m_layer/(1+model.nof_myelin_layers_dendrite)
        # axonal internodes
        model.g_m[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] == 1] = model.g_m_layer/(1+model.nof_myelin_layers_axon)
        
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
        model.g_Na_vector = np.zeros(model.nof_comps)*msiemens/cm**2
        model.g_Na_vector[:] = model.g_Na
        model.g_Na_vector[model.structure == 1] = 0*msiemens/cm**2
        model.g_Na_vector[model.structure == 4] = model.g_Na_soma
        model.noise_term = np.sqrt(model.A_surface*model.g_Na_vector)
        
        ##### Compartments to plot
        # get indexes of all compartments that are not segmented
        model.indexes_comps = np.where(np.logical_or(np.logical_or(np.logical_or(model.structure == 0, model.structure == 1), model.structure == 2), model.structure == 5))
        # calculate middle compartments of presomatic region and soma
        model.middle_comp_presomatic_region = int(model.start_index_presomatic_region + np.floor((model.nof_segments_presomatic_region)/2))
        model.middle_comp_soma = int(model.start_index_soma + np.floor((model.nof_segments_soma)/2))
        # create array with all compartments to plot
        model.comps_to_plot = np.sort(np.append(model.indexes_comps, [model.middle_comp_presomatic_region, model.middle_comp_soma]))
    
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
    neuron.n = model.n_init
    neuron.h = model.h_init
    neuron.w = model.w_init
    neuron.z = model.z_init
    neuron.r = model.r_init
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
    # conductances active compartments
    neuron.g_Na = model.g_Na
    neuron.g_K = model.g_K
    neuron.g_KLT = model.g_KLT
    neuron.g_HCN = model.g_HCN
    neuron.g_L = model.g_L
    
    # conductances soma
    neuron.g_Na[model.index_soma] = model.g_Na_soma
    neuron.g_K[model.index_soma] = model.g_K_soma
    neuron.g_KLT[model.index_soma] = model.g_KLT_soma
    neuron.g_HCN[model.index_soma] = model.g_HCN_soma
    neuron.g_L[model.index_soma] = model.g_L_soma
    
    # conductances presomatic region
    neuron.g_KLT[model.index_presomatic_region] = model.g_KLT_somatic_region
    neuron.g_HCN[model.index_presomatic_region] = model.g_HCN_somatic_region
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.g_Na[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_K[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_KLT[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_HCN[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_L[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    # Nernst potential for leakage current
    neuron.E_Leak = model.E_L
    neuron.E_Leak[index_presomatic_region] = E_L_presomatic_region
    neuron.E_Leak[model.index_soma] = E_L_soma
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = '''
    T_celsius = {}.T_celsius
    V_res = {}.V_res
    E_Na = {}.E_Na
    E_K = {}.E_K
    E_HCN = {}.E_HCN
    E_L = {}.E_L
    '''.format(model_name,model_name,model_name,model_name,model_name,model_name)
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string, model
