##### import needed packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np

##### import functions
import functions.calculations as calc

# =============================================================================
# Nernst potentials
# =============================================================================
##### Resting potential of cell
V_res = -65*mV
##### Nernst potential sodium
E_Na = 115*mV
##### Nernst potential potassium
E_K = -12*mV

# =============================================================================
# Conductivities
# =============================================================================
##### conductivities active compartments
g_Na = 1200*msiemens/cm**2
g_K = 360*msiemens/cm**2
g_L = 3*msiemens/cm**2
##### conductivities soma
g_Na_soma = 120*msiemens/cm**2
g_K_soma = 36*msiemens/cm**2
g_L_soma = 0.3*msiemens/cm**2

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
I_L = g_L*(E_L-(v-V_res)) : amp/meter**2
Im = I_Na + I_K + I_L + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = 12 * (alpha_m * (1-m) - beta_m * m) : 1
dn/dt = 12 * (alpha_n * (1-n) - beta_n * n) : 1
dh/dt = 12 * (alpha_h * (1-h) - beta_h * h) : 1
alpha_m = (0.1) * (-(v-V_res)/mV+25) / (exp((-(v-V_res)/mV+25) / 10) - 1)/ms : Hz
beta_m = 4 * exp(-(v-V_res)/mV/18)/ms : Hz
alpha_h = 0.07 * exp(-(v-V_res)/mV/20)/ms : Hz
beta_h = 1/(exp((-(v-V_res)/mV+30) / 10) + 1)/ms : Hz
alpha_n = (0.01) * (-(v-V_res)/mV+10) / (exp((-(v-V_res)/mV+10) / 10) - 1)/ms : Hz
beta_n = 0.125*exp(-(v-V_res)/mV/80)/ms : Hz
g_Na : siemens/meter**2
g_K : siemens/meter**2
g_L : siemens/meter**2
g_myelin : siemens/meter**2
V_res : volt
E_Na : volt
E_K : volt
E_L : volt
'''

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_segments_presomatic_region = 3
nof_segments_soma = 10
nof_axonal_internodes = 20
##### lengths
length_peripheral_terminal = 10*um
length_internodes_dendrite = [430,430,430,430,430,360]*um
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
electrode_distance = 500*um

# =============================================================================
# Display name
# =============================================================================
display_name = "Rattay et al. 2001"
display_name_plots = "Rattay et al. (2001)"
display_name_short = "Rattay 01"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(1.288, 1.35, num=40, endpoint = False),
                                  np.linspace(1.35, 5, num=20))*1e-3

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

##### calculate Nerst potential for leakage current
E_L = -(1/g_L)* (g_Na*m_init**3*h_init* E_Na + g_K*n_init**4*E_K)

##### structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# soma = 4
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
distance_comps_middle[0] = 0.5*compartment_lengths[0]
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
                                          
        ##### calculate Nerst potential for leakage current
        model.E_L = -(1/model.g_L)* (model.g_Na*model.m_init**3*model.h_init* model.E_Na + model.g_K*model.n_init**4*model.E_K)

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
        model.distance_comps_middle[0] = 0.5*model.compartment_lengths[0]
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
            
    ##### initial values of gating variables 
    neuron.v = model.V_res
    neuron.m = model.m_init
    neuron.n = model.n_init
    neuron.h = model.h_init
    
    ##### Set values of parameters in differential equations
    # conductances active compartments
    neuron.g_Na = model.g_Na
    neuron.g_K = model.g_K
    neuron.g_L = model.g_L
    
    # conductances soma
    neuron.g_Na[model.index_soma] = model.g_Na_soma
    neuron.g_K[model.index_soma] = model.g_K_soma
    neuron.g_L[model.index_soma] = model.g_L_soma
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.g_Na[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_K[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_L[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    # potentials
    neuron.V_res = model.V_res
    neuron.E_Na = model.E_Na
    neuron.E_K = model.E_K
    neuron.E_L = model.E_L    
    
    return neuron, model
