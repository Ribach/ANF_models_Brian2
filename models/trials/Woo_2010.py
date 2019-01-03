##### import needed packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F, electric_constant as e_0
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
##### Nernst potential potassium
E_K = -88*mV - V_res

# =============================================================================
# Ion concentrations
# =============================================================================
Na_i = 8.71*mM/liter
Na_e = 154*mM/liter
K_i = 159.5*mM/liter
K_e = 5.9*mM/liter
Cl_i = 18*mM/liter
Cl_e = 116*mM/liter

# =============================================================================
# Conductivities
# =============================================================================
##### dividing factor for conductances of peripheral terminal and somatic region (makes currents smalller)
dividing_factor_conductances = 15
##### conductances nodes
gamma_Na = 22.65*psiemens
gamma_K = 50*psiemens
##### conductances peripheral terminal
gamma_Na_terminal = gamma_Na / dividing_factor_conductances
gamma_K_terminal = gamma_K / dividing_factor_conductances

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_segments_soma = 20
nof_segments_internodes = 9
nof_axonal_internodes = 10
##### lengths
length_peripheral_terminal = 10*um
length_internodes_dendrite = 150*um
length_internodes_axon = [150,200,250,300,350]*um # the last value defines the lengths of further internodes
length_nodes_dendrite = 1*um
length_nodes_axon = 1*um
##### diameters
diameter_dendrite = 1.2*um
diameter_soma = 20*um
diameter_axon = 2.3*um
##### myelin sheath thicknes
thicknes_myelin_sheath = 1*um
##### myelin dielectric constant
e_r = 1.27

# =============================================================================
# Conductivity of leakage channels
# =============================================================================
g_L = 1/(166.2*kohm*mm**2)
g_L_terminal = g_L / dividing_factor_conductances
g_L_somatic_region = g_L / dividing_factor_conductances

# =============================================================================
# Resistivity of internodes
# =============================================================================
rho_m = 29.26*Gohm*mm

# =============================================================================
# Ion channel densities
# =============================================================================
rho_Na = 80/(um**2)
rho_K = 45/(um**2)

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 6378*ohm*mm
##### external resistivity
rho_out = 0.3*kohm*cm

# =============================================================================
# Constants for adaptation algorithm
# =============================================================================
##### amplitude constant
A_f = 0.335
##### decay time constant
tau_depl = 40*ms

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
I_Na = gamma_Na*rho_Na*m**3*h* (E_Na-(v-V_res)) : amp/meter**2
I_K = gamma_K*rho_K*n**4*(E_K-(v-V_res)) : amp/meter**2
I_L = g_L*(E_Leak-(v-V_res)) : amp/meter**2
Im = I_Na + I_K + I_L + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
alpha_m = 1.872/mV*((v-V_res)-25.41*mV)/(1-exp((25.41*mV-(v-V_res))/(6.06*mV)))/ms : Hz
beta_m = 3.793/mV*(21.001*mV-(v-V_res))/(1-exp(((v-V_res)-21.001*mV)/(9.41*mV)))/ms : Hz
alpha_h = -0.549/mV*(27.74*mV + (v-V_res))/(1-exp(((v-V_res)+27.74*mV)/(9.06*mV)))/ms : Hz
beta_h = 22.57/(1+exp((56.0*mV-(v-V_res))/(12.5*mV)))/ms : Hz
alpha_n = 0.129/mV*((v-V_res)-35*mV)/(1-exp((35*mV-(v-V_res))/(10*mV)))/ms : Hz
beta_n = 0.324/mV*(35*mV-(v-V_res))/(1-exp(((v-V_res)-35*mV)/(10*mV)))/ms : Hz
gamma_Na : siemens
gamma_K : siemens
g_L : siemens/meter**2
g_myelin : siemens/meter**2
E_Leak : volt
V_res : volt
T_celsius : 1
E_Na : volt
E_K : volt
rho_Na : 1/meter**2
rho_K : 1/meter**2
'''

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacity of axolemma
c_m_axolemma = 0.5125*nF/(mm**2)

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.006*uA/np.sqrt(mS)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Woo et al. 2010"
display_name_short = "Negm 10"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(1.5, 1.6, num=29, endpoint = False),
                                  np.linspace(1.6, 5, num=20))*1e-3

# =============================================================================
# Calculations
# =============================================================================
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

##### calculate Nerst potential for leakage current
E_L = -(1/g_L)* (gamma_Na*rho_Na*m_init**3*h_init* E_Na + gamma_K*rho_K*n_init**4*E_K)
E_L_terminal = -(1/g_L_terminal)* (gamma_Na_terminal*rho_Na*m_init**3*h_init* E_Na + gamma_K_terminal*rho_K*n_init**4*E_K)

#####  Structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5
structure = np.array([0] + list(np.tile(np.tile([1],nof_segments_internodes).tolist() + [2],3)) +
                     list(np.tile([4],nof_segments_soma)) + list(np.tile([2] + np.tile([1],nof_segments_internodes).tolist(),nof_axonal_internodes)))

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
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 1] = length_internodes_dendrite / nof_segments_internodes
# internodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 1] = np.repeat(list(list(length_internodes_axon/nof_segments_internodes) +
                         list(np.tile(length_internodes_axon[-1]/nof_segments_internodes,nof_axonal_internodes-5))),nof_segments_internodes) * meter
# nodes dendrite
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 2] = length_nodes_dendrite
# nodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 2] = length_nodes_axon
# soma
compartment_lengths[structure == 4] = diameter_soma/nof_segments_soma
# total length neuron
length_neuron = sum(compartment_lengths)

##### compartment diameters
# initialize
compartment_diameters = np.zeros(nof_comps+1)*um
# dendrite
compartment_diameters[0:start_index_soma] = diameter_dendrite
dendrite_outer_diameter = diameter_dendrite + thicknes_myelin_sheath*2
# soma
soma_comp_diameters = calc.get_soma_diameters(nof_segments_soma,
                                              diameter_dendrite,
                                              diameter_soma,
                                              diameter_axon)
compartment_diameters[start_index_soma:end_index_soma+2] = soma_comp_diameters
soma_outer_diameters = soma_comp_diameters + thicknes_myelin_sheath*2
# axon
compartment_diameters[end_index_soma+2:] = diameter_axon
axon_outer_diameter = diameter_axon + thicknes_myelin_sheath*2

##### Surface arias
# lateral surfaces
m = [np.sqrt(abs(compartment_diameters[i+1] - compartment_diameters[i])**2 + compartment_lengths[i]**2)
           for i in range(0,nof_comps)]
# total surfaces
A_surface = [(compartment_diameters[i+1] + compartment_diameters[i])*np.pi*m[i]*0.5
           for i in range(0,nof_comps)]
        
#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
distance_comps_middle[0] = 0.5*compartment_lengths[0]
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]

##### Capacities
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# peripheral terminal
c_m[structure == 0] = c_m_axolemma
# nodes
c_m[structure == 2] = c_m_axolemma
# dendritic internodes (formula in paper gives capacitances so its divided by the surface arias)
c_m[0:start_index_soma][structure[0:start_index_soma] == 1] = (2*e_0*e_r)/np.log(dendrite_outer_diameter/diameter_dendrite) / diameter_dendrite
# soma (formula in paper gives capacitances so its divided by the surface arias)
c_m_soma = np.transpose((2*e_0*e_r)/np.log(soma_outer_diameters/soma_comp_diameters)) / np.transpose(diameter_soma/nof_segments_soma)
c_m[index_soma] = (c_m_soma[:-1] + c_m_soma[1:])/2
# axonal internodes (formula in paper gives capacitances so its divided by the surface arias)
c_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = (2*e_0*e_r)/np.log(axon_outer_diameter/diameter_axon) / diameter_axon

##### Conductivities internodes and soma
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# dendritic internodes
g_m[0:start_index_soma][structure[0:start_index_soma] == 1] = 2/(rho_m*(dendrite_outer_diameter-diameter_dendrite))
# soma
g_m[index_soma] = np.transpose(2/(rho_m*(soma_outer_diameters-soma_comp_diameters)))[0]
# axonal internodes
g_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = 2/(rho_m*(axon_outer_diameter-diameter_axon))

##### Axoplasmatic resistances
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

##### Noise term
gamma_Na_vector = np.zeros(nof_comps)*psiemens
gamma_Na_vector[structure == 2] = gamma_Na
gamma_Na_vector[structure == 0] = gamma_Na / dividing_factor_conductances
noise_term = np.sqrt(A_surface*gamma_Na_vector*rho_Na)

##### Compartments to plot
# get indexes of all compartments that are not segmented
indexes_comps = np.where(np.logical_or(structure == 0, structure == 2))[0]
# calculate middle compartments of internodes
middle_comps_internodes = np.ceil(indexes_comps[:-1] + nof_segments_internodes/2).astype(int)
middle_comps_internodes = middle_comps_internodes[np.logical_or(middle_comps_internodes < start_index_soma, middle_comps_internodes > end_index_soma)]
# calculate middle compartments of somatic region
middle_comp_soma = int(start_index_soma + np.floor((nof_segments_soma)/2))
# create array with all compartments to plot
comps_to_plot = np.sort(np.append(np.append(indexes_comps, middle_comps_internodes), middle_comp_soma))

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
        #####  Structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        # postsomatic region = 5
        model.structure = np.array(list(np.tile([2] + np.tile([1],model.nof_segments_internodes).tolist(),model.nof_internodes)) + [2])
        model.nof_comps = len(model.structure)
        
        ##### compartment lengths
        # initialize
        model.compartment_lengths = np.zeros_like(structure)*um
        # length internodes
        model.compartment_lengths[model.structure == 1] = model.length_internodes / model.nof_segments_internodes
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
                
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle[0] = 0.5*model.compartment_lengths[0]
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
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
        model.middle_comps_internodes = np.ceil(model.indexes_comps[:-1] + model.nof_segments_internodes/2).astype(int)
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
    
    ##### Set parameter values of differential equations
    # conductances nodes
    neuron.gamma_Na = model.gamma_Na
    neuron.gamma_K = model.gamma_K
    neuron.g_L = model.g_L
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.gamma_Na[np.asarray(np.where(model.structure == 1))] = 0*psiemens
    neuron.gamma_K[np.asarray(np.where(model.structure == 1))] = 0*psiemens
    neuron.g_L[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    # conductances peripheral terminal
    neuron.gamma_Na[np.where(model.structure == 0)[0]] = model.gamma_Na_terminal
    neuron.gamma_K[np.where(model.structure == 0)[0]] = model.gamma_K_terminal
    neuron.g_L[np.where(model.structure == 0)[0]] = model.g_L_terminal
    
    # conductances soma
    neuron.gamma_Na[index_soma] = 0*psiemens
    neuron.gamma_K[index_soma] = 0*psiemens
    neuron.g_L[index_soma] = 0*msiemens/cm**2
    
    # Nernst potential for leakage current
    neuron.E_Leak = model.E_L
    neuron.E_Leak[np.where(model.structure == 0)[0]] = E_L_terminal
    
    # other parameters
    neuron.V_res = model.V_res
    neuron.T_celsius = model.T_celsius
    neuron.E_Na = model.E_Na
    neuron.E_K = model.E_K
    neuron.rho_Na = model.rho_Na
    neuron.rho_K = model.rho_K
    
    return neuron, model
