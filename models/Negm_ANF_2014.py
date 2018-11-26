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
##### Nernst potential potassium (normal and low threshold potassium (KLT) channels)
E_K = -88*mV - V_res
##### Reversal potential hyperpolarization-activated cation (HCN) channels
E_HCN = -43*mV - V_res

# =============================================================================
# Conductivities
# =============================================================================
##### dividing factor for conductances of peripheral terminal and somatic region (makes currents smalller)
dividing_factor_conductances = 15
##### conductances nodes
gamma_Na = 25.69*psiemens
gamma_K = 50*psiemens
gamma_KLT = 13*psiemens
gamma_HCN = 13*psiemens
##### conductances peripheral terminal
gamma_Na_terminal = gamma_Na / dividing_factor_conductances
gamma_K_terminal = gamma_K / dividing_factor_conductances
gamma_KLT_terminal = gamma_KLT / dividing_factor_conductances
gamma_HCN_terminal = gamma_HCN / dividing_factor_conductances
##### conductances somatic region
gamma_Na_somatic_region = gamma_Na / dividing_factor_conductances
gamma_K_somatic_region = gamma_K / dividing_factor_conductances
gamma_KLT_somatic_region = gamma_KLT / dividing_factor_conductances *0
gamma_HCN_somatic_region = gamma_HCN / dividing_factor_conductances *0

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_segments_presomatic_region = 3
nof_segments_soma = 20
nof_segments_internodes = 9
nof_axonal_internodes = 10
##### lengths
length_peripheral_terminal = 10*um
length_internodes_dendrite = 150*um
length_internodes_axon = [150,200,250,300,350]*um # the last value defines the lengths of further internodes
length_nodes_dendrite = 1*um
length_nodes_axon = 1*um
length_presomatic_region = 60*um
##### diameters
diameter_dendrite = 1.2*um
diameter_soma = 30*um
diameter_axon = 2.3*um
##### myelin sheath thicknes
thicknes_myelin_sheath = 1*um
##### myelin dielectric constant
e_r = 1.27
##### usual nodal suface aria (as in some cases absolute values were given and values per aria needed
# and nodal dimesions were not defined in the paper by Negm and Bruce 2014)
aria = 10*um*1*um*np.pi

# =============================================================================
# Conductivity of leakage channels
# =============================================================================
g_L = (1953.49*Mohm)**-1/aria
g_L_terminal = (1953.49*Mohm)**-1/aria / dividing_factor_conductances
g_L_somatic_region = (1953.49*Mohm)**-1/aria / dividing_factor_conductances

# =============================================================================
# Resistivity of internodes
# =============================================================================
rho_m = 29.26*Gohm*mm

# =============================================================================
# Ion channel densities
# =============================================================================
rho_Na = 1000/aria
rho_K = 166/aria
rho_KLT = 166/aria
rho_HCN = 100/aria

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
I_L = g_L*(E_Leak-(v-V_res)) : amp/meter**2
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
E_Leak : volt
'''

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacity of axolemma
c_m_axolemma = 0.0714*pF/aria
##### dividing factor for somatic region
dividing_factor_capacitances = 4

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
display_name = "Negm and Bruce 2014 ANF"
display_name_short = "Negm 14 ANF"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(1.5, 1.6, num=29, endpoint = False),
                                  np.linspace(1.6, 5, num=20))*1e-3

# =============================================================================
# Calculations
# =============================================================================
##### rates for resting potential
alpha_m_0 = 1.875*(-25.41)/(1-np.exp(25.41/6.6))
beta_m_0 = 3.973*(21.001)/(1-np.exp(-21.001/9.41))
alpha_h_0 = -0.549*27.74/(1-np.exp(27.74/9.06))
beta_h_0 = 22.57/(1+np.exp(56.0/12.5))
alpha_n_0 = 0.129*(-35)/(1-np.exp((35)/10))
beta_n_0 = 0.3236*35/(1-np.exp(-35/10))

##### initial values for gating variables
m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  
w_init = 1/(np.exp(13/5)+1)**(1/4)
z_init = 1/(2*(np.exp(0.74)+1))+0.5
r_init = 1/(np.exp(+62/35)+1)


##### calculate Nerst potential for leakage current
E_L = -(1/g_L)* (gamma_Na*rho_Na*m_init**3*h_init* E_Na + gamma_K*rho_K*n_init**4*E_K + gamma_KLT*rho_KLT*w_init**4*z_init*E_K + gamma_HCN*rho_HCN*r_init*E_HCN)
E_L_terminal = -(1/g_L_terminal)* (gamma_Na_terminal*rho_Na*m_init**3*h_init* E_Na + gamma_K_terminal*rho_K*n_init**4*E_K +
                gamma_KLT_terminal*rho_KLT*w_init**4*z_init*E_K + gamma_HCN_terminal*rho_HCN*r_init*E_HCN)
E_L_somatic_region = -(1/g_L_somatic_region)* (gamma_Na_somatic_region*rho_Na*m_init**3*h_init* E_Na + gamma_K_somatic_region*rho_K*n_init**4*E_K +
                      gamma_KLT_somatic_region*rho_KLT*w_init**4*z_init*E_K + gamma_HCN_somatic_region*rho_HCN*r_init*E_HCN)

#####  Structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5
structure = np.array([0] + list(np.tile(np.tile([1],nof_segments_internodes).tolist() + [2],5)) + list(np.tile([3],nof_segments_presomatic_region)) +
                     list(np.tile([4],nof_segments_soma)) + list(np.tile([2] + np.tile([1],nof_segments_internodes).tolist(),nof_axonal_internodes)))

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
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 1] = length_internodes_dendrite / nof_segments_internodes
# internodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 1] = np.repeat(list(list(length_internodes_axon/nof_segments_internodes) +
                         list(np.tile(length_internodes_axon[-1]/nof_segments_internodes,nof_axonal_internodes-5))),nof_segments_internodes) * meter
# nodes dendrite
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 2] = length_nodes_dendrite
# nodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 2] = length_nodes_axon
# presomatic region
compartment_lengths[structure == 3] = length_presomatic_region/nof_segments_presomatic_region
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
# internodes (formula in paper gives capacitances so its divided by the internodal surface arias)
c_m[structure == 1] = (2*e_0*e_r)/np.log((compartment_diameters[:-1][structure == 1] + thicknes_myelin_sheath*2)/compartment_diameters[:-1][structure == 1])/\
                      compartment_diameters[:-1][structure == 1]
# somatic region
c_m[index_presomatic_region] = c_m_axolemma/dividing_factor_capacitances
c_m[index_soma] = c_m_axolemma/dividing_factor_capacitances

##### Conductivities internodes and soma
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# internodes (formula in paper gives conductance so its divided by the internodal surface arias)
g_m[structure == 1] = (2*compartment_diameters[:-1][structure == 1])/(rho_m*thicknes_myelin_sheath*2)/compartment_diameters[:-1][structure == 1]

##### Axoplasmatic resistances
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

##### Noise term
gamma_Na_vector = np.zeros(nof_comps)*psiemens
gamma_Na_vector[structure == 2] = gamma_Na
gamma_Na_vector[structure == 0] = gamma_Na / dividing_factor_conductances
gamma_Na_vector[structure == 3] = gamma_Na / dividing_factor_conductances
gamma_Na_vector[structure == 4] = gamma_Na / dividing_factor_conductances
noise_term = np.sqrt(A_surface*gamma_Na_vector*rho_Na)

##### Compartments to plot
# get indexes of all compartments that are not segmented
indexes_comps = np.where(np.logical_or(structure == 0, structure == 2))[0]
# calculate middle compartments of internodes
middle_comps_internodes = np.ceil(indexes_comps[:-1] + nof_segments_internodes/2).astype(int)
middle_comps_internodes = middle_comps_internodes[np.logical_or(middle_comps_internodes < start_index_soma, middle_comps_internodes > end_index_soma)]
# calculate middle compartments of somatic region
middle_comp_presomatic_region = int(start_index_presomatic_region + np.floor((nof_segments_presomatic_region)/2))
middle_comp_soma = int(start_index_soma + np.floor((nof_segments_soma)/2))
# create array with all compartments to plot
comps_to_plot = np.sort(np.append(np.append(indexes_comps, middle_comps_internodes), middle_comp_soma))

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
    neuron.w = model.w_init
    neuron.z = model.z_init
    neuron.r = model.r_init
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
    # conductances nodes
    neuron.gamma_Na = model.gamma_Na
    neuron.gamma_K = model.gamma_K
    neuron.gamma_KLT = model.gamma_KLT
    neuron.gamma_HCN = model.gamma_HCN
    neuron.g_L = model.g_L
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.gamma_Na[np.asarray(np.where(model.structure == 1))] = 0*psiemens
    neuron.gamma_K[np.asarray(np.where(model.structure == 1))] = 0*psiemens
    neuron.gamma_KLT[np.asarray(np.where(model.structure == 1))] = 0*psiemens
    neuron.gamma_HCN[np.asarray(np.where(model.structure == 1))] = 0*psiemens
    neuron.g_L[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    # conductances peripheral terminal
    neuron.gamma_Na[np.where(model.structure == 0)[0]] = model.gamma_Na_terminal
    neuron.gamma_K[np.where(model.structure == 0)[0]] = model.gamma_K_terminal
    neuron.gamma_KLT[np.where(model.structure == 0)[0]] = model.gamma_KLT_terminal
    neuron.gamma_HCN[np.where(model.structure == 0)[0]] = model.gamma_HCN_terminal
    neuron.g_L[np.where(model.structure == 0)[0]] = model.g_L_terminal
    
    # conductances presomatic terminal
    neuron.gamma_Na[index_presomatic_region] = model.gamma_Na_somatic_region
    neuron.gamma_K[index_presomatic_region] = model.gamma_K_somatic_region
    neuron.gamma_KLT[index_presomatic_region] = model.gamma_KLT_somatic_region
    neuron.gamma_HCN[index_presomatic_region] = model.gamma_HCN_somatic_region
    neuron.g_L[index_presomatic_region] = model.g_L_somatic_region
    
    # conductances soma
    neuron.gamma_Na[index_soma] = model.gamma_Na_somatic_region
    neuron.gamma_K[index_soma] = model.gamma_K_somatic_region
    neuron.gamma_KLT[index_soma] = model.gamma_KLT_somatic_region
    neuron.gamma_HCN[index_soma] = model.gamma_HCN_somatic_region
    neuron.g_L[index_soma] = model.g_L_somatic_region
    
    # Nernst potential for leakage current
    neuron.E_Leak = model.E_L
    neuron.E_Leak[np.where(model.structure == 0)[0]] = E_L_terminal
    neuron.E_Leak[model.index_presomatic_region] = E_L_somatic_region
    neuron.E_Leak[model.index_soma] = E_L_somatic_region
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = '''
    V_res = {}.V_res
    T_celsius = {}.T_celsius
    E_Na = {}.E_Na
    E_K = {}.E_K
    E_HCN = {}.E_HCN
    rho_Na = {}.rho_Na
    rho_K = {}.rho_K
    rho_KLT = {}.rho_KLT
    rho_HCN = {}.rho_HCN
    '''.format(model_name,model_name,model_name,model_name,model_name,model_name,model_name,model_name,model_name)
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string, model
