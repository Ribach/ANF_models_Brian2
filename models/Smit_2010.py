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
# Ionic concentrations
# =============================================================================
##### Na_e / Na_i
Na_ratio = 7.210
K_ratio = 0.036
Leak_ratio = 0.0367

# =============================================================================
# Resting potential
# =============================================================================
V_res = -79.4*mV *1.035**((T_celsius-6.3)/10)

# =============================================================================
# Nernst potentials Rattay
# =============================================================================
##### Nernst potential sodium
E_Na_Rat = 115*mV
##### Nernst potential potassium
E_K_Rat = -12*mV

# =============================================================================
# Conductivities Smit + Rattay
# =============================================================================
##### conductivities active compartments Smit
g_Na_Smit = 640*msiemens/cm**2 * 1.02**((T_celsius-24)/10)
g_K_Smit = 60*msiemens/cm**2 * 1.16**((T_celsius-20)/10)
g_L_Smit = 57.5*msiemens/cm**2 * 1.418**((T_celsius-24)/10)
##### conductivities active compartments Rattay
g_Na_Rat = 1200*msiemens/cm**2
g_K_Rat = 360*msiemens/cm**2
g_L_Rat = 3*msiemens/cm**2
##### conductivities soma Rattay
g_Na_soma = 120*msiemens/cm**2
g_K_soma = 36*msiemens/cm**2
g_L_soma = 0.3*msiemens/cm**2

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 25*ohm*cm * (1/1.35)**((T_celsius-37)/10)
##### external resistivity
rho_out = 300*ohm*cm

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
I_Na_persistent_Smit = 0.975*g_Na_Smit*m_t_Smit**3*h_Smit*(E_Na_Smit-(v-V_res)) : amp/meter**2
I_Na_transient_Smit = 0.025*g_Na_Smit*m_p_Smit**3*h_Smit*(E_Na_Smit-(v-V_res)) : amp/meter**2
I_K_Smit = g_K_Smit*n_Smit**4*(E_K_Smit-(v-V_res)) : amp/meter**2
I_L_Smit = g_L_Smit*(E_L_Smit-(v-V_res)) : amp/meter**2
I_Na_Rat = g_Na_Rat*m_Rat**3*h_Rat*(E_Na_Rat-(v-V_res)) : amp/meter**2
I_K_Rat = g_K_Rat*n_Rat**4*(E_K_Rat-(v-V_res)) : amp/meter**2
I_L_Rat = g_L_Rat*(E_L_Rat-(v-V_res)) : amp/meter**2
Im = I_Na_persistent_Smit + I_Na_transient_Smit + I_K_Smit + I_L_Smit + I_Na_Rat + I_K_Rat + I_L_Rat + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm_t_Smit/dt = alpha_m_t_Smit * (1-m_t_Smit) - beta_m_t_Smit * m_t_Smit : 1
dm_p_Smit/dt = alpha_m_p_Smit * (1-m_p_Smit) - beta_m_p_Smit * m_p_Smit : 1
dn_Smit/dt = alpha_n_Smit * (1-n_Smit) - beta_n_Smit * n_Smit : 1
dh_Smit/dt = alpha_h_Smit * (1-h_Smit) - beta_h_Smit * h_Smit : 1
alpha_m_t_Smit = 4.42*(2.5-0.1*(v-V_res)/mV)/(1*(exp(2.5-0.1*(v-V_res)/mV))-1) * 2.23**(0.1*(T_celsius-20))/ms : Hz
alpha_m_p_Smit = 2.06*(2.5-0.1*((v-V_res)/mV-20))/(1*(exp(2.5-0.1*((v-V_res)/mV-20)))-1) * 1.99**(0.1*(T_celsius-20))/ms : Hz
alpha_n_Smit = 0.2*(1.0-0.1*(v-V_res)/mV)/(10*(exp(1-0.1*(v-V_res)/mV)-1)) * 1.5**(0.1*(T_celsius-20))/ms : Hz
alpha_h_Smit = 1.47*0.07*exp(-(v-V_res)/mV/20) * 1.5**(0.1*(T_celsius-20))/ms : Hz
beta_m_t_Smit = 4.42*4.0*exp(-(v-V_res)/mV/18) * 2.23**(0.1*(T_celsius-20))/ms : Hz
beta_m_p_Smit = 2.06*4.0*exp(-((v-V_res)/mV-20)/18) * 1.99**(0.1*(T_celsius-20))/ms : Hz
beta_n_Smit = 0.2*0.125*exp(-(v-V_res)/mV/80) * 1.5**(0.1*(T_celsius-20))/ms : Hz
beta_h_Smit = 1.47/(1+exp(3.0-0.1*(v-V_res)/mV)) * 1.5**(0.1*(T_celsius-20))/ms : Hz
dm_Rat/dt = 12 * (alpha_m_Rat * (1-m_Rat) - beta_m_Rat * m_Rat) : 1
dn_Rat/dt = 12 * (alpha_n_Rat * (1-n_Rat) - beta_n_Rat * n_Rat) : 1
dh_Rat/dt = 12 * (alpha_h_Rat * (1-h_Rat) - beta_h_Rat * h_Rat) : 1
alpha_m_Rat = (0.1) * (-(v-V_res)/mV+25) / (exp((-(v-V_res)/mV+25) / 10) - 1)/ms : Hz
beta_m_Rat = 4 * exp(-(v-V_res)/mV/18)/ms : Hz
alpha_h_Rat = 0.07 * exp(-(v-V_res)/mV/20)/ms : Hz
beta_h_Rat = 1/(exp((-(v-V_res)/mV+30) / 10) + 1)/ms : Hz
alpha_n_Rat = (0.01) * (-(v-V_res)/mV+10) / (exp((-(v-V_res)/mV+10) / 10) - 1)/ms : Hz
beta_n_Rat = 0.125*exp(-(v-V_res)/mV/80)/ms : Hz
g_Na_Rat : siemens/meter**2
g_K_Rat : siemens/meter**2
g_L_Rat : siemens/meter**2
g_myelin : siemens/meter**2
g_Na_Smit : siemens/meter**2
g_K_Smit : siemens/meter**2
g_L_Smit : siemens/meter**2
g_myelin_Smit : siemens/meter**2
V_res : volt
E_Na_Smit : volt
E_K_Smit : volt
E_L_Smit : volt
E_Na_Rat : volt
E_K_Rat : volt
E_L_Rat : volt
T_celsius : 1
'''

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_segments_presomatic_region = 3
nof_segments_soma = 10
nof_axonal_internodes = 100
##### lengths
length_peripheral_terminal = 10*um
length_internodes_dendrite = [210,440,350,430,360]*um
length_internodes_axon = 77.4*um
length_nodes_dendrite = 2.5*um
length_nodes_axon = 1.061*um
length_presomatic_region = 100*um
length_postsomatic_region = 5*um
##### diameters
diameter_dendrite = 1*um
dendrite_outer_diameter = 1.68*um
diameter_soma = 27*um
diameter_axon = 2.02*um
axon_outer_diameter = 3.75*um
##### myelin layer thickness axon
myelin_layer_thicknes_axon = 16*nmeter

# =============================================================================
# Myelin data dendrite and soma
# =============================================================================
nof_myelin_layers_dendrite = 40
nof_myelin_layers_soma = 3

# =============================================================================
# Capacities
# =============================================================================
##### capacaty one layer (membrane and myelin as in Rattay's model)
c_m_layer = 1*uF/cm**2
##### cell membrane capacitiy one layer
c_mem = 2.8*uF/cm**2
##### myelin layer capacity
c_my = 0.6*uF/cm**2

# =============================================================================
# Condactivities and resistivities internodes
# =============================================================================
##### membrane conductivity internodes one layer
g_m_layer = 1*msiemens/cm**2
##### axolemma resistivity internodes
r_mem = 4.871*10**4*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)
##### myelin layer resistivity internodes
r_my = 104*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.002*uA/np.sqrt(mS)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 500*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Smit et al. 2010"
display_name_plots = "Smit et al. (2010)"
display_name_short = "Smit 10"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(2.09, 2.32, num=40, endpoint = False),
                                  np.linspace(2.32, 5, num=20))*1e-3

# =============================================================================
# Calculations
# =============================================================================
##### Temperature
T_kelvin = zero_celsius + T_celsius*kelvin

##### Nernst potentials Smit
# Nernst potential sodium
E_Na_Smit = R*T_kelvin/F * np.log(Na_ratio) - V_res
# Nernst potential potassium
E_K_Smit = R*T_kelvin/F * np.log(K_ratio) - V_res

##### rates for resting potential
alpha_m_Rat_0 = 0.1*25/(np.exp(25/10) - 1)
beta_m_Rat_0 = 4
alpha_h_Rat_0 = 0.07
beta_h_Rat_0 = 1/(np.exp(3) + 1)
alpha_n_Rat_0 = 0.01 * 10 / (np.exp(1) - 1)
beta_n_Rat_0 = 0.125
alpha_m_t_Smit_0 = 4.42*2.5/(np.exp(2.5)-1) * 2.23**(0.1*(T_celsius-20))
alpha_m_p_Smit_0 = 2.06*(2.5-0.1*(-20))/(1*(np.exp(2.5-0.1*(-20)))-1) * 1.99**(0.1*(T_celsius-20))
alpha_n_Smit_0 = 0.2*1.0/(10*(np.exp(1)-1)) * 1.5**(0.1*(T_celsius-20))
alpha_h_Smit_0 = 1.47*0.07 * 1.5**(0.1*(T_celsius-20))
beta_m_t_Smit_0 = 4.42*4.0 * 2.23**(0.1*(T_celsius-20))
beta_m_p_Smit_0 = 2.06*4.0*np.exp(20/18) * 1.99**(0.1*(T_celsius-20))
beta_n_Smit_0 = 0.2*0.125*1 * 1.5**(0.1*(T_celsius-20))
beta_h_Smit_0 = 1.47/(1+np.exp(3.0)) * 1.5**(0.1*(T_celsius-20))

##### initial values for gating variables
m_init_Rat = alpha_m_Rat_0 / (alpha_m_Rat_0 + beta_m_Rat_0)
n_init_Rat = alpha_n_Rat_0 / (alpha_n_Rat_0 + beta_n_Rat_0)
h_init_Rat = alpha_h_Rat_0 / (alpha_h_Rat_0 + beta_h_Rat_0)  
m_t_init_Smit = alpha_m_t_Smit_0 / (alpha_m_t_Smit_0 + beta_m_t_Smit_0)
m_p_init_Smit = alpha_m_p_Smit_0 / (alpha_m_p_Smit_0 + beta_m_p_Smit_0)
n_init_Smit = alpha_n_Smit_0 / (alpha_n_Smit_0 + beta_n_Smit_0)
h_init_Smit = alpha_h_Smit_0 / (alpha_h_Smit_0 + beta_h_Smit_0)               

##### calculate Nerst potential for leakage current
E_L_Rat = -(1/g_L_Rat)* (g_Na_Rat*m_init_Rat**3*h_init_Rat* E_Na_Rat + g_K_Rat*n_init_Rat**4*E_K_Rat)
E_L_Smit = -(1/g_L_Smit)* (0.975*g_Na_Smit*m_t_init_Smit**3*h_init_Smit* E_Na_Smit + 0.025*g_Na_Smit*m_p_init_Smit**3*h_init_Smit* E_Na_Smit +
            g_K_Smit*n_init_Smit**4*E_K_Smit)

##### structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5)
structure = np.array([0] + list(np.tile([1,2],4)) + [1] + list(np.tile([3],nof_segments_presomatic_region)) +
                     list(np.tile([4],nof_segments_soma)) + [5] + list(np.tile([1,2],nof_axonal_internodes)) + [1])
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

#####  Compartment lengths
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
compartment_lengths[np.where(structure == 3)] = length_presomatic_region/nof_segments_presomatic_region
# soma
compartment_lengths[np.where(structure == 4)] = diameter_soma/nof_segments_soma
# postsomatic region
compartment_lengths[np.where(structure == 5)] = length_postsomatic_region
# total length neuron
length_neuron = sum(compartment_lengths)

##### Compartment diameters
# initialize
compartment_diameters = np.zeros(nof_comps+1)*um
# dendrite
compartment_diameters[0:start_index_soma] = diameter_dendrite
# soma
soma_comp_diameters = calc.get_soma_diameters(nof_segments_soma,
                                              diameter_dendrite,
                                              diameter_soma,
                                              diameter_axon)
compartment_diameters[start_index_soma:end_index_soma+2] = soma_comp_diameters
# axon
compartment_diameters[end_index_soma+2:] = diameter_axon

##### number of axonal myelin layers
nof_myelin_layers_axon = np.floor(0.5*(axon_outer_diameter-diameter_axon)/myelin_layer_thicknes_axon)

#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
distance_comps_middle[0] = 0.5*compartment_lengths[0]
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]

##### Capacities
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# all but internodes dendrite
c_m[0:start_index_soma][structure[0:start_index_soma] != 1] = c_m_layer
# dendritic internodes
c_m[0:start_index_soma][structure[0:start_index_soma] == 1] = c_m_layer/(1+nof_myelin_layers_dendrite)
# soma
c_m[np.where(structure == 4)] = c_m_layer/(1+nof_myelin_layers_soma)
# all but internodes axon
c_m[end_index_soma+1:][structure[end_index_soma+1:] != 1] = c_mem
# axonal internodes
c_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = 1/(1/c_mem + nof_myelin_layers_axon/c_my)

##### Condactivities internodes
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# dendritic internodes
g_m[0:start_index_soma][structure[0:start_index_soma] == 1] = g_m_layer/(1+nof_myelin_layers_dendrite)
# axonal internodes
g_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = 1/(r_mem + nof_myelin_layers_axon*r_my)

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
g_Na_vector[0:start_index_soma][structure[0:start_index_soma] != 1] = g_Na_Rat
g_Na_vector[np.where(structure == 4)] = g_Na_soma
g_Na_vector[end_index_soma+1:][structure[end_index_soma+1:] != 1] = g_Na_Smit
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
        ##### Temperature
        model.T_kelvin = model.zero_celsius + model.T_celsius*kelvin
        
        ##### Nernst potentials Smit
        # Nernst potential sodium
        model.E_Na_Smit = model.R*model.T_kelvin/model.F * np.log(model.Na_ratio) - model.V_res
        # Nernst potential potassium
        model.E_K_Smit = model.R*model.T_kelvin/model.F * np.log(model.K_ratio) - model.V_res
        
        ##### rates for resting potential
        alpha_m_Rat_0 = 0.1*25/(np.exp(25/10) - 1)
        beta_m_Rat_0 = 4
        alpha_h_Rat_0 = 0.07
        beta_h_Rat_0 = 1/(np.exp(3) + 1)
        alpha_n_Rat_0 = 0.01 * 10 / (np.exp(1) - 1)
        beta_n_Rat_0 = 0.125
        alpha_m_t_Smit_0 = 4.42*2.5/(np.exp(2.5)-1) * 2.23**(0.1*(model.T_celsius-20))
        alpha_m_p_Smit_0 = 2.06*(2.5-0.1*(-20))/(1*(np.exp(2.5-0.1*(-20)))-1) * 1.99**(0.1*(model.T_celsius-20))
        alpha_n_Smit_0 = 0.2*1.0/(10*(np.exp(1)-1)) * 1.5**(0.1*(model.T_celsius-20))
        alpha_h_Smit_0 = 1.47*0.07 * 1.5**(0.1*(model.T_celsius-20))
        beta_m_t_Smit_0 = 4.42*4.0 * 2.23**(0.1*(model.T_celsius-20))
        beta_m_p_Smit_0 = 2.06*4.0*np.exp(20/18) * 1.99**(0.1*(model.T_celsius-20))
        beta_n_Smit_0 = 0.2*0.125*1 * 1.5**(0.1*(model.T_celsius-20))
        beta_h_Smit_0 = 1.47/(1+np.exp(3.0)) * 1.5**(0.1*(model.T_celsius-20))
        
        ##### initial values for gating variables
        model.m_init_Rat = alpha_m_Rat_0 / (alpha_m_Rat_0 + beta_m_Rat_0)
        model.n_init_Rat = alpha_n_Rat_0 / (alpha_n_Rat_0 + beta_n_Rat_0)
        model.h_init_Rat = alpha_h_Rat_0 / (alpha_h_Rat_0 + beta_h_Rat_0)  
        model.m_t_init_Smit = alpha_m_t_Smit_0 / (alpha_m_t_Smit_0 + beta_m_t_Smit_0)
        model.m_p_init_Smit = alpha_m_p_Smit_0 / (alpha_m_p_Smit_0 + beta_m_p_Smit_0)
        model.n_init_Smit = alpha_n_Smit_0 / (alpha_n_Smit_0 + beta_n_Smit_0)
        model.h_init_Smit = alpha_h_Smit_0 / (alpha_h_Smit_0 + beta_h_Smit_0)               
        
        ##### calculate Nerst potential for leakage current
        model.E_L_Rat = -(1/model.g_L_Rat)* (model.g_Na_Rat*model.m_init_Rat**3*model.h_init_Rat* model.E_Na_Rat + model.g_K_Rat*model.n_init_Rat**4*model.E_K_Rat)
        model.E_L_Smit = -(1/model.g_L_Smit)* (0.975*model.g_Na_Smit*model.m_t_init_Smit**3*model.h_init_Smit* model.E_Na_Smit + 0.025*model.g_Na_Smit*model.m_p_init_Smit**3*model.h_init_Smit* model.E_Na_Smit +
                    model.g_K_Smit*model.n_init_Smit**4*model.E_K_Smit)
        
        ##### structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        # postsomatic region = 5)
        model.structure = np.array([0] + list(np.tile([1,2],4)) + [1] + list(np.tile([3],model.nof_segments_presomatic_region)) +\
                             list(np.tile([4],model.nof_segments_soma)) + [5] + list(np.tile([1,2],model.nof_axonal_internodes)) + [1])
        # indexes presomatic region
        model.index_presomatic_region = np.argwhere(model.structure == 3)
        model.start_index_presomatic_region = int(model.index_presomatic_region[0])
        # indexes of soma
        model.index_soma = np.argwhere(model.structure == 4)
        model.start_index_soma = int(model.index_soma[0])
        model.end_index_soma = int(model.index_soma[-1])
        # further structural data
        model.nof_comps = len(model.structure)
        model.nof_comps_dendrite = len(model.structure[:model.start_index_soma])
        model.nof_comps_axon = len(model.structure[model.end_index_soma+1:])
        
        #####  Compartment lengths
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
        model.compartment_lengths[np.where(model.structure == 3)] = model.length_presomatic_region/model.nof_segments_presomatic_region
        # soma
        model.compartment_lengths[np.where(model.structure == 4)] = model.diameter_soma/model.nof_segments_soma
        # postsomatic region
        model.compartment_lengths[np.where(model.structure == 5)] = model.length_postsomatic_region
        # total length neuron
        model.length_neuron = sum(model.compartment_lengths)
        
        ##### Compartment diameters
        # initialize
        model.compartment_diameters = np.zeros(model.nof_comps+1)*um
        # dendrite
        model.compartment_diameters[0:model.start_index_soma] = model.diameter_dendrite
        # soma
        model.soma_comp_diameters = calc.get_soma_diameters(model.nof_segments_soma,
                                                            model.diameter_dendrite,
                                                            model.diameter_soma,
                                                            model.diameter_axon)
        model.compartment_diameters[model.start_index_soma:model.end_index_soma+2] = model.soma_comp_diameters
        # axon
        model.compartment_diameters[model.end_index_soma+2:] = model.diameter_axon
        
        ##### number of axonal myelin layers
        model.nof_myelin_layers_axon = np.floor(0.5*(model.axon_outer_diameter-model.diameter_axon)/model.myelin_layer_thicknes_axon)
        
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
        model.distance_comps_middle[0] = 0.5*model.compartment_lengths[0]
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5* model.compartment_lengths[ii] + 0.5* model.compartment_lengths[ii+1]
        
        ##### Capacities
        # initialize
        model.c_m = np.zeros_like(model.structure)*uF/cm**2
        # all but internodes dendrite
        model.c_m[0:model.start_index_soma][model.structure[0:model.start_index_soma] != 1] = model.c_m_layer
        # dendritic internodes
        model.c_m[0:model.start_index_soma][model.structure[0:model.start_index_soma] == 1] = model.c_m_layer/(1+model.nof_myelin_layers_dendrite)
        # soma
        model.c_m[np.where(model.structure == 4)] = model.c_m_layer/(1+model.nof_myelin_layers_soma)
        # all but internodes axon
        model.c_m[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] != 1] = model.c_mem
        # axonal internodes
        model.c_m[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] == 1] = 1/(1/model.c_mem + model.nof_myelin_layers_axon/model.c_my)
        
        ##### Condactivities internodes
        # initialize
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        # dendritic internodes
        model.g_m[0:model.start_index_soma][model.structure[0:model.start_index_soma] == 1] = model.g_m_layer/(1+model.nof_myelin_layers_dendrite)
        # axonal internodes
        model.g_m[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] == 1] = 1/(model.r_mem + model.nof_myelin_layers_axon*model.r_my)
        
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
        model.g_Na_vector[0:model.start_index_soma][model.structure[0:model.start_index_soma] != 1] = model.g_Na_Rat
        model.g_Na_vector[np.where(model.structure == 4)] = model.g_Na_soma
        model.g_Na_vector[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] != 1] = model.g_Na_Smit
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
    neuron.v = V_res
    neuron.m_t_Smit = model.m_t_init_Smit
    neuron.m_p_Smit = model.m_p_init_Smit
    neuron.n_Smit = model.n_init_Smit
    neuron.h_Smit = model.h_init_Smit
    neuron.m_Rat = model.m_init_Rat
    neuron.n_Rat = model.n_init_Rat
    neuron.h_Rat = model.h_init_Rat
    
    ##### Set parameter values of differential equations
    # conductances dentritic nodes and peripheral terminal 
    neuron.g_Na_Rat[0:model.start_index_soma] = model.g_Na_Rat
    neuron.g_K_Rat[0:model.start_index_soma] = model.g_K_Rat
    neuron.g_L_Rat[0:model.start_index_soma] = model.g_L_Rat
    
    neuron.g_Na_Smit[0:model.start_index_soma] = 0*msiemens/cm**2
    neuron.g_K_Smit[0:model.start_index_soma] = 0*msiemens/cm**2
    neuron.g_L_Smit[0:model.start_index_soma] = 0*msiemens/cm**2
    
    # conductances axonal nodes
    neuron.g_Na_Smit[model.end_index_soma+1:] = model.g_Na_Smit
    neuron.g_K_Smit[model.end_index_soma+1:] = model.g_K_Smit
    neuron.g_L_Smit[model.end_index_soma+1:] = model.g_L_Smit
    
    neuron.g_Na_Rat[model.end_index_soma+1:] = 0*msiemens/cm**2
    neuron.g_K_Rat[model.end_index_soma+1:] = 0*msiemens/cm**2
    neuron.g_L_Rat[model.end_index_soma+1:] = 0*msiemens/cm**2
    
    # conductances soma
    neuron.g_Na_Rat[model.index_soma] = model.g_Na_soma
    neuron.g_K_Rat[model.index_soma] = model.g_K_soma
    neuron.g_L_Rat[model.index_soma] = model.g_L_soma
    
    neuron.g_Na_Smit[model.index_soma] = 0*msiemens/cm**2
    neuron.g_K_Smit[model.index_soma] = 0*msiemens/cm**2
    neuron.g_L_Smit[model.index_soma] = 0*msiemens/cm**2
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    
    neuron.g_Na_Rat[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_K_Rat[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_L_Rat[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    neuron.g_Na_Smit[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_K_Smit[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_L_Smit[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    # other parameters
    neuron.V_res = model.V_res
    neuron.E_Na_Smit = model.E_Na_Smit
    neuron.E_K_Smit = model.E_K_Smit
    neuron.E_L_Smit = model.E_L_Smit
    neuron.E_Na_Rat = model.E_Na_Rat
    neuron.E_K_Rat = model.E_K_Rat
    neuron.E_L_Rat = model.E_L_Rat
    neuron.T_celsius = model.T_celsius
    
    return neuron, model
