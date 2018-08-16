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
# Resting potential
# =============================================================================
V_res = -79.4*mV *1.035**((T_celsius-6.3)/10)

# =============================================================================
# Nernst potentials Smit
# =============================================================================
##### Nernst potential sodium
E_Na_Smit = R*T_kelvin/F * log(Na_ratio) - V_res
##### Nernst potential potassium
E_K_Smit = R*T_kelvin/F * log(K_ratio) - V_res
##### Nerst potential for leakage current
E_L_Smit = R*T_kelvin/F * log(Leak_ratio) - V_res

# =============================================================================
# Nernst potentials Rattay
# =============================================================================
##### Nernst potential sodium
E_Na_Rat = 115*mV
##### Nernst potential potassium
E_K_Rat = -12*mV
##### Nerst potential for leakage current
E_L_Rat = 10.6*mV

# =============================================================================
# Conductivities Smit + Rattay
# =============================================================================
##### conductivities active compartments Rattay
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
# Initial values for gating variables
# =============================================================================
##### Smit
m_t_init_Smit = 0.05
m_p_init_Smit = 0.05
n_init_Smit = 0.32
h_init_Smit = 0.6
##### Rattay
m_init_Rat = 0.05
n_init_Rat = 0.3
h_init_Rat = 0.6

# =============================================================================
# Differential equations
# =============================================================================
#Im = 1*g_Na*m_t**3*h* ((v-V_res)-E_Na) + 0*g_Na*m_p**3*h* ((v-V_res)-E_Na) + g_K*n**4*((v-V_res)-E_K) + g_L*((v-V_res)-E_L) + g_myelin*(-(v-V_res)): amp/meter**2

eqs = '''
Im = 0.975*g_Na_Smit*m_t_Smit**3*h_Smit*(E_Na_Smit-(v-V_res)) + 0.025*g_Na_Smit*m_p_Smit**3*h_Smit*(E_Na_Smit-(v-V_res)) + g_K_Smit*n_Smit**4*(E_K_Smit-(v-V_res)) + g_L_Smit*(E_L_Smit-(v-V_res)) + g_myelin*(-(v-V_res)) + g_Na_Rat*m_Rat**3*h_Rat*(E_Na_Rat-(v-V_res)) + g_K_Rat*n_Rat**4*(E_K_Rat-(v-V_res)) + g_L_Rat*(E_L_Rat-(v-V_res)): amp/meter**2
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

##### number of segments for presomatic region
nof_segments_presomatic_region = 5
##### number of segments for soma
nof_segments_soma = 20
##### number of modeled axonal internodes (at least 5)
nof_axonal_internodes = 10
##### build structure
structure = np.array([0] + list(np.tile([1,2],4)) + [1] + list(np.tile([3],nof_segments_presomatic_region)) + list(np.tile([4],nof_segments_soma)) + [5] + list(np.tile([1,2],nof_axonal_internodes)) + [1])

##### indexes presomatic region
index_presomatic_region = np.argwhere(structure == 3)
start_index_presomatic_region = int(index_presomatic_region[0])
##### indexes of soma
index_soma = np.argwhere(structure == 4)
start_index_soma = int(index_soma[0])
end_index_soma = int(index_soma[-1])
##### further structural data
nof_comps = len(structure)
nof_comps_dendrite = len(structure[:start_index_soma])
nof_comps_axon = len(structure[end_index_soma+1:])

# =============================================================================
#  Compartment lengths
# ============================================================================= 
##### initialize
compartment_lengths = np.zeros_like(structure)*um
##### peripheral terminal
compartment_lengths[np.where(structure == 0)] = 10*um
##### internodes dendrite
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 1] = [210,440,350,430,360]*um
##### internodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 1] = 77.4*um
##### nodes dendrite
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 2] = 2.5*um
##### nodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 2] = 1.061*um
##### presomatic region
compartment_lengths[np.where(structure == 3)] = (100/3)*um
##### soma
compartment_lengths[np.where(structure == 4)] = 27*um/nof_segments_soma
##### postsomatic region
compartment_lengths[np.where(structure == 5)] = 5*um

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
soma_diameter = 27*um
axon_diameter = 2.02*um
##### initialize
compartment_diameters = np.zeros(nof_comps+1)*um
##### dendrite
compartment_diameters[0:start_index_soma] = dendrite_diameter
##### soma
soma_comp_diameters = calc.get_soma_diameters(nof_segments_soma,
                                                dendrite_diameter,
                                                soma_diameter,
                                                axon_diameter)

compartment_diameters[start_index_soma:end_index_soma+2] = soma_comp_diameters

##### axon
compartment_diameters[end_index_soma+2:] = axon_diameter

# =============================================================================
# Myelin data
# =============================================================================
myelin_layer_thicknes = 16*nmeter
myelin_layers_dendrite = 40
myelin_layers_soma = 3
myelin_layers_axon = 35

# =============================================================================
# Capacities
# =============================================================================
##### capacaty one layer (membrane and myelin as in Rattay's model)
c_m_layer = 1*uF/cm**2
##### cell membrane capacitiy one layer
c_mem = 2.8*uF/cm**2
##### myelin layer capacity
c_my = 0.6*uF/cm**2

##### initialize
c_m = np.zeros_like(structure)*uF/cm**2
##### all but internodes dendrite
c_m[0:start_index_soma][structure[0:start_index_soma] != 1] = c_m_layer
##### dendritic internodes
c_m[0:start_index_soma][structure[0:start_index_soma] == 1] = c_m_layer/(1+myelin_layers_dendrite)
##### soma
c_m[np.where(structure == 4)] = c_m_layer/(1+myelin_layers_soma)
##### all but internodes axon
c_m[end_index_soma+1:][structure[end_index_soma+1:] != 1] = c_mem
##### axonal internodes
c_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = 1/(1/c_mem + myelin_layers_axon/c_my)

# =============================================================================
# Condactivities internodes
# =============================================================================
##### membrane conductivity internodes one layer
g_m_layer = 1*msiemens/cm**2
##### cell membrane conductivity internodes
r_mem = 4.871*10**4*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)
##### cell membrane conductivity internodes
r_my = 104*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)

##### initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
##### dendritic internodes
g_m[0:start_index_soma][structure[0:start_index_soma] == 1] = g_m_layer/(1+myelin_layers_dendrite)
##### axonal internodes
g_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = 1/(r_mem + myelin_layers_axon*r_my)

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
k_noise = 0.0008*uA/np.sqrt(mS)
noise_term = np.sqrt(A_surface*g_Na_Rat)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Smit et al. 2010"

# =============================================================================
# Compartments to plot
# =============================================================================
##### get indexes of all compartments that are not segmented
indexes_comps = np.where(np.logical_or(np.logical_or(np.logical_or(structure == 0, structure == 1), structure == 2), structure == 5))
##### calculate middle compartments of presomatic region and soma
middle_comp_presomatic_region = int(start_index_presomatic_region + floor((nof_segments_presomatic_region)/2))
middle_comp_soma = int(start_index_soma + floor((nof_segments_soma)/2))
##### create array with all compartments to plot
comps_to_plot = sort(np.append(indexes_comps, [middle_comp_presomatic_region, middle_comp_soma]))

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
    neuron.m_t_Smit = model.m_t_init_Smit
    neuron.m_p_Smit = model.m_p_init_Smit
    neuron.n_Smit = model.n_init_Smit
    neuron.h_Smit = model.h_init_Smit
    neuron.m_Rat = model.m_init_Rat
    neuron.n_Rat = model.n_init_Rat
    neuron.h_Rat = model.h_init_Rat
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
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
    neuron.g_K_Smit[model.end_index_soma+1:] = model.g_L_Smit
    
    neuron.g_Na_Rat[model.end_index_soma+1:] = 0*msiemens/cm**2
    neuron.g_K_Rat[model.end_index_soma+1:] = 0*msiemens/cm**2
    neuron.g_K_Rat[model.end_index_soma+1:] = 0*msiemens/cm**2
    
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
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = f'''
    V_res = {model_name}.V_res
    E_Na_Smit = {model_name}.E_Na_Smit
    E_K_Smit = {model_name}.E_K_Smit
    E_L_Smit = {model_name}.E_L_Smit
    E_Na_Rat = {model_name}.E_Na_Rat
    E_K_Rat = {model_name}.E_K_Rat
    E_L_Rat = {model_name}.E_L_Rat
    T_celsius = {model_name}.T_celsius
    '''
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string
