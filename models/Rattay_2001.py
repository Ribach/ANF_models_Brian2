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
##### Nerst potential for leakage current
E_L = 10.6*mV

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
myelin_layers_dendrite = 40
myelin_layers_soma = 3
myelin_layers_axon = 80

# =============================================================================
# Initial values for gating variables
# =============================================================================
m_init = 0.05
n_init = 0.3
h_init = 0.6

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
Im = g_Na*m**3*h* (E_Na-(v-V_res)) + g_K*n**4*(E_K-(v-V_res)) + g_L*(E_L-(v-V_res)) + g_myelin*(-(v-V_res)): amp/meter**2
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
nof_segments_presomatic_region = 3
##### number of segments for soma
nof_segments_soma = 20
##### number of modeled axonal internodes (at least 5)
nof_axonal_internodes = 10
##### build structure
structure = np.array([0] + list(np.tile([1,2],5)) + [1] + list(np.tile([3],nof_segments_presomatic_region)) + list(np.tile([4],nof_segments_soma)) + [5] + list(np.tile([1,2],nof_axonal_internodes-1)) + [1])

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
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 1] = 350*um
##### internodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 1] = 500*um
##### nodes dendrite
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 2] = 2.5*um
##### nodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 2] = 2.5*um
##### presomatic region
compartment_lengths[np.where(structure == 3)] = (100/3)*um
##### soma
compartment_lengths[np.where(structure == 4)] = 30*um/nof_segments_soma
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
soma_diameter = 30*um
axon_diameter = 2*um
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
# Capacities
# =============================================================================
##### membrane capacitivity one layer
c_m_layer = 1*uF/cm**2

##### initialize
c_m = np.zeros_like(structure)*uF/cm**2
##### all but internodes
c_m[np.where(structure != 1)] = c_m_layer
##### dendrite internodes
c_m[0:start_index_soma][structure[0:start_index_soma] == 1] = c_m_layer/(1+myelin_layers_dendrite)
##### soma
c_m[np.where(structure == 4)] = c_m_layer/(1+myelin_layers_soma)
##### axon internodes
c_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = c_m_layer/(1+myelin_layers_axon)

# =============================================================================
# Condactivities internodes
# =============================================================================
##### membrane conductivity internodes one layer
g_m_layer = 1*msiemens/cm**2

##### initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
##### dendritic internodes
g_m[0:start_index_soma][structure[0:start_index_soma] == 1] = g_m_layer/(1+myelin_layers_dendrite)
##### axonal internodes
g_m[end_index_soma+1:][structure[end_index_soma+1:] == 1] = g_m_layer/(1+myelin_layers_axon)

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
k_noise = 0.0003*uA/np.sqrt(mS)
noise_term = np.sqrt(A_surface*g_Na)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Rattay et al. 2001"

# =============================================================================
# Compartments to plot
# =============================================================================
##### get indexes of all compartments that are not segmented
indexes_comps = np.where(np.logical_or(np.logical_or(np.logical_or(structure == 0, structure == 1), structure == 2), structure == 5))
##### calculate middle compartments of presomatic region and soma
middle_comp_presomatic_region = int(start_index_presomatic_region + np.floor((nof_segments_presomatic_region)/2))
middle_comp_soma = int(start_index_soma + np.floor((nof_segments_soma)/2))
##### create array with all compartments to plot
comps_to_plot = np.sort(np.append(indexes_comps, [middle_comp_presomatic_region, middle_comp_soma]))

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
    neuron.n = model.n_init
    neuron.h = model.h_init
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
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
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = f'''
    V_res = {model_name}.V_res
    E_Na = {model_name}.E_Na
    E_K = {model_name}.E_K
    E_L = {model_name}.E_L
    '''
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string
