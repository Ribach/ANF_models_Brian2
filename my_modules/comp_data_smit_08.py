from brian2 import *
import numpy as np
import my_modules.my_functions as my_fun
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F

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
# Nernst potentials
# =============================================================================
##### Resting potential of cell
V_res = -79.4*mV *1.035**((T_celsius-6.3)/10)
##### Nernst potential sodium
E_Na = R*T_kelvin/F * log(Na_ratio) - V_res
##### Nernst potential potassium
E_K = R*T_kelvin/F * log(K_ratio) - V_res
##### Nerst potential for leakage current
E_L = R*T_kelvin/F * log(Leak_ratio) - V_res

# =============================================================================
# Conductivities
# =============================================================================
##### conductivities active compartments
g_Na = 640*msiemens/cm**2 * 1.02**((T_celsius-24)/10)
g_K = 60*msiemens/cm**2 * 1.16**((T_celsius-20)/10)
g_L = 57.5*msiemens/cm**2 * 1.418**((T_celsius-24)/10)

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
m_t_init = 0.05
m_p_init = 0.05
n_init = 0.32
h_init = 0.6

# =============================================================================
# Differential equations
# =============================================================================
#Im = 1*g_Na*m_t**3*h* ((v-V_res)-E_Na) + 0*g_Na*m_p**3*h* ((v-V_res)-E_Na) + g_K*n**4*((v-V_res)-E_K) + g_L*((v-V_res)-E_L) + g_myelin*(-(v-V_res)): amp/meter**2

eqs = '''
Im = 0.975*g_Na*m_t**3*h*(E_Na-(v-V_res)) + 0.025*g_Na*m_p**3*h*(E_Na-(v-V_res)) + g_K*n**4*(E_K-(v-V_res)) + g_L*(E_L-(v-V_res)) + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm_t/dt = alpha_m_t * (1-m_t) - beta_m_t * m_t : 1
dm_p/dt = alpha_m_p * (1-m_p) - beta_m_p * m_p : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m_t = 4.42*(2.5-0.1*(v-V_res)/mV)/(1*(exp(2.5-0.1*(v-V_res)/mV))-1) * 2.23**(0.1*(T_celsius-20))/ms : Hz
alpha_m_p = 2.06*(2.5-0.1*((v-V_res)/mV-20))/(1*(exp(2.5-0.1*((v-V_res)/mV-20)))-1) * 1.99**(0.1*(T_celsius-20))/ms : Hz
alpha_n = 0.2*(1.0-0.1*(v-V_res)/mV)/(10*(exp(1-0.1*(v-V_res)/mV)-1)) * 1.5**(0.1*(T_celsius-20))/ms : Hz
alpha_h = 1.47*0.07*exp(-(v-V_res)/mV/20) * 1.5**(0.1*(T_celsius-20))/ms : Hz
beta_m_t = 4.42*4.0*exp(-(v-V_res)/mV/18) * 2.23**(0.1*(T_celsius-20))/ms : Hz
beta_m_p = 2.06*4.0*exp(-((v-V_res)/mV-20)/18) * 1.99**(0.1*(T_celsius-20))/ms : Hz
beta_n = 0.2*0.125*exp(-(v-V_res)/mV/80) * 1.5**(0.1*(T_celsius-20))/ms : Hz
beta_h = 1.47/(1+exp(3.0-0.1*(v-V_res)/mV)) * 1.5**(0.1*(T_celsius-20))/ms : Hz
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
dendrite_diameter = 15*um
soma_diameter = 27*um
axon_diameter = 15*um
##### initialize
compartment_diameters = np.zeros(nof_comps+1)*um
##### dendrite
compartment_diameters[0:start_index_soma] = dendrite_diameter
##### soma
soma_comp_diameters = my_fun.get_soma_diameters(nof_segments_soma,
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
myelin_layers = 35

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacities one layer
c_mem = 2.8*uF/cm**2
##### myelin layer capacity
c_my = 0.6*uF/cm**2

##### initialize
c_m = np.zeros_like(structure)*uF/cm**2
##### all but internodes
c_m[np.where(structure != 1)] = c_mem
##### values for all compartments (Smit 2008)
c_m[structure == 1] = 1/(1/c_mem + myelin_layers/c_my)

# =============================================================================
# Condactivities internodes
# =============================================================================
##### cell membrane conductivity internodes
r_mem = 4.871*10**4*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)
##### cell membrane conductivity internodes
r_my = 104*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)

##### initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
##### calculate values
g_m[structure == 1] = 1/(r_mem + myelin_layers*r_my)

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
# Compartments to plot
# =============================================================================
##### get indexes of all compartments that are not segmented
indexes_comps = np.where(np.logical_or(np.logical_or(np.logical_or(structure == 0, structure == 1), structure == 2), structure == 5))
##### calculate middle compartments of presomatic region and soma
middle_comp_presomatic_region = int(start_index_presomatic_region + floor((nof_segments_presomatic_region)/2))
middle_comp_soma = int(start_index_soma + floor((nof_segments_soma)/2))
##### create array with all compartments to plot
comps_to_plot = sort(np.append(indexes_comps, [middle_comp_presomatic_region, middle_comp_soma]))
