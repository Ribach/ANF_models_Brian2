from brian2 import *
import numpy as np
import my_modules.my_functions as my_fun
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F

## =============================================================================
## Temperature
## =============================================================================
T_celsius = 37
T_kelvin = zero_celsius + T_celsius*kelvin

# =============================================================================
# Permeabilities
# =============================================================================
P_Na = 51.5*um/second
P_K = 2.0*um/second

# =============================================================================
# Conductivities
# =============================================================================
# conductivity of leakage channels; not needed, as the leakage channels were excluded
g_L = 728*siemens/meter**2

# =============================================================================
# Ion concentrations
# =============================================================================
Na_i = 10*mole/meter**3
Na_e = 142*mole/meter**3
K_i = 141*mole/meter**3
K_e = 4.2*mole/meter**3

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 70*ohm*cm
##### external resistivity
rho_out = 300*ohm*cm

# =============================================================================
# Initial values for gating variables (steady state values at resting potential)
# =============================================================================
m_init = 0.00775
n_init = 0.0268
h_init = 0.7469

# =============================================================================
# Other parameters
# =============================================================================
##### Myelin layers soma and presomatic region
myelin_layers_somatic_region = 4
##### Dividing factor for ion currentsin the somatic region (makes currents smaller)
dividing_factor = 30

# =============================================================================
# Potentials
# =============================================================================
##### Resting potential (calculated with Goldman equation)
V_res = (R*T_kelvin)/F * np.log((P_K*n_init**2*K_e + P_Na*h_init*m_init**3*Na_e)/(P_K*n_init**2*K_i + P_Na*h_init*m_init**3*Na_i))

##### Nerst potential for leakage current; leakage chanels were excluded but could be added by using: g_L*(E_L-(v-V_res))  
E_L = (-1/g_L)*(P_Na*m_init**3*h_init*(V_res*F**2)/(R*T_kelvin) * (Na_e-Na_i*exp(V_res*F/(R*T_kelvin)))/(1-exp(V_res*F/(R*T_kelvin))) + P_K*n_init**2*(V_res*F**2)/(R*T_kelvin) * (K_e-K_i*exp(V_res*F/(R*T_kelvin)))/(1-exp(V_res*F/(R*T_kelvin))))

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
Im = P_Na*m**3*h*(v*F**2)/(R*T_kelvin) * (Na_e-Na_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) + P_K*n**2*(v*F**2)/(R*T_kelvin) * (K_e-K_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m = 0.49/mV*((v-V_res)-25.41*mV)/(1-exp((25.41*mV-(v-V_res))/(6.06*mV))) * 2.2**(0.1*(T_celsius-20))/ms : Hz
alpha_n = 0.02/mV*((v-V_res)-35*mV)/(1-exp((35*mV-(v-V_res))/(10*mV))) * 3**(0.1*(T_celsius-20))/ms : Hz
alpha_h = 0.09/mV*(-27.74*mV-(v-V_res))/(1-exp(((v-V_res)+27.74*mV)/(9.06*mV))) * 2.9**(0.1*(T_celsius-20))/ms : Hz
beta_m = 1.04/mV*(21*mV-(v-V_res))/(1-exp(((v-V_res)-21*mV)/(9.41*mV))) * 2.2**(0.1*(T_celsius-20))/ms : Hz
beta_n = 0.05/mV*(10*mV-(v-V_res))/(1-exp(((v-V_res)-10*mV)/(10*mV))) * 3**(0.1*(T_celsius-20))/ms : Hz
beta_h = 3.7/(1+exp((56*mV-(v-V_res))/(12.5*mV))) * 2.9**(0.1*(T_celsius-20))/ms : Hz
P_Na : meter/second
P_K : meter/second
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

##### number of segments for presomatic region
nof_segments_presomatic_region = 10
##### number of segments for soma
nof_segments_soma = 10
##### number of modeled axonal internodes (at least 5)
nof_axonal_internodes = 10
##### build structure
structure = np.array([0] + list(np.tile([1,2],6)) + list(np.tile([3],nof_segments_presomatic_region)) + [2] + list(np.tile([4],nof_segments_soma)) + [2] + list(np.tile([1,2],nof_axonal_internodes-1)) + [1])

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
compartment_lengths[0:start_index_soma][structure[0:start_index_soma] == 1] = [175,175,175,175,175,50]*um
##### internodes axon
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 1] = list([150,200,250,300,350] + list(np.tile([350],nof_axonal_internodes-5)))*um
##### nodes
compartment_lengths[np.where(structure == 2)] = 1*um
##### presomatic region
compartment_lengths[np.where(structure == 3)] = 100*um/nof_segments_presomatic_region
##### soma
compartment_lengths[np.where(structure == 4)] = 30*um/nof_segments_soma

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
dendrite_diameter = 3*um
region_around_soma_diameter = 2*um
soma_diameter = 10*um
axon_diameter = 3*um
##### initialize
compartment_diameters = np.zeros(nof_comps+1)*um
##### dendrite
compartment_diameters[0:start_index_presomatic_region+1] = dendrite_diameter
##### region before soma
compartment_diameters[start_index_presomatic_region+1:start_index_soma+1] = region_around_soma_diameter
##### soma
compartment_diameters[start_index_soma+1:end_index_soma+1] = soma_diameter
##### node after soma
compartment_diameters[end_index_soma+1] = region_around_soma_diameter
##### axon
compartment_diameters[end_index_soma+2:] = axon_diameter

# =============================================================================
# Capacitivites
# =============================================================================
##### membrane capacitivity one layer (calculated with the values given in Frijns et al. 2005 page 146)
c_m_layer = 2.801*uF/cm**2

##### initialize
c_m = np.zeros_like(structure)*uF/cm**2
##### peripheral terminal and nodes
c_m[np.where(np.logical_or(structure == 0, structure == 2))] = c_m_layer
##### somatic region
c_m[np.where(np.logical_or(structure == 3, structure == 4))] = c_m_layer/(1+myelin_layers_somatic_region)
##### values for internodes are zero

# =============================================================================
# Condactivities internodes
# =============================================================================
##### membrane conductivity internodes one layer (calculated with the values given in Frijns et al. 2005 page 146)
g_m_layer = 0.6*msiemens/cm**2

##### initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
##### somatic region
g_m[np.where(np.logical_or(structure == 3, structure == 4))] = g_m_layer/(1+myelin_layers_somatic_region)
##### values for internodes are zero

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
indexes_comps = np.where(np.logical_or(np.logical_or(structure == 0, structure == 1), structure == 2))
##### calculate middle compartments of presomatic region and soma
middle_comp_presomatic_region = int(start_index_presomatic_region + floor((nof_segments_presomatic_region)/2))
middle_comp_soma = int(start_index_soma + floor((nof_segments_soma)/2))
##### create array with all compartments to plot
comps_to_plot = sort(np.append(indexes_comps, [middle_comp_presomatic_region, middle_comp_soma]))
