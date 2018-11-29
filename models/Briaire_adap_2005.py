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
# Permeabilities
# =============================================================================
##### Dividing factor for permeabilities in the somatic region (makes currents smaller)
dividing_factor = 30
##### permeabilites active compartments
P_Na = 51.5*um/second
P_K = 2.0*um/second
P_KLT = 0.018*um/second
P_HCN = 0.0108*um/second
##### permeabilites somatic region
P_Na_somatic_region = P_Na / dividing_factor
P_K_somatic_region = P_K / dividing_factor
P_KLT_somatic_region = P_KLT / dividing_factor
P_HCN_somatic_region = P_HCN / dividing_factor

# =============================================================================
# Conductivities
# =============================================================================
# conductivity of leakage channels
g_L = 728*siemens/meter**2
g_L_somatic_region = g_L / dividing_factor

# =============================================================================
# Ion concentrations
# =============================================================================
Na_i = 10*mole/meter**3
Na_e = 142*mole/meter**3
K_i = 141*mole/meter**3
K_e = 4.2*mole/meter**3
C_i = 60*mole/meter**3
C_e = 15*mole/meter**3

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 70*ohm*cm
##### external resistivity
rho_out = 300*ohm*cm

# =============================================================================
# Other parameters
# =============================================================================
##### Myelin layers soma and presomatic region
myelin_layers_somatic_region = 4

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
I_Na = P_Na*m**3*h*(v*F**2)/(R*T_kelvin) * (Na_e-Na_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) : amp/meter**2
I_K = P_K*n**2*(v*F**2)/(R*T_kelvin) * (K_e-K_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) : amp/meter**2
I_KLT = P_KLT*w**4*z*(v*F**2)/(R*T_kelvin) * (K_e-K_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) : amp/meter**2
I_HCN = P_HCN*r*(v*F**2)/(R*T_kelvin) * (C_e-C_i*exp(v*F/(R*T_kelvin)))/(exp(v*F/(R*T_kelvin))-1) : amp/meter**2
I_L = g_L*(E_Leak-(v-V_res)) : amp/meter**2
Im = I_Na + I_K + I_KLT + I_HCN + I_L + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dn/dt = alpha_n * (1-n) - beta_n * n : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
dw/dt = alpha_w * (1-w) - beta_w * w : 1
dz/dt = alpha_z * (1-z) - beta_z * z : 1
dr/dt = alpha_r * (1-r) - beta_r * r : 1
alpha_m = 0.49/mV*((v-V_res)-25.41*mV)/(1-exp((25.41*mV-(v-V_res))/(6.06*mV))) * 2.2**(0.1*(T_celsius-20))/ms : Hz
alpha_n = 0.02/mV*((v-V_res)-35*mV)/(1-exp((35*mV-(v-V_res))/(10*mV))) * 3**(0.1*(T_celsius-20))/ms : Hz
alpha_h = 0.09/mV*(-27.74*mV-(v-V_res))/(1-exp(((v-V_res)+27.74*mV)/(9.06*mV))) * 2.9**(0.1*(T_celsius-20))/ms : Hz
beta_m = 1.04/mV*(21*mV-(v-V_res))/(1-exp(((v-V_res)-21*mV)/(9.41*mV))) * 2.2**(0.1*(T_celsius-20))/ms : Hz
beta_n = 0.05/mV*(10*mV-(v-V_res))/(1-exp(((v-V_res)-10*mV)/(10*mV))) * 3**(0.1*(T_celsius-20))/ms : Hz
beta_h = 3.7/(1+exp((56*mV-(v-V_res))/(12.5*mV))) * 2.9**(0.1*(T_celsius-20))/ms : Hz
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
P_Na : meter/second
P_K : meter/second
P_KLT : meter/second
P_HCN : meter/second
g_L : siemens/meter**2
g_myelin : siemens/meter**2
E_Leak : volt
V_res : volt
T_celsius : 1
T_kelvin : kelvin
Na_i : mole/meter**3
Na_e : mole/meter**3
K_i : mole/meter**3
K_e : mole/meter**3
C_i : mole/meter**3
C_e : mole/meter**3
'''

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_segments_presomatic_region = 10
nof_segments_soma = 10
nof_axonal_internodes = 10 # have to be at least 5
##### lengths
length_peripheral_terminal = 10*um
length_internodes_dendrite = [175,175,175,175,175,50]*um
length_internodes_axon = [150,200,250,300,350]*um # the last value defines the lengths of further internodes
length_nodes = 1*um
length_presomatic_region = 100*um
length_soma = 30*um
##### diameters
diameter_dendrite = 3*um
diameter_somatic_region = 2*um
diameter_soma = 10*um
diameter_axon = 3*um

# =============================================================================
# Capacitivites
# =============================================================================
##### membrane capacitivity one layer (calculated with the values given in Briaire and Frijns 2005 page 146)
c_m_layer = 2.801*uF/cm**2

# =============================================================================
# Condactivities internodes
# =============================================================================
##### membrane conductivity somatic region one layer (calculated with the values given in Briaire and Frijns 2005 page 146)
g_m_layer = 0.6*msiemens/cm**2

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.00001*uA*np.sqrt(second/um**3)
    
# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Briaire and Frijns 2005 a."
display_name_short = "Briaire 05 a."

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.append(np.linspace(0.8, 0.85, num=20, endpoint = False),
                                            np.linspace(0.85, 1.3, num=15, endpoint = False)),
                                            np.linspace(1.3, 5, num=15))*1e-3

# =============================================================================
# Calculations
# =============================================================================
##### Temperature
T_kelvin = zero_celsius + T_celsius*kelvin

##### rates for resting potential
alpha_m_0 = 0.49*(-25.41)/(1-np.exp(25.41/6.06)) * 2.2**(0.1*(T_celsius-20))
alpha_n_0 = 0.02*(-35)/(1-np.exp(35/10)) * 3**(0.1*(T_celsius-20))
alpha_h_0 = 0.09*(-27.74)/(1-np.exp(27.74/9.06)) * 2.9**(0.1*(T_celsius-20))
beta_m_0 = 1.04*21/(1-np.exp(-21/9.41)) * 2.2**(0.1*(T_celsius-20))
beta_n_0 = 0.05*10/(1-np.exp(-10/10)) * 3**(0.1*(T_celsius-20))
beta_h_0 = 3.7/(1+np.exp(56/12.5)) * 2.9**(0.1*(T_celsius-20))

##### initial values for gating variables
m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  
w_init = 1/(np.exp(13/5)+1)**(1/4)
z_init = 1/(2*(np.exp(0.74)+1))+0.5
r_init = 1/(np.exp(+62/35)+1)

##### Potentials
# Resting potential (calculated with Goldman equation)
V_res = (R*T_kelvin)/F * np.log((P_K*n_init**2*K_e + P_Na*h_init*m_init**3*Na_e + P_KLT*z_init*w_init**4*K_e + P_HCN*r_init*C_e)/
         (P_K*n_init**2*K_i + P_Na*h_init*m_init**3*Na_i + P_KLT*z_init*w_init**4*K_i + P_HCN*r_init*C_i))

# Nerst potential for leakage current
E_L = (-1/g_L)*(P_Na*m_init**3*h_init*(V_res*F**2)/(R*T_kelvin) *
             (Na_e-Na_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))) +
             P_K*n_init**2*(V_res*F**2)/(R*T_kelvin) *
             (K_e-K_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))) +
             P_KLT*z_init*w_init**4*(V_res*F**2)/(R*T_kelvin) *
             (K_e-K_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))) +
             P_HCN*r_init*(V_res*F**2)/(R*T_kelvin) *
             (C_e-C_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))))

E_L_somatic_region = (-1/g_L_somatic_region)*(P_Na_somatic_region*m_init**3*h_init*(V_res*F**2)/(R*T_kelvin) *
                      (Na_e-Na_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))) +
                      P_K_somatic_region*n_init**2*(V_res*F**2)/(R*T_kelvin) *
                      (K_e-K_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))) +
                      P_KLT_somatic_region*z_init*w_init**4*(V_res*F**2)/(R*T_kelvin) *
                      (K_e-K_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))) +
                      P_HCN_somatic_region*r_init*(V_res*F**2)/(R*T_kelvin) *
                      (C_e-C_i*np.exp(V_res*F/(R*T_kelvin)))/(1-np.exp(V_res*F/(R*T_kelvin))))

##### structure of ANF
# structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
structure = np.array([0] + list(np.tile([1,2],6)) + list(np.tile([3],nof_segments_presomatic_region)) + [2] + 
                           list(np.tile([4],nof_segments_soma)) + [2] + list(np.tile([1,2],nof_axonal_internodes-1)) + [1])
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
compartment_lengths[end_index_soma+1:][structure[end_index_soma+1:] == 1] = list(list(length_internodes_axon) +
                         list(np.tile(length_internodes_axon[-1],nof_axonal_internodes-5)))
# nodes
compartment_lengths[np.where(structure == 2)] = length_nodes
# presomatic region
compartment_lengths[np.where(structure == 3)] = length_presomatic_region/nof_segments_presomatic_region
# soma
compartment_lengths[np.where(structure == 4)] = length_soma/nof_segments_soma
# total length neuron
length_neuron = sum(compartment_lengths)
  
##### Compartment diameters
# initialize
compartment_diameters = np.zeros(nof_comps+1)*um
# dendrite
compartment_diameters[0:start_index_presomatic_region+1] = diameter_dendrite
dendrite_outer_diameter = diameter_dendrite / 0.6 # see Gillespie 1983
# region before soma
compartment_diameters[start_index_presomatic_region+1:start_index_soma+1] = diameter_somatic_region
# soma
compartment_diameters[start_index_soma+1:end_index_soma+1] = diameter_soma
# node after soma
compartment_diameters[end_index_soma+1] = diameter_somatic_region
# axon
compartment_diameters[end_index_soma+2:] = diameter_axon
axon_outer_diameter = diameter_axon / 0.6 # see Gillespie 1983

#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
distance_comps_middle[0] = 0.5*compartment_lengths[0]
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5*compartment_lengths[ii] + 0.5*compartment_lengths[ii+1]

##### Capacitivites
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# peripheral terminal and nodes
c_m[np.where(np.logical_or(structure == 0, structure == 2))] = c_m_layer
# somatic region
c_m[np.where(np.logical_or(structure == 3, structure == 4))] = c_m_layer/(1+myelin_layers_somatic_region)
# values for internodes are zero

##### Condactivities internodes
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# somatic region
g_m[np.where(np.logical_or(structure == 3, structure == 4))] = g_m_layer/(1+myelin_layers_somatic_region)
# values for internodes are zero

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
P_Na_vector = np.zeros(nof_comps)*0*meter/second
P_Na_vector[:] = P_Na
P_Na_vector[structure == 1] = 0*meter/second
P_Na_vector[structure == 3] = P_Na_somatic_region
P_Na_vector[structure == 4] = P_Na_somatic_region
noise_term = np.sqrt(A_surface*P_Na_vector)

##### Compartments to plot
# get indexes of all compartments that are not segmented
indexes_comps = np.where(np.logical_or(np.logical_or(structure == 0, structure == 1), structure == 2))
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

        ##### rates for resting potential
        alpha_m_0 = 0.49*(-25.41)/(1-np.exp(25.41/6.06)) * 2.2**(0.1*(model.T_celsius-20))
        alpha_n_0 = 0.02*(-35)/(1-np.exp(35/10)) * 3**(0.1*(model.T_celsius-20))
        alpha_h_0 = 0.09*(-27.74)/(1-np.exp(27.74/9.06)) * 2.9**(0.1*(model.T_celsius-20))
        beta_m_0 = 1.04*21/(1-np.exp(-21/9.41)) * 2.2**(0.1*(model.T_celsius-20))
        beta_n_0 = 0.05*10/(1-np.exp(-10/10)) * 3**(0.1*(model.T_celsius-20))
        beta_h_0 = 3.7/(1+np.exp(56/12.5)) * 2.9**(0.1*(model.T_celsius-20))
        
        ##### initial values for gating variables
        model.m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
        model.n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
        model.h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)
        
        ##### Potentials
        # Resting potential (calculated with Goldman equation)
        model.V_res = (model.R*model.T_kelvin)/model.F * np.log((model.P_K*model.n_init**2*model.K_e + model.P_Na*model.h_init*model.m_init**3*model.Na_e +
                      model.P_KLT*model.z_init*model.w_init**4*model.K_e + model.P_HCN*model.r_init*model.C_e)/
                 (model.P_K*model.n_init**2*model.K_i + model.P_Na*model.h_init*model.m_init**3*model.Na_i +
                  model.P_KLT*model.z_init*model.w_init**4*model.K_i + model.P_HCN*model.r_init*model.C_i))
        
        # Nerst potential for leakage current
        model.E_L = (-1/model.g_L)*(model.P_Na*model.m_init**3*model.h_init*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                     (model.Na_e-model.Na_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) +
                     model.P_K*model.n_init**2*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                     (model.K_e-model.K_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) +
                     model.P_KLT*model.z_init*model.w_init**4*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                     (model.K_e-model.K_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) +
                     model.P_HCN*model.r_init*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                     (model.C_e-model.C_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))))
        
        model.E_L_somatic_region = (-1/model.g_L_somatic_region)*(model.P_Na_somatic_region*model.m_init**3*model.h_init*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                                     (model.Na_e-model.Na_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) +
                                     model.P_K_somatic_region*model.n_init**2*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                                     (model.K_e-model.K_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) +
                                     model.P_KLT_somatic_region*model.z_init*model.w_init**4*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                                     (model.K_e-model.K_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) +
                                     model.P_HCN_somatic_region*model.r_init*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *
                                     (model.C_e-model.C_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))))
        
        ##### Potentials
        # Resting potential (calculated with Goldman equation)
        model.V_res = (model.R*model.T_kelvin)/model.F * np.log((model.P_K*model.n_init**2*model.K_e + model.P_Na*model.h_init*model.m_init**3*model.Na_e)/\
                 (model.P_K*model.n_init**2*model.K_i + model.P_Na*model.h_init*model.m_init**3*model.Na_i))
        
        # Nerst potential for leakage current; leakage chanels were excluded but could be added by using: g_L*(E_L-(v-V_res))  
        model.E_L = (-1/model.g_L)*(model.P_Na*model.m_init**3*model.h_init*(model.V_res*model.F**2)/(model.R*model.T_kelvin) *\
                     (model.Na_e-model.Na_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))) +\
                     model.P_K*model.n_init**2*(model.V_res*F**2)/(model.R*model.T_kelvin) *\
                     (model.K_e-model.K_i*np.exp(model.V_res*model.F/(model.R*model.T_kelvin)))/(1-np.exp(model.V_res*model.F/(model.R*model.T_kelvin))))
        
        ##### structure of ANF
        # structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        model.structure = np.array([0] + list(np.tile([1,2],6)) + list(np.tile([3],model.nof_segments_presomatic_region)) + [2] + \
                                   list(np.tile([4],model.nof_segments_soma)) + [2] + list(np.tile([1,2],model.nof_axonal_internodes-1)) + [1])
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
        model.compartment_lengths[model.end_index_soma+1:][model.structure[model.end_index_soma+1:] == 1] = list(list(model.length_internodes_axon) +
                         list(np.tile(model.length_internodes_axon[-1],model.nof_axonal_internodes-5)))
        # nodes
        model.compartment_lengths[np.where(model.structure == 2)] = model.length_nodes
        # presomatic region
        model.compartment_lengths[np.where(model.structure == 3)] = model.length_presomatic_region/model.nof_segments_presomatic_region
        # soma
        model.compartment_lengths[np.where(model.structure == 4)] = model.length_soma/model.nof_segments_soma
        # total length neuron
        model.length_neuron = sum(model.compartment_lengths)
          
        ##### Compartment diameters
        # initialize
        model.compartment_diameters = np.zeros(model.nof_comps+1)*um
        # dendrite
        model.compartment_diameters[0:model.start_index_presomatic_region+1] = model.diameter_dendrite
        # region before soma
        model.compartment_diameters[model.start_index_presomatic_region+1:model.start_index_soma+1] = model.diameter_somatic_region
        # soma
        model.compartment_diameters[model.start_index_soma+1:model.end_index_soma+1] = model.diameter_soma
        # node after soma
        model.compartment_diameters[model.end_index_soma+1] = model.diameter_somatic_region
        # axon
        model.compartment_diameters[model.end_index_soma+2:] = model.diameter_axon
        
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
        model.distance_comps_middle[0] = 0.5*model.compartment_lengths[0]
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5*model.compartment_lengths[ii] + 0.5*model.compartment_lengths[ii+1]
        
        ##### Capacitivites
        # initialize
        model.c_m = np.zeros_like(model.structure)*uF/cm**2
        # peripheral terminal and nodes
        model.c_m[np.where(np.logical_or(model.structure == 0, model.structure == 2))] = model.c_m_layer
        # somatic region
        model.c_m[np.where(np.logical_or(model.structure == 3, model.structure == 4))] = model.c_m_layer/(1+model.myelin_layers_somatic_region)
        # values for internodes are zero
        
        ##### Condactivities internodes
        # initialize
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        # somatic region
        model.g_m[np.where(np.logical_or(model.structure == 3, model.structure == 4))] = model.g_m_layer/(1+model.myelin_layers_somatic_region)
        # values for internodes are zero
        
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
        model.P_Na_vector = np.zeros(model.nof_comps)*0*meter/second
        model.P_Na_vector[:] = model.P_Na
        model.P_Na_vector[model.structure == 1] = 0*meter/second
        model.P_Na_vector[model.structure == 3] = model.P_Na/model.dividing_factor
        model.P_Na_vector[model.structure == 4] = model.P_Na/model.dividing_factor
        model.noise_term = np.sqrt(model.A_surface*model.P_Na_vector)
        
        ##### Compartments to plot
        # get indexes of all compartments that are not segmented
        model.indexes_comps = np.where(np.logical_or(np.logical_or(model.structure == 0, model.structure == 1), model.structure == 2))
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
    
    ##### Set parameter values of differential equations
    # permeabilities peripheral terminal and active compartments
    neuron.P_Na[np.asarray(np.where(np.logical_or(model.structure == 0, model.structure == 2)))] = model.P_Na
    neuron.P_K[np.asarray(np.where(np.logical_or(model.structure == 0, model.structure == 2)))] = model.P_K
    neuron.P_KLT[np.asarray(np.where(np.logical_or(model.structure == 0, model.structure == 2)))] = model.P_KLT
    neuron.P_HCN[np.asarray(np.where(np.logical_or(model.structure == 0, model.structure == 2)))] = model.P_HCN
    neuron.g_L[np.asarray(np.where(np.logical_or(model.structure == 0, model.structure == 2)))] = model.g_L
    
    # permeabilities internodes
    neuron.P_Na[np.asarray(np.where(model.structure == 1))] = 0*meter/second
    neuron.P_K[np.asarray(np.where(model.structure == 1))] = 0*meter/second
    neuron.P_KLT[np.asarray(np.where(model.structure == 1))] = 0*meter/second
    neuron.P_HCN[np.asarray(np.where(model.structure == 1))] = 0*meter/second
    neuron.g_L[np.asarray(np.where(model.structure == 1))] = 0*siemens/meter**2
    
    # conductances
    neuron.g_myelin = model.g_m
    
    # permeabilities presomatic region
    neuron.P_Na[index_presomatic_region] = model.P_Na_somatic_region
    neuron.P_K[index_presomatic_region] = model.P_K_somatic_region
    neuron.P_KLT[index_presomatic_region] = model.P_KLT_somatic_region
    neuron.P_HCN[index_presomatic_region] = model.P_HCN_somatic_region
    neuron.g_L[index_presomatic_region] = model.g_L_somatic_region
    
    # permeabilities soma
    neuron.P_Na[index_soma] = model.P_Na_somatic_region
    neuron.P_K[index_soma] = model.P_K_somatic_region
    neuron.P_KLT[index_soma] = model.P_KLT_somatic_region
    neuron.P_HCN[index_soma] = model.P_HCN_somatic_region
    neuron.g_L[index_soma] = model.g_L_somatic_region
    
    # Nernst potential for leakage current
    neuron.E_Leak = model.E_L
    neuron.E_Leak[index_presomatic_region] = E_L_somatic_region
    neuron.E_Leak[model.index_soma] = E_L_somatic_region
    
    # other parameters
    neuron.V_res = model.V_res
    neuron.T_celsius = model.T_celsius
    neuron.T_kelvin = model.T_kelvin
    neuron.Na_i = model.Na_i
    neuron.Na_e = model.Na_e
    neuron.K_i = model.K_i
    neuron.K_e = model.K_e
    neuron.C_i = model.C_i
    neuron.C_e = model.C_e
    
    return neuron, model
