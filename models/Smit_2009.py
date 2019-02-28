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
Na_ratio = 7.2102
K_ratio = 0.036645
Leak_ratio = 0.036645

# =============================================================================
# Resting potential
# =============================================================================
##### Resting potential of cell
V_res = -79.4*mV *1.035**((T_celsius-6.3)/10)

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
# Differential equations
# =============================================================================
eqs = '''
I_Na = 0.975*g_Na*m_t**3*h*(E_Na-(v-V_res)) : amp/meter**2
I_Na_p = 0.025*g_Na*m_p**3*h*(E_Na-(v-V_res)) : amp/meter**2
I_K = g_K*n**4*(E_K-(v-V_res)) : amp/meter**2
I_L = g_L*(E_L-(v-V_res)) : amp/meter**2
Im = I_Na + I_Na_p + I_K + I_L + g_myelin*(-(v-V_res)): amp/meter**2
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
V_res : volt
E_Na : volt
E_K : volt
E_L : volt
T_celsius : 1
'''

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_internodes = 30
##### lengths
length_nodes = 1.061*um
##### diameters
fiber_outer_diameter = 15*um
##### myelin layer thickness
myelin_layer_thicknes = 16*nmeter

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacities one layer
c_mem = 2.8*uF/cm**2
##### myelin layer capacity
c_my = 0.6*uF/cm**2

# =============================================================================
# Condactivities internodes
# =============================================================================
##### cell membrane conductivity internodes
r_mem = 4.871*10**4*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)
##### cell membrane conductivity internodes
r_my = 104*ohm*cm**2 * (1/1.3)**((T_celsius-25)/10)

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.04*uA/np.sqrt(mS)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 500*um #1.5*mm

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Smit et al. 2009"
display_name_plots = "Smit et al. (2009)"
display_name_short = "Smit 09"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.append(np.linspace(2, 3, num=30, endpoint = False),
                                  np.linspace(3, 8, num=20))*1e-3

# =============================================================================
# Calculations
# =============================================================================
##### Temperature
T_kelvin = zero_celsius + T_celsius*kelvin

##### Nernst potentials
# Nernst potential sodium
E_Na = R*T_kelvin/F * np.log(Na_ratio) - V_res
# Nernst potential potassium
E_K = R*T_kelvin/F * np.log(K_ratio) - V_res

##### rates for resting potential
alpha_m_t_0 = 4.42*2.5/(np.exp(2.5)-1) * 2.23**(0.1*(T_celsius-20))
alpha_m_p_0 = 2.06*(2.5-0.1*(-20))/(1*(np.exp(2.5-0.1*(-20)))-1) * 1.99**(0.1*(T_celsius-20))
alpha_n_0 = 0.2*1.0/(10*(np.exp(1)-1)) * 1.5**(0.1*(T_celsius-20))
alpha_h_0 = 1.47*0.07 * 1.5**(0.1*(T_celsius-20))
beta_m_t_0 = 4.42*4.0 * 2.23**(0.1*(T_celsius-20))
beta_m_p_0 = 2.06*4.0*np.exp(20/18) * 1.99**(0.1*(T_celsius-20))
beta_n_0 = 0.2*0.125*1 * 1.5**(0.1*(T_celsius-20))
beta_h_0 = 1.47/(1+np.exp(3.0)) * 1.5**(0.1*(T_celsius-20))

##### initial values for gating variables
m_t_init = alpha_m_t_0 / (alpha_m_t_0 + beta_m_t_0)
m_p_init = alpha_m_p_0 / (alpha_m_p_0 + beta_m_p_0)
n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)               

##### calculate Nerst potential for leakage current
E_L = -(1/g_L)* (0.975*g_Na*m_t_init**3*h_init* E_Na + 0.025*g_Na*m_p_init**3*h_init* E_Na + g_K*n_init**4*E_K)

##### structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5)
structure = np.array(list(np.tile([2,1],nof_internodes)) + [2])
nof_comps = len(structure)

#####  Compartment lengths
# initialize
compartment_lengths = np.zeros_like(structure)*um
# internodes
compartment_lengths[np.where(structure == 1)] = 7.9*10**-2*np.log((fiber_outer_diameter/cm)/(3.4*1e-4))*cm
# nodes
compartment_lengths[np.where(structure == 2)] = length_nodes
# total length neuron
length_neuron = sum(compartment_lengths)

##### Compartment diameters
# internode inner diameter
internode_inner_diameter = 0.63*fiber_outer_diameter - 3.4*1e-5*cm
# initialize
compartment_diameters = np.zeros(nof_comps+1)*um
# internodes
compartment_diameters[:] = internode_inner_diameter
#diameter_nodes calculateion in paper:
#(8.502*10**5*(fiber_outer_diameter/cm)**3 - 1.376*10**3*(fiber_outer_diameter/cm)**2 + 8.202*10**-1*(fiber_outer_diameter/cm) - 3.622*10**-5)*cm

##### Number of myelin layers
nof_myelin_layers = np.floor(0.5*(fiber_outer_diameter-internode_inner_diameter)/myelin_layer_thicknes)

#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
distance_comps_middle[0] = 0.5*compartment_lengths[0]
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]
    
###### Capacities
# initialize
c_m = np.zeros_like(structure)*uF/cm**2
# nodes
c_m[np.where(structure == 2)] = c_mem
# internodes
c_m[structure == 1] = 1/(1/c_mem + nof_myelin_layers/c_my)

###### Condactivities internodes
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# calculate values
g_m[structure == 1] = 1/(r_mem + nof_myelin_layers*r_my)

###### Axoplasmatic resistances
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

###### Surface arias
# lateral surfaces
m = [np.sqrt(abs(compartment_diameters[i+1] - compartment_diameters[i])**2 + compartment_lengths[i]**2)
           for i in range(0,nof_comps)]
# total surfaces
A_surface = [(compartment_diameters[i+1] + compartment_diameters[i])*np.pi*m[i]*0.5
           for i in range(0,nof_comps)]

##### Noise term
g_Na_vector = np.zeros(nof_comps)*msiemens/cm**2
g_Na_vector[structure == 2] = g_Na
noise_term = np.sqrt(A_surface*g_Na_vector)

##### Compartments to plot
comps_to_plot = range(1,nof_comps)

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
        
        ##### Nernst potentials
        # Nernst potential sodium
        model.E_Na = model.R*model.T_kelvin/model.F * np.log(model.Na_ratio) - model.V_res
        # Nernst potential potassium
        model.E_K = model.R*model.T_kelvin/model.F * np.log(model.K_ratio) - model.V_res
        
        ##### rates for resting potential
        alpha_m_t_0 = 4.42*2.5/(np.exp(2.5)-1) * 2.23**(0.1*(model.T_celsius-20))
        alpha_m_p_0 = 2.06*(2.5-0.1*(-20))/(1*(np.exp(2.5-0.1*(-20)))-1) * 1.99**(0.1*(model.T_celsius-20))
        alpha_n_0 = 0.2*1.0/(10*(np.exp(1)-1)) * 1.5**(0.1*(model.T_celsius-20))
        alpha_h_0 = 1.47*0.07 * 1.5**(0.1*(model.T_celsius-20))
        beta_m_t_0 = 4.42*4.0 * 2.23**(0.1*(model.T_celsius-20))
        beta_m_p_0 = 2.06*4.0*np.exp(20/18) * 1.99**(0.1*(model.T_celsius-20))
        beta_n_0 = 0.2*0.125*1 * 1.5**(0.1*(model.T_celsius-20))
        beta_h_0 = 1.47/(1+np.exp(3.0)) * 1.5**(0.1*(model.T_celsius-20))
        
        ##### initial values for gating variables
        model.m_t_init = alpha_m_t_0 / (alpha_m_t_0 + beta_m_t_0)
        model.m_p_init = alpha_m_p_0 / (alpha_m_p_0 + beta_m_p_0)
        model.n_init = alpha_n_0 / (alpha_n_0 + beta_n_0)
        model.h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)               
        
        ##### calculate Nerst potential for leakage current
        model.E_L = -(1/model.g_L)* (0.975*model.g_Na*model.m_t_init**3*model.h_init* model.E_Na +
                     0.025*model.g_Na*model.m_p_init**3*model.h_init* model.E_Na + model.g_K*model.n_init**4*model.E_K)

        ##### structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        # postsomatic region = 5)
        model.structure = np.array(list(np.tile([2,1],model.nof_internodes)) + [2])
        model.nof_comps = len(model.structure)
        
        #####  Compartment lengths
        # initialize
        model.compartment_lengths = np.zeros_like(model.structure)*um
        # internodes
        model.compartment_lengths[np.where(model.structure == 1)] = 7.9*10**-2*np.log((model.fiber_outer_diameter/cm)/(3.4*10**-4))*cm
        # nodes
        model.compartment_lengths[np.where(model.structure == 2)] = model.length_nodes
        # total length neuron
        model.length_neuron = sum(model.compartment_lengths)
        
        ##### Compartment diameters
        # internode inner diameter
        model.internode_inner_diameter = 0.63*model.fiber_outer_diameter - 3.4*10**-5*cm
        # initialize
        model.compartment_diameters = np.zeros(model.nof_comps+1)*um
        # internodes
        model.compartment_diameters[:] = model.internode_inner_diameter
        # diameter_nodes calculateion in paper: (8.502*10**5*(fiber_outer_diameter/cm)**3 - 1.376*10**3*(fiber_outer_diameter/cm)**2 + 8.202*10**-1*(fiber_outer_diameter/cm) - 3.622*10**-5)*cm
        
        ##### Number of myelin layers
        model.nof_myelin_layers = np.floor(0.5*(model.fiber_outer_diameter-model.internode_inner_diameter)/model.myelin_layer_thicknes)
        
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
        model.distance_comps_middle[0] = 0.5*model.compartment_lengths[0]
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5* model.compartment_lengths[ii] + 0.5* model.compartment_lengths[ii+1]
            
        ###### Capacities
        # initialize
        model.c_m = np.zeros_like(model.structure)*uF/cm**2
        # nodes
        model.c_m[np.where(model.structure == 2)] = model.c_mem
        # internodes
        model.c_m[model.structure == 1] = 1/(1/model.c_mem + model.nof_myelin_layers/model.c_my)
        
        ###### Condactivities internodes
        # initialize
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        # calculate values
        g_m[model.structure == 1] = 1/(model.r_mem + model.nof_myelin_layers*model.r_my)
        
        ###### Axoplasmatic resistances
        model.compartment_center_diameters = np.zeros(model.nof_comps)*um
        model.compartment_center_diameters = (model.compartment_diameters[0:-1] + model.compartment_diameters[1:]) / 2
        model.R_a = (model.compartment_lengths*model.rho_in) / ((model.compartment_center_diameters*0.5)**2*np.pi)
        
        ###### Surface arias
        # lateral surfaces
        m = [np.sqrt(abs(model.compartment_diameters[i+1] - model.compartment_diameters[i])**2 + model.compartment_lengths[i]**2)
                   for i in range(0,model.nof_comps)]
        # total surfaces
        model.A_surface = [(model.compartment_diameters[i+1] + model.compartment_diameters[i])*np.pi*m[i]*0.5
                   for i in range(0,model.nof_comps)]
        
        ##### Noise term
        model.g_Na_vector = np.zeros(model.nof_comps)*msiemens/cm**2
        model.g_Na_vector[model.structure == 2] = model.g_Na
        model.noise_term = np.sqrt(model.A_surface*model.g_Na_vector)
        
        ##### Compartments to plot
        model.comps_to_plot = range(1,model.nof_comps)
    
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
    neuron.m_t = model.m_t_init
    neuron.m_p = model.m_p_init
    neuron.n = model.n_init
    neuron.h = model.h_init
    
    ##### Set parameter values of differential equations
    # conductances active compartments
    neuron.g_Na = model.g_Na
    neuron.g_K = model.g_K
    neuron.g_L = model.g_L
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.g_Na[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_K[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    neuron.g_L[np.asarray(np.where(model.structure == 1))] = 0*msiemens/cm**2
    
    # other parameters
    neuron.V_res = model.V_res
    neuron.E_Na = model.E_Na
    neuron.E_K = model.E_K
    neuron.E_L = model.E_L
    neuron.T_celsius = model.T_celsius    
    
    return neuron, model
