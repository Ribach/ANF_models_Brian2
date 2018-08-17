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
V_res = -84*mV
##### Nernst potential sodium
E_Na = 50*mV - V_res
##### Nernst potential potassium
E_K = -84*mV - V_res
##### Nerst potential for leakage current
E_L = 10.6*mV

# =============================================================================
# Conductivities
# =============================================================================
##### conductivities active compartments
gamma_Na = 20*psiemens
gamma_Ks = 10*psiemens
gamma_Kf = 10*psiemens

##### cell membrane conductivity nodes
g_L = 1/(8310*ohm*mm**2)

# =============================================================================
# Numbers of channels per aria
# =============================================================================
rho_Na = 618/um**2
rho_Kf = 20.3/um**2
rho_Ks = 41.2/um**2

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 733*ohm*mm
##### external resistivity
rho_out = 25000*ohm*mm

# =============================================================================
# Initial values for gating variables
# =============================================================================
m_init = 0.00083
ns_init = 0.926
nf_init = 0.256
h_init = 0.627

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
Im = gamma_Na*rho_Na*m**3*h* (E_Na-(v-V_res)) + gamma_Ks*rho_Ks*ns**4*(E_K-(v-V_res)) + gamma_Kf*rho_Kf*nf**4*(E_K-(v-V_res)) + g_L*(E_L-(v-V_res)) + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dns/dt = alpha_ns * (1-ns) - beta_ns * ns : 1
dnf/dt = alpha_nf * (1-nf) - beta_nf * nf : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m = 6.57/mV*((v-V_res)-27.4*mV)/(1-exp((27.4*mV-(v-V_res))/(10.3*mV)))/ms : Hz
alpha_ns = 0.3/mV*((v-V_res)-(-12.5*mV))/(1-exp(((-12.5*mV)-(v-V_res))/(23.6*mV)))/ms : Hz
alpha_nf = 0.0462/mV*((v-V_res)-(-93.2*mV))/(1-exp(((-93.2*mV)-(v-V_res))/(1.1*mV)))/ms : Hz
alpha_h = 0.34/mV*(-114*mV-(v-V_res))/(1-exp(((v-V_res)-(-114*mV))/(11*mV)))/ms : Hz
beta_m = 0.304/mV*((-25.7*mV)-(v-V_res))/(1-exp(((v-V_res)-(-25.7*mV))/(9.6*mV)))/ms : Hz
beta_ns = 0.003631/mV*((-80.1*mV)-(v-V_res))/(1-exp(((v-V_res)-(-80.1*mV))/(21.8*mV)))/ms : Hz
beta_nf = 0.0824/mV*((-76*mV)-(v-V_res))/(1-exp(((v-V_res)-(-76*mV))/(10.5*mV)))/ms : Hz
beta_h = 12.6/(1+exp(((-31.8*mV)-(v-V_res))/(13.4*mV)))/ms : Hz
gamma_Na : siemens
gamma_Ks : siemens
gamma_Kf : siemens
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

##### number of segments per internode
nof_segments_internode = 9

##### build structure
structure = np.array(list(np.tile([2] + np.tile([1],nof_segments_internode).tolist(),1)) + [2])
nof_comps = len(structure)

# =============================================================================
#  Compartment lengths
# ============================================================================= 
##### initialize
compartment_lengths = np.zeros_like(structure)*um
##### length internodes
compartment_lengths[structure == 1] = 230*um / nof_segments_internode
##### length nodes
compartment_lengths[structure == 2] = 1*um

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
##### initialize
compartment_diameters = np.zeros(nof_comps+1)*um
##### dendrite
compartment_diameters[:] = 2.5*um

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacities one layer
c_mem = 2.05e-5*nF/mm**2
##### myelin layer capacity
c_my = 1.45e-10*nF/(2.5*np.pi*mm**2)

##### initialize
c_m = np.zeros_like(structure)*nF/mm**2
##### nodes
c_m[structure == 2] = c_mem
##### internodes
c_m[structure == 1] = c_my

# =============================================================================
# Condactivities internodes
# =============================================================================
##### cell membrane + myelin sheath resistivity internodes
r_my = 1254e6*ohm*2.5*np.pi*mm**2

##### initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
##### calculate values
g_m[structure == 1] = 1/r_my

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
k_noise = 0.0005*uA/np.sqrt(mS)
noise_term = np.sqrt(A_surface*gamma_Na*rho_Na)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 300*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Imennov and Rubinstein 2009"

# =============================================================================
# Compartments to plot
# =============================================================================
##### get indexes of all compartments that are not segmented
indexes_comps = np.where(structure == 2)[0]
##### calculate middle compartments of internodes
middle_comps_internodes = np.ceil(indexes_comps[:-1] + nof_segments_internode/2).astype(int)
##### create array with all compartments to plot
comps_to_plot = np.sort(np.append(indexes_comps, middle_comps_internodes))

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
    neuron.ns = model.ns_init
    neuron.nf = model.nf_init
    neuron.h = model.h_init
    
    ##### Set parameter values (parameters that were initialised in the equations eqs and which are different for different compartment types)
    # conductances active compartments
    neuron.gamma_Na = model.gamma_Na
    neuron.gamma_Ks = model.gamma_Ks
    neuron.gamma_Kf = model.gamma_Kf
    neuron.g_L = model.g_L
    
    # conductances internodes
    neuron.g_myelin = model.g_m
    neuron.gamma_Na[np.where(model.structure == 1)[0]] = 0*psiemens
    neuron.gamma_Ks[np.where(model.structure == 1)[0]] = 0*psiemens
    neuron.gamma_Kf[np.where(model.structure == 1)[0]] = 0*psiemens
    neuron.g_L[np.where(model.structure == 1)[0]] = 0*msiemens/cm**2
    
    ##### save parameters that are part of the equations in eqs to load them in the workspace before a simulation  
    param_string = f'''
    V_res = {model_name}.V_res
    E_Na = {model_name}.E_Na
    E_K = {model_name}.E_K
    E_L = {model_name}.E_L
    rho_Na = {model_name}.rho_Na
    rho_Ks = {model_name}.rho_Ks
    rho_Kf = {model_name}.rho_Kf
    '''
    
    ##### remove spaces to avoid complications
    param_string = param_string.replace(" ", "")
    
    return neuron, param_string
