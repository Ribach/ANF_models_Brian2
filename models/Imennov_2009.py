##### import needed packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np

##### import functions
import functions.calculations as calc

# =============================================================================
# Constriction factor (constricts surface aria of nodes)
# =============================================================================
cons_fac = 1/2

# =============================================================================
# Nernst potentials
# =============================================================================
##### Resting potential of cell
V_res = -84*mV
##### Nernst potential sodium
E_Na = 50*mV - V_res
##### Nernst potential potassium
E_K = -84*mV - V_res

# =============================================================================
# Conductivities
# =============================================================================
##### conductivities active compartments
gamma_Na = 20*psiemens
gamma_Ks = 10*psiemens
gamma_Kf = 10*psiemens

##### cell membrane conductivity nodes
g_L = 1/(8310*ohm*mm**2) * cons_fac

# =============================================================================
# Numbers of channels per aria
# =============================================================================
rho_Na = 618/um**2 * cons_fac
rho_Kf = 20.3/um**2 * cons_fac
rho_Ks = 41.2/um**2 * cons_fac

# =============================================================================
# Resistivities
# =============================================================================
##### axoplasmatic resistivity
rho_in = 733*ohm*mm
##### external resistivity
rho_out = 25000*ohm*mm

# =============================================================================
# Differential equations
# =============================================================================
eqs = '''
I_Na = gamma_Na*rho_Na*m**3*h* (E_Na-(v-V_res)) : amp/meter**2
I_Ks = gamma_Ks*rho_Ks*ns**4*(E_K-(v-V_res)) : amp/meter**2
I_Kf = gamma_Kf*rho_Kf*nf**4*(E_K-(v-V_res)) : amp/meter**2
I_L = g_L*(E_L-(v-V_res)) : amp/meter**2
Im = I_Na + I_Ks + I_Kf + I_L + g_myelin*(-(v-V_res)): amp/meter**2
I_stim = stimulus(t,i) : amp (point current)
dm/dt = alpha_m * (1-m) - beta_m * m : 1
dns/dt = alpha_ns * (1-ns) - beta_ns * ns : 1
dnf/dt = alpha_nf * (1-nf) - beta_nf * nf : 1
dh/dt = alpha_h * (1-h) - beta_h * h : 1
alpha_m = 6.57/mV*(v-(-27.4*mV))/(1-exp(((-27.4*mV)-v)/(10.3*mV)))/ms : Hz
alpha_ns = 0.3/mV*(v-(-12.5*mV))/(1-exp(((-12.5*mV)-v)/(23.6*mV)))/ms : Hz
alpha_nf = 0.0462/mV*(v-(-93.2*mV))/(1-exp(((-93.2*mV)-v)/(1.1*mV)))/ms : Hz
alpha_h = 0.34/mV*(-(v + 114*mV))/(1-exp((v+114*mV)/(11*mV)))/ms : Hz
beta_m = 0.304/mV*((-25.7*mV)-v)/(1-exp((v-(-25.7*mV))/(9.16*mV)))/ms : Hz
beta_ns = 0.003631/mV*((-80.1*mV)-v)/(1-exp((v-(-80.1*mV))/(21.8*mV)))/ms : Hz
beta_nf = 0.0824/mV*((-76*mV)-v)/(1-exp((v-(-76*mV))/(10.5*mV)))/ms : Hz
beta_h = 12.6/(1+exp(((-31.8*mV)-v)/(13.4*mV)))/ms : Hz
gamma_Na : siemens
gamma_Ks : siemens
gamma_Kf : siemens
g_L : siemens/meter**2
g_myelin : siemens/meter**2
V_res : volt
E_Na : volt
E_K : volt
E_L : volt
rho_Na : 1/meter**2
rho_Ks : 1/meter**2
rho_Kf : 1/meter**2
'''

# =============================================================================
#  Morphologic data
# =============================================================================
##### structure
nof_segments_internodes = 9
nof_internodes = 38
##### lengths
length_internodes = 230*um
length_nodes = 1*um
##### diameters
fiber_outer_diameter = 2.5*um

# =============================================================================
# Capacities
# =============================================================================
##### cell membrane capacity of nodes
c_mem = 2.05e-5*nF/um**2 * cons_fac
##### myelin sheath capacity per mm
c_my_per_mm = 0.145e-12*farad/mm

# =============================================================================
# resistivity internodes
# =============================================================================
##### resistivity internodes per mm
r_my_per_mm = 1254*1e6*ohm*mm

# =============================================================================
# Noise factor
# =============================================================================
k_noise = 0.01*uA/np.sqrt(mS)

# =============================================================================
# Electrode
# =============================================================================
electrode_distance = 500*um

# =============================================================================
# Display name for plots
# =============================================================================
display_name = "Imennov and Rubinstein 2009"
display_name_plots = "Imennov and Rubinstein (2009)"
display_name_short = "Imennov 09"

# =============================================================================
# Define inter-pulse intervalls for refractory curve calculation
# =============================================================================
inter_pulse_intervals = np.hstack([np.linspace(0.6, 1.3, num=40, endpoint = False),
                                   np.linspace(1.3, 5, num=20)])*1e-3

# =============================================================================
# Calculations
# =============================================================================
##### rates for resting potential
alpha_m_0 = 6.57*((V_res/mV)-(-27.4))/(1-np.exp(((-27.4)-(V_res/mV))/(10.3)))
alpha_ns_0 = 0.3*((V_res/mV)-(-12.5))/(1-np.exp(((-12.5)-(V_res/mV))/(23.6)))
alpha_nf_0 = 0.0462*((V_res/mV)-(-93.2))/(1-np.exp(((-93.2)-(V_res/mV))/(1.1)))
alpha_h_0 = 0.34*(-((V_res/mV) + 114))/(1-np.exp(((V_res/mV)+114)/(11)))
beta_m_0 = 0.304*((-25.7)-(V_res/mV))/(1-np.exp(((V_res/mV)-(-25.7))/(9.16)))
beta_ns_0 = 0.003631*((-80.1)-(V_res/mV))/(1-np.exp(((V_res/mV)-(-80.1))/(21.8)))
beta_nf_0 = 0.0824*((-76)-(V_res/mV))/(1-np.exp(((V_res/mV)-(-76))/(10.5)))
beta_h_0 = 12.6/(1+np.exp(((-31.8)-(V_res/mV))/(13.4)))

##### initial values for gating variables
m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
ns_init = alpha_ns_0 / (alpha_ns_0 + beta_ns_0)
nf_init = alpha_nf_0 / (alpha_nf_0 + beta_nf_0)
h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  

##### calculate Nerst potential for leakage current
E_L = -(1/g_L)* (gamma_Na*rho_Na*m_init**3*h_init* E_Na + gamma_Ks*rho_Ks*nf_init**4*E_K + gamma_Kf*rho_Kf*ns_init**4*E_K)

##### structure of ANF
# terminal = 0
# internode = 1
# node = 2
# presomatic region = 3
# Soma = 4
# postsomatic region = 5)
structure = np.array(list(np.tile([2] + np.tile([1],nof_segments_internodes).tolist(),nof_internodes)) + [2])
nof_comps = len(structure)

#####  Compartment lengths
# initialize
compartment_lengths = np.zeros_like(structure)*um
# length internodes
compartment_lengths[structure == 1] = length_internodes / nof_segments_internodes
# length nodes
compartment_lengths[structure == 2] = length_nodes
# total length neuron
length_neuron = sum(compartment_lengths)

##### Compartment diameters
# initialize
compartment_diameters = np.zeros(nof_comps+1)*um
# same diameter for whole fiber
fiber_inner_diameter = fiber_outer_diameter*0.6
compartment_diameters[:] = fiber_inner_diameter

# makes nodal diameters smaller
#nodes1 = np.append(structure == 2, True)
#nodes2 = np.append(True,structure == 2)
#compartment_diameters[nodes1 | nodes2] = fiber_inner_diameter * 0.5

#####  Compartment middle point distances (needed for plots)
distance_comps_middle = np.zeros_like(compartment_lengths)
distance_comps_middle[0] = 0.5*compartment_lengths[0]
for ii in range(0,nof_comps-1):
    distance_comps_middle[ii+1] = 0.5* compartment_lengths[ii] + 0.5* compartment_lengths[ii+1]

##### internodal surface aria per 1 mm
surface_aria_1mm = np.pi*fiber_inner_diameter*1*mm

##### Capacities
# initialize
c_m = np.zeros_like(structure)*nF/mm**2
# nodes
c_m[structure == 2] = c_mem
# internodes
c_my = c_my_per_mm/(surface_aria_1mm/mm)
c_m[structure == 1] = c_my

##### Condactivities internodes
# initialize
g_m = np.zeros_like(structure)*msiemens/cm**2
# calculate values
r_my = r_my_per_mm*(surface_aria_1mm/mm)
g_m[structure == 1] = 1/r_my

##### Axoplasmatic resistances
compartment_center_diameters = np.zeros(nof_comps)*um
compartment_center_diameters = (compartment_diameters[0:-1] + compartment_diameters[1:]) / 2                         
R_a = (compartment_lengths*rho_in) / ((compartment_center_diameters*0.5)**2*np.pi)

##### Surface arias
# list of constriction factors
cons_fac_list = np.ones(nof_comps)
cons_fac_list[structure == 2] = cons_fac

# lateral lengths
m = [np.sqrt(abs(compartment_diameters[i+1] - compartment_diameters[i])**2 + compartment_lengths[i]**2)
           for i in range(nof_comps)]
# total surfaces
A_surface = [(compartment_diameters[i+1] + compartment_diameters[i])*np.pi*m[i]*0.5*cons_fac_list[i]
           for i in range(nof_comps)]

##### Noise term
gamma_Na_vector = np.zeros(nof_comps)*psiemens
gamma_Na_vector[structure == 2] = gamma_Na
noise_term = np.sqrt(A_surface*gamma_Na_vector*rho_Na)

##### Compartments to plot
# get indexes of all compartments that are not segmented
indexes_comps = np.where(structure == 2)[0]
# calculate middle compartments of internodes
middle_comps_internodes = np.ceil(indexes_comps[:-1] + nof_segments_internodes/2).astype(int)
# create array with all compartments to plot
comps_to_plot = np.sort(np.append(indexes_comps, middle_comps_internodes))

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
        ##### rates for resting potential
        alpha_m_0 = 6.57*((model.V_res/mV)-(-27.4))/(1-np.exp(((-27.4)-(model.V_res/mV))/(10.3)))
        alpha_ns_0 = 0.3*((model.V_res/mV)-(-12.5))/(1-np.exp(((-12.5)-(model.V_res/mV))/(23.6)))
        alpha_nf_0 = 0.0462*((model.V_res/mV)-(-93.2))/(1-np.exp(((-93.2)-(model.V_res/mV))/(1.1)))
        alpha_h_0 = 0.34*(-((model.V_res/mV) + 114))/(1-np.exp(((model.V_res/mV)+114)/(11)))
        beta_m_0 = 0.304*((-25.7)-(model.V_res/mV))/(1-np.exp(((model.V_res/mV)-(-25.7))/(9.16)))
        beta_ns_0 = 0.003631*((-80.1)-(model.V_res/mV))/(1-np.exp(((model.V_res/mV)-(-80.1))/(21.8)))
        beta_nf_0 = 0.0824*((-76)-(model.V_res/mV))/(1-np.exp(((model.V_res/mV)-(-76))/(10.5)))
        beta_h_0 = 12.6/(1+np.exp(((-31.8)-(model.V_res/mV))/(13.4)))
        
        ##### initial values for gating variables
        model.m_init = alpha_m_0 / (alpha_m_0 + beta_m_0)
        model.ns_init = alpha_ns_0 / (alpha_ns_0 + beta_ns_0)
        model.nf_init = alpha_nf_0 / (alpha_nf_0 + beta_nf_0)
        model.h_init = alpha_h_0 / (alpha_h_0 + beta_h_0)                                  
        
        ##### calculate Nerst potential for leakage current
        model.E_L = -(1/model.g_L)* (model.gamma_Na*model.rho_Na*model.m_init**3*model.h_init* model.E_Na +
                     model.gamma_Ks*model.rho_Ks*model.nf_init**4*model.E_K + model.gamma_Kf*model.rho_Kf*model.ns_init**4*model.E_K)

        ##### structure of ANF
        # terminal = 0
        # internode = 1
        # node = 2
        # presomatic region = 3
        # Soma = 4
        # postsomatic region = 5)
        model.structure = np.array(list(np.tile([2] + np.tile([1],model.nof_segments_internodes).tolist(),model.nof_internodes)) + [2])
        model.nof_comps = len(model.structure)
        
        #####  Compartment lengths
        # initialize
        model.compartment_lengths = np.zeros_like(model.structure)*um
        # length internodes
        model.compartment_lengths[model.structure == 1] = model.length_internodes / model.nof_segments_internodes
        # length nodes
        model.compartment_lengths[model.structure == 2] = model.length_nodes
        # total length neuron
        model.length_neuron = sum(model.compartment_lengths)
        
        ##### Compartment diameters
        # initialize
        model.compartment_diameters = np.zeros(nof_comps+1)*um
        # same diameter for whole fiber
        model.fiber_inner_diameter = model.fiber_outer_diameter*0.6
        model.compartment_diameters[:] = model.fiber_inner_diameter
        
        #####  Compartment middle point distances (needed for plots)
        model.distance_comps_middle = np.zeros_like(model.compartment_lengths)
        model.distance_comps_middle[0] = 0.5*model.compartment_lengths[0]
        for ii in range(0,model.nof_comps-1):
            model.distance_comps_middle[ii+1] = 0.5* model.compartment_lengths[ii] + 0.5* model.compartment_lengths[ii+1]
        
        ##### internodal surface aria per 1 mm
        model.surface_aria_1mm = np.pi*model.fiber_inner_diameter*1*mm
        
        ##### Capacities
        # initialize
        model.c_m = np.zeros_like(model.structure)*nF/mm**2
        # nodes
        model.c_m[model.structure == 2] = model.c_mem
        # internodes
        model.c_my = model.c_my_per_mm/(model.surface_aria_1mm/mm)
        model.c_m[model.structure == 1] = model.c_my
        
        ##### Condactivities internodes
        # initialize
        model.g_m = np.zeros_like(model.structure)*msiemens/cm**2
        # calculate values
        model.r_my = model.r_my_per_mm*(model.surface_aria_1mm/mm)
        model.g_m[model.structure == 1] = 1/model.r_my
        
        ##### Axoplasmatic resistances
        model.compartment_center_diameters = np.zeros(model.nof_comps)*um
        model.compartment_center_diameters = (model.compartment_diameters[0:-1] + model.compartment_diameters[1:]) / 2                         
        model.R_a = (model.compartment_lengths*model.rho_in) / ((model.compartment_center_diameters*0.5)**2*np.pi)
        
        ##### Surface arias
        # lateral surfaces
        m = [np.sqrt(abs(model.compartment_diameters[i+1] - model.compartment_diameters[i])**2 + model.compartment_lengths[i]**2)
                   for i in range(0,nof_comps)]
        # total surfaces
        model.A_surface = [(model.compartment_diameters[i+1] + model.compartment_diameters[i])*np.pi*m[i]*0.5
                   for i in range(0,model.nof_comps)]
        
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
    neuron.ns = model.ns_init
    neuron.nf = model.nf_init
    neuron.h = model.h_init
    
    ##### Set parameter values of differential equations
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
    
    # other parameters
    neuron.V_res = model.V_res
    neuron.E_Na = model.E_Na
    neuron.E_K = model.E_K
    neuron.E_L = model.E_L
    neuron.rho_Na = model.rho_Na
    neuron.rho_Ks = model.rho_Ks
    neuron.rho_Kf = model.rho_Kf    

    return neuron, model
