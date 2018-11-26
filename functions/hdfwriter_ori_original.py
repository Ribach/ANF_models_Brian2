import numpy as np
import tables as tbls
import scipy.io as io
# import pickle as pkl
import os
import os.path as path

#root = os.getcwd()
#os.chdir(path.join(root, 'data'))
files = []

coor_data = io.loadmat('ci_refine_list_mdl.mat')
coordinates = coor_data['fibre_loc']*1e-3

for el in range(1, 13):

    filename = ('ori_el' + '{:02d}'.format(el) + '.mat')
    if path.isfile(filename):
        files.append(filename)

h5file = tbls.open_file("original_mdl.h5", mode = "w", title = "neuron_data")

pots_array = np.array([len(files), len(coordinates)])
# potentials_group = h5file.create_group(where='/', name='potentials', title='neuron coordinates')
pots = np.zeros([coordinates.shape[0], coordinates.shape[1], len(files)])
for i, f_name in enumerate(files):
    data = io.loadmat(f_name)
    v_vals = data['V_vals']
    name = 'electrode%i' % i
    pots[:, :, i] = v_vals

for i,n_coords in enumerate(coordinates):
    name = 'neuron%i' % i
    n_nan = np.sum(np.isnan(n_coords)[:, 0])
    neuron_group = h5file.create_group(where='/', name=name, title='neuron %i' % i)
    if n_nan:
        h5file.create_array('/' + name, 'coordinates', n_coords[:-n_nan])
        h5file.create_array('/' + name, 'potentials', pots[i][:-n_nan])
    else:
        h5file.create_array('/' + name, 'coordinates', n_coords[:])
        h5file.create_array('/' + name, 'potentials', pots[i][:])
#    h5file.create_array('/' + name, 'soma', soma_data[i])

h5file.close()
