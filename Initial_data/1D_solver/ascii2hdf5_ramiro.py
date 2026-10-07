import numpy as np
import codecs
import h5py
import matplotlib.pyplot as plt

#{Riso, lapse, lapse'(R), scalar, scalar'(R), psi, psi'(R), pressure, internal energy, baryonic density}

with codecs.open('Inidat.dat', encoding='utf-8-sig') as f:
	X = [[float(x) for x in line.split()] for line in f]



coord = np.array(X)[:,0]

counter = 1
for field in ['phir1d', 'drphir1d']:
	if field != '':
		data = np.array(X)[:,counter]
		hdf = h5py.File(field + '.h5', mode='w')
		dset = hdf.require_dataset('var0', data.shape, dtype='f')
		dset[:] = data[:]

		dset = hdf.require_dataset('nvars', (1,), dtype='f')
		dset[:] = 1

		dsetC = hdf.require_dataset('coord0', data.shape, dtype='f')
		dsetC[:] = coord[:]

		hdf.close()
	counter = counter + 1


def visualize_hdf5(hdf5_filename):
    with h5py.File(hdf5_filename, 'r') as f:
        data = f['var0'][:]  # Assuming your data is stored in a dataset named 'var0'
        coord = f['coord0'][:]  # Assuming your coordinates are stored in a dataset named 'coord0'
        
        plt.plot(coord, data)  # Plotting the data
        plt.xlabel('Coordinate')
        plt.ylabel('Data')
        plt.title('Visualization of Data from ' + hdf5_filename)
        plt.show()

if __name__ == "__main__":
    for field in ['phir1d.h5', 'drphir1d.h5']:  # Assuming your fields are saved as phi.h5 and dxphi.h5
        visualize_hdf5(field)


'''
with codecs.open('phiRF4.0.0.000.dat', encoding='utf-8-sig') as f:
	X = [[float(x) for x in line.split()] for line in f]


data = np.array(X)

xcoord = np.unique(data[:, 0])
ycoord = np.unique(data[:, 1])
zcoord = np.unique(data[:, 2])

values = np.zeros((xcoord.shape[0], ycoord.shape[0], zcoord.shape[0]))

for (i,j,k), v in np.ndenumerate(values):
	values[i,j,k] = data[np.where(((data[:, 0] == xcoord[i]) & (data[:, 1] == ycoord[j]) & (data[:, 2] == zcoord[k])))[0]][:,3]

hdf = h5py.File('phir3D.h5', mode='w')
dset = hdf.require_dataset('data', values.shape, dtype='f')
dset[:] = values[:]

dsetC1 = hdf.require_dataset('coord1', xcoord.shape, dtype='f')
dsetC1[:] = xcoord[:]

dsetC2 = hdf.require_dataset('coord2', ycoord.shape, dtype='f')
dsetC2[:] = ycoord[:]

dsetC3 = hdf.require_dataset('coord3', zcoord.shape, dtype='f')
dsetC3[:] = zcoord[:]

hdf.close()


'''
