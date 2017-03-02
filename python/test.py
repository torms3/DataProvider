#!/usr/bin/env python
import h5py

from data_provider import VolumeDataProvider
import time
from vector import Vec3d

if __name__ == "__main__":

    # Data spec path.
    dspec_path = 'test_spec/pinky.spec'

    # Net specification.
    net_spec = dict(input=(32,158,158), label=(32,158,158))

    # Parameters.
    params = dict()
    params['drange']  = [0]
    params['augment'] = [dict(type='flip')]

    # Create VolumeDataProvider.
    dp = VolumeDataProvider(dspec_path, net_spec, params)

    # Loop.
    # for i in range(10000):
    #     start = time.time()
    #     sample = dp.random_sample()
    #     print time.time() - start

    # Dump a single random sample.
    sample = dp.random_sample()
    print 'Save as file...'
    f = h5py.File('sample.h5')
    for name, data in sample.iteritems():
        f.create_dataset('/' + name, data=data)
    f.close()

    # # Dump the whole dataset.
    # print 'Save as file...'
    # dataset = dp.datasets[0]._data  # Illegal access. Don't try this at home.
    # f = h5py.File('dataset.h5')
    # for name, data in dataset.iteritems():
    #     f.create_dataset('/' + name, data=data.get_data())
    # f.close()
