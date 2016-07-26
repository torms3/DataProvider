#!/usr/bin/env python
import h5py
from data_provider import *

if __name__ == "__main__":

    # Data spec path
    dspec_path = 'zfish.spec'

    # Net specification
    net_spec = {}
    net_spec['input'] = (18,208,208)
    net_spec['label'] = (10,100,100)

    # Parameters
    params = {}
    params['border']  = 'mirror'
    params['augment'] = [{'type':'flip'}]
    params['drange'] = list(range(7))

    # VolumeDataProvider
    dp = VolumeDataProvider(dspec_path, net_spec, params)

    sample = dp.random_sample()

    # Save as file.
    # print 'Save as file...'
    # f = h5py.File('sample.h5')
    # for name, data in sample.iteritems():
    #     f.create_dataset('/' + name, data=data)
    # f.close()