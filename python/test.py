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

    # VolumeDataProvider
    dp = VolumeDataProvider(dspec_path, net_spec, params, [0])

    samples = []
    for x in range(16):
        print 'Iteration %d...' % (x+1)
        sample = dp.random_sample()
        for name, _ in iter(sorted(sample.iteritems())):
            print name
        # samples.append(sample)
        # # Save as file.
        # print 'Save as file...'
        # f = h5py.File('sample%d.h5' % (x+1))
        # for name, data in sample.iteritems():
        #     f.create_dataset('/' + name, data=data)
        # f.close()