#!/usr/bin/env python
if __name__ == "__main__":

    # Data spec path
    dspec_path = 'zfish.spec'

    # Net specification
    net_spec = {}
    net_spec['input'] = (18,208,208)
    net_spec['label'] = (10,100,100)

    # Parameters
    params = {}
    params['border_mode'] = 'mirror'
    params['augment'] = [{'type':'flip'}]

    # VolumeDataProvider
    from data_provider import *
    dp = VolumeDataProvider(dspec_path, net_spec, params, [1,3])
    sample = dp.random_sample()

    import h5py
    f = h5py.File('sample.h5')
    for name, data in sample.iteritems():
        f.create_dataset('/' + name, data=data)
    f.close()