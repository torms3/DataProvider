#!/usr/bin/env python
import h5py

from data_provider import VolumeDataProvider
from vector import Vec3d

if __name__ == "__main__":

    # Data spec path.
    dspec_path = 'test_spec/piriform.spec'

    # Net specification.
    fov   = Vec3d(9,109,109)
    outsz = Vec3d(10,100,100)
    insz  = outsz + fov - Vec3d(1,1,1)
    net_spec = dict(input=tuple(insz), label=tuple(outsz))

    # Parameters.
    params = dict()
    params['border']  = dict(type='mirror_border', fov=fov)
    params['augment'] = [dict(type='grey',mode='mix',skip_ratio=0.2)]
    params['drange']  = [1]

    # Create VolumeDataProvider.
    dp = VolumeDataProvider(dspec_path, net_spec, params)

    # for _ in range(10000):
    #     sample = dp.random_sample()

    # Dump a single randome sample.
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
