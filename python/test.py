#!/usr/bin/env python
import h5py
import os

from data_provider import VolumeDataProvider
import time
from vector import Vec3d

if __name__ == "__main__":

    # Data spec path.
    dspec_path = 'test_spec/pinky.spec'

    # Net specification.
    net_spec = dict(input=(18,160,160), label=(18,160,160))

    # Parameters.
    params = dict()
    params['drange'] = [0]

    # Data augmentation.
    aug_list = list()
    aug_list.append(dict(type='misalign'))
    aug_list.append(dict(type='missing',max_sec=3,mode='mix',consecutive=True,skip_ratio=0.5))
    aug_list.append(dict(type='missing',max_sec=5,mode='mix',random_color=True,skip_ratio=0.5))
    aug_list.append(dict(type='blur',mode='mix',max_sec=5))
    aug_list.append(dict(type='warp'))
    aug_list.append(dict(type='grey',mode='mix'))
    aug_list.append(dict(type='box',min_dim=20,max_dim=60,aspect_ratio=6,density=0.3,skip_ratio=0.3,random_color=True))
    aug_list.append(dict(type='flip'))
    params['augment'] = aug_list

    # Create VolumeDataProvider.
    dp = VolumeDataProvider(dspec_path, net_spec, params)

    # Loop.
    for i in range(10):
        start = time.time()
        sample = dp.random_sample()
        if True:
            print 'Save as file...'
            f = h5py.File('sample%.2d.h5' % (i+1))
            for name, data in sample.iteritems():
                f.create_dataset('/' + name, data=data)
            f.close()
        print time.time() - start


    # # Dump the whole dataset.
    # print 'Save as file...'
    # dataset = dp.datasets[0]._data  # Illegal access. Don't try this at home.
    # f = h5py.File('dataset.h5')
    # for name, data in dataset.iteritems():
    #     f.create_dataset('/' + name, data=data.get_data())
    # f.close()
