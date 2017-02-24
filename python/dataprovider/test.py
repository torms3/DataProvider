#!/usr/bin/env python
import argparse
import h5py
import os
import time

from dataprovider import VolumeDataset, Augmentor
import emio

if __name__ == "__main__":

    dsc = 'Test.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('z', type=int, help='sample z dim.')
    parser.add_argument('y', type=int, help='sample y dim.')
    parser.add_argument('x', type=int, help='sample x dim.')
    parser.add_argument('img', help='image file (h5 or tif) path.')
    parser.add_argument('lbl', help='label file (h5 or tif) path.')

    args = parser.parse_args()

    # Load data.
    img = emio.imread(args.img)
    lbl = emio.imread(args.lbl)

    # Create dataset and add data.
    vdset = VolumeDataset()
    vdset.add_raw_data(key='input', data=img)
    vdset.add_raw_data(key='label', data=lbl)

    # Set sample spec.
    size = (args.z, args.y, args.x)
    spec = dict(input=size, label=size)
    vdset.set_spec(spec)

    # Data augmentation.
    augment = Augmentor()
    augment.add_augment('misalign', max_trans=30, slip_ratio=1.0)
    augment.add_augment('missing', max_sec=5, mode='mix', consecutive=True, skip_ratio=0.0)
    augment.add_augment('flip')

    # Random sample.
    spec = augment.prepare(vdset.get_spec())
    sample = vdset.random_sample(spec=spec)
    sample = augment(sample, imgs=['input'])

    # Dump a single random sample.
    print "\nSave as file..."
    fname = 'sample.h5'
    if os.path.exists(fname):
        os.remove(fname)
    f = h5py.File(fname)
    for key, data in sample.iteritems():
        f.create_dataset('/' + key, data=data)
    f.close()
