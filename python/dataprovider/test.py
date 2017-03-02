#!/usr/bin/env python
import argparse
import h5py
import os
import time

from dataprovider import VolumeDataset, Augmentor, Affinity
import emio
import transform

if __name__ == "__main__":

    dsc = 'Test.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('z', type=int, help='sample z dim.')
    parser.add_argument('y', type=int, help='sample y dim.')
    parser.add_argument('x', type=int, help='sample x dim.')
    parser.add_argument('img', help='image file (h5 or tif) path.')
    parser.add_argument('lbl', help='label file (h5 or tif) path.')
    parser.add_argument('-iter', type=int, default=1)
    parser.add_argument('-s', '--save', action='store_true')

    args = parser.parse_args()

    # Load data.
    img = emio.imread(args.img)
    lbl = emio.imread(args.lbl)

    # Preprocess.
    img = transform.divideby(img, val=255.0, dtype='float32')

    # Create dataset and add data.
    vdset = VolumeDataset()
    vdset.add_raw_data(key='input', data=img)
    vdset.add_raw_data(key='label', data=lbl)

    # Set sample spec.
    size = (1+args.z, 1+args.y, 1+args.x)
    spec = dict(input=size, label=size)
    vdset.set_spec(spec)

    # Data augmentation.
    augment = Augmentor()
    augment.add_augment('misalign', max_trans=30, slip_ratio=0.5)
    augment.add_augment('warp')
    augment.add_augment('missing', max_sec=3, mode='mix', consecutive=True, skip_ratio=0.3)
    augment.add_augment('blur', max_sec=5, mode='mix', skip_ratio=0.3)
    augment.add_augment('greyscale', mode='mix', skip_ratio=0.3)
    augment.add_augment('flip')
    # augment.add_augment('box', min_dim=50, max_dim=100, aspect_ratio=10, density=0.2)

    # Data transformation.
    dst = list()
    for i in xrange(1):
        dst.append((i+1,3**i,3**i))
    transform = Affinity(dst, 'label', 'label', crop=(1,1,1), rebalance=True)

    # Failure test.
    elapsed = 0.0
    for i in range(args.iter):
        t0 = time.time()
        # Sample & augment.
        spec   = augment.prepare(vdset.get_spec(), imgs=['input'])
        sample = vdset.random_sample(spec=spec)
        sample = augment(sample, imgs=['input'])
        sample = transform(sample)
        elapsed += time.time() - t0
        print "Iteration %7d, elapsed: %.3f" % (i+1, elapsed/(i+1))

    # Dump a single random sample.
    if args.save:
        print "\nSave as file..."
        fname = 'sample.h5'
        if os.path.exists(fname):
            os.remove(fname)
        f = h5py.File(fname)
        for key, data in sample.iteritems():
            f.create_dataset('/' + key, data=data)
        f.close()
