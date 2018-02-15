from __future__ import print_function
import argparse
import h5py
import os
import time

from dataprovider import *
import emio
import transform

if __name__ == "__main__":

    dsc = 'Test.'
    parser = argparse.ArgumentParser(description=dsc)

    parser.add_argument('z', type=int, help='sample z dim.')
    parser.add_argument('y', type=int, help='sample y dim.')
    parser.add_argument('x', type=int, help='sample x dim.')
    # parser.add_argument('img', help='image file (h5 or tif) path.')
    # parser.add_argument('lbl', help='label file (h5 or tif) path.')
    # parser.add_argument('prob', help='prob file (h5 or tif) path.')
    # parser.add_argument('-iter', type=int, default=1)
    parser.add_argument('-s', '--save', action='store_true')

    args = parser.parse_args()

    # Temp.
    img = '~/Data_local/datasets/SNEMI3D/original/train.img.h5'
    lbl = '~/Data_local/datasets/SNEMI3D/original/train.seg.h5'
    args.img = os.path.expanduser(img)
    args.lbl = os.path.expanduser(lbl)

    # Load data.
    img = emio.imread(args.img);   print("Load image...")
    lbl = emio.imread(args.lbl);   print("Load label...")
    # prob = emio.imread(args.prob);  print("Load probability map...")

    # Preprocess.
    img = transform.divideby(img, val=255.0, dtype='float32')

    # Create dataset and add data.
    vdset = VolumeDataset()
    vdset.add_raw_data(key='input', data=img)
    vdset.add_raw_data(key='label', data=lbl)

    # Create sample sequence.
    # seq = InstanceSequence(prob, length=100000)
    # vdset.set_sequence(seq)

    # Set sample spec.
    size = (1+args.z, 1+args.y, 1+args.x)
    spec = dict(input=size, label=size)
    vdset.set_spec(spec)

    # Data augmentation.
    augment = Augmentor()
    # augment.append('misalign', max_trans=17, slip_ratio=0.5)
    # augment.append('warp')
    # augment.append('missing', max_sec=3, mode='mix', consecutive=True, skip_ratio=0.3)
    # augment.append('blur', max_sec=5, mode='mix', skip_ratio=0.3)
    # augment.append('greyscale', mode='mix', skip_ratio=0.3)
    # augment.append('flip')

    # Data transformation.
    transform = Transformer()
    # transform.append(ObjectInstance(source='label', target='object'))
    # transform.append(Boundary(source='label', target='boundary'))
    dst = list()
    dst.append((0,0,1))
    dst.append((0,1,0))
    dst.append((1,0,0))
    dst.append((0,0,3))
    dst.append((0,3,0))
    dst.append((1,3,3))
    dst.append((0,0,5))
    dst.append((0,5,0))
    dst.append((1,5,5))
    dst.append((0,0,10))
    dst.append((0,10,0))
    dst.append((2,0,0))
    dst.append((0,0,15))
    dst.append((0,15,0))
    dst.append((3,0,0))
    dst.append((0,0,20))
    dst.append((0,20,0))
    dst.append((4,0,0))
    transform.append(Affinity1(dst, 'label', 'affinity', crop=(1,1,1)))

    # Sampler.
    while True:
        try:
            print('try sample...')
            spec = vdset.get_spec()
            # spec['mask'] = spec['input']
            params = vdset.get_params()
            spec = augment.prepare(spec, **params)
            sample = vdset.next_sample(spec=spec)
            break
        except:
            pass
    # Object instance mask.
    # z, y, x = sample['label'].shape[-3:]
    # object_id = sample['label'][...,z//2,y//2,x//2]
    # mask = np.zeros((z,y,x), dtype='float32')
    # mask[z//2,y//2,x//2] = 1
    # sample['mask'] = mask
    # Apply data augmentation.
    sample = augment(sample, imgs=['input'])
    # Apply transformation.
    # sample = transform(sample, object_id=object_id)
    sample = transform(sample)

    # Failure test.
    # elapsed = 0.0
    # for i in range(args.iter):
    #     t0 = time.time()
    #     # Sample & augment.
    #     sample = sampler('next', imgs=['input'])
    #     elapsed += time.time() - t0
    #     print("Iteration %7d, elapsed: %.3f" % (i+1, elapsed/(i+1)))

    # Dump a single random sample.
    if args.save:
        print("\nSave as file...")
        fname = 'sample.h5'
        if os.path.exists(fname):
            os.remove(fname)
        f = h5py.File(fname)
        for key, data in sample.items():
            f.create_dataset('/' + key, data=data)
        f.close()
