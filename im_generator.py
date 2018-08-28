# Author: Zhongnan Fang, zhongnanf@gmail.com, 2017 July
# Modified: Akshay Chaudhari, akshaysc@stanford.edu 2017 August

from __future__ import print_function, division
from deprecated import deprecated

import numpy as np
from os import listdir
from os.path import splitext
from random import shuffle
from re import split
import h5py

from deeplab_2d.deeplab_model import preprocess_input

def preprocess_input_scale(im):
    p_min = np.amin(im)
    im -= p_min # minimum is 0
    p_max = np.amax(im)
    im = im / p_max * 255.0
    assert(np.amin(im) == 0)
    assert(np.amax(im) <= 255)

    return im


# find unique data regardless of the file prefix
def calc_generator_info(data_path, batch_size, learn_files=[], pids=None):
    if pids is not None:
        learn_files=[]

    files = listdir(data_path)
    unique_filename = {}

    for file in files:
        file, _ = splitext(file)
        if add_file(file, unique_filename, pids):
            unique_filename[file] = file

    files = list(unique_filename.keys())

    # Set total number of files based on argument for limiting trainign size
    if learn_files:
        nfiles = learn_files
    else:
        nfiles = len(files)
    
    batches_per_epoch = nfiles // batch_size
    
    return (files, batches_per_epoch)


def add_file(file, unique_filename, pids):
    if pids is None:
        return file not in unique_filename

    if (not file in unique_filename):
        for pid in pids:
            pid_str = str(pid)
            if pid_str in file:
                return True

    return False


def dess_generator(data_path, batch_size, img_size, file_types, tag,
                   tissue=0, learn_files=[], shuffle_epoch=True):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size, learn_files)
    files = sort_files(files, tag)

    x = np.zeros((batch_size,) + img_size + (1,))
    y = np.zeros((batch_size,) + img_size)

    while True:

        # If limiting size of training data, sort through file names and fix batches_per_epoch
        if learn_files:
            arr = np.arange(0, learn_files)
            files = [files[i] for i in arr]

        if shuffle_epoch:
            shuffle(files)

        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):
                for type_cnt in range(len(file_types)):
                    file_ind = batch_cnt * batch_size + file_cnt

                    #                     Python 3+ does not support indexing dicts
                    path = '%s/%s.%s' % (data_path, files[file_ind], file_types[type_cnt])
                    # path = '%s/%s.%s'%(data_path, next(iter(files)), file_types[type_cnt])
                    with h5py.File(path, 'r') as f:
                        im = f['data'][:]
                        im = np.expand_dims(im, axis=2)
                        x[file_cnt, ..., 0] = im

                    # Python 3+ does not support indexing dicts
                    seg_path = '%s/%s.seg' % (data_path, files[file_ind])
                    # seg_path = '%s/%s.seg'%(data_path, next(iter(files)))
                    with h5py.File(seg_path, 'r') as f:
                        seg = f['data'][:].astype('float32')
                        seg = np.expand_dims(seg, axis=2)
                        y[file_cnt, ..., 0] = seg

            yield (x, y)


def dess_generator_test(data_path, batch_size, img_size, file_types, tissue=0, shuffle_epoch=True):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size)

    x = np.zeros((batch_size,) + img_size + (len(file_types),))
    y = np.zeros((batch_size,) + img_size + (1,))

    while True:

        if shuffle_epoch:
            shuffle(files)
        else:
            files = sort_files(files)

        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):
                for type_cnt in range(len(file_types)):

                    file_ind = batch_cnt * batch_size + file_cnt

                    #                     Python 3+ does not support indexing dicts
                    path = '%s/%s.%s' % (data_path, files[file_ind], file_types[type_cnt])
                    # path = '%s/%s.%s'%(data_path, next(iter(files)), file_types[type_cnt])
                    with h5py.File(path, 'r') as f:
                        x[file_cnt, ..., type_cnt] = f['data'][:]

                    # Python 3+ does not support indexing dicts
                    seg_path = '%s/%s.seg' % (data_path, files[file_ind])
                    fname = files[file_ind]
                    # seg_path = '%s/%s.seg'%(data_path, next(iter(files)))
                    with h5py.File(seg_path, 'r') as f:
                        seg = f['data'][:].astype('float32')
                        if (np.ndim(seg) == 4):
                            y[file_cnt, ..., 0] = seg[..., tissue]
                        else:
                            y[file_cnt, ..., 0] = seg

            yield (x, y, fname)


def img_generator_dess(data_path, batch_size, img_size, shuffle_epoch=True):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size)

    x = np.zeros((batch_size,) + img_size + (2,))
    y = np.zeros((batch_size,) + img_size + (1,))

    while True:

        if shuffle_epoch:
            shuffle(files)
        else:
            files = sort_files(files)

        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):
                file_ind = batch_cnt * batch_size + file_cnt
                im_path = '%s/%s.im' % (data_path, files[file_ind])
                with h5py.File(im_path, 'r') as f:
                    im = f['data'][:]

                t2m_path = '%s/%s.t2m' % (data_path, files[file_ind])
                with h5py.File(t2m_path, 'r') as f:
                    t2m = f['data'][:]

                seg_path = '%s/%s.seg' % (data_path, files[file_ind])
                with h5py.File(seg_path, 'r') as f:
                    seg = f['data'][:].astype('float32')

                x[file_cnt, ..., 0] = im
                x[file_cnt, ..., 1] = t2m
                y[file_cnt, ..., 0] = seg
            yield (x, y)

def add_background_layer(seg):
    sum_seg = np.sum(seg, axis=-1)
    background = sum_seg == 0
    background = np.multiply(background, sum_seg+1)

    temp_seg = seg.squeeze(axis=2)
    a = np.concatenate([background, temp_seg], axis=-1)
    return a

def img_generator(data_path, batch_size, img_size, tag, tissue_inds, shuffle_epoch=True, pids=None):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size, pids=pids)

    # img_size must be 3D
    assert(len(img_size) == 3)
    total_classes = len(tissue_inds)
    mask_size = (img_size[0], img_size[1], total_classes)

    x = np.zeros((batch_size,) + img_size)
    y = np.zeros((batch_size,) + mask_size)
    count = 0
    while True:

        if shuffle_epoch:
            shuffle(files)
        else:
            files = sort_files(files, tag)

        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):
                file_ind = batch_cnt * batch_size + file_cnt
                im_path = '%s/%s.im' % (data_path, files[file_ind])
                with h5py.File(im_path, 'r') as f:
                    im = f['data'][:]

                seg_path = '%s/%s.seg' % (data_path, files[file_ind])
                with h5py.File(seg_path, 'r') as f:
                    seg = f['data'][:].astype(np.float32)

                #x[file_cnt, ..., 0] = preprocess_input_scale(im)
                x[file_cnt, ..., 0] = im
                y[file_cnt, ...] = seg[..., np.newaxis]
                
            #yield (preprocess_input(x), y)
            yield (x, y)

def img_generator_test(data_path, batch_size, img_size, tag, tissue_inds, shuffle_epoch=False):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size)

    # img_size must be 3D
    assert(len(img_size) == 3)
    total_classes = len(tissue_inds)
    mask_size = (img_size[0], img_size[1], total_classes)

    x = np.zeros((batch_size,) + img_size)
    y = np.zeros((batch_size,) + mask_size)

    while True:

        if shuffle_epoch:
            shuffle(files)
        else:
            files = sort_files(files, tag)

        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):
                file_ind = batch_cnt * batch_size + file_cnt
                im_path = '%s/%s.im' % (data_path, files[file_ind])
                with h5py.File(im_path, 'r') as f:
                    im = f['data'][:]

                seg_path = '%s/%s.seg' % (data_path, files[file_ind])
                with h5py.File(seg_path, 'r') as f:
                    seg = f['data'][:].astype('float32')

                #x[file_cnt, ..., 0] = preprocess_input_scale(im)
                x[file_cnt, ..., 0] = im
                y[file_cnt, ...] = seg[..., np.newaxis]
                
                #inspect_vals(seg)

                fname = files[file_ind]

            #yield (preprocess_input(x), y, fname)
            yield (x, y, fname)

def inspect_vals(x):
    print('0: %0.2f, 1: %0.2f' %(np.sum(x==0), np.sum(x==1)))

def img_generator_oai(data_path, batch_size, img_size, tissue, tag=None, shuffle_epoch=True, pids=None):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size, pids=pids)
    
    # img_size must be 3D
    assert(len(img_size) == 3)
    total_classes = len(tissue)
    mask_size = (img_size[0], img_size[1], total_classes)

    x = np.zeros((batch_size,) + img_size)
    y = np.zeros((batch_size,) + mask_size)

    while True:

        if shuffle_epoch:
            shuffle(files)
        else:
            files = sort_files(files, tag)

        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):
                file_ind = batch_cnt * batch_size + file_cnt
                im_path = '%s/%s.im' % (data_path, files[file_ind])
                with h5py.File(im_path, 'r') as f:
                    im = f['data'][:]

                seg_path = '%s/%s.seg' % (data_path, files[file_ind])
                with h5py.File(seg_path, 'r') as f:
                    seg = f['data'][:].astype('float32')
                
                if (len(im.shape) == 2):
                    im = im[..., np.newaxis]
                
                x[file_cnt, ...] = im
                y[file_cnt, ...] = seg[..., 0, tissue]


            yield (x, y)

def get_file_pid(fname):
    f_pid = fname.split('-')
    return f_pid[0]

def img_generator_oai_test(data_path, batch_size, img_size, tissue, tag):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size)
    files = sort_files(files, tag)

    # img_size must be 3D
    assert (len(img_size) == 3)
    total_classes = len(tissue)
    mask_size = (img_size[0], img_size[1], total_classes)

    x = np.zeros((batch_size,) + img_size)
    y = np.zeros((batch_size,) + mask_size)

    pids = []
    for fname in files:
        pid = get_file_pid(fname)
        pids.append(pid)

    pids_unique = list(set(pids))
    pids_unique.sort()

    pids_dict = dict()
    for pid in pids_unique:
        indices = [i for i, x in enumerate(pids) if x == pid]
        pids_dict[pid] = indices

    for pid in list(pids_unique):
        inds = pids_dict[pid]
        num_slices = len(inds)
        x = np.zeros((num_slices,) + img_size)
        y = np.zeros((num_slices,) + mask_size)
        for file_cnt in range(num_slices):
            ind = inds[file_cnt]
            fname = files[ind]

            # Make sure that this pid is actually in the filename
            assert(pid in fname)

            im_path = '%s/%s.im' % (data_path, fname)
            with h5py.File(im_path, 'r') as f:
                im = f['data'][:]

            seg_path = '%s/%s.seg' % (data_path, fname)
            with h5py.File(seg_path, 'r') as f:
                seg = f['data'][:].astype('float32')

            if (len(im.shape) == 2):
                im = im[..., np.newaxis]

            x[file_cnt, ...] = im
            y[file_cnt, ...] = seg[..., 0, tissue]

        yield (x, y, pid, num_slices)

def sort_files(files, tag):
    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    file_id = [None] * len(files)
    for cnt1 in range(len(files)):
        tmp = split('_', files[cnt1])

        if (tag == 'dess'):
            tmp = int(tmp[1])
        elif (tag == 'oai'):
            tmp = int(tmp[-1][0])
        elif (tag == 'oai_new'):
            tmp = int(tmp[0] + tmp[1][1:])
        elif (tag == 'oai_aug'):
            tmp = int(tmp[0] + tmp[1][2:3] + tmp[2])
        else:
            raise ValueError('Specified tag (%s) is unsupported' % tag)

        file_id[cnt1] = int(tmp)

    order = argsort(file_id)

    return [files[cnt1] for cnt1 in order]


if __name__ == '__main__':
    filename = '9967358_V01-Aug00_072'
    tmp = split('_', filename)
    print(tmp)

    #tmp = int(tmp[0] + tmp[1][2:3])

    print(tmp[1][2:3])
