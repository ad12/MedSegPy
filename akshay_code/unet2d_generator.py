# Author: Zhongnan Fang, zhongnanf@gmail.com, 2017 July
# Modified: Akshay Chaudhari, akshaysc@stanford.edu 2017 August

from __future__ import print_function, division

import numpy as np
from os import listdir
from os.path import splitext
from random import shuffle
from re import split
import h5py


# find unique data regardless of the file prefix
def calc_generator_info(data_path, batch_size, learn_files=[]):
    files = listdir(data_path)
    unique_filename = {}

    for file in files:
        file, _ = splitext(file)
        if not file in unique_filename:
            unique_filename[file] = file

    files = unique_filename.keys()

    # Set total number of files based on argument for limiting trainign size
    if learn_files:
        nfiles = learn_files
    else:
        nfiles = len(files)

    batches_per_epoch = nfiles // batch_size

    return (files, batches_per_epoch)


def dess_generator(data_path, batch_size, img_size, tag,
                   combine_echoes=False,
                   inference_mode=False,
                   learn_files=[],
                   shuffle_epoch=True):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size, learn_files)
    files = sort_files(files, tag)

    x = np.zeros((batch_size,) + img_size)
    # This is to deal with inconsistent input and output sizes
    y = np.zeros((batch_size,) + (img_size[0:-1]) + (1,))

    while True:

        # If limiting size of training data, sort through file names and fix batches_per_epoch
        if learn_files:
            arr = np.arange(0, learn_files)
            files = [files[i] for i in arr]

        if shuffle_epoch:
            shuffle(files)

        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):

                file_ind = batch_cnt * batch_size + file_cnt

                #                     Python 3+ does not support indexing dicts
                path = '%s/%s.im' % (data_path, files[file_ind])
                # path = '%s/%s.%s'%(data_path, next(iter(files)), file_types[type_cnt])

                with h5py.File(path, 'r') as f:

                    if combine_echoes is False:
                        # Last dimension is whitened sum of squares when implemented
                        im = f['data'][...]
                    else:
                        im = f['data'][..., -1]
                    x[file_cnt, ...] = im

                #                     Python 3+ does not support indexing dicts
                seg_path = '%s/%s.seg' % (data_path, files[file_ind])
                # seg_path = '%s/%s.seg'%(data_path, next(iter(files)))

                with h5py.File(seg_path, 'r') as f:
                    seg = f['data'][:].astype('float32')
                    y[file_cnt, ..., 0] = seg

            if inference_mode is True:
                yield (x, y, fname)
            else:
                yield (x, y)


def oai_generator(data_path, batch_size, img_size, tissue, tag,
                  inference_mode=False, shuffle_epoch=True):
    files, batches_per_epoch = calc_generator_info(data_path, batch_size)

    x = np.zeros((batch_size,) + img_size)
    y = np.zeros((batch_size,) + img_size)

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
                    tmp = f['data'][..., tissue].astype('float32')
                    tmp = np.sum(tmp, axis=-1)

                # seg = np.zeros(img_size)
                # for ii in tissue:
                #     seg = seg + tmp[...,ii]

                x[file_cnt, ..., 0] = im
                y[file_cnt, ...] = tmp

                fname = files[file_ind]

            if inference_mode is False:
                yield (x, y)
            else:
                fname = files[file_ind]
                yield (x, y, fname)


def sort_files(files, tag):
    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    file_id = [None] * len(files)
    for cnt1 in range(len(files)):
        tmp = split('_', files[cnt1])

        if (tag == 'dess'):
            tmp = int(tmp[1])
        elif (tag == 'dess_aug'):
            tmp = int(tmp[0] + tmp[2][-1] + tmp[-1])
        elif (tag == 'oai'):
            tmp = int(tmp[-1][0])
        elif (tag == 'oai_new'):
            tmp = int(tmp[0] + tmp[1][1:])
        elif (tag == 'oai_aug'):
            tmp = int(tmp[0] + tmp[1][2:3])
        else:
            raise ValueError('Specified tag (%s) is unsupported' % tag)

        file_id[cnt1] = int(tmp)

    order = argsort(file_id)

    return [files[cnt1] for cnt1 in order]