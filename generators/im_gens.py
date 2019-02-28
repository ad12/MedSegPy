import os
from abc import ABC, abstractmethod
from os import listdir
from random import shuffle
from re import split

import h5py
import numpy as np

from config import Config


# TODO: add support for cross validation
def get_generator(config: Config):
    for generator in [OAIGenerator, OAI3DGenerator]:
        try:
            gen = generator(config)
            if gen:
                return gen
        except ValueError:
            continue

    raise ValueError('No generator found for tag `%s`' % config.TAG)


class Generator(ABC):
    SUPPORTED_TAGS = ['']

    def __init__(self, config: Config):
        if config.TAG not in self.SUPPORTED_TAGS:
            raise ValueError('Tag mismatch: config must have tag in %s' % self.SUPPORTED_TAGS)
        self.config = config

    @abstractmethod
    def get_file_id(self, fname):
        pass

    @abstractmethod
    def __load_inputs__(self, data_path, file):
        pass

    @abstractmethod
    def img_generator(self, state='training'):
        if state not in ['training', 'validation']:
            raise ValueError('state must be in [\'training\', \'validation\']')

    @abstractmethod
    def im_generator_test(self):
        pass

    @abstractmethod
    def __get_file_info__(self, file: str, dirpath: str):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def num_steps(self):
        return 0, 0

    def sort_files(self, files):
        def argsort(seq):
            return sorted(range(len(seq)), key=seq.__getitem__)

        file_id = [None] * len(files)
        for cnt1 in range(len(files)):
            file_id[cnt1] = self.get_file_id(files[cnt1])

        order = argsort(file_id)

        return [files[cnt1] for cnt1 in order]

    def __add_file__(self, file: str, unique_filename, pids, augment_data: bool):
        should_add_file = file not in unique_filename
        file_info = self.__get_file_info__(file, '')
        if pids is not None:
            contains_pid = [str(x) in file for x in pids]

            # if any pid is included, only 1 can be included
            assert (sum(contains_pid) in {0, 1})
            contains_pid = any(contains_pid)

            should_add_file &= contains_pid

        if not augment_data:
            should_add_file &= (file_info['aug'] == 0)

        return should_add_file

    def __add_background_labels__(self, segs: np.ndarray):
        """
        Generate background labels based on segmentations
        :param segs: a binary ndarray with last dimension corresponding to different classes
                     i.e. if 3 classes, segs.shape[-1] = 3
        :return:
        """
        all_tissues = np.sum(segs, axis=-1, dtype=np.bool)
        background = np.asarray(~all_tissues, dtype=np.float)
        background = background[..., np.newaxis]
        seg_total = np.concatenate([background, segs], axis=-1)

        return seg_total


class OAIGenerator(Generator):
    SUPPORTED_TAGS = ['oai_aug', 'oai', 'oai_2d', 'oai_aug_2d', 'oai_2.5d', 'oai_aug_2.5d']

    def get_file_id(self, fname):
        # sample fname: 9311328_V01-Aug04_072.im
        fname_info = self.__get_file_info__(fname, '')
        return str(fname_info['pid']) + str(fname_info['timepoint']) + str(fname_info['aug']) + str(fname_info['slice'])

    def img_generator(self, state='training'):
        super().img_generator(state)

        config = self.config

        if state == 'training':
            data_path = config.TRAIN_PATH
            batch_size = config.TRAIN_BATCH_SIZE
            shuffle_epoch = True
            pids = config.PIDS
            augment_data = config.AUGMENT_DATA
        else:
            data_path = config.VALID_PATH
            batch_size = config.VALID_BATCH_SIZE
            shuffle_epoch = False
            pids = None
            augment_data = False

        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND
        num_neighboring_slices = config.num_neighboring_slices()

        files, batches_per_epoch, max_slice_num = self.__calc_generator_info__(data_path, batch_size,
                                                                               pids=pids,
                                                                               augment_data=augment_data)

        # img_size must be 3D
        if len(img_size) != 3:
            raise ValueError('Image size must be 3D')

        total_classes = config.get_num_classes()
        mask_size = (img_size[0], img_size[1], total_classes)

        x = np.zeros((batch_size,) + img_size)
        y = np.zeros((batch_size,) + mask_size)

        while True:
            if shuffle_epoch:
                shuffle(files)
            else:
                files = self.sort_files(files)

            for batch_cnt in range(batches_per_epoch):
                for file_cnt in range(batch_size):
                    file_ind = batch_cnt * batch_size + file_cnt

                    if num_neighboring_slices:
                        im, seg = self.__get_neighboring_ims__(num_slices=num_neighboring_slices,
                                                               data_path=data_path,
                                                               filename=files[file_ind],
                                                               max_slice=max_slice_num)
                    else:
                        im, seg = self.__load_inputs__(data_path, files[file_ind])

                    seg_tissues = seg[..., 0, tissues]
                    seg_total = seg_tissues

                    # if considering background, add class
                    # background should mark every other pixel that is not already accounted for in segmentation
                    if include_background:
                        seg_total = self.__add_background_labels__(seg_tissues)

                    x[file_cnt, ...] = im
                    y[file_cnt, ...] = seg_total

                yield (x, y)

    def im_generator_test(self):
        config = self.config

        img_size = config.IMG_SIZE
        tissue = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND
        data_path = config.TEST_PATH
        batch_size = config.TEST_BATCH_SIZE

        files, batches_per_epoch, _ = calc_generator_info(data_path, batch_size)

        files = self.sort_files(files)

        num_neighboring = config.num_neighboring_slices()
        print(num_neighboring)
        # img_size must be 3D
        assert (len(img_size) == 3)
        total_classes = config.get_num_classes()
        mask_size = (img_size[0], img_size[1], total_classes)

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
                assert (pid in fname)

                if num_neighboring is not None:
                    im, seg = get_neighboring_ims(num_slices=num_neighboring, data_path=data_path, filename=fname)
                else:
                    im, seg = load_inputs(data_path, fname)
                    if (len(im.shape) == 2):
                        im = im[..., np.newaxis]

                seg_tissues = seg[..., 0, tissue]
                seg_total = seg_tissues

                # if considering background, add class
                # background should mark every other pixel that is not already accounted for in segmentation
                if include_background:
                    seg_total = add_background_labels(seg_tissues)

                x[file_cnt, ...] = im
                y[file_cnt, ...] = seg_total

            yield (x, y, pid, num_slices)

    def __get_neighboring_ims__(self, num_slices, data_path, filename, max_slice=72):
        """
        Assumes that there are at most slices go from 1-max_slice
        :param num_slices:
        :param data_path:
        :param filename:
        :return:
        """

        d_slice = num_slices // 2
        filename_split = filename.split('_')
        central_slice_no = int(filename_split[-1])

        slice_nos = np.asarray(list(range(central_slice_no - d_slice, central_slice_no + d_slice + 1)))
        slice_nos[slice_nos < 1] = 1
        slice_nos[slice_nos > max_slice] = max_slice

        assert len(slice_nos) == num_slices

        base_filename = '_'.join(filename_split[:-1]) + '_%03d'

        ims = []
        segs = []
        for i in range(num_slices):
            slice_no = slice_nos[i]
            slice_filename = base_filename % slice_no

            im, seg = self.__load_inputs__(data_path, slice_filename)
            ims.append(im)
            segs.append(seg)

        # segmentation is central slice segmentation
        im = np.stack(ims)
        im = np.transpose(im, (1, 2, 0))
        seg = segs[len(segs) // 2]

        return im, seg

    def __calc_generator_info__(self, data_path, batch_size, pids=None, augment_data=False):
        files = listdir(data_path)
        unique_filename = {}  # use dict to avoid having to reconstruct set every time

        # track the largest slice number that we see - assume it is the same for all scans
        max_slice_num = 0

        for file in files:
            file, _ = os.path.splitext(file)

            if self.__add_file__(file, unique_filename, pids, augment_data):
                file_info = self.__get_file_info__(file, data_path)
                if max_slice_num < file_info['slice']:
                    max_slice_num = file_info['slice']

                unique_filename[file] = file

        files = list(unique_filename.keys())

        # Set total number of files based on argument for limiting trainign size
        nfiles = len(files)

        batches_per_epoch = nfiles // batch_size

        return files, batches_per_epoch, max_slice_num

    def __load_inputs__(self, data_path: str, file: str):
        im_path = '%s/%s.im' % (data_path, file)
        with h5py.File(im_path, 'r') as f:
            im = f['data'][:]
            if len(im.shape) == 2:
                im = im[..., np.newaxis]

        seg_path = '%s/%s.seg' % (data_path, file)
        with h5py.File(seg_path, 'r') as f:
            seg = f['data'][:].astype('float32')

        assert len(im.shape) == 3
        assert len(seg.shape) == 4 and seg.shape[-2] == 1

        return im, seg

    def __get_file_info__(self, fname: str, dirpath: str):
        fname, ext = os.path.splitext(fname)
        f_data = fname.split('-')
        scan_id = f_data[0].split('_')
        f_aug_slice = f_data[1].split('_')
        data = {'pid': scan_id[0],
                'timepoint': int(scan_id[1][1:]),
                'aug': int(f_aug_slice[0][3:]),
                'slice': int(f_aug_slice[1]),
                'fname': fname,
                'impath': os.path.join(dirpath, '%s.%s' % (fname, 'im')),
                'segpath': os.path.join(dirpath, '%s.%s' % (fname, 'seg')),
                'scanid': scan_id}
        assert data['pid'] == fname[:7], str(data)

        return data

    def summary(self):
        train_files, train_batches_per_epoch, _ = self.__calc_generator_info__(data_path=self.config.TRAIN_PATH,
                                                                               batch_size=self.config.TRAIN_BATCH_SIZE,
                                                                               pids=self.config.PIDS,
                                                                               augment_data=self.config.AUGMENT_DATA)

        valid_files, valid_batches_per_epoch, _ = self.__calc_generator_info__(data_path=self.config.VALID_PATH,
                                                                               batch_size=self.config.VALID_BATCH_SIZE)

        print('INFO: Train size: %d, batch size: %d' % (len(train_files), self.config.TRAIN_BATCH_SIZE))
        print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), self.config.VALID_BATCH_SIZE))
        print('INFO: Image size: %s' % (self.config.IMG_SIZE,))
        print('INFO: Image types included in training: %s' % (self.config.FILE_TYPES,))

        # not supported
        # print('INFO: Number of frozen layers: %s' % len(self.config.))

    def num_steps(self):
        train_files, train_batches_per_epoch, _ = self.__calc_generator_info__(data_path=self.config.TRAIN_PATH,
                                                                               batch_size=self.config.TRAIN_BATCH_SIZE,
                                                                               pids=self.config.PIDS,
                                                                               augment_data=self.config.AUGMENT_DATA)

        valid_files, valid_batches_per_epoch, _ = self.__calc_generator_info__(data_path=self.config.VALID_PATH,
                                                                               batch_size=self.config.VALID_BATCH_SIZE)

        return train_batches_per_epoch, valid_batches_per_epoch


class OAI3DGenerator(Generator):
    SUPPORTED_TAGS = ['oai_aug_3d_pixel', 'oai_3d']

    def get_file_id(self, fname):
        # sample fname: 9146462_V01-Aug0_9.im
        tmp = split('_', fname)
        int(tmp[0] + tmp[1][1:3] + tmp[1][-1:] + tmp[2])

    def __load_inputs__(self, data_path, file):
        pass

    def img_generator(self, state='training'):
        super().img_generator(state)

    def __get_file_info__(self, file: str, dirpath: str):
        pass
