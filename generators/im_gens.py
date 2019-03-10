import os
from abc import ABC, abstractmethod
from os import listdir
from random import shuffle
from re import split
from typing import Union

import h5py
import numpy as np

from config import Config


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
    def img_generator_test(self):
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

    def __add_file__(self, file: str, unique_filenames, pids, augment_data: bool):
        should_add_file = file not in unique_filenames
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
            data_path_or_files = config.__CV_TRAIN_FILES__ if config.USE_CROSS_VALIDATION else config.TRAIN_PATH
            batch_size = config.TRAIN_BATCH_SIZE
            shuffle_epoch = True
            pids = config.PIDS
            augment_data = config.AUGMENT_DATA
        else:
            data_path_or_files = config.__CV_VALID_FILES__ if config.USE_CROSS_VALIDATION else config.VALID_PATH
            batch_size = config.VALID_BATCH_SIZE
            shuffle_epoch = False
            pids = None
            augment_data = False

        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND
        num_neighboring_slices = config.num_neighboring_slices()

        files, batches_per_epoch, max_slice_num = self.__calc_generator_info__(data_path_or_files=data_path_or_files,
                                                                               batch_size=batch_size,
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
                    filepath = files[file_ind]

                    im, seg_total = self.__load_input_helper__(filepath=filepath,
                                                               tissues=tissues,
                                                               num_neighboring_slices=num_neighboring_slices,
                                                               max_slice_num=max_slice_num,
                                                               include_background=include_background)
                    x[file_cnt, ...] = im
                    y[file_cnt, ...] = seg_total

                yield (x, y)

    def img_generator_test(self):
        config = self.config

        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND
        data_path_or_files = config.__CV_TEST_FILES__ if config.USE_CROSS_VALIDATION else config.TEST_PATH
        num_neighboring_slices = config.num_neighboring_slices()
        batch_size = config.TEST_BATCH_SIZE

        # img_size must be 3D
        if len(img_size) != 3:
            raise ValueError('Image size must be 3D')

        files, batches_per_epoch, _ = self.__calc_generator_info__(data_path_or_files=data_path_or_files,
                                                                   batch_size=batch_size,
                                                                   pids=None,
                                                                   augment_data=False)

        files = self.sort_files(files)
        scan_id_to_files = self.__map_files_to_scan_id__(files)
        scan_ids = sorted(scan_id_to_files.keys())

        mask_size = (img_size[0], img_size[1], config.get_num_classes())

        for scan_id in list(scan_ids):
            scan_id_files = scan_id_to_files[scan_id]
            num_slices = len(scan_id_files)

            x = np.zeros((num_slices,) + img_size)
            y = np.zeros((num_slices,) + mask_size)

            for fcount in range(num_slices):
                filepath = scan_id_files[fcount]

                # Make sure that this pid is actually in the filename
                assert scan_id in filepath, "scan_id missing: id %s not in %s" % (scan_id, filepath)

                im, seg_total = self.__load_input_helper__(filepath=filepath,
                                                           tissues=tissues,
                                                           num_neighboring_slices=num_neighboring_slices,
                                                           max_slice_num=num_slices,
                                                           include_background=include_background)

                x[fcount, ...] = im
                y[fcount, ...] = seg_total

            yield (x, y, scan_id, num_slices)

    def __map_files_to_scan_id__(self, files):
        scan_id_files = dict()
        for f in files:
            filename = os.path.basename(f)
            file_info = self.__get_file_info__(filename, '')
            scan_id = file_info['scanid']

            if scan_id in scan_id_files.keys():
                scan_id_files[scan_id].append(f)
            else:
                scan_id_files[scan_id] = [f]

        for k in scan_id_files.keys():
            scan_id_files[k] = sorted(scan_id_files[k])

        return scan_id_files

    def __load_input_helper__(self, filepath, tissues, num_neighboring_slices, max_slice_num, include_background):
        # check that fname contains a dirpath
        assert os.path.dirname(filepath) is not ''

        if num_neighboring_slices:
            im, seg = self.__load_neighboring_slices__(num_slices=num_neighboring_slices,
                                                       filepath=filepath,
                                                       max_slice=max_slice_num)
        else:
            im, seg = self.__load_inputs__(os.path.dirname(filepath), os.path.basename(filepath))

        # support multi class
        if len(tissues) > 1:
            seg_tissues = self.__compress_multi_class_mask__(seg, tissues)
        else:
            seg_tissues = seg[..., 0, tissues]
        seg_total = seg_tissues

        # if considering background, add class
        # background should mark every other pixel that is not already accounted for in segmentation
        if include_background:
            seg_total = self.__add_background_labels__(seg_tissues)

        return im, seg_total

    def __compress_multi_class_mask__(self, seg, tissues):
        o_seg = []

        for t_inds in tissues:
            c_seg = seg[..., 0, t_inds]
            if c_seg.ndim == 3:
                c_seg = np.sum(c_seg, axis=-1)
            o_seg.append(c_seg)

        return np.stack(o_seg, axis=-1)

    def __load_neighboring_slices__(self, num_slices, filepath, max_slice=72):
        """
        Assumes that there are at most slices go from 1-max_slice
        :param num_slices:
        :param filepath:
        :return:
        """
        data_path, filename = os.path.dirname(filepath), os.path.basename(filepath)

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

    def __calc_generator_info__(self, data_path_or_files: Union[str, list], batch_size, pids=None, augment_data=False):
        if type(data_path_or_files) is str:
            data_path = data_path_or_files
            files = listdir(data_path)
            filepaths = [os.path.join(data_path, f) for f in files]
        elif type(data_path_or_files) is list:
            filepaths = data_path_or_files
        else:
            raise ValueError('data_path_or_files must be type str or list')

        unique_filepaths = {}  # use dict to avoid having to reconstruct set every time

        # track the largest slice number that we see - assume it is the same for all scans
        max_slice_num = 0

        for fp in filepaths:
            fp, _ = os.path.splitext(fp)
            dirpath, filename = os.path.dirname(fp), os.path.basename(fp)

            if self.__add_file__(fp, unique_filepaths, pids, augment_data):
                file_info = self.__get_file_info__(filename, dirpath)
                if max_slice_num < file_info['slice']:
                    max_slice_num = file_info['slice']

                unique_filepaths[fp] = fp

        files = list(unique_filepaths.keys())

        # Set total number of files based on argument for limiting training size
        nfiles = len(files)

        batches_per_epoch = nfiles // batch_size

        return files, batches_per_epoch, max_slice_num

    def __get_file_info__(self, fname: str, dirpath: str=''):
        fname, ext = os.path.splitext(fname)
        dirpath = os.path.dirname(fname)
        fname = os.path.basename(fname)

        f_data = fname.split('-')
        scan_id = f_data[0]
        pid_timepoint_split = scan_id.split('_')
        pid = pid_timepoint_split[0]
        f_aug_slice = f_data[1].split('_')
        try:
            data = {'pid': pid,
                    'timepoint': int(pid_timepoint_split[1][1:]),
                    'aug': int(f_aug_slice[0][3:]),
                    'slice': int(f_aug_slice[1]),
                    'fname': fname,
                    'impath': os.path.join(dirpath, '%s.%s' % (fname, 'im')),
                    'segpath': os.path.join(dirpath, '%s.%s' % (fname, 'seg')),
                    'scanid': scan_id}
        except Exception as e:
            import pdb; pdb.set_trace()
            raise e
        assert data['pid'] == fname[:7], str(data)

        return data

    def summary(self):
        config = self.config

        if config.STATE == 'training':
            train_data_path_or_files = config.__CV_TRAIN_FILES__ if config.USE_CROSS_VALIDATION else config.TRAIN_PATH
            valid_data_path_or_files = config.__CV_VALID_FILES__ if config.USE_CROSS_VALIDATION else config.VALID_PATH

            train_files, train_batches_per_epoch, _ = self.__calc_generator_info__(
                data_path_or_files=train_data_path_or_files,
                batch_size=self.config.TRAIN_BATCH_SIZE,
                pids=self.config.PIDS,
                augment_data=self.config.AUGMENT_DATA)

            valid_files, valid_batches_per_epoch, _ = self.__calc_generator_info__(
                data_path_or_files=valid_data_path_or_files,
                batch_size=self.config.VALID_BATCH_SIZE,
                pids=None,
                augment_data=False)

            # Get number of subjects training and validation sets
            train_pids = []
            for f in train_files:
                file_info = self.__get_file_info__(os.path.dirname(f))
                train_pids.append(file_info['pid'])

            valid_pids = []
            for f in valid_files:
                file_info = self.__get_file_info__(os.path.dirname(f))
                valid_pids.append(file_info['pid'])

            num_train_subjects = len(set(train_pids))
            num_valid_subjects = len(set(valid_pids))

            print('INFO: Train size: %d (%d subjects), batch size: %d' % (len(train_files), num_train_subjects, self.config.TRAIN_BATCH_SIZE))
            print('INFO: Valid size: %d (%d subjects), batch size: %d' % (len(valid_files), num_valid_subjects, self.config.VALID_BATCH_SIZE))
            print('INFO: Image size: %s' % (self.config.IMG_SIZE,))
            print('INFO: Image types included in training: %s' % (self.config.FILE_TYPES,))
        else:  # config in Testing state
            test_data_path_or_files = config.__CV_TEST_FILES__ if config.USE_CROSS_VALIDATION else config.TEST_PATH

            test_files, test_batches_per_epoch, _ = self.__calc_generator_info__(
                data_path_or_files=test_data_path_or_files,
                batch_size=self.config.TEST_BATCH_SIZE,
                pids=None,
                augment_data=False)

            scanset_info = self.__get_scanset_data__(test_files)
            print('INFO: Test size: %d, batch size: %d, # subjects: %d, # scans: %d' % (len(test_files),
                                                                                        config.TEST_BATCH_SIZE,
                                                                                        len(scanset_info['pid']),
                                                                                        len(scanset_info['scanid'])))
            if not config.USE_CROSS_VALIDATION:
                print('Test path: %s' % (config.TEST_PATH))

        # not supported
        # print('INFO: Number of frozen layers: %s' % len(self.config.))

    def __get_scanset_data__(self, files, keys=['pid', 'scanid']):
        info_dict = dict()
        for k in keys:
            info_dict[k] = []

        for f in files:
            file_info = self.__get_file_info__(os.path.basename(f))
            for k in keys:
                info_dict[k].append(file_info[k])

        for k in keys:
            info_dict[k] = list(set(info_dict[k]))

        return info_dict

    def num_steps(self):
        config = self.config

        if config.STATE is not 'training':
            raise ValueError('Method is only active when config is in training state')

        train_data_path_or_files = config.__CV_TRAIN_FILES__ if config.USE_CROSS_VALIDATION else config.TRAIN_PATH
        valid_data_path_or_files = config.__CV_VALID_FILES__ if config.USE_CROSS_VALIDATION else config.VALID_PATH

        train_files, train_batches_per_epoch, _ = self.__calc_generator_info__(
            data_path_or_files=train_data_path_or_files,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            pids=self.config.PIDS,
            augment_data=self.config.AUGMENT_DATA)

        valid_files, valid_batches_per_epoch, _ = self.__calc_generator_info__(
            data_path_or_files=valid_data_path_or_files,
            batch_size=self.config.VALID_BATCH_SIZE,
            pids=None,
            augment_data=False)

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
