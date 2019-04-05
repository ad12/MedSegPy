import multiprocessing as mp
import os
import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from os import listdir
from random import shuffle
from typing import Tuple

import h5py
import numpy as np

from config import Config
from generators.fname_parsers import OAISliceWise


class GeneratorState(Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3


def get_generator(config: Config):
    for generator in [OAIGenerator, OAI3DGenerator, OAI3DBlockGenerator, OAI3DGeneratorFullVolume]:
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
        self.fname_parser = OAISliceWise()

    @abstractmethod
    def __load_inputs__(self, data_path, file):
        pass

    @abstractmethod
    def img_generator(self, state: GeneratorState):
        accepted_states = [GeneratorState.TRAINING, GeneratorState.VALIDATION]
        if state not in accepted_states:
            raise ValueError('Generator must be either %s' % accepted_states)

    @abstractmethod
    def img_generator_test(self, model):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def num_steps(self) -> Tuple[int, int]:
        return 0, 0

    def sort_files(self, files):
        def argsort(seq):
            return sorted(range(len(seq)), key=seq.__getitem__)

        file_id = [None] * len(files)
        for cnt1 in range(len(files)):
            file_id[cnt1] = self.fname_parser.get_file_id(files[cnt1])

        order = argsort(file_id)

        return [files[cnt1] for cnt1 in order]

    def __add_file__(self, file: str, unique_filenames, pids, augment_data: bool):
        """
        Check if file should be added to list of files
        :param file: a string (filename or filepath accepted)
        :param unique_filenames: list of unique filenames
        :param pids: list of pids to include
        :param augment_data: If data should be augmented, ignored if 'aug' not found in filename
        :return: a boolean
        """
        should_add_file = file not in unique_filenames
        file_info = self.fname_parser.get_file_info(file)

        if pids is not None:
            contains_pid = [str(x) in file for x in pids]

            # if any pid is included, only 1 can be included
            assert (sum(contains_pid) in {0, 1})
            contains_pid = any(contains_pid)

            should_add_file &= contains_pid

        if not augment_data:
            if 'aug' not in file_info.keys():
                warnings.simplefilter('once')
                warnings.warn("Augmentation index not found in filename. Will add all files regardless of name.")
            else:
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

    def __img_generator_base_info__(self, state: GeneratorState):
        config = self.config
        if state == GeneratorState.TRAINING:
            # training
            data_path_or_files = config.__CV_TRAIN_FILES__ if config.USE_CROSS_VALIDATION else config.TRAIN_PATH
            batch_size = config.TRAIN_BATCH_SIZE
            shuffle_epoch = True
            pids = config.PIDS
            augment_data = config.AUGMENT_DATA
        elif state == GeneratorState.VALIDATION:
            # validation
            data_path_or_files = config.__CV_VALID_FILES__ if config.USE_CROSS_VALIDATION else config.VALID_PATH
            batch_size = config.VALID_BATCH_SIZE
            shuffle_epoch = False
            pids = None
            augment_data = False
        elif state == GeneratorState.TESTING:
            data_path_or_files = config.__CV_TEST_FILES__ if config.USE_CROSS_VALIDATION else config.TEST_PATH
            batch_size = config.TEST_BATCH_SIZE
            shuffle_epoch = False
            pids = None
            augment_data = False

        base_info = {'data_path_or_files': data_path_or_files,
                     'batch_size': batch_size,
                     'shuffle_epoch': shuffle_epoch,
                     'pids': pids,
                     'augment_data': augment_data}

        return base_info


class OAIGenerator(Generator):
    SUPPORTED_TAGS = ['oai_aug', 'oai', 'oai_2d', 'oai_aug_2d', 'oai_2.5d', 'oai_aug_2.5d']
    __EXPECTED_IMG_SIZE_DIMS__ = 3

    def img_generator(self, state: GeneratorState):
        super().img_generator(state)

        config = self.config
        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND
        num_neighboring_slices = config.num_neighboring_slices()

        base_info = self.__img_generator_base_info__(state)
        batch_size = base_info['batch_size']
        shuffle_epoch = base_info['shuffle_epoch']
        files, batches_per_epoch, max_slice_num = self.__calc_generator_info__(state)

        total_classes = config.get_num_classes()
        mask_size = img_size[:-1] + (total_classes,)
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

                    im, seg = self.__load_input_helper__(filepath=filepath,
                                                         tissues=tissues,
                                                         num_neighboring_slices=num_neighboring_slices,
                                                         max_slice_num=max_slice_num,
                                                         include_background=include_background)

                    assert im.shape == img_size, "Input shape mismatch. Expected %s, got %s" % (img_size, im.shape)
                    assert seg.shape == mask_size, "Ouput shape mismatch. Expected %s, got %s" % (mask_size, seg.shape)

                    x[file_cnt, ...] = im
                    y[file_cnt, ...] = seg

                yield (x, y)

    def img_generator_test(self, model):
        config = self.config
        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND
        num_neighboring_slices = config.num_neighboring_slices()

        base_info = self.__img_generator_base_info__(GeneratorState.TESTING)
        batch_size = base_info['batch_size']

        if len(img_size) != self.__EXPECTED_IMG_SIZE_DIMS__:
            raise ValueError('Image size must be %dD' % self.__EXPECTED_IMG_SIZE_DIMS__)

        total_classes = config.get_num_classes()
        mask_size = img_size[:-1] + (total_classes,)

        files, batches_per_epoch, _ = self.__calc_generator_info__(GeneratorState.TESTING)
        files = self.sort_files(files)
        scan_id_to_files = self.__map_files_to_scan_id__(files)
        scan_ids = sorted(scan_id_to_files.keys())

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

            recon = model.predict(x, batch_size=batch_size)
            x, y, recon = self.__reformat_testing_scans__(x, y, recon)
            yield (x, y, recon, scan_id)
    
    def __reformat_testing_scans__(self, x, y, recon):
        return x, y, recon

    def __map_files_to_scan_id__(self, files):
        scan_id_files = dict()
        for f in files:
            filename = os.path.basename(f)
            file_info = self.fname_parser.get_file_info(filename)
            scan_id = file_info['scanid']

            if scan_id in scan_id_files.keys():
                scan_id_files[scan_id].append(f)
            else:
                scan_id_files[scan_id] = [f]

        for k in scan_id_files.keys():
            scan_id_files[k] = sorted(scan_id_files[k])

        return scan_id_files

    def __load_input_helper__(self, filepath, tissues, num_neighboring_slices, max_slice_num, include_background):
        """

        :param filepath:
        :param tissues:
        :param num_neighboring_slices: The number of slices to load - None for 2D networks
        :param max_slice_num:
        :param include_background:
        :return:
        """
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

    def __load_neighboring_slices__(self, num_slices, filepath, max_slice):
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

    def __calc_generator_info__(self, state: GeneratorState):
        base_info = self.__img_generator_base_info__(state)
        data_path_or_files = base_info['data_path_or_files']
        batch_size = base_info['batch_size']
        pids = base_info['pids']
        augment_data = base_info['augment_data']

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
                file_info = self.fname_parser.get_file_info(filename)
                if max_slice_num < file_info['slice']:
                    max_slice_num = file_info['slice']

                unique_filepaths[fp] = fp

        files = list(unique_filepaths.keys())

        # Set total number of files based on argument for limiting training size
        nfiles = len(files)

        batches_per_epoch = nfiles // batch_size

        return files, batches_per_epoch, max_slice_num

    def summary(self):
        config = self.config

        if config.STATE == 'training':
            train_files, train_batches_per_epoch, _ = self.__calc_generator_info__(GeneratorState.TRAINING)
            valid_files, valid_batches_per_epoch, _ = self.__calc_generator_info__(GeneratorState.VALIDATION)

            # Get number of subjects training and validation sets
            train_pids = []
            for f in train_files:
                file_info = self.fname_parser.get_file_info(os.path.basename(f))
                train_pids.append(file_info['pid'])

            valid_pids = []
            for f in valid_files:
                file_info = self.fname_parser.get_file_info(os.path.basename(f))
                valid_pids.append(file_info['pid'])

            num_train_subjects = len(set(train_pids))
            num_valid_subjects = len(set(valid_pids))

            print('INFO: Train size: %d slices (%d subjects), batch size: %d' % (
                len(train_files), num_train_subjects, self.config.TRAIN_BATCH_SIZE))
            print('INFO: Valid size: %d slices (%d subjects), batch size: %d' % (
                len(valid_files), num_valid_subjects, self.config.VALID_BATCH_SIZE))
            print('INFO: Image size: %s' % (self.config.IMG_SIZE,))
            print('INFO: Image types included in training: %s' % (self.config.FILE_TYPES,))
        else:  # config in Testing state
            test_files, test_batches_per_epoch, _ = self.__calc_generator_info__(GeneratorState.TESTING)
            scanset_info = self.__get_scanset_data__(test_files)

            print('INFO: Test size: %d slices, batch size: %d, # subjects: %d, # scans: %d' % (len(test_files),
                                                                                               config.TEST_BATCH_SIZE,
                                                                                               len(scanset_info['pid']),
                                                                                               len(scanset_info[
                                                                                                       'scanid'])))
            if not config.USE_CROSS_VALIDATION:
                print('Test path: %s' % config.TEST_PATH)

    def __get_scanset_data__(self, files, keys=['pid', 'scanid']):
        info_dict = dict()
        for k in keys:
            info_dict[k] = []

        for f in files:
            file_info = self.fname_parser.get_file_info(os.path.basename(f))
            for k in keys:
                info_dict[k].append(file_info[k])

        for k in keys:
            info_dict[k] = list(set(info_dict[k]))

        return info_dict

    def num_steps(self):
        config = self.config

        if config.STATE != 'training':
            raise ValueError('Method is only active when config is in training state')

        _, train_batches_per_epoch, _ = self.__calc_generator_info__(GeneratorState.TRAINING)
        _, valid_batches_per_epoch, _ = self.__calc_generator_info__(GeneratorState.VALIDATION)

        return train_batches_per_epoch, valid_batches_per_epoch


class OAI3DGenerator(OAIGenerator):
    """
    Generator for training 3D networks where data is stored per slice
    """
    SUPPORTED_TAGS = ['oai_3d']
    __EXPECTED_IMG_SIZE_DIMS__ = 4

    def __init__(self, config: Config):
        if config.TAG == 'oai_3d_block-train_full-test':
            assert config.testing, "Config must be in testing state to use tag %s" % config.TAG

        super().__init__(config)

    def __validate_img_size__(self, slices_per_scan):
        # only accept image sizes where slices can be perfectly disjoint
        if len(self.config.IMG_SIZE) != self.__EXPECTED_IMG_SIZE_DIMS__:
            raise ValueError('`IMG_SIZE` must be in format (y, x, z, #channels) - got %s' % str(self.config.IMG_SIZE))

        input_volume_num_slices = self.config.IMG_SIZE[2]
        if input_volume_num_slices == 1:
            raise ValueError('For 2D/2.5D networks, use `OAIGenerator`')

        if slices_per_scan % input_volume_num_slices != 0:
            raise ValueError('All input volumes must be disjoint. %d slices per scan, but %d slices in input' % (
                slices_per_scan, input_volume_num_slices))

    def __get_corresponding_files__(self, fname: str):
        num_slices = self.config.IMG_SIZE[2]
        file_info = self.fname_parser.get_file_info(fname)
        f_slice = file_info['slice']

        volume_index = int((f_slice - 1) / num_slices)

        base_info = {'pid': file_info['pid'], 'timepoint': file_info['timepoint'],
                     'aug': file_info['aug'], 'slice': file_info['slice']}

        # slices are 1-indexed
        slices = list(range(volume_index * num_slices + 1, (volume_index + 1) * num_slices + 1))
        slice_fnames = []
        for s in slices:
            base_info['slice'] = s
            slice_fnames.append(self.fname_parser.get_fname(base_info))

        return slice_fnames

    def __load_input_helper__(self, filepath, tissues, num_neighboring_slices, max_slice_num, include_background):
        # check that fname contains a dirpath
        assert os.path.dirname(filepath) is not ''

        if num_neighboring_slices:
            im, seg = self.__load_corresponding_slices__(filepath)
        else:
            raise ValueError('`num_neighboring_slices` must be initialized')

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

    def __load_corresponding_slices__(self, filepath):
        """
        Loads volume that slice (filepath) corresponds to
        e.g. If each input volume consists of 4 slices, slices 1-4 will be in the same volume.
             If `filepath` corresponds to slice 2, the volume (slices 1-4) and corresponding segmentations will be returned
        :param filepath: Filepath corresponding to single slice
        :return: tuple of 2 3d numpy array corresponding to 3d input volume and 3d segmentation volume (im_vol, seg_vol)
        """
        data_path, filename = os.path.dirname(filepath), os.path.basename(filepath)
        corresponding_files = self.__get_corresponding_files__(filename)

        ims = []
        segs = []
        for f in corresponding_files:
            im, seg = self.__load_inputs__(data_path, f)
            ims.append(im)
            segs.append(seg)

        im_vol = np.stack(ims, axis=-1)
        seg_vol = np.stack(segs, axis=-1)
        im_vol = np.squeeze(im_vol)
        im_vol = im_vol[..., np.newaxis]
        seg_vol = np.transpose(np.squeeze(seg_vol), [0, 1, 3, 2])
        seg_vol = seg_vol[..., np.newaxis]

        assert im_vol.shape == self.config.IMG_SIZE, "Loaded volume of size %s. Expected %s" % (im_vol.shape,
                                                                                                self.config.IMG_SIZE)

        return im_vol, seg_vol

    def __calc_generator_info__(self, state: GeneratorState):
        base_info = self.__img_generator_base_info__(state)
        data_path_or_files = base_info['data_path_or_files']
        batch_size = base_info['batch_size']
        pids = base_info['pids']
        augment_data = base_info['augment_data']

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
        slice_ids = []

        for fp in filepaths:
            fp, _ = os.path.splitext(fp)
            dirpath, filename = os.path.dirname(fp), os.path.basename(fp)

            if self.__add_file__(fp, unique_filepaths, pids, augment_data):
                file_info = self.fname_parser.get_file_info(filename)
                slice_ids.append(file_info['slice'])

                unique_filepaths[fp] = fp

        files = list(unique_filepaths.keys())

        min_slice_id = min(slice_ids)
        max_slice_id = max(slice_ids)

        # validate image size
        self.__validate_img_size__(max_slice_id - min_slice_id + 1)

        # Remove files corresponding to the same volume
        # e.g. If input volume has 4 slices, slice 1 and 2 will be part of the same volume
        #      We only include file corresponding to slice 1, because other files are accounted for by default
        slices_to_include = range(min_slice_id, max_slice_id + 1, self.config.IMG_SIZE[2])
        files_refined = []
        for filepath in files:
            fname = os.path.basename(filepath)
            f_info = self.fname_parser.get_file_info(fname)
            if f_info['slice'] in slices_to_include:
                files_refined.append(filepath)

        files = list(set(files_refined))

        # Set total number of volumes based on argument for limiting training size
        nvolumes = len(files)

        batches_per_epoch = nvolumes // batch_size

        return files, batches_per_epoch, max_slice_id

    def __add_file__(self, file: str, unique_filenames, pids, augment_data: bool):
        add_file = super().__add_file__(file, unique_filenames, pids, augment_data)

        # If only subset of slices should be selected, then return
        if hasattr(self.config, 'SLICE_SUBSET') and self.config.SLICE_SUBSET is not None:
            file_info = self.fname_parser.get_file_info(file)
            add_file &= file_info['slice'] in range(self.config.SLICE_SUBSET[0], self.config.SLICE_SUBSET[1] + 1)

        return add_file


class OAI3DGeneratorFullVolume(OAI3DGenerator):
    SUPPORTED_TAGS = ['oai_3d_block-train_full-vol-test', 'full-vol-test', 'oai_3d_full-vol-test']

    def __init__(self, config: Config):
        if not config.testing:
            raise ValueError('%s only available for testing 3D volumes' % self.__class__.__name__)

        super().__init__(config)

        # update image size for config to be total volume
        img_size = self.__get_img_dims__(GeneratorState.TESTING)
        assert len(img_size) == 3, "Loaded image size must be a tuple of 3 integers (y, x, #slices)"
        self.config.IMG_SIZE = img_size + self.config.IMG_SIZE[3:]

    def __img_generator_base_info__(self, state: GeneratorState):
        assert state == GeneratorState.TESTING, "Only testing state is supported for this generator"
        return super().__img_generator_base_info__(state)

    def __get_img_dims__(self, state: GeneratorState):
        base_info = self.__img_generator_base_info__(state)
        data_path_or_files = base_info['data_path_or_files']
        batch_size = base_info['batch_size']
        pids = base_info['pids']
        augment_data = base_info['augment_data']

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
        slice_ids = []

        for fp in filepaths:
            fp, _ = os.path.splitext(fp)
            dirpath, filename = os.path.dirname(fp), os.path.basename(fp)

            if self.__add_file__(fp, unique_filepaths, pids, augment_data):
                file_info = self.fname_parser.get_file_info(filename)
                slice_ids.append(file_info['slice'])

                unique_filepaths[fp] = fp

        min_slice_id = min(slice_ids)
        max_slice_id = max(slice_ids)
        num_slices = max_slice_id - min_slice_id + 1

        # load 1 file to get inplane resolution
        filepath = list(unique_filepaths.keys())[0]
        im_vol, _ = self.__load_inputs__(os.path.dirname(filepath), os.path.basename(filepath))
        return im_vol.shape[:2] + (num_slices,)
    
    def __reformat_testing_scans__(self, x, y, recon):
        vols = [x, y, recon]
        vols_updated = []
        for v in vols:
            assert v.ndim == 5 and v.shape[0] == 1 and v.shape[1:] == self.config.IMG_SIZE, "img dims must be %s" % str((1,) + self.config.IMG_SIZE)
            v_updated = np.squeeze(v, axis=0)
            v_updated = np.transpose(v_updated, [2, 0, 1, 3])
            vols_updated.append(v_updated)
        return tuple(vols_updated)

    def num_steps(self):
        raise ValueError('This method is not supported for a testing-only generator')


class OAI3DBlockGenerator(OAI3DGenerator):
    """
    Generator for 3D networks where data is stored in blocks
    """
    SUPPORTED_TAGS = ['oai_3d_block']

    def __init__(self, config):
        super().__init__(config=config)
        self._cached_data = dict()

    def cached_data(self, state: GeneratorState):
        if state not in self._cached_data.keys():
            start_time = time.time()
            print('Computing %s blocks' % state.name)
            self._cached_data[state] = self.__calc_generator_info__(state)
            print('%0.2f seconds' % (time.time() - start_time))

        return self._cached_data[state]

    def process_filepath(self, fp):
        dirpath = os.path.dirname(fp)
        fname = os.path.basename(fp)

        im, seg = self.__load_inputs__(data_path=dirpath, file=fname)
        im = np.squeeze(im)
        seg = np.squeeze(seg)
        assert im.ndim == 2 and seg.ndim == 3, "image must be 2D (Y,X) and segmentation must be 3D (Y,X,#masks)"

        info = self.fname_parser.get_file_info(fname)
        volume_id = info['volume_id']
        slice_num = info['slice']
        return volume_id, slice_num, im, seg

    def __load_all_volumes__(self, filepaths):
        parallelize = False
        if parallelize:
            with mp.Pool() as pool:
                loaded_data_list = pool.map(self.process_filepath, filepaths)
        else:
            loaded_data_list = []
            for fp in filepaths:
                loaded_data_list.append(self.process_filepath(fp))
        # sort list first by volume_id, then by slice_num
        loaded_data_list = sorted(loaded_data_list, key=(lambda x: (x[0], x[1])))

        scans_data = dict()
        for volume_id, slice_num, im, seg in loaded_data_list:
            if volume_id not in scans_data.keys():
                scans_data[volume_id] = {'ims': [], 'segs': []}

            scans_data[volume_id]['ims'].append(im)
            scans_data[volume_id]['segs'].append(seg)

        packaged_scan_volumes = {}
        for scan_id in scans_data.keys():
            im_vol = np.stack(scans_data[scan_id]['ims'], axis=-1)
            seg_vol = np.stack(scans_data[scan_id]['segs'], axis=-1)
            seg_vol = np.transpose(seg_vol, [0, 1, 3, 2])

            packaged_scan_volumes[scan_id] = (im_vol, seg_vol)

        return packaged_scan_volumes

    def unify_blocks(self, blocks, reshape_dims):
        # blocks are stored in decreasing order of fastest changing axes: x, y, z
        # inverse of blockify volume
        """
        z= 0:
        -----------------
        | 1 | 2 | 3 | 4 |
        -----------------
        | 5 | 6 | 7 | 8 |
        -----------------
        | 9 | 10| 11| 12|
        -----------------
        | 13| 14| 15| 16|
        -----------------
        z= 1:
        -----------------
        | 17| 18| 19| 20|
        -----------------
        | 21| 22| 23| 24|
        -----------------
        | 25| 26| 27| 28|
        -----------------
        | 29| 30| 31| 32|
        -----------------
        :param blocks: a list of tuples of numpy arrays [(im_block1, seg_block1), (im_block2, seg_block2)]
        :param reshape_dims:
        :return:
        """
        yb, xb, zb, _ = self.config.IMG_SIZE

        num_masks = blocks[0][1].shape[-1]
        im_volume = np.zeros(reshape_dims)
        seg_volume = np.zeros(reshape_dims + (num_masks,))

        ind = 0
        for z in range(0, reshape_dims[2], zb):
            for y in range(0, reshape_dims[1], yb):
                for x in range(0, reshape_dims[0], xb):
                    im, seg = blocks[ind]
                    im_volume[y:y + yb, x:x + xb, z:z + zb] = im
                    seg_volume[y:y + yb, x:x + xb, z:z + zb, :] = seg
                    ind += 1
        assert ind == len(blocks), "Blocks not appropriately portioned"

        return im_volume, seg_volume

    def blockify_volume(self, im_volume: np.ndarray, seg_volume: np.ndarray):
        assert im_volume.ndim == 3, "Dimension mismatch. im_volume must be 3D (y, x, z)."
        assert seg_volume.ndim == 4, "Dimension mismatch. seg_volume must be 4D (y, x, z, #classes)."

        yb, xb, zb, _ = self.config.IMG_SIZE
        expected_block_size = (yb, xb, zb)

        # verify that the sizes of im_volume and seg_volume are as expected
        assert im_volume.shape == seg_volume.shape[:-1], "Input volume of size %s. Masks of size %s" % (im_volume.shape,
                                                                                                        seg_volume.shape)
        self.__validate_img_size__(im_volume.shape)

        expected_num_blocks = self.__calc_num_blocks__(im_volume.shape)
        blocks = []
        for z in range(0, im_volume.shape[2], zb):
            for y in range(0, im_volume.shape[1], yb):
                for x in range(0, im_volume.shape[0], xb):
                    im = im_volume[y:y + yb, x:x + xb, z:z + zb]
                    seg = seg_volume[y:y + yb, x:x + xb, z:z + zb, :]
                    assert im.shape[:3] == seg.shape[:3], "Block shape mismatch. im_block %s, seg_block %s" % (im.shape,
                                                                                                               seg.shape)
                    assert im.shape[:3] == expected_block_size, "Block shape error. Expected %s, but got %s" % (
                        expected_block_size, im.shape)
                    blocks.append((im, seg))
        assert len(blocks) == expected_num_blocks, "Expected %d blocks, got %d" % (expected_num_blocks, len(blocks))

        return blocks

    def __blockify_volumes__(self, scan_volumes: dict):
        """
        Split all volumes into blocks of size config.IMG_SIZE
        Throws error if volume is not perfectly divisible into blocks of size config.IMG_SIZE (Yi, Xi, Zi)
        :param scan_volumes: A dict of scan_ids --> (im_volume, seg_volume)
                                    im_volume: 3D numpy array (Y, X, Z)
                                    seg_volume: 4D numpy array (Y, X, Z, #masks)
        :return: a list of tuples im_volume and corresponding seg_volume blocks
                    im_volume block: a 3D numpy array of size Y/Yi, X/Xi, Z/Zi
                    seg_volume block: a 3D numpy array of size Y/Yi, X/Xi, Z/Zi, #masks
        """
        ordered_keys = sorted(scan_volumes.keys())
        scan_to_blocks = dict()
        for scan_id in ordered_keys:
            im_volume, seg_volume = scan_volumes[scan_id]
            scan_to_blocks[scan_id] = self.blockify_volume(im_volume, seg_volume)

        return scan_to_blocks

    def __get_num_blocks__(self, scan_to_blocks):
        temp = []
        for k in scan_to_blocks:
            temp.extend(scan_to_blocks[k])
        return len(temp)

    def __calc_num_blocks__(self, total_volume_shape):
        num_blocks = 1
        for dim in range(3):
            assert total_volume_shape[dim] % self.config.IMG_SIZE[
                dim] == 0, "Verify volumes prior to calling this function using `__validate_img_size__`"
            num_blocks *= total_volume_shape[dim] // self.config.IMG_SIZE[dim]
        assert int(num_blocks) == num_blocks, "Num_blocks is %0.2f, must be an integer" % num_blocks

        return int(num_blocks)

    def __calc_generator_info__(self, state: GeneratorState) -> Tuple[dict, int, dict]:
        base_info = self.__img_generator_base_info__(state)
        data_path_or_files = base_info['data_path_or_files']
        batch_size = base_info['batch_size']
        pids = base_info['pids']
        augment_data = base_info['augment_data']

        if type(data_path_or_files) is str:
            data_path = data_path_or_files
            files = os.listdir(data_path)
            filepaths = [os.path.join(data_path, f) for f in files]
        elif type(data_path_or_files) is list:
            filepaths = data_path_or_files
        else:
            raise ValueError('data_path_or_files must be type str or list')

        unique_filepaths = {}  # use dict to avoid having to reconstruct set every time

        # track the largest slice number that we see - assume it is the same for all scans
        slice_ids = []

        for fp in filepaths:
            fp, _ = os.path.splitext(fp)
            dirpath, filename = os.path.dirname(fp), os.path.basename(fp)

            if self.__add_file__(fp, unique_filepaths, pids, augment_data):
                file_info = self.fname_parser.get_file_info(filename)
                slice_ids.append(file_info['slice'])

                unique_filepaths[fp] = fp

        files = list(unique_filepaths.keys())
        scan_to_volumes = self.__load_all_volumes__(files)

        # store original volume sizes in dict
        scan_to_im_size = dict()
        for scan in scan_to_volumes.keys():
            scan_to_im_size[scan] = scan_to_volumes[scan][0].shape

        scan_to_blocks = self.__blockify_volumes__(scan_to_volumes)

        # Set total number of volumes based on argument for limiting training size
        nblocks = 0
        for k in scan_to_blocks.keys():
            nblocks += len(scan_to_blocks[k])

        batches_per_epoch = nblocks // batch_size

        return scan_to_blocks, batches_per_epoch, scan_to_im_size

    def __validate_img_size__(self, total_volume_shape):
        """
        Validate that image input size goes perfectly into volume
        :param total_volume_shape:
        :return:
        """
        super().__validate_img_size__(total_volume_shape[2])

        # check that all blocks will be disjoint
        # this means shape of total volume must be perfectly divisible into cubes of size IMG_SIZE
        for dim in range(3):
            assert total_volume_shape[dim] % self.config.IMG_SIZE[
                dim] == 0, "Cannot divide volume of size %s to blocks of size %s" % (
            total_volume_shape, self.config.IMG_SIZE)

    def img_generator_test(self, model):
        state = GeneratorState.TESTING
        base_info = self.__img_generator_base_info__(state)
        batch_size = base_info['batch_size']

        config = self.config
        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND

        scan_to_blocks, batches_per_epoch, scan_to_im_size = self.cached_data(state)

        total_classes = config.get_num_classes()
        mask_size = img_size[:-1] + (total_classes,)

        volume_ids = sorted(scan_to_blocks.keys())

        for vol_id in volume_ids:
            blocks = scan_to_blocks[vol_id]
            num_blocks = len(blocks)

            x = np.zeros((num_blocks,) + img_size)
            y = np.zeros((num_blocks,) + mask_size)

            for block_cnt in range(num_blocks):
                im, seg = blocks[block_cnt]
                if im.ndim == 3:
                    im = im[..., np.newaxis]

                seg = self.__format_seg_helper__(seg, tissues, include_background)
                assert im.shape == img_size, "Input shape mismatch. Expected %s, got %s" % (img_size, im.shape)
                assert seg.shape == mask_size, "Ouput shape mismatch. Expected %s, got %s" % (mask_size, seg.shape)

                x[block_cnt, ...] = im
                y[block_cnt, ...] = seg

            recon = model.predict(x, batch_size=batch_size)

            # reshape into original volume shape
            ytrue_blocks = [(np.squeeze(x[b, ...]), y[b, ...]) for b in range(num_blocks)]
            im_vol, ytrue_vol = self.unify_blocks(ytrue_blocks, scan_to_im_size[vol_id])

            ypred_blocks = [(np.squeeze(x[b, ...]), recon[b, ...]) for b in range(num_blocks)]
            _, recon_vol = self.unify_blocks(ypred_blocks, scan_to_im_size[vol_id])

            # reshape to expected output shape
            im_vol = np.transpose(im_vol, [2, 0, 1])
            im_vol = im_vol[..., np.newaxis]
            ytrue_vol = np.transpose(ytrue_vol, [2, 0, 1, 3])
            recon_vol = np.transpose(recon_vol, [2, 0, 1, 3])

            yield (im_vol, ytrue_vol, recon_vol, vol_id)

    def img_generator(self, state):
        accepted_states = [GeneratorState.TRAINING, GeneratorState.VALIDATION]
        if state not in accepted_states:
            raise ValueError('state must be either %s' % accepted_states)

        base_info = self.__img_generator_base_info__(state)
        batch_size = base_info['batch_size']
        shuffle_epoch = base_info['shuffle_epoch']

        config = self.config
        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND

        scan_to_blocks, batches_per_epoch, _ = self.cached_data(state)

        total_classes = config.get_num_classes()
        mask_size = img_size[:-1] + (total_classes,)

        x = np.zeros((batch_size,) + img_size)
        y = np.zeros((batch_size,) + mask_size)

        blocks = []
        for k in scan_to_blocks.keys():
            blocks.extend(scan_to_blocks[k])

        while True:
            if shuffle_epoch:
                shuffle(blocks)

            for batch_cnt in range(batches_per_epoch):
                for block_cnt in range(batch_size):
                    block_ind = batch_cnt * batch_size + block_cnt

                    im, seg = blocks[block_ind]
                    if im.ndim == 3:
                        im = im[..., np.newaxis]

                    seg = self.__format_seg_helper__(seg, tissues, include_background)
                    assert im.shape == img_size, "Input shape mismatch. Expected %s, got %s" % (img_size, im.shape)
                    assert seg.shape == mask_size, "Ouput shape mismatch. Expected %s, got %s" % (mask_size, seg.shape)

                    x[block_cnt, ...] = im
                    y[block_cnt, ...] = seg

                yield (x, y)

    def __format_seg_helper__(self, seg, tissues, include_background):
        seg = seg[..., np.newaxis]
        seg = np.transpose(seg, [0, 1, 2, 4, 3])

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

        return seg_total

    def __compress_multi_class_mask__(self, seg, tissues):
        o_seg = []

        for t_inds in tissues:
            c_seg = seg[..., 0, t_inds]
            if c_seg.ndim == 3:
                c_seg = np.sum(c_seg, axis=-1)
            o_seg.append(c_seg)

        return np.stack(o_seg, axis=-1)

    def summary(self):
        config = self.config

        if config.STATE == 'training':
            self.__state_summary(GeneratorState.TRAINING)
            self.__state_summary(GeneratorState.VALIDATION)

            print('INFO: Image size: %s' % (self.config.IMG_SIZE,))
            print('INFO: Image types included in training: %s' % (self.config.FILE_TYPES,))
        else:  # config in Testing state
            self.__state_summary(GeneratorState.TESTING)
            if not config.USE_CROSS_VALIDATION:
                print('Test path: %s' % config.TEST_PATH)

    def __state_summary(self, state: GeneratorState):
        scan_to_blocks, batches_per_epoch, _ = self.cached_data(state)
        volume_ids = scan_to_blocks.keys()
        num_volumes = len(volume_ids)
        num_blocks = self.__get_num_blocks__(scan_to_blocks)

        pids = [self.fname_parser.get_pid_from_volume_id(vol_id) for vol_id in volume_ids]
        num_pids = len(pids)

        base_info = self.__img_generator_base_info__(state)
        batch_size = base_info['batch_size']

        print('INFO: %s size: %d blocks (%d volumes), batch size: %d, # subjects: %d' % (state.name,
                                                                                         num_blocks,
                                                                                         num_volumes,
                                                                                         batch_size,
                                                                                         num_pids))

    def num_steps(self):
        config = self.config

        if config.STATE != 'training':
            raise ValueError('Method is only active when config is in training state')

        _, train_batches_per_epoch, _ = self.cached_data(GeneratorState.TRAINING)
        _, valid_batches_per_epoch, _ = self.cached_data(GeneratorState.VALIDATION)

        return train_batches_per_epoch, valid_batches_per_epoch


