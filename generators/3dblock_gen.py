import os
import numpy as np
from typing import Union

from generators.im_gens import OAI3DGenerator
from utils.io_utils import load_h5

class OAI3DBlockGenerator(OAI3DGenerator):
    """
    Generator for 3D networks where data is stored in blocks
    """
    SUPPORTED_TAGS = ['oai_3d_block']

    def __load_inputs__(self, data_path, file):
        im_path = '%s/%s.im' % (data_path, file)
        im = load_h5(im_path)['data'][:]
        if len(im.shape) == 2:
            im = im[..., np.newaxis]

        seg_path = '%s/%s.seg' % (data_path, file)
        seg = load_h5(seg_path)['data'][:].astype('float32')

        assert len(im.shape) == 3
        assert len(seg.shape) == 4 and seg.shape[-2] == 1

        return im, seg

    def __calc_generator_info__(self, data_path_or_files: Union[str, list], batch_size, pids=None, augment_data=False):
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
                file_info = self.__get_file_info__(filename, dirpath)
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
            f_info = self.__get_file_info__(fname)
            if f_info['slice'] in slices_to_include:
                files_refined.append(filepath)

        files = list(set(files_refined))

        # Set total number of volumes based on argument for limiting training size
        nvolumes = len(files)

        batches_per_epoch = nvolumes // batch_size

        return files, batches_per_epoch, max_slice_id

    def __validate_img_size__(self, total_volume_shape):
        """
        Validate that image input size goes perfectly into volume
        :param total_volume_shape:
        :return:
        """
        super().__validate_img_size__(total_volume_shape[2])

        # check that all blocks will be disjoint
        for dim in range(3):
            assert total_volume_shape[dim] % self.config.IMG_SIZE[dim] == 0, "Cannot divide volume of size %s to blocks of size %s" % (total_volume_shape, self.config.IMG_SIZE)

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
        # if len(img_size) != 3:
        #    raise ValueError('Image size must be 3D')

        total_classes = config.get_num_classes()
        mask_size = img_size[:-1] + (total_classes,)
        # mask_size = (img_size[0], img_size[1], total_classes)

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
