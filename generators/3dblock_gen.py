import os
import numpy as np
from random import shuffle
from typing import Union

from generators.im_gens import OAI3DGenerator, Generator
from utils.io_utils import load_h5


class OAI3DBlockGenerator(OAI3DGenerator):
    """
    Generator for 3D networks where data is stored in blocks
    """
    SUPPORTED_TAGS = ['oai_3d_block']

    def __init__(self, config):
        super().__init__(config=config)

    def __load_all_volumes__(self, filepaths):
        fps_sorted = sorted(filepaths)
        scans_data = dict()
        for fp in fps_sorted:
            dirpath = os.path.dirname(fp)
            fname = os.path.basename(fp)

            im, seg = self.__load_inputs__(data_path=dirpath, file=fname)
            im = np.squeeze(im)
            seg = np.squeeze(seg)
            assert im.ndim == 2 and seg.ndim == 3, "image must be 2D (Y,X) and segmentation must be 3D (Y,X,#masks)"

            info = self.__get_file_info__(fname)
            scan_id = info['scanid']
            if scan_id not in scans_data:
                scans_data[scan_id] = {'ims': [], 'segs': []}

            scans_data[scan_id]['ims'].append(im)
            scans_data[scan_id]['segs'].append(seg)

        packaged_scan_volumes = {}
        for scan_id in scans_data.keys():
            im_vol = np.stack(scans_data[scan_id]['ims'], axis=-1)
            seg_vol = np.stack(scans_data[scan_id]['segs'], axis=-1)
            seg_vol = np.transpose(seg_vol, [0, 1, 3, 2])

            packaged_scan_volumes[scan_id] = (im_vol, seg_vol)

        return packaged_scan_volumes

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
        yb, xb, zb = tuple(np.squeeze(self.config.IMG_SIZE))

        ordered_keys = sorted(scan_volumes.keys())
        scan_to_blocks = dict()
        for scan_id in ordered_keys:
            im_volume, seg_volume = scan_volumes[scan_id]

            # verify that the sizes of im_volume and seg_volume are as expected
            assert im_volume.shape == seg_volume.shape[:-1], "Input volume of size %s. Masks of size %s" % (im_volume.shape,
                                                                                                            seg_volume.shape)
            self.__validate_img_size__(im_volume.shape)

            expected_num_blocks = self.__calc_num_blocks__(im_volume.shape)
            blocks = []
            for z in range(0, im_volume.shape[2], zb):
                for y in range(0, im_volume.shape[1], yb):
                    for x in range(0, im_volume.shape[0], xb):
                        im = im_volume[y:y+yb, x:x+xb, x:z+zb]
                        seg = seg_volume[y:y+yb, x:x+xb, x:z+zb, :]
                        blocks.append((im, seg))
            assert len(blocks) == expected_num_blocks, "Expected %d blocks, got %d" % (expected_num_blocks, len(blocks))

            scan_to_blocks[scan_id] = blocks

        return scan_to_blocks

    def __calc_num_blocks__(self, total_volume_shape):
        num_blocks = 1
        for dim in range(3):
            assert total_volume_shape[dim] % self.config.IMG_SIZE[dim] == 0, "Verify volumes prior to calling this function using `__validate_img_size__`"
            num_blocks *= total_volume_shape[dim] // self.config.IMG_SIZE[dim]
        assert int(num_blocks) == num_blocks, "Num_blocks is %0.2f, must be an integer" % num_blocks

        return int(num_blocks)

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
        scan_to_volumes = self.__load_all_volumes__(files)
        scan_to_blocks = self.__blockify_volumes__(scan_to_volumes)

        # Set total number of volumes based on argument for limiting training size
        nblocks = 0
        for k in scan_to_blocks.keys():
            nblocks += len(scan_to_blocks[k])

        batches_per_epoch = nblocks // batch_size

        return scan_to_blocks, batches_per_epoch

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
            assert total_volume_shape[dim] % self.config.IMG_SIZE[dim] == 0, "Cannot divide volume of size %s to blocks of size %s" % (total_volume_shape, self.config.IMG_SIZE)

    def img_generator(self, state='training'):
        if state not in ['training', 'validation']:
            raise ValueError('state must be in [\'training\', \'validation\']')

        base_info = self.__img_generator_base_info(state)
        data_path_or_files = base_info['data_path_or_files']
        batch_size = base_info['batch_size']
        pids = base_info['pids']
        augment_data = base_info['augment_data']
        shuffle_epoch = base_info['shuffle_epoch']

        config = self.config
        img_size = config.IMG_SIZE
        tissues = config.TISSUES
        include_background = config.INCLUDE_BACKGROUND

        scan_to_blocks, batches_per_epoch = self.__calc_generator_info__(data_path_or_files=data_path_or_files,
                                                                               batch_size=batch_size,
                                                                               pids=pids,
                                                                               augment_data=augment_data)

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
                    seg = self.__format_seg_helper__(seg, tissues, include_background)

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
