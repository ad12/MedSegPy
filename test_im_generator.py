import im_generator
from akshay_code import unet2d_generator


if __name__ == '__main__':
    data_path = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'
    batch_size = 72

    files1, _ = im_generator.calc_generator_info(data_path, batch_size)
    files2, batches_per_epoch = unet2d_generator.calc_generator_info(data_path, batch_size)

    files2 = unet2d_generator.sort_files(files2, 'oai_aug')

    assert len(files2) % batch_size == 0

    for batch_cnt in range(batches_per_epoch):
        for file_cnt in range(batch_size):
            file_ind = batch_cnt * batch_size + file_cnt
            f = files2[file_ind]
            print(f)
        print('')