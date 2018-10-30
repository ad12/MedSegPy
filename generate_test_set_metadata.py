import pandas as pd
from scan_metadata import ScanMetadata
import utils

if __name__ == '__main__':
    TEST_SET_METADATA = '/bmrNAS/people/arjun/msk_seg_networks/oai_data_test/oai_test_data.xlsx'
    TEST_SET_METADATA_PIK = '/bmrNAS/people/arjun/msk_seg_networks/oai_data_test/oai_test_data.dat'
    df = pd.read_excel(pd.ExcelFile(TEST_SET_METADATA))
    test_set_metadata_dict = dict()

    for test_scan_data in df.values:
        scan_id = test_scan_data[0]
        slice_direction = test_scan_data[1]
        kl_grade = test_scan_data[2]
        test_set_metadata_dict[scan_id] = (scan_id, slice_direction, kl_grade)

    utils.save_pik(test_set_metadata_dict, TEST_SET_METADATA_PIK)