from medsegpy.utils.cluster import Cluster, CLUSTER

DATA_CATALOG = {}

if CLUSTER in (Cluster.ROMA, Cluster.VIGATA):
    DATA_CATALOG = {
        "oai_2d_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/train",
        "oai_2d_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/valid",
        "oai_2d_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/test",

        "oai_2d_whitened_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/train",
        "oai_2d_whitened_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/valid",
        "oai_2d_whitened_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_whitened_2d/test",

        "oai_3d_train": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/train",
        "oai_3d_val": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/val",
        "oai_3d_test": "/bmrNAS/people/arjun/data/oai_data/h5_files_3d/test",

        "abCT_v0.0.1_train": "/bmrNAS/people/akshay/dl/abct_data/train",
        "abCT_v0.0.1_val": "/bmrNAS/people/akshay/dl/abct_data/valid",
        "abCT_v0.0.1_test": "/bmrNAS/people/akshay/dl/abct_data/test",
    }
elif CLUSTER == CLUSTER.NERO:
    DATA_CATALOG = {
        "oai_2d_train": "/share/pi/bah/data/oai_data/h5_files_2d/train",
        "oai_2d_val": "/share/pi/bah/data/oai_data/h5_files_2d/val",
        "oai_2d_test": "/share/pi/bah/data/oai_data/h5_files_2d/test",
    }
else:
    raise ValueError("Data not found on cluster {}".format(CLUSTER))


def convert_path_to_dataset(path):
    catalog = {v: k for k, v in DATA_CATALOG.items()}
    return catalog[path]
