import os
import shutil
import socket
import unittest

from fvcore.common.file_io import PathManager

from medsegpy.utils.cluster import Cluster


class TestCluster(unittest.TestCase):
    _orig_env = None
    _settings_path = PathManager.get_local_path("mock_data://.settings")

    @classmethod
    def setUpClass(cls):
        cls._orig_env = dict(os.environ)
        os.environ["MEDSEGPY_SETTINGS"] = cls._settings_path

    @classmethod
    def tearDownClass(cls):
        os.environ.clear()
        os.environ.update(cls._orig_env)
        shutil.rmtree(cls._settings_path)
    
    def test_save_load_delete(self):
        hostname = socket.gethostname()
        cluster = Cluster("SAMPLE", [hostname])
        cluster.save()
        assert os.path.isfile(cluster.filepath()), cluster.filepath()
        
        cluster2 = cluster.from_config(cluster.filepath())
        assert all(cluster.__dict__[k] == cluster2.__dict__[k] for k in cluster.__dict__)

        cluster.delete()
        assert not os.path.isfile(cluster.filepath()), cluster.filepath()
    
    def test_cluster(self):
        hostname = socket.gethostname()
        cluster = Cluster("SAMPLE", [hostname])
        cluster.save()

        cluster2 = Cluster.cluster()
        assert all(cluster.__dict__[k] == cluster2.__dict__[k] for k in cluster.__dict__), (cluster.__dict__, cluster2.__dict__)
    
    def test_all_clusters(self):
        hostname = socket.gethostname()
        cluster_names = []
        for i in range(4):
            name = f"SAMPLE-{i}"
            cluster_names.append(name)
            Cluster(name, [hostname]).save()
        all_clusters = Cluster.all_clusters()
        shutil.rmtree(self._settings_path)


if __name__ == "__main__":
    unittest.main()