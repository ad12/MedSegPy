import os

from fvcore.common.file_io import PathManager

from medsegpy.utils.io_utils import GeneralPathHandler

_LOCAL_FOLDER = os.path.dirname(os.path.abspath(__file__))


class MockDataPathHandler(GeneralPathHandler):
    PREFIX = "mock_data://"

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        return os.path.join(_LOCAL_FOLDER, "mock_data", name)


PathManager.register_handler(MockDataPathHandler())
