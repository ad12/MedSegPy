from abc import ABC, abstractmethod
import re
import os


class FnameParser(ABC):
    """Abstract class for parsing filenames into comprehensible information"""
    FNAME_REGEX = ''

    @abstractmethod
    def get_file_info(self, fname_or_filepath: str) -> dict:
        """
        Returns dictionary containing parsed file information
        Dictionary must contain keys ['pid', 'aug', 'slice', 'scanid']
        :param fname_or_filepath:
        :return:
        """
        pass

    @abstractmethod
    def get_file_id(self, fname: str):
        pass

    @abstractmethod
    def get_fname(self, file_info: dict):
        pass


class OAISliceWise(FnameParser):
    # sample fname: 9311328_V01-Aug04_072.im
    FNAME_FORMAT = '%7d_V%02d-Aug%02d_%03d'
    FNAME_REGEX = '([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)'

    def get_file_info(self, fname_or_filepath: str) -> dict:
        fname, ext = os.path.splitext(fname_or_filepath)
        dirpath = os.path.dirname(fname)
        fname = os.path.basename(fname)

        _, pid, timepoint, augmentation, slice_num, _ = tuple(re.split(self.FNAME_REGEX, fname_or_filepath))
        pid = int(pid)
        timepoint = int(timepoint)
        augmentation = int(augmentation)
        slice_num = int(slice_num)

        return {'pid': pid, 'timepoint': timepoint, 'aug': augmentation, 'slice': slice_num,
                'scanid':'%d_V%d' % (pid, timepoint)}
                # 'fname': fname,
                # 'impath': os.path.join(dirpath, '%s.%s' % (fname, 'im')),
                # 'segpath': os.path.join(dirpath, '%s.%s' % (fname, 'seg')),
                # 'scanid': scan_id}

    def get_file_id(self, fname):
        fname_info = self.get_file_info(fname)
        return str(fname_info['pid']) + str(fname_info['timepoint']) + str(fname_info['aug']) + str(fname_info['slice'])

    def get_fname(self, file_info: dict):
        return self.FNAME_FORMAT % (file_info['pid'], file_info['timepoint'], file_info['aug'], file_info['slice'])
