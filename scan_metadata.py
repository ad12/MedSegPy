class ScanMetadata():
    def __init__(self, data):
        self.scan_id, self.slice_dir, self.kl_grade = data

        self.cv = None
        self.dsc = None
        self.voe = None
