import os, sys
import unittest

sys.path.append('../')
from analysis import exp_filepaths
from utils import utils

GPU = None

class TestResults(unittest.TestCase):
    experiment_names = [item for item in dir(exp_filepaths) if not item.startswith("__")]

    def read_file(self, filepath):
        with open(filepath) as f:
            content = f.readlines()
        return [x.strip() for x in content]

    def test_best_weights(self):
        # Check if results were created from best weights (i.e. weights with lowest validation loss)
        for exp in self.experiment_names:
            exp_filepath = exp_filepaths.__dict__[exp]
            results_summary_path = os.path.join(exp_filepath, 'results.txt')
            file_lines = self.read_file(results_summary_path)

            weights = ''
            for l in file_lines:
                if 'Weights Loaded:' in l:
                    if weights:
                        raise ValueError('Multiple lines with `Weights Loaded:` keyword- rerun test for %s: %s' % (exp, exp_filepath))

                    weights = l.split(':')[1].strip()

            if not weights:
                raise ValueError('`Weights Loaded:` keyword not found - rerun test for %s: %s' % (exp, exp_filepath))

            # get best weights in the base folder
            ind = exp_filepath.find('test_results')
            base_folder = exp_filepath[:ind]
            expected_best_weights = os.path.basename(utils.get_weights(base_folder))

            assert weights == expected_best_weights, 'weights %s not expected %s- rerun test for %s: %s' % (weights, expected_best_weights, exp, exp_filepath)
            print('%s valid' % exp)

if __name__ == '__main__':
    print(sys.argv)
    unittest.main()
