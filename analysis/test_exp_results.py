import os, sys
import unittest

sys.path.append('../')
from analysis import exp_filepaths
from utils import utils


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
                        raise ValueError('Multiple lines with `Weights Loaded:` keyword - rerun test for %s' % exp_filepath)

                    weights = l.split(':')[1].strip()

            if not weights:
                raise ValueError('`Weights Loaded:` keyword not found - rerun test for %s' % exp_filepath)

            # get best weights in the folder
            expected_best_weights = utils.get_weights(os.path.dirname(exp_filepath))

            assert weights == expected_best_weights, 'weights %s not expected %s' % (weights, expected_best_weights)

if __name__ == '__main__':
    unittest.main()