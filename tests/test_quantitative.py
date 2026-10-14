import math
import tempfile
import unittest
from pathlib import Path

from evaluate_quantitative import emotion_metrics
from emoagent.source_images import discover_source_images


class QuantitativeTests(unittest.TestCase):
    def test_known_distributions(self):
        result = emotion_metrics({'a': .75, 'b': .25}, {'a': .25, 'b': .75}, 'b')
        self.assertEqual(result['Emo-A_ResNet'], 1)
        self.assertEqual(result['Emo-S_ResNet_probability_delta'], .5)
        self.assertAlmostEqual(result['ESR_ResNet_KL_edited_source'], .5 * math.log(3))

    def test_missing_classifier_is_not_zero_score(self):
        with self.assertRaises(ValueError):
            emotion_metrics({'a': 0, 'b': 0}, {'a': 0, 'b': 0}, 'b')

    def test_checkpoints_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '1.png').touch()
            (root / '.ipynb_checkpoints').mkdir()
            (root / '.ipynb_checkpoints' / '1-checkpoint.png').touch()
            self.assertEqual([s.assignment_key for s in discover_source_images(root)], ['1.png'])


if __name__ == '__main__':
    unittest.main()
