import csv
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from research.data import FIELDS, read_manifest
from research.metrics import classification_metrics
from research.models import EngagementModel


class ResearchTests(unittest.TestCase):
    def test_daisee_adapter_preserves_official_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'Labels').mkdir()
            for i, folder in enumerate(['Train', 'Validation', 'Test']):
                clip_id = f'clip{i}'
                video = root / 'DataSet' / folder / f'person{i}' / clip_id / f'{clip_id}.avi'
                video.parent.mkdir(parents=True)
                video.touch()  # Adapter checks identity/existence, decoder tested in smoke.
                with (root / 'Labels' / f'{folder}Labels.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ClipID', 'Engagement '])
                    writer.writerow([f'{clip_id}.avi', i])
            out = root / 'manifest.csv'
            subprocess.run([sys.executable, '-m', 'research.prepare', '--root', str(root),
                            '--out', str(out)], check=True, capture_output=True)
            rows = read_manifest(out)
            self.assertEqual([r['split'] for r in rows], ['train', 'val', 'test'])
            self.assertEqual([r['label'] for r in rows], [0, 1, 2])
            self.assertTrue(out.with_suffix('.audit.json').is_file())

    def test_metrics_known_values(self):
        p = np.eye(4)
        m = classification_metrics([0, 1, 2, 3], p)
        self.assertEqual(m['macro_f1'], 1.)
        self.assertEqual(m['ece_15_bins'], 0.)
        m = classification_metrics([0, 1, 2, 3], np.tile([1., 0, 0, 0], (4, 1)))
        self.assertAlmostEqual(m['macro_f1'], .1)
        self.assertEqual(m['balanced_accuracy'], .25)

    def test_subject_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = []
            for i, split in enumerate(['train', 'val', 'test']):
                (root / f'{i}.avi').touch()
                rows.append(dict(clip_id=str(i), subject_id='same_person', split=split,
                                 path=f'{i}.avi', label=i))
            with (root / 'm.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Subject leakage'):
                read_manifest(root / 'm.csv')

    def test_forward_backward_all_models(self):
        torch.set_num_threads(2)
        for name in ['tiny_mean', 'tiny_tcn', 'resnet18_tcn']:
            with self.subTest(model=name):
                model = EngagementModel(name)
                out = model(torch.randn(2, 2, 3, 32, 32))
                self.assertEqual(tuple(out.shape), (2, 4))
                loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 3]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
