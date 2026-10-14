"""Generate tiny artificial videos to test plumbing, never to measure model quality."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from research.data import FIELDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('runs/smoke'))
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Use a new smoke directory')
    a.out.mkdir(parents=True)
    rng = np.random.default_rng(42)
    rows = []
    for split in ['train', 'val', 'test']:
        for label in range(4):
            clip_id = f'{split}_{label}'
            video = (a.out / f'{clip_id}.avi').resolve()
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 4, (32, 32))
            if not writer.isOpened():
                raise RuntimeError('MJPG codec unavailable')
            for _ in range(4):
                writer.write(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
            writer.release()
            rows.append(dict(clip_id=clip_id, subject_id=f'synthetic_{split}', split=split,
                             path=str(video), label=label))
    manifest = a.out / 'synthetic.csv'
    with manifest.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    subprocess.run([sys.executable, '-m', 'research.train', '--manifest', str(manifest),
                    '--out', str(a.out / 'train'), '--epochs', '1', '--frames', '2',
                    '--size', '32', '--batch-size', '2', '--device', 'cpu', '--weighted-loss'], check=True)
    subprocess.run([sys.executable, '-m', 'research.evaluate', '--manifest', str(manifest),
                    '--checkpoint', str(a.out / 'train' / 'best.pt'), '--out', str(a.out / 'eval'),
                    '--device', 'cpu', '--calibrate'], check=True)
    print('SMOKE PASSED. Synthetic metrics are not research results.')


if __name__ == '__main__':
    main()
