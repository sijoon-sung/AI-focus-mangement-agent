"""Manifest validation and deterministic full-clip sampling."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = ['clip_id', 'subject_id', 'split', 'path', 'label']


def read_manifest(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires {FIELDS}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty manifest')
    seen, paths, subjects = set(), set(), {}
    for row in rows:
        row['label'] = int(row['label'])
        if row['label'] not in range(4) or row['split'] not in {'train', 'val', 'test'}:
            raise ValueError(f'Invalid label/split: {row}')
        if not row['subject_id'].strip() or not row['clip_id'].strip():
            raise ValueError('Empty subject or clip ID')
        if row['clip_id'] in seen:
            raise ValueError(f'Duplicate clip: {row["clip_id"]}')
        seen.add(row['clip_id'])
        p = Path(row['path'])
        if not p.is_absolute():
            p = Path(path).resolve().parent / p
        row['path'] = str(p.resolve())
        if not p.is_file():
            raise FileNotFoundError(p)
        if row['path'] in paths:
            raise ValueError(f'Duplicate video path: {p}')
        paths.add(row['path'])
        old = subjects.setdefault(row['subject_id'], row['split'])
        if old != row['split']:
            raise ValueError(f'Subject leakage: {row["subject_id"]}: {old}/{row["split"]}')
    if {r['split'] for r in rows} != {'train', 'val', 'test'}:
        raise ValueError('All three splits are required')
    return rows


def load_clip(path, frames, size):
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise ValueError(f'Empty/unreadable video: {path}')
        indices = np.linspace(0, count - 1, frames).round().astype(int)
        images = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Failed decoding {path}, frame {index}; repair explicitly')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))
        return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
    finally:
        cap.release()


class Clips(Dataset):
    def __init__(self, rows, split, frames=16, size=112):
        self.rows = [r for r in rows if r['split'] == split]
        self.frames, self.size = frames, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = load_clip(row['path'], self.frames, self.size)
        # Same input normalization for scratch and pretrained comparisons.
        mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        return (x - mean) / std, row['label'], row['clip_id']
