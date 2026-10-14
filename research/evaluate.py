"""Evaluate frozen best weights; fit optional temperature on validation only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from research.data import Clips, read_manifest
from research.metrics import classification_metrics
from research.models import EngagementModel
from research.train import collect, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--calibrate', action='store_true')
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Use an empty evaluation output directory')
    rows = read_manifest(a.manifest)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    split = [{k: r[k] for k in ['clip_id', 'subject_id', 'split', 'label']} for r in rows]
    fingerprint = hashlib.sha256(json.dumps(sorted(split, key=lambda r: r['clip_id']), sort_keys=True).encode()).hexdigest()
    if fingerprint != cfg['split_sha256']:
        raise ValueError('Split/label identity differs from training manifest')
    model = EngagementModel(cfg['model'], pretrained=False).to(a.device)
    model.load_state_dict(checkpoint['state_dict'])
    def loader(split):
        return DataLoader(Clips(rows, split, cfg['frames'], cfg['size']),
                          batch_size=a.batch_size, num_workers=a.workers)
    temperature = 1.
    if a.calibrate:
        val_logits, val_labels, _ = collect(model, loader('val'), a.device)
        # Bounded scalar grid: no test labels used for fitting.
        candidates = torch.logspace(-1, 1, 101)
        losses = torch.stack([torch.nn.functional.cross_entropy(val_logits / t, val_labels) for t in candidates])
        temperature = float(candidates[losses.argmin()])
    start = time.perf_counter()
    logits, labels, ids = collect(model, loader('test'), a.device)
    elapsed = time.perf_counter() - start
    raw = logits.softmax(1).numpy()
    probs = (logits / temperature).softmax(1).numpy()
    y = labels.numpy()
    counts = np.bincount([r['label'] for r in rows if r['split'] == 'train'], minlength=4)
    majority = np.zeros_like(probs)
    majority[:, counts.argmax()] = 1.
    prior = np.tile(counts / counts.sum(), (len(y), 1))
    result = {'epoch': checkpoint['epoch'], 'temperature': temperature,
              'checkpoint_sha256': hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
              'raw': classification_metrics(y, raw), 'calibrated': classification_metrics(y, probs),
              'train_majority': classification_metrics(y, majority),
              'train_class_prior': classification_metrics(y, prior),
              'parameters': sum(p.numel() for p in model.parameters()),
              'test_clips': len(y), 'end_to_end_seconds_per_clip': elapsed / len(y),
              'timing_note': 'Includes video decode, batching, transfers and inference; not batch=1 latency.',
              'device': a.device, 'batch_size': a.batch_size}
    # Descriptive risk/coverage curve; not a deployment threshold selection procedure.
    result['selective_prediction'] = []
    for threshold in [.25, .5, .6, .7, .8, .9]:
        mask = probs.max(1) >= threshold
        result['selective_prediction'].append({'threshold': threshold, 'coverage': float(mask.mean()),
              'error_rate': float((probs[mask].argmax(1) != y[mask]).mean()) if mask.any() else None})
    a.out.mkdir(parents=True, exist_ok=True)
    write_json(a.out / 'metrics.json', result)
    subjects = {r['clip_id']: r['subject_id'] for r in rows}
    with (a.out / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'subject_id', 'label', 'prediction', 'p0', 'p1', 'p2', 'p3'])
        for clip_id, label, probability in zip(ids, y, probs):
            writer.writerow([clip_id, subjects[clip_id], int(label), int(probability.argmax()), *probability.tolist()])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
