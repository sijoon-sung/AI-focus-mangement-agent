"""Single GPU/CPU training. Validation selects checkpoints; test is never scored here."""
import argparse
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from research.data import Clips, read_manifest
from research.metrics import classification_metrics
from research.models import EngagementModel


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def collect(model, loader, device):
    model.eval()
    logits, labels, ids = [], [], []
    with torch.inference_mode():
        for x, y, clip_ids in loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
            ids.extend(clip_ids)
    return torch.cat(logits), torch.cat(labels), ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--model', choices=['tiny_mean', 'tiny_tcn', 'resnet18_tcn'], default='tiny_tcn')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--size', type=int, default=112)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--weighted-loss', action='store_true')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.frames, a.size) < 1 or a.workers < 0 or a.lr <= 0:
        p.error('Positive epochs/batch/frames/size/lr and nonnegative workers required')
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Output directory must be empty; use a fresh run name')
    rows = read_manifest(a.manifest)
    counts = np.bincount([r['label'] for r in rows if r['split'] == 'train'], minlength=4)
    if (counts == 0).any():
        raise ValueError(f'Train must include every class: {counts}')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a.out.mkdir(parents=True, exist_ok=True)
    config = vars(a).copy()
    config['out'] = str(a.out)
    config['manifest'] = str(Path(a.manifest).resolve())
    config['manifest_sha256'] = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    # Portable identity of the split; absolute paths need not match across machines.
    split = [{k: r[k] for k in ['clip_id', 'subject_id', 'split', 'label']} for r in rows]
    config['split_sha256'] = hashlib.sha256(json.dumps(sorted(split, key=lambda r: r['clip_id']), sort_keys=True).encode()).hexdigest()
    write_json(a.out / 'config.json', config)
    write_json(a.out / 'split.json', split)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    write_json(a.out / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
               'cuda_runtime': torch.version.cuda, 'device': a.device, 'git_commit': commit,
               'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    (a.out / 'packages.txt').write_text(subprocess.check_output(
        [__import__('sys').executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    gen = torch.Generator().manual_seed(a.seed)
    train = DataLoader(Clips(rows, 'train', a.frames, a.size), batch_size=a.batch_size,
                       shuffle=True, num_workers=a.workers, worker_init_fn=seed_worker, generator=gen)
    val = DataLoader(Clips(rows, 'val', a.frames, a.size), batch_size=a.batch_size, num_workers=a.workers)
    model = EngagementModel(a.model, a.pretrained).to(a.device)
    weights = torch.tensor(counts.sum() / (4 * counts), dtype=torch.float32, device=a.device) if a.weighted_loss else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best = -1.
    for epoch in range(a.epochs):
        model.train()
        total, n = 0., 0
        for x, y, _ in train:
            x, y = x.to(a.device), y.to(a.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        logits, labels, _ = collect(model, val, a.device)
        metrics = classification_metrics(labels.numpy(), logits.softmax(1).numpy())
        record = {'epoch': epoch + 1, 'train_loss_batch_average': total / n, 'val': metrics}
        with (a.out / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        checkpoint = {'state_dict': model.state_dict(), 'config': config, 'epoch': epoch + 1,
                      'val_macro_f1': metrics['macro_f1']}
        torch.save(checkpoint, a.out / 'last.pt')
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            torch.save(checkpoint, a.out / 'best.pt')
        print(f'epoch={epoch+1} loss={total/n:.4f} val_macro_f1={metrics["macro_f1"]:.4f}', flush=True)
    print(f'Finished. Best validation macro-F1={best:.4f}; test not evaluated.')


if __name__ == '__main__':
    main()
