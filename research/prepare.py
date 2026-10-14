"""Build a manifest from an unmodified DAiSEE distribution."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from research.data import FIELDS, read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='DAiSEE folder containing DataSet and Labels')
    p.add_argument('--out', type=Path, default=Path('data/daisee.csv'))
    a = p.parse_args()
    rows = []
    for folder, split in [('Train', 'train'), ('Validation', 'val'), ('Test', 'test')]:
        label_file = a.root / 'Labels' / f'{folder}Labels.csv'
        with label_file.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            labels = {}
            for raw in reader:
                r = {k.strip(): v.strip() for k, v in raw.items()}
                key = Path(r['ClipID']).stem
                if key in labels:
                    raise ValueError(f'Duplicate label: {key}')
                labels[key] = int(r['Engagement'])
        split_root = a.root / 'DataSet' / folder
        videos = sorted(x for x in split_root.rglob('*') if x.suffix.lower() in {'.avi', '.mp4', '.mov'})
        found = set()
        for video in videos:
            rel = video.relative_to(split_root)
            if len(rel.parts) < 3:
                raise ValueError(f'Expected split/subject/clip/video: {video}')
            key = video.stem
            if key not in labels:
                raise ValueError(f'No engagement label: {video}')
            found.add(key)
            rows.append(dict(clip_id=key, subject_id=rel.parts[0], split=split,
                             path=str(video.resolve()), label=labels[key]))
        missing = set(labels) - found
        if missing:
            raise ValueError(f'{split}: {len(missing)} labelled videos missing; examples {sorted(missing)[:5]}. Do not silently drop.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Validate before publishing the final manifest.
    temp = a.out.with_suffix('.pending.csv')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    checked = read_manifest(temp)
    temp.replace(a.out)
    report = {s: {'clips': sum(r['split'] == s for r in checked),
                  'subjects': len({r['subject_id'] for r in checked if r['split'] == s}),
                  'labels': dict(Counter(r['label'] for r in checked if r['split'] == s))}
              for s in ['train', 'val', 'test']}
    a.out.with_suffix('.audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
