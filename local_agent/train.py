"""Run python -m local_agent.train --help. No third-party dependencies."""
import argparse
import json
from pathlib import Path
from local_agent.learning import FEATURES, KINDS, fit, temporal_split, evaluate
from local_agent.store import Store


def synthetic():
    # Deliberately separable artificial fixture, NOT a real-world benchmark.
    rows = []
    for day in range(5):
        for i in range(50):
            k = i % len(KINDS)
            x = [0.] * len(FEATURES)
            x[k] = x[k+6] = 1.
            x[-2] = 1.
            rows.append(dict(at=day*86400+i*600, day=f'2000-01-{day+1:02d}', kind=KINDS[k], features=x))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, default=Path('local_data/planwise.sqlite3'))
    p.add_argument('--output', type=Path, default=Path('runs/activity-model.json'))
    p.add_argument('--epochs', type=int, default=180)
    p.add_argument('--smoke', action='store_true', help='synthetic plumbing test; app refuses this model')
    args = p.parse_args()
    if args.smoke:
        rows = synthetic()
    else:
        if not args.db.is_file():
            p.error('DB가 없습니다. 앱에서 실제 사용자 확인 데이터를 먼저 수집하세요.')
        store = Store(args.db)
        try:
            store.prune()
            rows = [dict(at=r['at'], day=r['day'], kind=r['kind'], features=json.loads(r['features']))
                    for r in store.db.execute('SELECT * FROM learning_examples ORDER BY at')]
        finally:
            store.close()
    try:
        train, test = temporal_split(rows)
        model = fit(train, args.epochs)
    except ValueError as exc:
        p.error(str(exc))
    model['synthetic'] = args.smoke
    model['training_count'] = len(train)
    model['evaluation'] = evaluate(model, test, train)
    model['split'] = dict(train_days=sorted({r['day'] for r in train}), test_days=sorted({r['day'] for r in test}))
    # Export the exact model evaluated on the untouched future split. No refit.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(synthetic=args.smoke, train=len(train), test=len(test), metrics=model['evaluation'], output=str(args.output)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
