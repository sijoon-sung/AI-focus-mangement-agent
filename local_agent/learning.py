"""Small supervised activity-kind model; stdlib training and exact logit explanations.

This predicts a user-confirmed work kind, never concentration or productivity.
No text, task IDs, executable names or screenshots are used as model features.
"""
import json
import math
import random
import time

CATEGORIES = ('editor', 'browser', 'terminal', 'documents', 'meeting', 'other')
KINDS = ('general', 'coding', 'writing', 'research', 'admin')
FEATURES = tuple('current_' + c for c in CATEGORIES) + tuple('recent_' + c for c in CATEGORIES) + ('idle_fraction', 'switch_fraction', 'coverage', 'age_fraction')
VERSION = 'activity-kind-v1'


def context(store, now=None):
    now = time.time() if now is None else now
    rows = list(store.db.execute('SELECT at,category,idle FROM observations WHERE at>=? AND at<=? ORDER BY at,id', (now - 300, now)))
    excluded = store.setting('excluded_categories', ['meeting'])
    rows = [r for r in rows if r['category'] not in excluded]
    if not rows or now - rows[-1]['at'] > 60:
        return None
    n = len(rows)
    current = rows[-1]['category']
    values = [float(current == c) for c in CATEGORIES]
    values += [sum(r['category'] == c for r in rows) / n for c in CATEGORIES]
    values += [sum(r['idle'] for r in rows) / n,
               sum(a['category'] != b['category'] for a, b in zip(rows, rows[1:])) / max(1, n-1),
               min(1., n / 20), min(1., (now-rows[-1]['at'])/60)]
    return values


def capture(store, task_id, now=None):
    """Called ONLY after the user explicitly confirms the observed activity."""
    now = time.time() if now is None else now
    values = context(store, now)
    if values is None:
        return False
    last = store.db.execute('SELECT MAX(at) FROM learning_examples').fetchone()[0]
    if last is not None and now - last < 300:
        return False  # one label per non-overlapping five-minute window
    kind = store.task(task_id)['kind']
    with store.db:
        store.db.execute('INSERT INTO learning_examples(at,day,kind,features) VALUES (?,?,?,?)',
                         (now, time.strftime('%Y-%m-%d', time.localtime(now)), kind, json.dumps(values)))
    return True


def probabilities(logits):
    exps = [math.exp(v-max(logits)) for v in logits]
    total = sum(exps)
    return [v/total for v in exps]


def validate(model):
    if model.get('version') != VERSION or model.get('features') != list(FEATURES) or model.get('kinds') != list(KINDS):
        raise ValueError('지원하지 않는 모델 스키마입니다.')
    weights, bias = model.get('weights'), model.get('bias')
    if not isinstance(weights, list) or len(weights) != len(KINDS) or not isinstance(bias, list) or len(bias) != len(KINDS):
        raise ValueError('모델 차원이 잘못되었습니다.')
    for row in weights:
        if not isinstance(row, list) or len(row) != len(FEATURES):
            raise ValueError('모델 특징 차원이 잘못되었습니다.')
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v)>1e6 for v in bias + [v for row in weights for v in row]):
        raise ValueError('유효하지 않은 모델 가중치입니다.')


def infer(model, x):
    validate(model)
    if len(x) != len(FEATURES) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in x):
        raise ValueError('유효하지 않은 특징입니다.')
    scores = [b + sum(w*v for w, v in zip(row, x)) for row, b in zip(model['weights'], model['bias'])]
    p = probabilities(scores)
    order = sorted(range(len(KINDS)), key=lambda i: -scores[i])
    top, runner = order[:2]
    terms = [{'feature': name, 'input': v, 'contribution': model['weights'][top][j]*v,
              'contrast': (model['weights'][top][j]-model['weights'][runner][j])*v}
             for j, (name, v) in enumerate(zip(FEATURES, x))]
    return {'kind': KINDS[top], 'runner_up': KINDS[runner], 'score': scores[top],
            'bias': model['bias'][top], 'terms': terms,
            'contrast_bias': model['bias'][top]-model['bias'][runner],
            'margin': scores[top]-scores[runner], 'softmax': p[top],
            'abstain': p[top] < .6 or p[top]-p[runner] < .15,
            'note': 'softmax는 검증된 정확도나 집중도 확률이 아닙니다. 기여도는 점수 계산 설명이며 인과 설명이 아닙니다.'}


def fit(rows, epochs=180, seed=7):
    if not rows or epochs < 1:
        raise ValueError('학습 데이터와 양의 epoch가 필요합니다.')
    model = dict(version=VERSION, features=list(FEATURES), kinds=list(KINDS),
                 weights=[[0.] * len(FEATURES) for _ in KINDS], bias=[0.] * len(KINDS))
    rng = random.Random(seed)
    data = list(rows)
    for epoch in range(epochs):
        rng.shuffle(data)
        rate = .08 / (1 + epoch/60)
        for row in data:
            x, y = row['features'], KINDS.index(row['kind'])
            if len(x) != len(FEATURES) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in x):
                raise ValueError('학습 특징이 잘못되었습니다.')
            p = probabilities([b+sum(w*v for w,v in zip(ws,x)) for ws,b in zip(model['weights'],model['bias'])])
            for k in range(len(KINDS)):
                error = p[k] - float(k == y)
                model['bias'][k] -= rate * error
                for j in range(len(x)):
                    model['weights'][k][j] -= rate * (error*x[j] + .001*model['weights'][k][j])
    return model


def temporal_split(rows):
    days = sorted({r['day'] for r in rows})
    if len(days) < 3:
        raise ValueError('서로 다른 3일 이상의 사용자 확인 기록이 필요합니다.')
    cut = max(1, min(len(days)-1, int(len(days)*.8)))
    train = [r for r in rows if r['day'] < days[cut]]
    # Purge five minutes at the boundary to prevent overlapping feature windows.
    boundary = min(r['at'] for r in rows if r['day'] >= days[cut])
    train = [r for r in train if r['at'] < boundary-300]
    test = [r for r in rows if r['day'] >= days[cut]]
    if len(train) < 30 or len({r['kind'] for r in train}) < 2 or len(test) < 10:
        raise ValueError('학습 30개/2종류 이상, 시간 분리 평가 10개 이상이 필요합니다. 이는 실행 최소 조건이지 충분한 성능의 보장이 아닙니다.')
    return train, test


def evaluate(model, rows, training):
    majority = max(KINDS, key=lambda k: sum(r['kind']==k for r in training))
    matrix = [[0]*len(KINDS) for _ in KINDS]
    accepted = correct_accepted = correct = baseline = 0
    for row in rows:
        pred = infer(model, row['features'])
        truth = KINDS.index(row['kind'])
        matrix[truth][KINDS.index(pred['kind'])] += 1
        correct += pred['kind'] == row['kind']
        baseline += majority == row['kind']
        if not pred['abstain']:
            accepted += 1
            correct_accepted += pred['kind'] == row['kind']
    f1 = []
    for k in range(len(KINDS)):
        denom = sum(matrix[k])+sum(r[k] for r in matrix)
        f1.append(2*matrix[k][k]/denom if denom else 0.)
    return dict(n=len(rows), accuracy=correct/len(rows), macro_f1_all_five=sum(f1)/len(f1),
                majority_accuracy=baseline/len(rows), coverage=accepted/len(rows),
                accepted_accuracy=correct_accepted/accepted if accepted else None,
                confusion_matrix=matrix, class_order=list(KINDS),
                unseen_test_kinds=sorted({r['kind'] for r in rows}-{r['kind'] for r in training}))
