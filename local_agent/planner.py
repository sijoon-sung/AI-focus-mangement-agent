"""Auditable rules and small, explicit-feedback category counts. No hidden LLM reasoning."""
import math
import statistics
import time

VERSION = 'rules-v1'


def recommend(store, budget=60, now=None):
    now = time.time() if now is None else now
    tasks = store.tasks()
    by_id = {t['id']: t for t in tasks}
    active = store.active()
    candidates, excluded = [], []
    for task in tasks:
        reason = None
        if task['state'] == 'done':
            reason = '완료됨'
        elif task['state'] == 'waiting':
            reason = '사용자가 대기로 지정함; 시간이 지나도 완료로 추정하지 않음'
        elif task['dependency'] and by_id[task['dependency']]['state'] != 'done':
            reason = f'선행 작업 #{task["dependency"]} 미완료'
        if reason:
            excluded.append({'task_id': task['id'], 'reason': reason})
            continue
        samples = [store.elapsed_minutes(t['id'], now) for t in tasks
                   if t['state'] == 'done' and t['kind'] == task['kind']]
        samples = [x for x in samples if x >= 1]
        estimate = float(task['estimate'])
        source = '사용자 입력'
        if len(samples) >= 3:
            estimate = .5 * estimate + .5 * statistics.median(samples)
            source = f'사용자 예상 50% + 같은 종류 완료 작업 {len(samples)}개의 세션 시간 중앙값 50%'
        remaining = max(1., estimate - store.elapsed_minutes(task['id'], now))
        terms = [dict(name='사용자 우선순위', value=task['priority'] * 10),
                 dict(name='현재 세션 유지', value=8 if active and active['task_id'] == task['id'] else 0),
                 dict(name='중단 작업 복귀', value=3 if task['state'] == 'paused' else 0),
                 dict(name='가용 시간에 맞음', value=5 if remaining <= budget else -5)]
        candidates.append({'task_id': task['id'], 'title': task['title'],
            'score': sum(x['value'] for x in terms), 'terms': terms,
            'remaining_minutes': round(remaining, 1), 'estimate_source': source,
            'evidence': {'priority': task['priority'], 'state': task['state'],
                         'entered_estimate': task['estimate'], 'history_count': len(samples),
                         'tracked_minutes': round(store.elapsed_minutes(task['id'], now), 1)}})
    candidates.sort(key=lambda x: (-x['score'], x['remaining_minutes'], x['task_id']))
    return {'version': VERSION, 'at': now, 'budget_minutes': budget,
            'candidates': candidates, 'excluded': excluded,
            'limitations': ['점수는 확률이 아닌 우선순위 규칙 합계입니다.',
                           '세션 시간은 실제 집중·진척·생산성 측정값이 아닙니다.',
                           '동점이면 예상 잔여 시간이 짧은 작업, 그 다음 ID 순입니다.']}


def suggest_link(store, category):
    """Dirichlet-smoothed category frequencies from explicit corrections only."""
    eligible = {t['id']: t for t in store.tasks() if t['state'] != 'done'}
    rows = [dict(r) for r in store.db.execute('SELECT * FROM links WHERE category=?', (category,))
            if r['task_id'] in eligible]
    if not rows:
        return None
    rows.sort(key=lambda r: (-r['count'], r['task_id']))
    if rows[0]['count'] < 2 or (len(rows) > 1 and rows[0]['count'] == rows[1]['count']):
        return None
    top = rows[0]
    total = sum(r['count'] for r in rows)
    return {'task_id': top['task_id'], 'title': eligible[top['task_id']]['title'],
            'confirmed_count': top['count'], 'category': category,
            'relative_score': (top['count'] + 1) / (total + len(eligible)),
            'note': '사용자가 확인한 범주 연결 빈도입니다. 현재 작업의 확률이 아니며 자동 전환하지 않습니다.'}


def daily_summary(store, now=None):
    now = time.time() if now is None else now
    local = time.localtime(now)
    start = time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1))
    duration = 0.
    for row in store.db.execute('SELECT started,ended FROM sessions WHERE ended IS NULL OR ended>=?', (start,)):
        duration += max(0, min(now, row['ended'] if row['ended'] is not None else now) - max(start, row['started']))
    done = store.db.execute('SELECT COUNT(*) FROM tasks WHERE completed>=? AND completed<=?', (start, now)).fetchone()[0]
    return {'completed_tasks': done, 'session_minutes': round(duration / 60, 1),
            'note': '완료는 사용자 확인 기준이며 세션 시간은 집중 시간과 다릅니다.'}
