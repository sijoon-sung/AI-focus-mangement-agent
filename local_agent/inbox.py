"""Local scratchpad -> reviewable task drafts and conservative time allocations."""
import difflib
import json
import math
import re
import statistics
from local_agent.note_parser import duration

KINDS = ('general', 'coding', 'writing', 'research', 'admin')


def guess_kind(title):
    for kind, words in [('coding', ('코드', '개발', '구현', '버그', '테스트', '학습 실행')),
                        ('writing', ('작성', '보고서', '초안', '글쓰기', '발표')),
                        ('research', ('논문', '조사', '실험', '분석', '자료')),
                        ('admin', ('메일', '신청', '예약', '정산'))]:
        if any(w in title for w in words):
            return kind
    return 'general'


def parse_notes(text):
    if len(text) > 20000:
        raise ValueError('메모는 20,000자 이내로 나누어 주세요.')
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        line = re.sub(r'^\s*(?:[-*•]\s*|\d+[.)]\s*)', '', line)
        if re.match(r'^\[[xX]\]', line):
            continue
        line = re.sub(r'^\[ \]\s*', '', line)
        title, minutes = duration(line)
        priority = 3 if title.startswith('!') else 2
        title = title.lstrip('!').strip()
        if not title or len(title) > 200 or (minutes is not None and not 1 <= minutes <= 10080):
            raise ValueError('한 줄은 1~200자, [예상 분]은 1~10080분이어야 합니다.')
        if title in [i['title'] for i in items]:
            raise ValueError('동일한 작업명이 있습니다. 구분할 내용을 덧붙여 주세요: ' + title)
        items.append(dict(title=title, kind=guess_kind(title), priority=priority, explicit_minutes=minutes))
    if len(items) > 100:
        raise ValueError('한 번에 최대 100개 작업을 정리합니다.')
    return items



def estimate(store, item, task_id=None):
    samples = []
    for t in store.tasks():
        if t['state'] != 'done' or t['id'] == task_id:
            continue
        duration = store.elapsed_minutes(t['id'])
        similarity = difflib.SequenceMatcher(None, item['title'], t['title']).ratio()
        if duration >= 1 and similarity >= .6:
            samples.append(duration)
    source = '비슷한 제목'
    if len(samples) < 3 and item['kind'] != 'general':
        samples = [store.elapsed_minutes(t['id']) for t in store.tasks()
                   if t['state']=='done' and t['kind']==item['kind'] and t['id'] != task_id and store.elapsed_minutes(t['id'])>=1]
        source = '같은 종류 (작업 규모는 다를 수 있음)'
    explicit = item.get('explicit_minutes')
    if explicit is not None:
        total, reason = explicit, '메모에 직접 입력'
    elif len(samples) >= 3:
        total, reason = statistics.median(samples), f'{source} 완료 {len(samples)}개 세션 시간 중앙값'
    elif task_id is not None:
        total, reason = store.task(task_id)['estimate'], '기존 계획 예상 · 충분한 이력 없음'
    else:
        total, reason = 25, '임시 25분 · 충분한 이력 없음'
    spent = store.elapsed_minutes(task_id) if task_id is not None else 0
    return dict(total=round(total), remaining=max(1, math.ceil(total-spent)), source=reason,
                sample_count=len(samples), observed_range=[round(min(samples)), round(max(samples))] if len(samples)>=3 else None)


def build_draft(store, text, budget=240):
    if not 1 <= budget <= 1440:
        raise ValueError('오늘 가용 시간은 1~1440분입니다.')
    items = parse_notes(text)
    old = store.setting('inbox_mapping', [])
    live = {t['id']: t for t in store.tasks()}
    old = [r for r in old if r['task_id'] in live]
    known = {r['title']: r for r in store.setting('inbox_known', []) if r['task_id'] in live}
    known.update({r['title']: r for r in old})
    assigned = {}
    # Exact matches survive reordered lines. A single replacement block can be
    # reviewed as an edit; ambiguous multi-line replacements become new tasks.
    for j, item in enumerate(items):
        match = known.get(item['title'])
        if match and match['task_id'] not in assigned.values():
            assigned[j] = match['task_id']
    matcher = difflib.SequenceMatcher(None, [r['title'] for r in old], [r['title'] for r in items])
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag=='replace' and b-a==d-c==1 and c not in assigned and old[a]['task_id'] not in assigned.values():
            task = live[old[a]['task_id']]
            if task['state'] != 'done' and difflib.SequenceMatcher(None, old[a]['title'], items[c]['title']).ratio() >= .6:
                assigned[c] = task['id']
    elapsed = 0
    for j, item in enumerate(items):
        task_id = assigned.get(j)
        task = live.get(task_id)
        item.update(task_id=task_id, operation='수정' if task and item['title']!=task['title'] else '유지' if task else '추가',
                    previous_title=task['title'] if task else None)
        if task and item['operation']=='유지':
            item['kind'] = task['kind']
        item['estimate'] = estimate(store, item, task_id)
        item['blocked'] = ('완료됨' if task and task['state']=='done' else '대기 중' if task and task['state']=='waiting'
                           else '선행 작업 미완료' if task and task['dependency'] and live[task['dependency']]['state']!='done' else None)
    for item in sorted(items, key=lambda r: -r['priority']):
        duration = item['estimate']['remaining']
        reserve = max(5, math.ceil(duration*.15))
        item.update(start=None, end=None, reserve=reserve)
        if not item['blocked'] and elapsed + duration + reserve <= budget:
            item.update(start=elapsed, end=elapsed+duration)
            elapsed += duration + reserve
    return dict(text=text, items=items, budget=budget, allocated=elapsed,
                removed=[r for r in old if r['task_id'] not in assigned.values()])


def apply_draft(store, draft):
    # Recompute using current DB before mutation so stale dependencies/history
    # are not accepted from a previously rendered preview.
    current = build_draft(store, draft['text'], draft['budget'])
    mapping = []
    # One transaction: do not call Store helpers which commit nested contexts.
    import time
    with store.db:
        for item in current['items']:
            task_id = item['task_id']
            if task_id is None:
                task_id = store.db.execute('INSERT INTO tasks(title,kind,estimate,priority,created) VALUES (?,?,?,?,?)',
                    (item['title'],item['kind'],item['estimate']['total'],item['priority'],time.time())).lastrowid
            elif store.task(task_id)['state'] != 'done':
                store.db.execute('UPDATE tasks SET title=?,kind=?,estimate=?,priority=? WHERE id=?',
                    (item['title'],item['kind'],item['estimate']['total'],item['priority'],task_id))
            mapping.append(dict(title=item['title'], task_id=task_id))
        store.db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', ('inbox_mapping', json.dumps(mapping, ensure_ascii=False)))
        known = {r['title']: r for r in store.setting('inbox_known', [])}
        known.update({r['title']: r for r in mapping})
        store.db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', ('inbox_known', json.dumps(list(known.values()), ensure_ascii=False)))
    return mapping
