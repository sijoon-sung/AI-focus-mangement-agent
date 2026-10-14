"""Local storage. Only app categories are accepted as passive observations."""
import json
import sqlite3
import time
from pathlib import Path

CATEGORIES = {'editor', 'browser', 'terminal', 'documents', 'meeting', 'other'}
STATES = {'todo', 'doing', 'paused', 'waiting', 'done'}


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA foreign_keys=ON;
        PRAGMA secure_delete=ON;
        CREATE TABLE IF NOT EXISTS tasks (
          id INTEGER PRIMARY KEY, title TEXT NOT NULL, completion TEXT NOT NULL DEFAULT '',
          next_action TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT 'general',
          estimate INTEGER NOT NULL DEFAULT 25, priority INTEGER NOT NULL DEFAULT 2,
          state TEXT NOT NULL DEFAULT 'todo', dependency INTEGER REFERENCES tasks(id),
          wait_until REAL, created REAL NOT NULL, completed REAL);
        CREATE TABLE IF NOT EXISTS sessions (
          id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id),
          started REAL NOT NULL, ended REAL, note TEXT NOT NULL DEFAULT '');
        CREATE UNIQUE INDEX IF NOT EXISTS one_open_session ON sessions((1)) WHERE ended IS NULL;
        CREATE TABLE IF NOT EXISTS observations (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, category TEXT NOT NULL, idle INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS links (
          task_id INTEGER REFERENCES tasks(id), category TEXT NOT NULL, count INTEGER NOT NULL,
          PRIMARY KEY(task_id, category));
        CREATE TABLE IF NOT EXISTS decisions (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, task_id INTEGER REFERENCES tasks(id),
          kind TEXT NOT NULL, trace TEXT NOT NULL, feedback TEXT);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS learning_examples (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, day TEXT NOT NULL,
          kind TEXT NOT NULL, features TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints (
          id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), at REAL NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS task_events (
          id INTEGER PRIMARY KEY, task_id INTEGER REFERENCES tasks(id), at REAL NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS concepts (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, modality TEXT NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS context_labels (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, participant TEXT NOT NULL, episode TEXT NOT NULL,
          label TEXT NOT NULL, data TEXT NOT NULL);
        ''')
        self.db.commit()

    def setting(self, key, default=None):
        row = self.db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (key, json.dumps(value)))

    def tasks(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM tasks ORDER BY priority DESC, created, id')]

    def task(self, task_id):
        row = self.db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        if not row:
            raise ValueError('작업을 찾을 수 없습니다.')
        return dict(row)

    def save_task(self, title, completion='', next_action='', kind='general', estimate=25,
                  priority=2, dependency=None, task_id=None):
        title = title.strip()
        if not title or len(title) > 200:
            raise ValueError('작업 이름은 1~200자로 입력하세요.')
        if not 1 <= int(estimate) <= 10080 or int(priority) not in {1, 2, 3}:
            raise ValueError('예상 시간은 1~10080분, 우선순위는 1~3입니다.')
        if kind not in {'general', 'coding', 'writing', 'research', 'admin'}:
            raise ValueError('알 수 없는 작업 종류입니다.')
        if dependency is not None:
            cursor, visited = dependency, {task_id}
            while cursor is not None:
                if cursor in visited:
                    raise ValueError('선행 작업이 순환합니다.')
                visited.add(cursor)
                cursor = self.task(cursor)['dependency']
        values = (title, completion[:2000], next_action[:2000], kind, int(estimate), int(priority), dependency)
        with self.db:
            if task_id is None:
                result = self.db.execute('''INSERT INTO tasks
                  (title,completion,next_action,kind,estimate,priority,dependency,created)
                  VALUES (?,?,?,?,?,?,?,?)''', values + (time.time(),))
                return result.lastrowid
            self.task(task_id)
            self.db.execute('''UPDATE tasks SET title=?,completion=?,next_action=?,kind=?,
              estimate=?,priority=?,dependency=? WHERE id=?''', values + (task_id,))
            return task_id

    def active(self):
        row = self.db.execute('SELECT * FROM sessions WHERE ended IS NULL').fetchone()
        return dict(row) if row else None

    def _end_active(self, now, note=''):
        active = self.active()
        if active:
            self.db.execute('UPDATE sessions SET ended=?,note=? WHERE id=?',
                            (now, note[:2000], active['id']))
            self.db.execute("UPDATE tasks SET state='paused' WHERE id=?", (active['task_id'],))

    def start(self, task_id, now=None):
        now = time.time() if now is None else now
        task = self.task(task_id)
        if task['state'] == 'done':
            raise ValueError('완료된 작업입니다.')
        if task['state'] == 'waiting':
            raise ValueError('대기를 먼저 해제하세요.')
        if task['dependency'] and self.task(task['dependency'])['state'] != 'done':
            raise ValueError('선행 작업이 아직 완료되지 않았습니다.')
        if self.active() and self.active()['task_id'] == task_id:
            return
        with self.db:
            self._end_active(now, '다른 작업으로 전환')
            self.db.execute('INSERT INTO sessions(task_id,started) VALUES (?,?)', (task_id, now))
            self.db.execute("UPDATE tasks SET state='doing' WHERE id=?", (task_id,))

    def pause(self, note='', next_action=None, now=None):
        active = self.active()
        if active:
            with self.db:
                self._end_active(time.time() if now is None else now, note)
                if next_action is not None:
                    self.db.execute('UPDATE tasks SET next_action=? WHERE id=?',
                                    (next_action[:2000], active['task_id']))

    def change_state(self, task_id, state, wait_minutes=None, now=None):
        if state not in STATES - {'doing'}:
            raise ValueError('지원하지 않는 상태입니다.')
        now = time.time() if now is None else now
        self.task(task_id)
        if wait_minutes is not None and not 1 <= wait_minutes <= 10080:
            raise ValueError('대기 시간은 1~10080분입니다.')
        with self.db:
            active = self.active()
            if active and active['task_id'] == task_id:
                self._end_active(now)
            self.db.execute('UPDATE tasks SET state=?,wait_until=?,completed=? WHERE id=?',
                            (state, now + wait_minutes * 60 if state == 'waiting' and wait_minutes else None,
                             now if state == 'done' else None, task_id))

    def elapsed_minutes(self, task_id, now=None):
        now = time.time() if now is None else now
        return sum(max(0, (r['ended'] if r['ended'] is not None else now) - r['started']) / 60
                   for r in self.db.execute('SELECT started,ended FROM sessions WHERE task_id=?', (task_id,)))

    def return_note(self, task_id):
        row = self.db.execute("SELECT note FROM sessions WHERE task_id=? AND note<>'' ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
        return row[0] if row else ''

    def observe(self, category, idle, at=None):
        if category not in CATEGORIES:
            raise ValueError('원본 앱명이나 화면 내용은 관측 저장 대상이 아닙니다.')
        with self.db:
            self.db.execute('INSERT INTO observations(at,category,idle) VALUES (?,?,?)',
                            (time.time() if at is None else at, category, int(bool(idle))))

    def learn_link(self, task_id, category):
        self.task(task_id)
        if category not in CATEGORIES:
            raise ValueError('유효한 앱 범주가 필요합니다.')
        with self.db:
            self.db.execute('''INSERT INTO links VALUES (?,?,1) ON CONFLICT(task_id,category)
              DO UPDATE SET count=count+1''', (task_id, category))

    def trace(self, kind, task_id, evidence, now=None):
        with self.db:
            return self.db.execute('INSERT INTO decisions(at,task_id,kind,trace) VALUES (?,?,?,?)',
                (time.time() if now is None else now, task_id, kind,
                 json.dumps(evidence, ensure_ascii=False))).lastrowid

    def feedback(self, decision_id, value):
        if value not in {'helpful', 'not_helpful', 'wrong_context'}:
            raise ValueError('알 수 없는 피드백입니다.')
        with self.db:
            self.db.execute('UPDATE decisions SET feedback=? WHERE id=?', (value, decision_id))

    def prune(self, now=None):
        days = max(1, min(365, int(self.setting('retention_days', 7))))
        cutoff = (time.time() if now is None else now) - days * 86400
        with self.db:
            self.db.execute('DELETE FROM observations WHERE at < ?', (cutoff,))
            self.db.execute('DELETE FROM decisions WHERE at < ?', (cutoff,))
            self.db.execute('DELETE FROM learning_examples WHERE at < ?', (cutoff,))
            self.db.execute('DELETE FROM concepts WHERE at < ?', (cutoff,))
            self.db.execute('DELETE FROM context_labels WHERE at < ?', (cutoff,))

    def clear_observations(self):
        with self.db:
            self.db.execute('DELETE FROM observations')
            self.db.execute('DELETE FROM links')
            self.db.execute('DELETE FROM decisions')
            self.db.execute('DELETE FROM learning_examples')
            self.db.execute('DELETE FROM concepts')
            self.db.execute('DELETE FROM context_labels')

    def reset(self):
        with self.db:
            for table in ['context_labels','concepts','checkpoints','task_events','learning_examples', 'decisions', 'observations', 'links', 'sessions', 'tasks', 'settings']:
                self.db.execute(f'DELETE FROM {table}')
        self.db.execute('VACUUM')

    def close(self):
        self.db.close()
