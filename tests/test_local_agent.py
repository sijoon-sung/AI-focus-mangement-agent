import json
import tempfile
import unittest
from pathlib import Path

from local_agent.store import Store
from local_agent.planner import recommend, suggest_link, daily_summary
from local_agent.sensors import category_for


class LocalAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'test.sqlite3')

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def task(self, title='A', **kwargs):
        return self.store.save_task(title, **kwargs)

    def test_dependency_blocks_start_and_recommendation(self):
        first = self.task()
        second = self.task('B', dependency=first, priority=3)
        with self.assertRaises(ValueError):
            self.store.start(second, now=100)
        result = recommend(self.store, now=100)
        self.assertEqual([r['task_id'] for r in result['candidates']], [first])
        self.store.change_state(first, 'done', now=101)
        self.store.start(second, now=102)
        self.assertEqual(self.store.active()['task_id'], second)

    def test_dependency_cycle_rejected(self):
        first = self.task()
        second = self.task('B', dependency=first)
        with self.assertRaisesRegex(ValueError, '순환'):
            self.store.save_task('A', dependency=second, task_id=first)

    def test_switch_has_one_active_session_and_preserves_time(self):
        first, second = self.task(), self.task('B')
        self.store.start(first, now=100)
        self.store.start(second, now=220)
        self.assertEqual(self.store.elapsed_minutes(first, now=300), 2.)
        self.assertEqual(self.store.task(first)['state'], 'paused')
        self.assertEqual(self.store.active()['task_id'], second)

    def test_failed_start_does_not_end_active_session(self):
        first = self.task()
        second = self.task('B', dependency=first)
        self.store.start(first, now=100)
        with self.assertRaises(ValueError):
            self.store.start(second, now=200)
        self.assertEqual(self.store.active()['task_id'], first)

    def test_pause_preserves_return_note_and_next_action(self):
        task = self.task()
        self.store.start(task, now=100)
        self.store.pause('표 1 작성 완료', '그림 작성', now=160)
        self.assertIsNone(self.store.active())
        self.assertEqual(self.store.return_note(task), '표 1 작성 완료')
        self.assertEqual(self.store.task(task)['next_action'], '그림 작성')

    def test_wait_timeout_does_not_mean_task_completed(self):
        task = self.task()
        self.store.change_state(task, 'waiting', wait_minutes=1, now=100)
        self.assertEqual(recommend(self.store, now=1000)['candidates'], [])
        self.assertEqual(self.store.task(task)['state'], 'waiting')

    def test_exact_score_explanation_and_trace(self):
        task = self.task(priority=3, estimate=20)
        result = recommend(self.store, budget=30, now=100)
        candidate = result['candidates'][0]
        self.assertEqual(candidate['score'], 35)
        self.assertEqual(candidate['score'], sum(t['value'] for t in candidate['terms']))
        trace_id = self.store.trace('next_task', task, result, now=100)
        stored = json.loads(self.store.db.execute('SELECT trace FROM decisions WHERE id=?', (trace_id,)).fetchone()[0])
        self.assertEqual(result, stored)
        self.store.feedback(trace_id, 'wrong_context')
        self.assertEqual(self.store.db.execute('SELECT feedback FROM decisions').fetchone()[0], 'wrong_context')

    def test_personal_duration_only_after_three_completed_samples(self):
        for index in range(3):
            task = self.task(str(index), kind='writing')
            self.store.start(task, now=100)
            self.store.change_state(task, 'done', now=1300)
        target = self.task('new', kind='writing', estimate=40)
        item = recommend(self.store, now=1400)['candidates'][0]
        self.assertEqual(item['task_id'], target)
        self.assertEqual(item['remaining_minutes'], 30.)
        self.assertEqual(item['evidence']['history_count'], 3)

    def test_category_links_explicit_only_and_ties_abstain(self):
        first, second = self.task(), self.task('B')
        self.store.observe('browser', False)
        self.assertIsNone(suggest_link(self.store, 'browser'))
        self.store.learn_link(first, 'browser')
        self.assertIsNone(suggest_link(self.store, 'browser'))
        self.store.learn_link(first, 'browser')
        self.assertEqual(suggest_link(self.store, 'browser')['task_id'], first)
        self.store.learn_link(second, 'browser')
        self.store.learn_link(second, 'browser')
        self.assertIsNone(suggest_link(self.store, 'browser'))

    def test_raw_sensitive_values_rejected(self):
        with self.assertRaises(ValueError):
            self.store.observe('private-url-or-window-title', False)
        self.assertFalse(self.store.setting('collection_enabled', False))

    def test_retention_does_not_delete_plan(self):
        task = self.task()
        self.store.observe('editor', False, at=0)
        self.store.trace('test', task, {}, now=0)
        self.store.observe('browser', False, at=10*86400)
        self.store.prune(now=10*86400)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0], 0)
        self.assertEqual(len(self.store.tasks()), 1)

    def test_clear_personalization_and_reset(self):
        first = self.task()
        self.task('B', dependency=first)
        self.store.learn_link(first, 'editor')
        self.store.start(first, now=100)
        self.store.clear_observations()
        self.assertEqual(len(self.store.tasks()), 2)
        self.store.reset()
        self.assertEqual(self.store.tasks(), [])
        self.assertIsNone(self.store.active())

    def test_category_mapping_uses_no_window_content(self):
        self.assertEqual(category_for('C:\\Program Files\\Code.exe'), 'editor')
        self.assertEqual(category_for('unknown.exe'), 'other')


if __name__ == '__main__':
    unittest.main()
