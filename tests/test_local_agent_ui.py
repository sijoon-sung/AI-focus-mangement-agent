"""Opt-in desktop integration tests; never collect actual user activity."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('PLANWISE_UI_TESTS') == '1', 'Requires desktop; set PLANWISE_UI_TESTS=1')
class DesktopTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from local_agent.app import AgentApp, configure_dpi
        from local_agent.store import Store
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'test.sqlite3')
        self.task = self.store.save_task('실험 보고서', '표 완성', '지표 추출')
        configure_dpi()
        self.root = tk.Tk()
        self.app = AgentApp(self.root, self.store, start_timer=False)
        self.root.update()
        self.app.tree.selection_set(str(self.task))

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def test_session_pause_complete_and_explanation(self):
        self.app.start_task()
        self.assertEqual(self.store.active()['task_id'], self.task)
        with patch('local_agent.app.simpledialog.askstring', side_effect=['표 작성함', '그래프 그리기']):
            self.app.pause_task()
        self.assertIsNone(self.store.active())
        self.assertEqual(self.store.return_note(self.task), '표 작성함')
        self.app.show_recommendation()
        self.assertIn('중단 작업 복귀 +3', self.app.explanation.get('1.0', 'end'))
        with patch('local_agent.app.messagebox.askyesno', return_value=True):
            self.app.complete_task()
        self.assertEqual(self.store.task(self.task)['state'], 'done')

    def test_timer_notification_preserves_waiting(self):
        self.store.change_state(self.task, 'waiting', wait_minutes=1, now=time.time()-120)
        with patch.object(self.app.sensor, 'sample', side_effect=AssertionError('capture must remain off')):
            self.app.tick()
        self.assertIn('대기 확인 시간', self.app.banner.get())
        self.assertEqual(self.store.task(self.task)['state'], 'waiting')
        self.assertIsNone(self.store.task(self.task)['wait_until'])

    def test_opt_in_category_only_and_exclusion(self):
        self.app.collection.set(True)
        self.app.change_collection()
        with patch.object(self.app.sensor, 'sample', return_value={'category': 'meeting', 'idle': False}):
            self.app.tick()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 0)
        self.root.after_cancel(self.app.tick_id)
        with patch.object(self.app.sensor, 'sample', return_value={'category': 'editor', 'idle': False}):
            self.app.tick()
        self.assertEqual(self.store.db.execute('SELECT category FROM observations').fetchone()[0], 'editor')
        self.app.collection.set(False)
        self.app.change_collection()
        self.assertIsNone(self.app.latest)


if __name__ == '__main__':
    unittest.main()
