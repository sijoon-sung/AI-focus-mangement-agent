"""Real widget actions and SQLite transitions; no user camera or screen capture."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('PLANWISE_UI_TESTS')=='1','Requires desktop')
class DesktopFlowTests(unittest.TestCase):
    def setUp(self):
        import customtkinter as ctk
        from local_agent.desktop import DesktopApp
        from local_agent.store import Store
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'state.sqlite3'
        self.store=Store(self.path)
        self.store.set_setting('desktop_auto_help',False)
        self.root=ctk.CTk()
        self.app=DesktopApp(self.root,self.store,start_timer=False,check_models=False)
        self.root.update()

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    def plan(self):
        self.app.notes.insert('1.0','보고서 쓰기 1시간\n메일 답장 10분')
        self.app.note_changed()
        self.app.preview(self.app.text())
        self.app.apply_button.invoke()
        self.root.update()

    def test_live_observation_stops_and_discards_pending_screen(self):
        self.plan()
        panel=self.app.open_observation()
        panel.set_enabled('screen',True)
        callbacks=[]
        with patch.object(self.app,'job',side_effect=lambda k,w,c:callbacks.append(c)):
            panel.screen_once()
        self.assertEqual(len(callbacks),1)
        panel.stop()
        callbacks[0](({'visible_text':'old screen','specialists':[],'specialist_errors':[],'source':'qwen_vision_unverified','seconds':1},time.time(),None),None)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM concepts').fetchone()[0],0)
        self.assertFalse(any(panel.gate.enabled.values()))

    def test_live_screen_records_real_result_as_unverified_evidence(self):
        self.plan()
        panel=self.app.open_observation();panel.set_enabled('screen',True)
        callbacks=[]
        with patch.object(self.app,'job',side_effect=lambda k,w,c:callbacks.append(c)):
            panel.screen_once()
        callbacks[0](({'visible_text':'experiment running','uncertainties':'','specialists':[],'specialist_errors':[],'source':'qwen_vision_unverified','seconds':1},time.time(),None),None)
        self.assertTrue(self.store.setting('screen_text_enabled',False))
        self.assertIn('experiment running',panel.output.get('1.0','end'))
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM concepts WHERE modality='screen_text'").fetchone()[0],1)
        self.assertNotEqual(self.store.task(self.app.selected)['state'],'done')

    def test_buttons_drive_complete_pause_resume_and_next(self):
        self.plan()
        first=self.store.active()['task_id']
        self.assertEqual(self.store.task(first)['estimate'],60)
        self.assertEqual(self.app.main_button.cget('text'),'잠깐 쉬기')
        self.app.next_note.insert(0,'결과 표부터 이어 쓰기')
        self.app.main_button.invoke()
        self.assertIsNone(self.store.active())
        self.assertIn('결과 표',self.app.facts.cget('text'))
        self.app.main_button.invoke()
        self.assertEqual(self.store.active()['task_id'],first)
        self.app.update_clock(time.time()+65)
        self.assertTrue(self.app.timer.cget('text').startswith('01:'))
        self.app.done_button.invoke()
        self.assertEqual(self.store.task(first)['state'],'done')
        self.assertIsNone(self.store.active())
        self.app.main_button.invoke()
        self.assertNotEqual(self.app.selected,first)
        self.app.main_button.invoke()
        self.assertEqual(self.store.active()['task_id'],self.app.selected)

    def test_no_start_for_budget_and_stale_preview(self):
        self.app.notes.insert('1.0','보고서 1시간')
        self.app.budget.set('15')
        self.app.preview(self.app.text())
        self.app.apply_button.invoke()
        self.assertIsNone(self.store.active())
        self.assertIn('자동 시작하지',self.app.notice.cget('text'))
        self.app.preview('보고서 1시간')
        self.app.notes.insert('end','\n다른 일')
        self.app.apply_button.invoke()
        self.assertEqual(len(self.store.tasks()),1)

    def test_parser_error_visible_and_stale_result_discarded(self):
        self.app.notes.insert('1.0','과목 A, B')
        self.app.parsed(self.app.text(),None,TimeoutError())
        self.assertIn('응답 시간',self.app.plan_status.cget('text'))
        self.assertIsNone(self.app.draft)
        self.app.parsed('예전 메모',{'outline':'잘못된 할 일'},None)
        self.assertIsNone(self.app.draft)

    def test_idle_pause_excludes_idle_time_and_does_not_auto_resume(self):
        self.plan()
        active=self.store.active()
        self.app.handle_idle(300,active['started']+400)
        self.assertIsNone(self.store.active())
        self.assertAlmostEqual(self.store.elapsed_minutes(self.app.selected),100/60,places=2)
        self.app.handle_idle(1,active['started']+401)
        self.assertIn('돌아오셨네요',self.app.notice.cget('text'))
        self.assertIsNone(self.store.active())

    def test_completed_can_be_undone_without_reopening_another_task(self):
        self.plan()
        first=self.app.selected
        self.app.done_button.invoke()
        self.app.done_button.invoke()
        self.assertEqual(self.store.task(first)['state'],'paused')
        self.assertIsNone(self.store.active())

    def test_help_result_visible_trace_and_stale_state_guard(self):
        self.plan()
        callbacks=[]
        with patch.object(self.app,'job',side_effect=lambda k,w,c:callbacks.append(c)):
            self.app.ask_help()
        result={'suggestion':'결과 표의 첫 행을 확인해 보세요.','action_id':'resume','evidence_ids':['task']}
        callbacks[0](result,None)
        self.assertIn('결과 표',self.app.ai_text.get('1.0','end'))
        self.assertIsNotNone(self.app.ai_trace_id)
        self.app.done_button.invoke()
        callbacks[0](result,None)
        self.assertIn('작업 상태가 바뀌어',self.app.ai_text.get('1.0','end'))

    def test_screen_result_then_help_and_clear_blocks_inflight_result(self):
        from PIL import Image
        self.plan()
        callbacks=[]
        with patch.object(self.app.capture,'capture',return_value=Image.new('RGB',(10,10))), \
             patch.object(self.app,'job',side_effect=lambda k,w,c:callbacks.append(c)):
            self.app.capture_screen({'handle':1})
        with patch.object(self.app,'ask_help') as help_call:
            callbacks[0]({'visible_text':'RUNNING','uncertainties':''},None)
            help_call.assert_called_once()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM concepts').fetchone()[0],1)
        self.app.privacy_revision+=1
        self.store.clear_observations()
        callbacks[0]({'visible_text':'STALE','uncertainties':''},None)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM concepts').fetchone()[0],0)

    def test_close_preserves_notes_checkpoint_and_ends_session(self):
        from local_agent.store import Store
        self.plan()
        task_id=self.app.selected
        self.app.next_note.insert(0,'표 확인')
        self.app.main_button.invoke()
        self.app.main_button.invoke()
        self.app.close()
        reopened=Store(self.path)
        self.assertIsNone(reopened.active())
        self.assertIn('보고서',reopened.setting('inbox_text'))
        self.assertEqual(reopened.task(task_id)['next_action'],'표 확인')
        reopened.close()
