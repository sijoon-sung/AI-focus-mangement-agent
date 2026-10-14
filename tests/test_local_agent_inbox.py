import tempfile
import unittest
from pathlib import Path
from local_agent.store import Store
from local_agent.inbox import parse_notes, build_draft, apply_draft, estimate


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'test.sqlite3')

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_parse_priority_time_and_completed_notes(self):
        rows = parse_notes('# 오늘\n- ! 논문 읽기 [40분]\n[x] 끝난 일\n1. 보고서 작성')
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['priority'],3)
        self.assertEqual(rows[0]['explicit_minutes'],40)
        with self.assertRaises(ValueError):
            parse_notes('같은 일\n같은 일')

    def test_repeat_rename_and_remove_keep_history(self):
        mapping = apply_draft(self.store,build_draft(self.store,'논문 읽기\n보고서 작성'))
        task = mapping[0]['task_id']
        self.store.start(task,now=100)
        self.store.pause('첫 장 읽음',now=700)
        apply_draft(self.store,build_draft(self.store,'논문 읽기\n보고서 작성'))
        self.assertEqual(len(self.store.tasks()),2)
        draft = build_draft(self.store,'논문 방법론 읽기\n보고서 작성')
        self.assertEqual(draft['items'][0]['operation'],'수정')
        apply_draft(self.store,draft)
        self.assertEqual(self.store.task(task)['title'],'논문 방법론 읽기')
        self.assertEqual(self.store.elapsed_minutes(task),10)
        apply_draft(self.store,build_draft(self.store,'보고서 작성'))
        self.assertEqual(len(self.store.tasks()),2)
        again = build_draft(self.store,'논문 방법론 읽기\n보고서 작성')
        self.assertEqual(again['items'][0]['task_id'],task)

    def test_completed_does_not_reopen_and_ambiguous_edit_does_not_reassign(self):
        mapping = apply_draft(self.store,build_draft(self.store,'A\nB'))
        self.store.change_state(mapping[0]['task_id'],'done')
        draft = build_draft(self.store,'A\nB')
        self.assertEqual(draft['items'][0]['blocked'],'완료됨')
        apply_draft(self.store,draft)
        self.assertEqual(self.store.task(mapping[0]['task_id'])['state'],'done')
        draft = build_draft(self.store,'C\nD')
        self.assertTrue(all(i['task_id'] is None for i in draft['items']))

    def test_history_prediction_and_budget(self):
        for i,minutes in enumerate([20,40,60]):
            task = self.store.save_task('보고서 작성 '+str(i),kind='writing')
            self.store.start(task,now=100)
            self.store.change_state(task,'done',now=100+minutes*60)
        item = parse_notes('보고서 초안 작성')[0]
        prediction = estimate(self.store,item)
        self.assertEqual(prediction['total'],40)
        self.assertEqual(prediction['sample_count'],3)
        self.assertEqual(prediction['observed_range'],[20,60])
        draft = build_draft(self.store,'보고서 초안 작성\n! 메일 보내기 [10분]\n코드 테스트 [60분]',budget=65)
        self.assertLessEqual(draft['allocated'],65)
        self.assertEqual(draft['items'][1]['start'],0)
        self.assertIsNone(draft['items'][2]['start'])

    def test_unfinished_dependency_excluded_from_schedule(self):
        mapping = apply_draft(self.store,build_draft(self.store,'A\nB'))
        a,b = [r['task_id'] for r in mapping]
        self.store.save_task('B',task_id=b,dependency=a)
        draft = build_draft(self.store,'A\nB')
        self.assertEqual(draft['items'][1]['blocked'],'선행 작업 미완료')
        self.assertIsNone(draft['items'][1]['start'])

    def test_empty_notes_no_delete_and_explicit_estimate(self):
        apply_draft(self.store,build_draft(self.store,'A [80분]'))
        apply_draft(self.store,build_draft(self.store,''))
        self.assertEqual(len(self.store.tasks()),1)
        self.assertEqual(self.store.tasks()[0]['estimate'],80)

    def test_unrelated_single_replacement_is_new_task(self):
        apply_draft(self.store,build_draft(self.store,'논문 방법론 읽기'))
        draft = build_draft(self.store,'우체국 방문')
        self.assertEqual(draft['items'][0]['operation'],'추가')
