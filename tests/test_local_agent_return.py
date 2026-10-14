import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from local_agent.store import Store
from local_agent.return_service import checkpoint,return_card,render_card,ProcessWatch
from local_agent.model_runtime import ModelRegistry,record_concept,sequence,personalize
from local_agent.model_contract import FEATURES
from local_agent.visual_capture import CaptureManager


class ReturnTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.temp.name)/'test.sqlite3')
        self.task=self.store.save_task('실험 보고서',next_action='결과 표 확인')

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_checkpoint_preserves_future_action_and_events_not_completion(self):
        checkpoint(self.store,self.task,'아직 표를 옮기지 않았음','표에 옮기기')
        with self.store.db:
            self.store.db.execute('INSERT INTO task_events(task_id,at,kind,detail) VALUES (?,?,?,?)',(self.task,time.time(),'process_exit','프로세스 종료. 성공 여부 미확인.'))
        card=return_card(self.store,self.task)
        self.assertIn('다음에 할 일: 표에 옮기기',render_card(card))
        self.assertEqual(self.store.task(self.task)['state'],'todo')
        self.assertEqual(len(card['events']),1)
        previous=card['revision']
        self.store.change_state(self.task,'waiting')
        card=return_card(self.store,self.task)
        self.assertNotEqual(previous,card['revision'])
        self.assertNotIn('resume',[a['id'] for a in card['actions']])

    def test_no_model_and_no_consent_abstain(self):
        self.assertEqual(ModelRegistry().predict('camera',None)['status'],'untrained')
        with self.assertRaises(ValueError):
            CaptureManager().capture('screen')
        values=sequence(self.store,self.task,now=100)[-1]
        self.assertEqual(len(values),len(FEATURES))
        self.assertEqual(values[:10],[0.]*10)

    def test_stale_disabled_and_future_observation_mask(self):
        data=dict(probabilities=[1.,0.,0.],abstain=False)
        record_concept(self.store,'camera',data,at=100)
        self.assertEqual(sequence(self.store,now=101)[-1][8],0)
        self.assertEqual(sequence(self.store,now=101,enabled={'camera':True})[-1][8],1)
        self.assertEqual(sequence(self.store,now=161,enabled={'camera':True})[-1][8],0)
        record_concept(self.store,'camera',dict(probabilities=[0.,1.,0.],abstain=False),at=200)
        self.assertEqual(sequence(self.store,now=101,enabled={'camera':True})[-1][0],1)

    def test_personal_corrections_and_cleanup(self):
        pred=dict(label='direct',probabilities=[.8,.05,.05,.05,.05])
        for _ in range(4):
            with self.store.db:
                self.store.db.execute('INSERT INTO context_labels(at,participant,episode,label,data) VALUES (?,?,?,?,?)',
                    (time.time(),'p','e','support',json.dumps(dict(predicted_label='direct'))))
        result=personalize(self.store,pred)
        self.assertGreater(result['probabilities'][1],pred['probabilities'][1])
        self.assertAlmostEqual(sum(result['probabilities']),1)
        self.store.clear_observations()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM context_labels').fetchone()[0],0)

    @unittest.skipUnless(os.name=='nt','Windows process handle check')
    def test_process_exit_uses_real_handle_without_completing_task(self):
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'])
        watcher=ProcessWatch()
        try:
            watcher.attach(process.pid,self.task)
            process.wait(timeout=10)
            watcher.poll(self.store)
            self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM task_events').fetchone()[0],1)
            self.assertEqual(self.store.task(self.task)['state'],'todo')
            watcher.poll(self.store)
            self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM task_events').fetchone()[0],1)
        finally:
            watcher.close()
            process.wait(timeout=10)
