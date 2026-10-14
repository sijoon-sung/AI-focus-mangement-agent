import json
import tempfile
import unittest
from pathlib import Path
from local_agent.store import Store
from local_agent.learning import capture, context, fit, infer, temporal_split, evaluate
from local_agent.train import synthetic


class LearningTests(unittest.TestCase):
    def test_training_and_faithful_explanation(self):
        train, test = temporal_split(synthetic())
        model = fit(train, epochs=30)
        result = infer(model, test[0]['features'])
        self.assertAlmostEqual(result['score'], result['bias']+sum(t['contribution'] for t in result['terms']))
        self.assertAlmostEqual(result['margin'], result['contrast_bias']+sum(t['contrast'] for t in result['terms']))
        metrics = evaluate(model, test, train)
        self.assertGreater(metrics['accuracy'], metrics['majority_accuracy'])
        self.assertEqual(infer(json.loads(json.dumps(model)), test[0]['features']), result)

    def test_temporal_split_and_minimum(self):
        train, test = temporal_split(synthetic())
        self.assertLess(max(r['at'] for r in train)+300, min(r['at'] for r in test))
        self.assertFalse({r['day'] for r in train} & {r['day'] for r in test})
        with self.assertRaises(ValueError):
            temporal_split(synthetic()[:20])

    def test_capture_staleness_dedup_retention_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/'test.sqlite3')
            try:
                task = store.save_task('secret title', kind='coding')
                self.assertIsNone(context(store, 1000))
                store.observe('editor', False, 1000)
                self.assertTrue(capture(store, task, 1001))
                self.assertFalse(capture(store, task, 1002))
                self.assertIsNone(context(store, 1061))
                row = dict(store.db.execute('SELECT * FROM learning_examples').fetchone())
                self.assertNotIn('secret title', str(row))
                self.assertNotIn('task_id', row)
                store.set_setting('excluded_categories', ['editor'])
                self.assertIsNone(context(store, 1001))
                store.prune(1001+8*86400)
                self.assertEqual(store.db.execute('SELECT COUNT(*) FROM learning_examples').fetchone()[0], 0)
                store.set_setting('excluded_categories', [])
                store.observe('browser', False, 2000)
                self.assertTrue(capture(store, task, 2001))
                store.clear_observations()
                self.assertEqual(store.db.execute('SELECT COUNT(*) FROM learning_examples').fetchone()[0], 0)
            finally:
                store.close()

    def test_corrupt_weights_and_nonfinite_input_rejected(self):
        model = fit(synthetic()[:10], epochs=1)
        x = synthetic()[0]['features']
        model['weights'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            infer(model, x)


if __name__ == '__main__':
    unittest.main()
