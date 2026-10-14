import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires torch research environment')
class ReturnModelTests(unittest.TestCase):
    def test_model_shapes_and_synthetic_refusal(self):
        import torch
        from research.return_models import make_model
        from local_agent.model_contract import FEATURES,LABELS,VERSION
        from local_agent.model_runtime import ModelRegistry
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ['camera','screen','context']:
                model=make_model(kind)
                x=torch.zeros(2,16,len(FEATURES)) if kind=='context' else torch.zeros(2,3,128,128)
                self.assertEqual(model(x).shape,(2,len(LABELS[kind])))
                path=Path(tmp)/(kind+'.pt')
                torch.save(dict(version=VERSION,kind=kind,labels=LABELS[kind],features=FEATURES,state_dict=model.state_dict(),trained=True,synthetic=True),path)
                with self.assertRaises(ValueError):
                    ModelRegistry().load(kind,path)

    def test_temporal_split_leakage_rejected(self):
        from research.train_return import validate_rows
        from local_agent.model_contract import FEATURES
        rows=[dict(participant='p',episode=s,at=i*1000,split=s,label='direct' if i else 'support',features=[[0.]*len(FEATURES)]) for i,s in enumerate(['train','val','test'])]
        rows.append(dict(rows[0],label='direct'))
        with self.assertRaises(ValueError):
            validate_rows(rows,'context','people')
        validate_rows(rows,'context','future')
        rows[-1]['episode']='val'
        with self.assertRaises(ValueError):
            validate_rows(rows,'context','future')
