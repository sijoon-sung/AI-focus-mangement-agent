import unittest
import tempfile
from pathlib import Path
import torch
from PIL import Image
from research.specialist_models import EyeGazeCNN,ScreenLayoutCNN,GoalHistoryGRU,contrastive_alignment
from research.specialist_data import audit_groups,screen_tensor,layout_target,HistoryRecords,eye_tensor
from local_agent.specialists import ScreenCalibration


class SpecialistTests(unittest.TestCase):
    def test_distinct_shapes_and_gradients(self):
        torch.set_num_threads(2)
        gaze=EyeGazeCNN()(torch.rand(2,1,36,60),torch.rand(2,4))
        self.assertTrue(torch.allclose(gaze.norm(dim=-1),torch.ones(2),atol=1e-5))
        gaze.sum().backward()
        layout=ScreenLayoutCNN()(torch.rand(2,3,256,256))
        self.assertEqual(tuple(layout.shape),(2,8,32,32));layout.mean().backward()
        aligned,op=GoalHistoryGRU()(torch.rand(2,16,386),torch.rand(2,384))
        self.assertEqual(tuple(op.shape),(2,3))
        contrastive_alignment(aligned,torch.rand(2,384),torch.tensor([1,1])).backward()

    def test_group_leakage_rejected(self):
        rows=[dict(group='same',split='train'),dict(group='same',split='test'),dict(group='val',split='val')]
        with self.assertRaises(ValueError):audit_groups(rows,'group')

    def test_unknown_annotation_mask_and_letterbox(self):
        x,t=screen_tensor(Image.new('RGB',(512,256)))
        y,v=layout_target([dict(bbox=[0,0,256,256],category=''),dict(bbox=[256,0,512,256],category='Button')],t)
        self.assertEqual(tuple(x.shape),(3,256,256))
        self.assertEqual(float(v[:,10,5].sum()),0)
        self.assertEqual(float(y[0,10,25]),1)
        self.assertEqual(float(v[:,0,0].sum()),0)

    def test_no_future_action_input(self):
        embeddings=torch.eye(384)[:4]
        task=dict(actions=['first','future'],operations=['CLICK','TYPE'],action_indices=[1,2],goal_index=0)
        x,g,op,_=HistoryRecords([task],embeddings)[0]
        self.assertEqual(int(op),0)
        self.assertEqual(float(x[:,:384][:,2].sum()),0)
        self.assertEqual(float(x[-1,1]),1)

    def test_eye_input_contract_and_calibration(self):
        with self.assertRaises(ValueError):eye_tensor(Image.new('RGB',(128,128)))
        cal=ScreenCalibration()
        angles=[[-1,-1],[-1,1],[1,-1],[1,1],[0,0]]
        points=[[0,0],[0,1],[1,0],[1,1],[.5,.5]]
        self.assertLess(cal.fit(angles,points),.001)
        self.assertAlmostEqual(cal.predict([0,0])['point'][0],.5)
        with self.assertRaises(ValueError):ScreenCalibration().predict([0,0])
