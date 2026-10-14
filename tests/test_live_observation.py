import unittest
from unittest.mock import patch,MagicMock
from PIL import Image
from local_agent.live_observation import ObservationGate,capture_foreground,capture_webcam,analyze_screen


class LiveObservationTests(unittest.TestCase):
    def test_face_model_loads_from_unicode_workspace(self):
        from local_agent.live_observation import face_detector
        self.assertFalse(face_detector().empty())

    def test_stop_and_restart_discard_old_results(self):
        gate=ObservationGate();old=gate.set('screen',True)
        self.assertTrue(gate.accepts('screen',old))
        gate.set('screen',False)
        self.assertFalse(gate.accepts('screen',old))
        current=gate.set('screen',True)
        self.assertFalse(gate.accepts('screen',old))
        self.assertTrue(gate.accepts('screen',current))

    def test_foreground_switch_discards_capture(self):
        image=MagicMock()
        image.convert.return_value=image
        first=dict(handle=1,pid=9,bounds=(0,0,100,100))
        second=dict(handle=2,pid=10,bounds=(0,0,100,100))
        with patch('local_agent.live_observation.foreground_window',side_effect=[first,second]),patch('PIL.ImageGrab.grab',return_value=image) as grab:
            with self.assertRaisesRegex(RuntimeError,'창이 바뀌어'):capture_foreground()
        grab.assert_called_once_with(bbox=first['bounds'],all_screens=True)
        image.close.assert_called_once()

    def test_camera_released_when_read_fails(self):
        camera=MagicMock();camera.isOpened.return_value=True;camera.read.return_value=(False,None)
        with patch('cv2.VideoCapture',return_value=camera):
            with self.assertRaisesRegex(RuntimeError,'영상을 받지'):capture_webcam()
        camera.release.assert_called_once()

    def test_specialist_failure_does_not_hide_reading(self):
        registry=MagicMock();registry.models={'screen_layout':1,'goal_history':1}
        registry.layout.side_effect=RuntimeError('bad checkpoint')
        registry.goal_alignment.return_value={'role':'goal_history'}
        with patch('local_agent.screen_text.read_screen',return_value={'visible_text':'result pending','uncertainties':'','source':'qwen_vision_unverified'}):
            result=analyze_screen(Image.new('RGB',(20,20)),['write report'],['prior'],registry)
        self.assertEqual(result['visible_text'],'result pending')
        self.assertEqual(len(result['specialist_errors']),1)
        registry.goal_alignment.assert_called_once_with(['prior','result pending'],['write report'])


if __name__=='__main__':unittest.main()
