import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

class VisionSensor:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.current_score = 100
        self.is_running = False
        self.frame_count = 0
        
        # EAR 임계값
        self.EAR_THRESHOLD = 0.20
        # 좌우(Yaw) 이탈 임계값
        self.YAW_RATIO_MIN = 0.4
        self.YAW_RATIO_MAX = 2.5
        
    def _calculate_ear(self, landmarks, indices):
        # indices: [왼쪽, 위1, 위2, 오른쪽, 아래2, 아래1]
        p_left = landmarks[indices[0]]
        p_top1 = landmarks[indices[1]]
        p_top2 = landmarks[indices[2]]
        p_right = landmarks[indices[3]]
        p_bot2 = landmarks[indices[4]]
        p_bot1 = landmarks[indices[5]]
        
        v1 = distance(p_top1, p_bot1)
        v2 = distance(p_top2, p_bot2)
        h = distance(p_left, p_right)
        
        if h == 0: return 0
        return (v1 + v2) / (2.0 * h)

    def _analyze_focus(self, landmarks):
        # 1. EAR (Eye Aspect Ratio)
        # 오른쪽 눈 (화면상 좌측)
        right_eye_indices = [33, 160, 158, 133, 153, 144]
        ear_right = self._calculate_ear(landmarks, right_eye_indices)
        # 왼쪽 눈 (화면상 우측)
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        ear_left = self._calculate_ear(landmarks, left_eye_indices)
        
        avg_ear = (ear_right + ear_left) / 2.0
        
        # 2. 간이 Head Pose (Yaw)
        nose_tip = landmarks[1]
        left_edge = landmarks[234]
        right_edge = landmarks[454]
        
        dist_left = distance(nose_tip, left_edge)
        dist_right = distance(nose_tip, right_edge)
        
        yaw_ratio = 1.0
        if dist_right > 0:
            yaw_ratio = dist_left / dist_right
            
        # 3. 점수 가감 로직
        is_focused = True
        
        if avg_ear < self.EAR_THRESHOLD:
            # 눈을 감고 있거나 졸음
            self.current_score -= 10
            is_focused = False
            
        if yaw_ratio < self.YAW_RATIO_MIN or yaw_ratio > self.YAW_RATIO_MAX:
            # 고개를 많이 돌림
            self.current_score -= 5
            is_focused = False
            
        if is_focused:
            # 집중 중이면 점수 회복
            self.current_score += 5
            
        # Clamp 0 ~ 100
        self.current_score = max(0, min(100, self.current_score))
        
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            print(f"[Vision] EAR: {avg_ear:.2f}, YAW: {yaw_ratio:.2f} | Score: {self.current_score} | Focus: {is_focused}")
            
        return avg_ear, yaw_ratio

    def run(self):
        self.is_running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        while self.is_running and cap.isOpened():
            success, image = cap.read()
            if not success:
                time.sleep(0.1)
                continue
                
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            results = self.detector.detect(mp_image)
            
            if results.face_landmarks and len(results.face_landmarks) > 0:
                self._analyze_focus(results.face_landmarks[0])
            else:
                # 얼굴이 화면 밖으로 사라짐
                self.current_score -= 10
                self.current_score = max(0, min(100, self.current_score))
                self.frame_count += 1
                if self.frame_count % 20 == 0:
                    print(f"[Vision] No face detected! Score: {self.current_score}")
                
            time.sleep(0.1) # 10fps
            
        cap.release()
        
    def get_score(self):
        return self.current_score
        
    def stop(self):
        self.is_running = False
