"""App-owned, opt-in foreground pixels and webcam snapshots. No raw-file storage."""
import os
import time


def foreground_window():
    if os.name != 'nt':
        raise RuntimeError('현재 화면 관측은 Windows에서 지원합니다.')
    import ctypes as c
    from ctypes import wintypes as w
    user=c.WinDLL('user32',use_last_error=True)
    user.GetForegroundWindow.restype=w.HWND
    user.GetWindowThreadProcessId.argtypes=[w.HWND,c.POINTER(w.DWORD)]
    user.GetWindowRect.argtypes=[w.HWND,c.POINTER(w.RECT)]
    hwnd=user.GetForegroundWindow()
    if not hwnd:raise RuntimeError('활성 창을 확인하지 못했어요.')
    pid=w.DWORD();user.GetWindowThreadProcessId(hwnd,c.byref(pid))
    if pid.value==os.getpid():raise RuntimeError('Planwise가 앞에 있어요. 작업할 창으로 전환해 주세요.')
    rect=w.RECT()
    if not user.GetWindowRect(hwnd,c.byref(rect)) or rect.right<=rect.left or rect.bottom<=rect.top:
        raise RuntimeError('활성 창 영역을 확인하지 못했어요.')
    return dict(handle=int(hwnd),pid=pid.value,bounds=(rect.left,rect.top,rect.right,rect.bottom))


def capture_foreground():
    from PIL import ImageGrab
    before=foreground_window()
    # Visible pixels only: do not read a covered/background browser tab or window.
    image=ImageGrab.grab(bbox=before['bounds'],all_screens=True).convert('RGB')
    after=foreground_window()
    if before!=after:
        image.close()
        raise RuntimeError('캡처 중 창이 바뀌어 이미지를 폐기했어요.')
    return image


def face_detector():
    import cv2
    from pathlib import Path
    # OpenCV's Windows filename API cannot reliably read non-ASCII venv paths.
    xml=(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml').read_text(encoding='utf8')
    storage=cv2.FileStorage(xml,cv2.FILE_STORAGE_READ|cv2.FILE_STORAGE_MEMORY)
    try:
        detector=cv2.CascadeClassifier()
        detector.read(storage.getFirstTopLevelNode())
        if detector.empty():raise RuntimeError('얼굴 검출 모델을 읽지 못했어요.')
        return detector
    finally:storage.release()


def capture_webcam(index=0):
    import cv2
    from PIL import Image
    camera=cv2.VideoCapture(index,cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
    try:
        if not camera.isOpened():raise RuntimeError('웹캠을 열지 못했어요. 다른 앱 사용 여부와 카메라 권한을 확인해 주세요.')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        # Discard initial frames while the device adjusts exposure.
        ok,frame=camera.read()
        for _ in range(4):
            if not ok:break
            ok,frame=camera.read()
        if not ok:raise RuntimeError('웹캠에서 영상을 받지 못했어요.')
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detector=face_detector()
        faces=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(40,40))
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        return dict(image=image,face_count=len(faces),mean_brightness=round(float(gray.mean()),1),captured_at=time.time(),
                    note='얼굴 검출 결과이며 부재·시선·집중도 판정이 아닙니다.')
    finally:
        camera.release()


def default_specialists():
    from pathlib import Path
    import torch
    from local_agent.specialists import Specialists
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]/'runs/public-models'
    registry=Specialists()
    for role in ['screen_layout','goal_history']:
        path=root/(role+'.pt')
        if path.exists():registry.load(path)
    return registry


def analyze_screen(image,goals,history,registry=None):
    from local_agent.screen_text import read_screen
    started=time.monotonic()
    result=read_screen(image)
    result['specialists']=[];result['specialist_errors']=[]
    if registry:
        for role in ['screen_layout','goal_history']:
            if role not in registry.models or role=='goal_history' and not goals:continue
            try:
                value=registry.layout(image) if role=='screen_layout' else registry.goal_alignment(history+[result['visible_text']],goals)
                value.pop('maps',None)
                result['specialists'].append(value)
            except Exception as exc:result['specialist_errors'].append(role+': '+str(exc)[:160])
    result['seconds']=round(time.monotonic()-started,2)
    return result


class ObservationGate:
    """Generation guard: stopping/restarting invalidates every pending callback."""
    def __init__(self):
        self.enabled={'screen':False,'camera':False}
        self.generation={'screen':0,'camera':0}

    def set(self,kind,enabled):
        self.generation[kind]+=1
        self.enabled[kind]=bool(enabled)
        return self.generation[kind]

    def accepts(self,kind,token):
        return self.enabled[kind] and self.generation[kind]==token
