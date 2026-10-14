"""Explicit, bounded single captures. Never falls back to a full-screen grab."""
import os
import time


def window_list():
    if os.name!='nt':
        raise ValueError('선택 창 캡처는 Windows에서 지원합니다.')
    import ctypes
    from ctypes import wintypes
    user=ctypes.WinDLL('user32',use_last_error=True)
    user.GetWindowTextLengthW.argtypes=[wintypes.HWND]
    user.GetWindowTextW.argtypes=[wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
    user.IsWindowVisible.argtypes=[wintypes.HWND]
    user.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    user.EnumWindows.argtypes=[callback_type,wintypes.LPARAM]
    rows=[]
    def callback(hwnd,param):
        if user.IsWindowVisible(hwnd):
            length=user.GetWindowTextLengthW(hwnd)
            if length:
                text=ctypes.create_unicode_buffer(length+1)
                user.GetWindowTextW(hwnd,text,length+1)
                pid=wintypes.DWORD()
                user.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
                rows.append(dict(handle=int(hwnd),pid=pid.value,title=text.value))
        return True
    user.EnumWindows(callback_type(callback),0)
    return rows


class CaptureManager:
    def __init__(self):
        self.enabled={'camera':False,'screen':False}
        self.history={'camera':[],'screen':[]}
        self.window=None

    def capture(self,kind):
        if not self.enabled.get(kind):
            raise ValueError('해당 관측을 직접 켜 주세요.')
        now=time.monotonic()
        recent=[t for t in self.history[kind] if now-t<3600]
        if recent and now-recent[-1]<30 or len(recent)>=12:
            raise ValueError('관측은 최소 30초 간격, 종류별 시간당 12회까지만 합니다.')
        self.history[kind]=recent+[now]
        if kind=='screen':
            if not self.window or self.window not in window_list():
                raise ValueError('선택한 창이 바뀌었어요. 캡처할 창을 다시 선택해 주세요.')
            from PIL import ImageGrab
            return ImageGrab.grab(window=self.window['handle']).convert('RGB')
        import cv2
        from PIL import Image
        camera=cv2.VideoCapture(0,cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
        try:
            ok,frame=camera.read()
            if not ok:
                raise ValueError('카메라 프레임을 읽지 못했어요.')
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            faces=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(40,40))
            if not len(faces):
                raise ValueError('분석 가능한 얼굴 영역을 찾지 못했어요. 부재나 비집중으로 판단하지 않습니다.')
            x,y,w,h=max(faces,key=lambda r:int(r[2])*int(r[3]))
            return Image.fromarray(cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB))
        finally:
            camera.release()
