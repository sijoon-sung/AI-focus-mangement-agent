"""Windows foreground category + idle only. No titles, keystrokes, screenshots or network."""
import os

GROUPS = {
    'editor': {'code.exe', 'devenv.exe', 'pycharm64.exe', 'idea64.exe', 'notepad.exe', 'cursor.exe'},
    'browser': {'chrome.exe', 'msedge.exe', 'firefox.exe', 'brave.exe'},
    'terminal': {'windowsterminal.exe', 'powershell.exe', 'pwsh.exe', 'cmd.exe', 'conhost.exe'},
    'documents': {'winword.exe', 'excel.exe', 'powerpnt.exe', 'acrord32.exe', 'sumatrapdf.exe'},
    'meeting': {'teams.exe', 'ms-teams.exe', 'zoom.exe', 'slack.exe', 'discord.exe'},
}


def category_for(executable):
    name = executable.replace('\\', '/').rsplit('/', 1)[-1].lower()
    return next((category for category, names in GROUPS.items() if name in names), 'other')


class WindowsSensor:
    def __init__(self):
        self.available = os.name == 'nt'
        if not self.available:
            return
        import ctypes as c
        from ctypes import wintypes as w
        self.c, self.w = c, w
        self.user = c.WinDLL('user32', use_last_error=True)
        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        self.user.GetForegroundWindow.restype = w.HWND
        self.user.GetWindowThreadProcessId.argtypes = [w.HWND, c.POINTER(w.DWORD)]
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD)]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.GetTickCount.restype = w.DWORD

    def sample(self):
        if not self.available:
            return None
        c, w = self.c, self.w
        hwnd = self.user.GetForegroundWindow()
        if not hwnd:
            return None
        pid = w.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, c.byref(pid))
        # Ignore this assistant's own windows rather than learning them as user work.
        if pid.value == os.getpid():
            return None
        handle = self.kernel.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return None
        try:
            buffer = c.create_unicode_buffer(32768)
            size = w.DWORD(len(buffer))
            if not self.kernel.QueryFullProcessImageNameW(handle, 0, buffer, c.byref(size)):
                return None
            category = category_for(buffer.value)
        finally:
            self.kernel.CloseHandle(handle)
        class LastInput(c.Structure):
            _fields_ = [('cbSize', w.UINT), ('dwTime', w.DWORD)]
        info = LastInput()
        info.cbSize = c.sizeof(info)
        if not self.user.GetLastInputInfo(c.byref(info)):
            return None
        elapsed = (self.kernel.GetTickCount() - info.dwTime) & 0xffffffff
        # Only a coarse flag leaves this function, not raw timings or process paths.
        return {'category': category, 'idle': elapsed >= 300000}
