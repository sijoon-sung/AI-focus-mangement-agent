import argparse
import os
from pathlib import Path
import sys
import tkinter as tk

from local_agent.store import Store
from local_agent.app import AgentApp, configure_dpi


def lock_directory(path):
    handle = (path / 'agent.lock').open('a+b')
    handle.seek(0)
    if handle.read(1) == b'':
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('이 데이터 폴더를 사용하는 Planwise가 이미 실행 중입니다.')
    return handle


def main():
    p = argparse.ArgumentParser(description='Planwise: local plan-aware productivity assistant')
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'local_data')
    p.add_argument('--demo', action='store_true', help='Add example tasks only if database is empty; capture stays off')
    p.add_argument('--ui-smoke', action='store_true', help='Build UI and exit without collecting activity')
    a = p.parse_args()
    a.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock = lock_directory(a.data_dir)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    try:
        store = Store(a.data_dir / 'planwise.sqlite3')
        if a.demo and not store.tasks():
            first = store.save_task('실험 결과 정리', '지표 비교표 1개 완성', '학습 로그에서 지표 추출', kind='research', priority=3)
            store.save_task('결과 해석 작성', '비교표 해석 3문단 작성', '오류 사례부터 정리', kind='writing', dependency=first)
            store.save_task('방법론 초안 작성', '데이터 분할과 실험 조건 기술', '실험 설정 파일 확인', kind='writing', estimate=40)
        configure_dpi()
        root = tk.Tk()
        app = AgentApp(root, store, start_timer=not a.ui_smoke)
        if a.ui_smoke:
            root.update()
            app.show_recommendation()
            root.update()
            print('UI smoke passed: scratchpad, task, return, review, privacy and research tabs rendered.')
            app.close()
        else:
            root.mainloop()
    finally:
        lock.close()


if __name__ == '__main__':
    main()
