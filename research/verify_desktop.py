"""Opt-in live Ollama/UI integration on synthetic data, isolated from user data.

python -m research.verify_desktop --output runs/desktop-live
Requires a desktop, CustomTkinter, Pillow, Ollama and both configured models.
No user window or webcam is captured; screenshots contain only this fixture UI.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    import customtkinter as ctk
    from PIL import ImageGrab
    from local_agent.desktop import DesktopApp
    from local_agent.store import Store
    from local_agent.app import configure_dpi
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/desktop-live'))
    p.add_argument('--specialists',type=Path,help='Also verify real screen_layout / goal_history checkpoints')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # Timestamped DB avoids affecting either user state or a previous run.
    store=Store(args.output/f'verify-{time.time_ns()}.sqlite3')
    configure_dpi()
    ctk.set_widget_scaling(1)
    ctk.set_window_scaling(1)
    root=ctk.CTk()
    app=DesktopApp(root,store,start_timer=True,check_models=True)
    if args.specialists:
        import torch
        torch.set_num_threads(2)
        from local_agent.specialists import Specialists
        app.specialists=Specialists()
        for role in ['screen_layout','goal_history']:app.specialists.load(args.specialists/(role+'.pt'))
    def pump(seconds=.3):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            root.update()
            time.sleep(.025)
    def wait_job(key,timeout=150):
        deadline=time.monotonic()+timeout
        while key in app.jobs and time.monotonic()<deadline:
            pump(.1)
        assert key not in app.jobs,f'{key} did not complete'
    def snap(name):
        pump()
        ImageGrab.grab(window=root.winfo_id()).save(args.output/name)
    result={}
    try:
        pump()
        wait_job('models',10)
        assert {'granite4:3b','qwen3.5:4b'} <= (app.models or set()), app.model_error
        result['models']=sorted(app.models)
        app.notes.insert('1.0','실험 보고서 작성 awarenet\n포스텍 자소서 쓰기 1시간\n시험 공부 분산 클라우드, 신호 및 시스템, 클라우드 응용 SW')
        app.organize_button.invoke()
        wait_job('parse')
        assert app.draft,app.plan_status.cget('text')
        assert len(app.draft['items'])==5,app.draft
        result['parsed']=[(i['title'],i['estimate']['total']) for i in app.draft['items']]
        snap('01-plan.png')
        app.apply_button.invoke()
        active=store.active()
        assert active,app.notice.cget('text')
        task_id=active['task_id']
        pump(2)
        assert app.timer.cget('text')!='00:00'
        result['timer']=app.timer.cget('text')
        app.next_note.insert(0,'결과 비교표 첫 행부터 이어 쓰기')
        app.main_button.invoke()
        assert store.active() is None
        wait_job('help')
        assert app.ai_trace_id,app.ai_text.get('1.0','end')
        result['return_suggestion']=app.ai_text.get('1.0','end-1c')
        snap('02-return.png')
        # Capture only a window created by this verifier, never a user window.
        from local_agent.visual_capture import window_list
        fixture=ctk.CTkToplevel(root)
        fixture.title('Planwise synthetic screen verification')
        fixture.geometry('560x220')
        ctk.CTkLabel(fixture,text='Experiment status: RUNNING\nResult: NOT CHECKED',font=('Arial',22)).pack(padx=25,pady=50)
        pump(.6)
        row=next(w for w in window_list() if w['title']=='Planwise synthetic screen verification')
        app.capture_screen(row)
        fixture.destroy()
        wait_job('screen')
        wait_job('help')
        concept=store.db.execute("SELECT data FROM concepts WHERE modality='screen_text' ORDER BY id DESC LIMIT 1").fetchone()
        assert concept,app.ai_text.get('1.0','end')
        result['screen_reading']=json.loads(concept['data'])['visible_text']
        assert 'RUNNING' in result['screen_reading'],result['screen_reading']
        trace=json.loads(store.db.execute('SELECT trace FROM decisions WHERE id=?',(app.ai_trace_id,)).fetchone()[0])
        assert any(e['source']=='vision_text_unverified' for e in trace['card']['evidence']),trace
        if args.specialists:
            specialist_evidence=[e for e in trace['card']['evidence'] if e['source']=='public_specialist_hypothesis']
            assert {e['value']['role'] for e in specialist_evidence}=={'screen_layout','goal_history'},trace
            assert all(e['value']['observation_ids'] for e in specialist_evidence)
            result['specialist_roles']=[e['value']['role'] for e in specialist_evidence]
        result['screen_suggestion']=trace['result']['suggestion']
        app.main_button.invoke()
        assert store.active()['task_id']==task_id
        app.done_button.invoke()
        assert store.task(task_id)['state']=='done'
        app.main_button.invoke()
        assert app.selected!=task_id
        app.main_button.invoke()
        assert store.active()['task_id']==app.selected
        result['flow']='parse → start → ticking → pause → live Qwen → selected synthetic window → vision → grounded suggestion → resume → complete → next → start'
        snap('03-next.png')
        (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    finally:
        app.close()


if __name__=='__main__':
    main()
