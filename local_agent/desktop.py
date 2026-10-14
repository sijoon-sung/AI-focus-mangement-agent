"""The everyday desktop: one workspace, live sessions, visible local AI jobs."""
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from tkinter import messagebox

import customtkinter as ctk

from local_agent.inbox import build_draft, apply_draft
from local_agent.note_parser import organize_with_granite
from local_agent.return_service import checkpoint, return_card, suggest, ProcessWatch
from local_agent.visual_capture import CaptureManager, window_list
from local_agent.model_runtime import record_concept

BG = '#F5F6F8'
WHITE = '#FFFFFF'
INK = '#202A3B'
MUTED = '#788397'
ACCENT = '#5264E8'
SOFT = '#EEF0FE'
GREEN = '#197D63'
STATE = {'todo':'시작 전', 'doing':'진행 중', 'paused':'잠깐 멈춤', 'waiting':'결과 기다리는 중', 'done':'완료'}


def local_models():
    req = urllib.request.Request('http://127.0.0.1:11434/api/tags')
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=5) as r:
        return {m['name'] for m in json.load(r).get('models', [])}


def idle_seconds():
    """Read only time since last input; no key, window or process data."""
    if os.name != 'nt':
        return None
    import ctypes as c
    from ctypes import wintypes as w
    class LastInput(c.Structure):
        _fields_ = [('cbSize', w.UINT), ('dwTime', w.DWORD)]
    info = LastInput()
    info.cbSize = c.sizeof(info)
    if not c.windll.user32.GetLastInputInfo(c.byref(info)):
        return None
    return ((c.windll.kernel32.GetTickCount() - info.dwTime) & 0xffffffff) / 1000


def error_text(exc):
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
        return '모델이 없어요. 설정에서 설치 명령을 확인한 뒤 다시 시도해 주세요.'
    if isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)):
        return 'Ollama에 연결하지 못했거나 응답 시간이 초과됐어요. Ollama 실행 후 다시 시도해 주세요.'
    return str(exc)[:250] or type(exc).__name__


class DesktopApp:
    def __init__(self, root, store, start_timer=True, check_models=True):
        self.root, self.store = root, store
        self.closed = False
        self.jobs = {}
        self.results = queue.Queue()
        self.after_ids = set()
        self.draft = None
        self.source = None
        self.rows_signature = None
        self.selected = store.setting('desktop_selected', None)
        self.last_text = store.setting('inbox_text', '')
        self.models = None
        self.model_error = ''
        self.watches = ProcessWatch()
        self.capture = CaptureManager()
        self.idle_paused = False
        self.break_reminded = set()
        self.lab = None
        self.lab_window = None
        self.ai_trace_id = None
        self.privacy_revision = 0
        self.specialists = None
        self.observation_panel = None
        self.store.set_setting('screen_text_enabled', False)
        active = store.active()
        if active:
            end = max(active['started'], min(time.time(), store.setting('last_heartbeat', active['started'])))
            store.pause('이전 실행의 마지막 확인 시각에서 중단', now=end)
        ctk.set_appearance_mode('light')
        root.title('Planwise · 오늘을 가볍게')
        root.geometry('1180x820')
        root.minsize(980, 720)
        root.configure(fg_color=BG)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        self.font = ('맑은 고딕', 13)
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.render()
        self.later(100, self.poll)
        if start_timer:
            self.later(1000, self.tick)
        if check_models:
            self.check_models()

    def later(self, ms, callback):
        if self.closed:
            return
        def run():
            self.after_ids.discard(token)
            if not self.closed:
                callback()
        token = self.root.after(ms, run)
        self.after_ids.add(token)
        return token

    def label(self, parent, text='', size=13, bold=False, color=INK, **kwargs):
        return ctk.CTkLabel(parent, text=text, font=('맑은 고딕', size, 'bold' if bold else 'normal'), text_color=color, **kwargs)

    def button(self, parent, text, command, primary=False, **kwargs):
        return ctk.CTkButton(parent, text=text, command=lambda:self.safe(command), height=kwargs.pop('height',40),
                             corner_radius=10, font=self.font, fg_color=ACCENT if primary else SOFT,
                             hover_color='#4355D2' if primary else '#E0E5FA',
                             text_color=WHITE if primary else ACCENT, **kwargs)

    def safe(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.notice.configure(text='처리하지 못했어요 · '+error_text(exc), text_color='#B44739')

    def _build(self):
        head = ctk.CTkFrame(self.root, fg_color='transparent')
        head.grid(row=0, column=0, sticky='ew', padx=30, pady=(22, 16))
        self.label(head, 'planwise', 25, True).pack(side='left')
        self.label(head, '  /  오늘을 가볍게', 13, color=MUTED).pack(side='left', padx=8)
        self.button(head, '설정', self.settings, width=70).pack(side='right')
        self.observation_button=self.button(head,'화면·웹캠',self.open_observation,width=110)
        self.observation_button.pack(side='right',padx=8)
        self.model_status = self.label(head, '로컬 AI 확인 중', 12, color=MUTED)
        self.model_status.pack(side='right', padx=18)
        body = ctk.CTkFrame(self.root, fg_color='transparent')
        body.grid(row=1, column=0, sticky='nsew', padx=30)
        body.grid_columnconfigure(0, weight=4, uniform='body')
        body.grid_columnconfigure(1, weight=5, uniform='body')
        body.grid_rowconfigure(0, weight=1)
        left = ctk.CTkFrame(body, fg_color=WHITE, corner_radius=18)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(6, weight=1)
        self.label(left, '머릿속 할 일을 내려놓으세요', 20, True, anchor='w').grid(row=0, column=0, sticky='ew', padx=24, pady=(24, 4))
        self.label(left, '순서나 형식은 신경 쓰지 않아도 괜찮아요.', color=MUTED, anchor='w').grid(row=1, column=0, sticky='ew', padx=24)
        self.notes = ctk.CTkTextbox(left, height=150, corner_radius=12, fg_color=BG, font=('맑은 고딕', 14), border_width=0, wrap='word')
        self.notes.grid(row=2, column=0, sticky='ew', padx=24, pady=(16, 8))
        self.notes.insert('1.0', self.last_text)
        self.notes.bind('<KeyRelease>', lambda e:self.note_changed())
        self.notes.bind('<<Paste>>', lambda e:self.later(30,self.note_changed()), add=True)
        self.notes.bind('<<Cut>>', lambda e:self.later(30,self.note_changed()), add=True)
        row = ctk.CTkFrame(left, fg_color='transparent')
        row.grid(row=3, column=0, sticky='ew', padx=24)
        self.organize_button = self.button(row, '할 일 정리하기  →', self.organize, True, width=170)
        self.organize_button.pack(side='left')
        self.budget = ctk.StringVar(value=str(store_budget(self.store)))
        self.budget_button = self.button(row, f'{self.budget.get()}분 여유', self.change_budget, width=105)
        self.budget_button.pack(side='right')
        self.plan_status = self.label(left, '메모는 이 컴퓨터에 자동 저장됩니다.', 12, color=MUTED, anchor='w', justify='left', wraplength=370)
        self.plan_status.grid(row=4, column=0, sticky='ew', padx=24, pady=(10, 12))
        bar = ctk.CTkFrame(left, fg_color='transparent')
        bar.grid(row=5, column=0, sticky='ew', padx=24, pady=(0, 8))
        self.list_title = self.label(bar, '오늘 할 일', 15, True)
        self.list_title.pack(side='left')
        self.button(bar, '목록 수정', self.edit_plan, width=82, height=32).pack(side='right')
        self.task_list = ctk.CTkScrollableFrame(left, fg_color='transparent', corner_radius=0)
        self.task_list.grid(row=6, column=0, sticky='nsew', padx=15, pady=(0, 8))
        self.task_list.grid_columnconfigure(0, weight=1)
        self.apply_button = self.button(left, '이 계획으로 시작하기', self.apply, True)
        self.apply_button.grid(row=7, column=0, sticky='ew', padx=24, pady=(0, 20))
        self.apply_button.grid_remove()
        right = ctk.CTkFrame(body, fg_color='transparent')
        right.grid(row=0, column=1, sticky='nsew')
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        hero = ctk.CTkFrame(right, fg_color=WHITE, corner_radius=18)
        hero.grid(row=0, column=0, sticky='ew')
        self.phase = self.label(hero, 'READY WHEN YOU ARE', 11, True, color=ACCENT)
        self.phase.pack(anchor='w', padx=28, pady=(24, 10))
        self.title = self.label(hero, '오늘은 무엇부터 해볼까요?', 23, True, anchor='w', justify='left', wraplength=470)
        self.title.pack(fill='x', padx=28)
        self.timer = self.label(hero, '00:00', 48, True)
        self.timer.pack(anchor='w', padx=25, pady=(8, 0))
        self.timer_caption = self.label(hero, '작업을 고르면 여기서 바로 시작해요.', 12, color=MUTED, anchor='w')
        self.timer_caption.pack(fill='x', padx=28)
        self.progress = ctk.CTkProgressBar(hero, height=5, fg_color=BG, progress_color=ACCENT)
        self.progress.pack(fill='x', padx=28, pady=(16, 18))
        self.progress.set(0)
        actions = ctk.CTkFrame(hero, fg_color='transparent')
        actions.pack(fill='x', padx=28)
        self.main_button = self.button(actions, '시작하기', self.primary, True, width=180)
        self.main_button.pack(side='left', fill='x', expand=True)
        self.done_button = self.button(actions, '완료했어요 ✓', self.complete, width=120)
        self.done_button.pack(side='left', padx=(10, 0))
        self.next_note = ctk.CTkEntry(hero, placeholder_text='다음에 할 일 한 줄 (선택) · 예: 결과 표부터 이어 쓰기', height=38,
                                       fg_color=BG, border_width=0, corner_radius=9, font=self.font)
        self.next_note.pack(fill='x', padx=28, pady=(14, 22))
        help_card = ctk.CTkFrame(right, fg_color=WHITE, corner_radius=18)
        help_card.grid(row=1, column=0, sticky='nsew', pady=(18, 0))
        self.label(help_card, '다시 시작하기 쉬워지도록', 16, True).pack(anchor='w', padx=24, pady=(20, 8))
        self.facts = self.label(help_card, '', 13, anchor='w', justify='left', wraplength=470)
        self.facts.pack(fill='x', padx=24, pady=(0, 10))
        self.ai_text = ctk.CTkTextbox(help_card, height=70, fg_color=SOFT, corner_radius=10, wrap='word', font=self.font, text_color=INK)
        self.ai_text.pack(fill='both', expand=True, padx=24, pady=(0, 12))
        self.set_ai('막막하면 다음 한 걸음만 같이 골라볼게요.')
        row = ctk.CTkFrame(help_card, fg_color='transparent')
        row.pack(fill='x', padx=24, pady=(0, 18))
        self.help_button = self.button(row, '다음 한 걸음', self.ask_help, width=130)
        self.help_button.pack(side='left')
        self.screen_button = self.button(row, '화면 보고 돕기', self.screen_dialog, width=130)
        self.screen_button.pack(side='left', padx=8)
        self.button(row, '근거', self.show_evidence, width=55).pack(side='right')
        self.notice = self.label(self.root, '메모를 적고 정리하기를 눌러보세요. 시작과 복귀는 오른쪽에서 이어집니다.', 12, color=MUTED, anchor='w', wraplength=1080)
        self.notice.grid(row=2, column=0, sticky='ew', padx=34, pady=(10, 14))

    def set_ai(self, text):
        self.ai_text.configure(state='normal')
        self.ai_text.delete('1.0', 'end')
        self.ai_text.insert('1.0', text)
        self.ai_text.configure(state='disabled')

    def text(self):
        return self.notes.get('1.0','end-1c')

    def note_changed(self):
        value = self.text()
        if value == self.last_text:
            return
        self.last_text = value
        self.store.set_setting('inbox_text', value)
        self.draft = None
        self.apply_button.grid_remove()
        self.plan_status.configure(text='저장했어요 · 다 적으면 정리하기를 눌러 주세요.')
        self.render_rows(force=True)

    def job(self, key, worker, callback):
        if key in self.jobs:
            return False
        self.jobs[key] = (time.monotonic(), callback)
        def run():
            try:
                self.results.put((key, worker(), None))
            except Exception as exc:
                self.results.put((key, None, exc))
        threading.Thread(target=run, daemon=True).start()
        return True

    def poll(self):
        while True:
            try:
                key, result, error = self.results.get_nowait()
            except queue.Empty:
                break
            item = self.jobs.pop(key, None)
            if item:
                self.safe(lambda:item[1](result,error))
        if 'parse' in self.jobs:
            elapsed = int(time.monotonic()-self.jobs['parse'][0])
            self.plan_status.configure(text=f'할 일을 나누고 있어요 · {elapsed}초\n처음에는 로컬 모델을 불러오는 시간이 필요해요.')
        if 'help' in self.jobs or 'screen' in self.jobs:
            key = 'screen' if 'screen' in self.jobs else 'help'
            elapsed = int(time.monotonic()-self.jobs[key][0])
            self.help_button.configure(text=f'도움 준비 중 · {elapsed}초', state='disabled')
        else:
            self.help_button.configure(text='다음 한 걸음', state='normal')
        self.later(150,self.poll)

    def check_models(self):
        self.model_status.configure(text='로컬 AI 확인 중', text_color=MUTED)
        def ready(models,error):
            self.models, self.model_error = models, error_text(error) if error else ''
            ok = models is not None and {'granite4:3b','qwen3.5:4b'} <= models
            self.model_status.configure(text='●  로컬 AI 준비됨' if ok else '●  AI 연결 확인 필요', text_color=GREEN if ok else '#AD6B22')
            if error:
                self.notice.configure(text=self.model_error, text_color='#AD6B22')
        self.job('models', local_models, ready)

    def organize(self):
        self.note_changed()
        source = self.text()
        if not source.strip():
            self.plan_status.configure(text='할 일 한 가지만 적어도 괜찮아요.')
            self.notes.focus_set()
            return
        if len(source)>4000:
            raise ValueError('자동 정리는 4,000자까지예요. 메모를 나눠 주세요.')
        self.organize_button.configure(state='disabled', text='정리 중…')
        self.draft = None
        self.apply_button.grid_remove()
        self.job('parse', lambda:organize_with_granite(source), lambda r,e:self.parsed(source,r,e))

    def parsed(self, source, result, error):
        self.organize_button.configure(state='normal', text='할 일 정리하기  →')
        if source != self.text():
            self.plan_status.configure(text='메모가 바뀌어 이전 결과는 쓰지 않았어요. 다시 정리해 주세요.')
            return
        if error:
            self.plan_status.configure(text=error_text(error)+'\n목록 수정에서 직접 나누어 시작할 수도 있어요.')
            return
        self.preview(result['outline'])
        if result.get('notices'):
            self.plan_status.configure(text=' '.join(result['notices']))

    def preview(self, outline):
        self.source = self.text()
        self.draft = build_draft(self.store, outline, int(self.budget.get()))
        count = len(self.draft['items'])
        self.plan_status.configure(text=f'{count}개로 정리했어요. 아래 목록을 확인하고 시작하세요.')
        self.apply_button.configure(state='normal' if count else 'disabled')
        self.apply_button.grid()
        self.render_rows(force=True)

    def apply(self):
        if not self.draft or self.source!=self.text():
            raise ValueError('메모가 바뀌었어요. 먼저 다시 정리해 주세요.')
        fresh = build_draft(self.store,self.draft['text'],int(self.budget.get()))
        mapping = apply_draft(self.store, fresh)
        first = next((mapping[i]['task_id'] for i,item in enumerate(fresh['items']) if item['start']==0),None)
        self.draft = None
        self.apply_button.grid_remove()
        if not self.store.active() and first:
            self.start(first)
        else:
            self.render()
        self.plan_status.configure(text='계획을 저장했어요 · 언제든 메모를 바꿔 다시 정리해도 돼요.')
        if not first:
            self.notice.configure(text='계획을 저장했어요. 오늘 여유 시간에 맞는 작업이 없어 자동 시작하지 않았어요.', text_color=MUTED)

    def render_rows(self, force=False):
        tasks = self.store.tasks()
        active = self.store.active()
        signature = (str(self.draft),str([(t['id'],t['title'],t['state'],t['estimate']) for t in tasks]),self.selected)
        if not force and signature==self.rows_signature:
            return
        self.rows_signature = signature
        for w in self.task_list.winfo_children():
            w.destroy()
        self.list_title.configure(text='이렇게 해볼까요?' if self.draft else f'오늘 할 일 · {sum(t["state"]=="done" for t in tasks)}/{len(tasks)}')
        if self.draft:
            rows = sorted(self.draft['items'],key=lambda i:i['start'] if i['start'] is not None else 10000)
            for i,item in enumerate(rows):
                suffix = item['blocked'] or (f'{item["estimate"]["remaining"]}분쯤' if item['start'] is not None else '나중에')
                self.task_row(i, item['title'], suffix, None, False)
        else:
            ordered = sorted(tasks,key=lambda t:(t['state']=='done', t['id'] != (active['task_id'] if active else self.selected),-t['priority'],t['created']))
            for i,t in enumerate(ordered):
                suffix = f'{STATE[t["state"]]} · {t["estimate"]}분 예상'
                self.task_row(i,t['title'],suffix,t['id'],t['id']==self.selected)
        if not tasks and not self.draft:
            self.label(self.task_list, '아직 비어 있어요.\n위에 하고 싶은 일을 적어보세요.', color=MUTED, justify='left').grid(row=0,column=0,sticky='w',padx=10,pady=24)

    def task_row(self, index, title, subtitle, task_id, selected):
        frame = ctk.CTkFrame(self.task_list,fg_color=SOFT if selected else BG,corner_radius=12)
        frame.grid(row=index,column=0,sticky='ew',padx=4,pady=4)
        frame.grid_columnconfigure(0,weight=1)
        title_widget=self.label(frame,title,14,True,anchor='w',justify='left',wraplength=320)
        title_widget.grid(row=0,column=0,sticky='ew',padx=14,pady=(12,2))
        detail=self.label(frame,subtitle,11,color=ACCENT if selected else MUTED,anchor='w')
        detail.grid(row=1,column=0,sticky='ew',padx=14,pady=(0,12))
        if task_id:
            for w in (frame,title_widget,detail):
                w.bind('<Button-1>',lambda e,k=task_id:self.safe(lambda:self.select(k)))
                w.configure(cursor='hand2')

    def select(self, task_id):
        self.save_next_action()
        self.selected = task_id
        self.store.set_setting('desktop_selected',task_id)
        self.next_note.delete(0,'end')
        self.ai_trace_id = None
        self.set_ai('필요하면 다음 한 걸음을 제안받거나, 선택한 화면을 한 번 확인할 수 있어요.')
        self.render()

    def save_next_action(self):
        value=self.next_note.get().strip()
        if value and self.selected:
            with self.store.db:
                self.store.db.execute('UPDATE tasks SET next_action=? WHERE id=?',(value[:2000],self.selected))

    def render(self):
        tasks = self.store.tasks()
        ids = {t['id'] for t in tasks}
        active = self.store.active()
        if self.selected not in ids:
            self.selected = active['task_id'] if active else next((t['id'] for t in tasks if t['state']!='done'),None)
        if self.selected:
            t=self.store.task(self.selected)
            self.title.configure(text=t['title'])
            self.phase.configure(text=STATE[t['state']].upper())
            label = {'todo':'이 일 시작하기','doing':'잠깐 쉬기','paused':'이어서 하기','waiting':'대기 끝 · 이어 하기','done':'다음 할 일로 →'}[t['state']]
            self.main_button.configure(text=label,state='normal')
            self.done_button.configure(state='normal',text='완료 취소' if t['state']=='done' else '완료했어요 ✓')
            card=return_card(self.store,self.selected)
            note=card['checkpoint']
            lines=[]
            if note and note['note']:
                lines.append('남겨 둔 메모 · '+note['note'][:160])
            if t['next_action']:
                lines.append('이어 할 일 · '+t['next_action'][:160])
            if card['events']:
                lines.append(card['events'][-1]['detail'][:160])
            self.facts.configure(text='\n'.join(lines) or '아직 남긴 메모가 없어요. 잠깐 쉴 때 한 줄만 남겨도 다음 시작이 쉬워져요.')
        else:
            self.title.configure(text='오늘은 무엇부터 해볼까요?')
            self.main_button.configure(state='disabled')
            self.done_button.configure(state='disabled')
            self.facts.configure(text='왼쪽에 메모를 적고 정리하면 여기서 바로 시작할 수 있어요.')
        self.render_rows()
        self.update_clock()

    def update_clock(self, now=None):
        now=time.time() if now is None else now
        if not self.selected:
            return
        active=self.store.active()
        t=self.store.task(self.selected)
        seconds=int(self.store.elapsed_minutes(self.selected,now)*60)
        self.timer.configure(text=f'{seconds//60:02d}:{seconds%60:02d}')
        is_active=active and active['task_id']==self.selected
        self.timer_caption.configure(text=('시간을 기록하고 있어요' if is_active else '이 작업에 쓴 시간')+f' · 예상 {t["estimate"]}분')
        self.progress.set(min(1,seconds/max(1,t['estimate']*60)))

    def start(self, task_id):
        self.store.start(task_id)
        self.store.set_setting('last_heartbeat',time.time())
        self.select(task_id)
        self.notice.configure(text='시작했어요. 시간이 흐르는 동안 이 창을 내려두고 작업해도 괜찮아요.',text_color=GREEN)

    def primary(self):
        if not self.selected:
            return
        t=self.store.task(self.selected)
        if t['state']=='doing':
            note=self.next_note.get().strip()
            self.store.pause(note=note or '잠깐 쉬기',next_action=note or None)
            checkpoint(self.store,t['id'],note or '잠깐 쉬고 돌아올게요.',note or t['next_action'])
            self.next_note.delete(0,'end')
            self.notice.configure(text='시간 기록을 멈췄어요. 돌아오면 같은 자리에서 이어 하면 돼요.',text_color=MUTED)
            self.render()
            if self.store.setting('desktop_auto_help',True):
                self.ask_help()
        elif t['state']=='done':
            self.next_task()
        else:
            if t['state']=='waiting':
                self.store.change_state(t['id'],'paused')
            self.start(t['id'])

    def complete(self):
        if not self.selected:
            return
        task_id=self.selected
        if self.store.task(task_id)['state']=='done':
            self.undo_complete(task_id)
            return
        self.store.change_state(task_id,'done')
        self.render()
        self.set_ai('하나 끝냈어요. 다음 할 일로 넘어가거나 오늘은 여기서 마쳐도 좋아요.')
        self.notice.configure(text='완료로 기록했어요. 잘못 눌렀다면 ‘완료 취소’로 되돌릴 수 있어요.',text_color=GREEN)

    def undo_complete(self, task_id):
        self.store.change_state(task_id,'paused')
        self.select(task_id)
        self.notice.unbind('<Button-1>')
        self.notice.configure(text='완료를 취소했어요. 이어서 시작할 수 있어요.',cursor='')

    def next_task(self):
        t=next((t for t in self.store.tasks() if t['state'] not in {'done','waiting'} and (not t['dependency'] or self.store.task(t['dependency'])['state']=='done')),None)
        if t:
            self.select(t['id'])
        else:
            self.notice.configure(text='지금 시작할 일이 없어요. 기다리는 결과를 확인하거나 메모에 새 일을 적어 주세요.',text_color=MUTED)

    def ask_help(self):
        if not self.selected or 'screen' in self.jobs or 'help' in self.jobs:
            return
        card=return_card(self.store,self.selected)
        privacy_revision=self.privacy_revision
        self.set_ai('저장한 메모와 작업 기록을 보고 다음 행동을 준비하고 있어요…')
        def ready(result,error):
            if privacy_revision!=self.privacy_revision:
                return
            if card['task']['id']!=self.selected or return_card(self.store,self.selected)['revision']!=card['revision']:
                self.set_ai('작업 상태가 바뀌어 이전 제안은 쓰지 않았어요. 필요하면 다시 요청해 주세요.')
            elif error:
                self.set_ai(error_text(error)+'\n아래 ‘다음 한 걸음’을 눌러 재시도할 수 있어요.')
            else:
                self.set_ai('이렇게 시작해볼까요?\n'+result['suggestion'])
                self.ai_trace_id=self.store.trace('return_suggestion',self.selected,dict(card=card,result=result))
        self.job('help',lambda:suggest(card),ready)

    def dialog(self, title, width=620, height=480):
        win=ctk.CTkToplevel(self.root)
        win.title(title)
        win.geometry(f'{width}x{height}')
        win.configure(fg_color=BG)
        win.transient(self.root)
        win.after(100,win.lift)
        return win

    def change_budget(self):
        dialog=ctk.CTkInputDialog(text='오늘 쓸 수 있는 시간은 대략 몇 분인가요?',title='오늘 여유')
        answer=dialog.get_input()
        if answer is None:
            return
        minutes=int(answer)
        if not 1<=minutes<=1440:
            raise ValueError('1~1440분으로 입력해 주세요.')
        self.budget.set(str(minutes))
        self.budget_button.configure(text=f'{minutes}분 여유')
        self.store.set_setting('inbox_budget',minutes)
        if self.draft:
            self.preview(self.draft['text'])

    def edit_plan(self):
        win=self.dialog('할 일 조금 고치기')
        self.label(win,'한 줄에 할 일 하나. 시간은 필요할 때만 적으세요.',16,True).pack(anchor='w',padx=24,pady=20)
        box=ctk.CTkTextbox(win,wrap='word',font=self.font)
        box.pack(fill='both',expand=True,padx=24)
        box.insert('1.0',self.draft['text'] if self.draft else self.text())
        def use():
            self.preview(box.get('1.0','end-1c'))
            win.destroy()
        self.button(win,'이 목록 사용하기',use,True).pack(padx=24,pady=20)

    def show_evidence(self):
        if not self.selected:
            return
        from local_agent.return_service import render_card
        card=return_card(self.store,self.selected)
        text=render_card(card)
        if self.ai_trace_id:
            row=self.store.db.execute('SELECT trace FROM decisions WHERE id=?',(self.ai_trace_id,)).fetchone()
            if row:
                result=json.loads(row['trace'])['result']
                text+='\n\nAI가 인용한 기록: '+', '.join(result['evidence_ids'])+'\n제안: '+result['suggestion']
        win=self.dialog('이 안내의 근거')
        box=ctk.CTkTextbox(win,font=self.font,wrap='word')
        box.pack(fill='both',expand=True,padx=20,pady=20)
        box.insert('1.0',text)
        box.configure(state='disabled')

    def screen_dialog(self):
        if not self.selected:
            raise ValueError('먼저 도와줄 작업을 골라 주세요.')
        if 'screen' in self.jobs or 'help' in self.jobs:
            raise ValueError('현재 AI 요청이 끝나면 다시 눌러 주세요.')
        rows=[w for w in window_list() if w['pid']!=os.getpid()]
        win=self.dialog('어느 화면을 같이 볼까요?',680,500)
        self.label(win,'선택한 창만 한 번 확인해요',19,True).pack(anchor='w',padx=24,pady=(20,8))
        self.label(win,'이미지는 저장하지 않고 이 컴퓨터의 Qwen에만 전달해요.\n창 이름은 이 목록에만 표시합니다.',color=MUTED,justify='left').pack(anchor='w',padx=24,pady=(0,12))
        listing=ctk.CTkScrollableFrame(win,fg_color=WHITE)
        listing.pack(fill='both',expand=True,padx=24,pady=(0,20))
        for row in rows:
            self.button(listing,row['title'][:65],lambda r=row:self.capture_screen(r,win)).pack(fill='x',pady=4)
        if not rows:
            self.label(listing,'열려 있는 작업 창이 없어요.').pack(pady=20)

    def capture_screen(self, window, dialog=None):
        self.capture.window=window
        self.capture.enabled['screen']=True
        try:
            image=self.capture.capture('screen')
        finally:
            self.capture.enabled['screen']=False
        if dialog:
            dialog.destroy()
        task_id=self.selected
        initial_revision=return_card(self.store,task_id)['revision']
        privacy_revision=self.privacy_revision
        captured_at=time.time()
        self.set_ai('선택한 화면을 읽고 있어요. 화면 읽기가 끝나면 복귀 제안으로 연결할게요…')
        from local_agent.screen_text import read_screen
        specialists=self.specialists
        goals=[t['title'] for t in self.store.tasks() if t['state']!='done'][:30]
        prior=[];prior_ids=[]
        for record in self.store.db.execute("SELECT id,data FROM concepts WHERE modality='screen_text' ORDER BY id DESC LIMIT 8"):
            item=json.loads(record['data'])
            if item.get('task_id')==task_id:
                prior.append(item['visible_text']);prior_ids.append(record['id'])
        prior.reverse();prior_ids.reverse()
        def analyze():
            result=read_screen(image)
            hypotheses=[]
            if specialists:
                errors=[]
                for role in ['screen_layout','goal_history']:
                    if role not in specialists.models:continue
                    try:
                        value=specialists.layout(image) if role=='screen_layout' else specialists.goal_alignment(prior+[result['visible_text']],goals)
                        value.pop('maps',None)
                        hypotheses.append(value)
                    except Exception as exc:
                        errors.append(role+': '+error_text(exc))
                result['specialist_errors']=errors
            result['specialists']=hypotheses
            return result
        def ready(result,error):
            if privacy_revision!=self.privacy_revision:
                return
            if task_id!=self.selected or return_card(self.store,task_id)['revision']!=initial_revision:
                self.set_ai('작업 상태가 바뀌어 이전 화면의 결과는 쓰지 않았어요.')
                return
            if error:
                self.set_ai('화면을 읽지 못했어요 · '+error_text(error))
                return
            self.store.set_setting('screen_text_enabled',True)
            hypotheses=result.pop('specialists',[])
            text_id=record_concept(self.store,'screen_text',dict(result,task_id=task_id,captured_at=captured_at),at=captured_at)
            for hypothesis in hypotheses:
                record_concept(self.store,'specialist',dict(hypothesis,task_id=task_id,captured_at=captured_at,observation_ids=prior_ids+[text_id]),at=captured_at)
            self.set_ai('읽은 내용 (확인 필요): '+result['visible_text'])
            self.notice.configure(text='화면 읽기 완료 · 내용을 복귀 제안에 연결했어요. 원본 이미지는 저장하지 않았어요.',text_color=GREEN)
            if result.get('specialist_errors'):
                self.notice.configure(text='화면은 읽었지만 일부 모델을 사용하지 못했어요 · '+'; '.join(result['specialist_errors']),text_color='#AD6B22')
            self.ask_help()
        self.job('screen',analyze,ready)

    def settings(self):
        win=self.dialog('설정',660,620)
        panel=ctk.CTkScrollableFrame(win,fg_color='transparent')
        panel.pack(fill='both',expand=True,padx=20,pady=20)
        self.label(panel,'필요한 만큼만 도와드릴게요',21,True).pack(anchor='w',pady=(0,20))
        auto=ctk.BooleanVar(value=self.store.setting('desktop_auto_help',True))
        ctk.CTkSwitch(panel,text='잠깐 쉴 때 다음 행동을 자동으로 제안',variable=auto,font=self.font,
                       command=lambda:self.store.set_setting('desktop_auto_help',auto.get())).pack(anchor='w',pady=10)
        idle=ctk.BooleanVar(value=self.store.setting('desktop_idle_assist',False))
        ctk.CTkSwitch(panel,text='5분간 입력이 없으면 시간 기록을 멈추고 복귀 안내',variable=idle,font=self.font,
                       command=lambda:self.store.set_setting('desktop_idle_assist',idle.get())).pack(anchor='w',pady=10)
        self.label(panel,'입력 유무만 확인해요. 키 내용·앱 제목·화면은 수집하지 않아요.\n읽기나 영상 시청 중에도 멈출 수 있어요. 필요한 경우에만 켜 주세요.',12,color=MUTED,justify='left').pack(anchor='w',pady=(0,16))
        self.label(panel,'로컬 AI',16,True).pack(anchor='w',pady=8)
        details='준비된 모델: '+', '.join(sorted(self.models or [])) if self.models else self.model_error or '연결 확인 중'
        self.label(panel,details,12,color=MUTED,wraplength=560,justify='left').pack(anchor='w')
        self.button(panel,'연결 다시 확인',self.check_models).pack(anchor='w',pady=10)
        box=ctk.CTkTextbox(panel,height=68,font=('Consolas',13),fg_color=WHITE)
        box.pack(fill='x')
        box.insert('1.0','ollama pull granite4:3b\nollama pull qwen3.5:4b')
        box.configure(state='disabled')
        self.label(panel,'메모·세션은 로컬 저장. 화면·웹캠은 직접 켰을 때만 관측해요.\n화면에서 읽은 문구·AI 근거는 기본 7일 보관, 계획과 메모는 유지합니다.',12,color=MUTED,justify='left').pack(anchor='w',pady=15)
        self.button(panel,'화면·AI 기록 지우기',self.clear_observations).pack(anchor='w',pady=6)
        self.label(panel,'연구 도구',16,True).pack(anchor='w',pady=(18,6))
        self.label(panel,'CNN·GRU 학습과 카메라는 선택 기능입니다. 일상 사용에 필요하지 않아요.',12,color=MUTED).pack(anchor='w')
        self.button(panel,'학습 및 관측 도구 열기',self.open_lab).pack(anchor='w',pady=10)
        self.button(panel,'공개 데이터 학습 모델 연결',self.load_specialists).pack(anchor='w',pady=6)
        self.button(panel,'실행 중인 작업의 종료 기다리기',self.watch_dialog).pack(anchor='w',pady=6)

    def clear_observations(self):
        if messagebox.askyesno('기록 지우기','화면에서 읽은 문구·AI 근거·관측과 교정 기록을 지울까요?\n계획과 세션은 유지합니다.',parent=self.root):
            if self.observation_panel:self.observation_panel.stop()
            self.store.clear_observations()
            self.privacy_revision+=1
            self.store.set_setting('screen_text_enabled',False)
            self.ai_trace_id=None
            self.set_ai('관측과 AI 근거를 지웠어요.')
            self.render()

    def open_observation(self):
        if self.observation_panel:
            self.observation_panel.window.lift()
            return self.observation_panel
        from local_agent.observation_panel import ObservationPanel
        self.observation_panel=ObservationPanel(self)
        return self.observation_panel

    def load_specialists(self):
        from tkinter import filedialog
        folder=filedialog.askdirectory(parent=self.root,title='screen_layout.pt / goal_history.pt 모델 폴더 선택')
        if not folder:return
        from pathlib import Path
        from local_agent.specialists import Specialists
        def load():
            registry=Specialists();loaded=[]
            for name in ['screen_layout','goal_history']:
                path=Path(folder)/(name+'.pt')
                if path.exists():loaded.append(registry.load(path))
            if not loaded:raise ValueError('이 폴더에 화면/목표 이력 모델이 없어요.')
            return registry,loaded
        def ready(result,error):
            if error:
                self.notice.configure(text=error_text(error),text_color='#B44739')
            else:
                self.specialists=result[0]
                names=', '.join(r['role']+(' (예비 학습)' if r['pilot'] else '') for r in result[1])
                self.notice.configure(text='연결했어요: '+names+' · 다음 화면 읽기부터 근거에 추가됩니다.',text_color=GREEN)
        self.job('specialist_load',load,ready)

    def open_lab(self):
        if self.lab_window and self.lab_window.winfo_exists():
            self.lab_window.lift()
            return
        from tkinter import ttk
        from local_agent.research_ui import ResearchPanel
        self.lab_window=self.dialog('연구 도구 · 선택 기능',1100,760)
        frame=ttk.Frame(self.lab_window,padding=20)
        frame.pack(fill='both',expand=True)
        self.lab=ResearchPanel(frame,self.store,lambda:self.selected)
        def close_lab():
            self.lab.reset()
            self.lab.close()
            self.lab_window.destroy()
            self.lab=None
        self.lab_window.protocol('WM_DELETE_WINDOW',close_lab)

    def watch_dialog(self):
        if not self.selected:
            raise ValueError('먼저 기다릴 작업을 골라 주세요.')
        task_id=self.selected
        rows=[w for w in window_list() if w['pid']!=os.getpid()]
        win=self.dialog('실행 중인 창 연결',680,480)
        self.label(win,'선택한 창의 프로그램이 종료되면 알려드려요.\n프로그램 종료를 작업 성공으로 판단하지는 않아요.',14,justify='left').pack(padx=24,pady=20)
        listing=ctk.CTkScrollableFrame(win)
        listing.pack(fill='both',expand=True,padx=24,pady=(0,20))
        def attach(row):
            if self.store.task(task_id)['state']=='done':
                raise ValueError('완료된 작업은 대기로 바꾸지 않아요. 먼저 완료를 취소해 주세요.')
            self.watches.attach(row['pid'],task_id)
            self.store.change_state(task_id,'waiting')
            self.render()
            self.notice.configure(text='프로그램 종료를 기다리고 있어요. 종료되면 이 화면에 표시할게요.',text_color=GREEN)
            win.destroy()
        for row in rows:
            self.button(listing,row['title'][:60],lambda r=row:attach(r)).pack(fill='x',pady=4)

    def tick(self):
        now=time.time()
        self.note_changed()
        active=self.store.active()
        if active:
            self.store.set_setting('last_heartbeat',now)
            if now-active['started']>=25*60 and active['id'] not in self.break_reminded:
                self.break_reminded.add(active['id'])
                self.notice.configure(text='25분 이어 했어요. 한 번 쉬어도 좋아요. 계속해도 괜찮아요.',text_color=GREEN)
                self.root.bell()
        if self.store.setting('desktop_idle_assist',False):
            self.handle_idle(idle_seconds(),now)
        before=self.store.db.execute('SELECT COALESCE(MAX(id),0) FROM task_events').fetchone()[0]
        self.watches.poll(self.store)
        latest=self.store.db.execute('SELECT * FROM task_events WHERE id>? ORDER BY id DESC LIMIT 1',(before,)).fetchone()
        if latest:
            self.notice.configure(text=latest['detail'],text_color=GREEN)
            self.root.bell()
            self.render()
        for t in self.store.tasks():
            if t['state']=='waiting' and t['wait_until'] and t['wait_until']<=now:
                with self.store.db:
                    self.store.db.execute('UPDATE tasks SET wait_until=NULL WHERE id=?',(t['id'],))
                self.notice.configure(text=t['title']+' · 기다리던 결과를 확인해 볼 시간이에요.',text_color=GREEN)
        self.update_clock(now)
        self.later(1000,self.tick)

    def handle_idle(self, seconds, now):
        if seconds is None:
            return
        active=self.store.active()
        if active and seconds>=300:
            task_id=active['task_id']
            self.store.pause('5분간 입력이 없어 시간 기록 중단',now=max(active['started'],now-seconds))
            with self.store.db:
                self.store.db.execute('INSERT INTO task_events(task_id,at,kind,detail) VALUES (?,?,?,?)',
                                      (task_id,now,'idle_pause','5분간 입력이 없어 시간 기록을 멈췄어요. 실제 집중 여부는 판단하지 않았어요.'))
            self.idle_paused=True
            self.selected=task_id
            self.render()
        elif self.idle_paused and seconds<10:
            self.idle_paused=False
            self.notice.configure(text='돌아오셨네요. 남겨 둔 메모를 보고 ‘이어서 하기’를 누르면 돼요.',text_color=GREEN)
            self.root.bell()

    def close(self):
        if self.closed:
            return
        if self.observation_panel:self.observation_panel.close()
        self.store.set_setting('inbox_text',self.text())
        self.save_next_action()
        self.store.set_setting('desktop_selected',self.selected)
        self.store.set_setting('screen_text_enabled',False)
        self.store.pause('앱 종료로 세션 중단')
        self.watches.close()
        if self.lab:
            self.lab.close()
        self.closed=True
        for token in self.after_ids:
            self.root.after_cancel(token)
        self.after_ids.clear()
        # CTk schedules its own animation/DPI callbacks in this Tcl interpreter.
        # Cancel those too before destroying widgets (including test windows).
        for token in self.root.tk.splitlist(self.root.tk.call('after','info')):
            self.root.tk.call('after','cancel',token)
        self.store.prune()
        self.store.close()
        self.root.destroy()


def store_budget(store):
    return max(1,min(1440,int(store.setting('inbox_budget',240))))
