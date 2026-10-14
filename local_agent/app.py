"""Tk desktop application. Start with python -m local_agent."""
import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

from local_agent.store import Store
from local_agent.planner import recommend, suggest_link, daily_summary
from local_agent.sensors import WindowsSensor
from local_agent.learning import capture, context, infer

STATE_NAMES = {'todo': '예정', 'doing': '진행 중', 'paused': '중단', 'waiting': '대기', 'done': '완료'}
KIND_NAMES = {'general': '일반', 'coding': '개발', 'writing': '작성', 'research': '자료 조사', 'admin': '행정'}
CAT_NAMES = {'editor': '편집기', 'browser': '브라우저', 'terminal': '터미널', 'documents': '문서', 'meeting': '회의/메신저', 'other': '기타'}


def configure_dpi():
    import os
    if os.name == 'nt':
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()


class AgentApp:
    def __init__(self, root, store, start_timer=True):
        self.root, self.store = root, store
        root.tk.call('tk', 'scaling', 1.5)
        self.sensor = WindowsSensor()
        self.latest = None
        self.decision_id = None
        self.tick_id = None
        root.title('Planwise Local · 계획을 실행으로')
        root.geometry(f'{min(1280, root.winfo_screenwidth()-60)}x{min(920, root.winfo_screenheight()-100)}')
        root.minsize(900, 650)
        root.configure(bg='#f2f5f8')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('맑은 고딕', 10))
        style.configure('TFrame', background='#f2f5f8')
        style.configure('TLabel', background='#f2f5f8', foreground='#172b40')
        style.configure('Title.TLabel', font=('맑은 고딕', 21, 'bold'))
        style.configure('Sub.TLabel', foreground='#52687a')
        style.configure('TButton', padding=(10, 7))
        style.configure('Treeview', rowheight=28, background='white', fieldbackground='white')
        style.configure('Treeview.Heading', font=('맑은 고딕', 10, 'bold'))
        shell = ttk.Frame(root, padding=22)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='Planwise Local', style='Title.TLabel').pack(anchor='w')
        ttk.Label(shell, text='계획 · 실행 · 복귀     |     외부 API 없이 내 컴퓨터에서', style='Sub.TLabel').pack(anchor='w', pady=(2, 12))
        self.banner = tk.StringVar(value='할 일을 추가하고 완료 조건과 다음 행동을 적어보세요.')
        ttk.Label(shell, textvariable=self.banner, wraplength=1040).pack(fill='x', pady=(0, 10))
        tabs = ttk.Notebook(shell)
        tabs.pack(fill='both', expand=True)
        self.today = ttk.Frame(tabs, padding=14)
        self.review = ttk.Frame(tabs, padding=14)
        self.privacy = ttk.Frame(tabs, padding=14)
        tabs.add(self.today, text='  계획과 실행  ')
        tabs.add(self.review, text='  근거와 회고  ')
        tabs.add(self.privacy, text='  개인정보  ')
        self._build_today()
        self._build_review()
        self._build_privacy()
        # A crash must never count all overnight downtime as work.
        active = store.active()
        if active:
            end = max(active['started'], min(time.time(), store.setting('last_heartbeat', active['started'])))
            store.pause('비정상 종료 후 마지막 확인 시각에서 세션 종료', now=end)
            self.banner.set('이전 세션을 마지막 확인 시각에서 종료했습니다. 작업을 선택해 다시 시작하세요.')
        store.prune()
        self.refresh()
        root.protocol('WM_DELETE_WINDOW', self.close)
        if start_timer:
            self.tick_id = root.after(1000, self.tick)

    def _build_today(self):
        toolbar = ttk.Frame(self.today)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='+ 작업 추가', command=self.edit_task).pack(side='left')
        ttk.Button(toolbar, text='수정', command=lambda: self.edit_task(self.selected())).pack(side='left', padx=5)
        ttk.Label(toolbar, text='지금 가용 시간(분)').pack(side='left', padx=(20, 5))
        self.budget = tk.StringVar(value=str(self.store.setting('budget', 60)))
        ttk.Spinbox(toolbar, from_=5, to=480, textvariable=self.budget, width=6).pack(side='left')
        ttk.Button(toolbar, text='다음 작업 추천', command=self.show_recommendation).pack(side='left', padx=8)
        frame = ttk.Frame(self.today)
        frame.pack(fill='both', expand=True, pady=12)
        cols = ('state', 'title', 'estimate', 'spent', 'priority')
        self.tree = ttk.Treeview(frame, columns=cols, show='headings', selectmode='browse', height=8)
        for name, label, width in [('state', '상태', 85), ('title', '작업', 430), ('estimate', '예상(분)', 85), ('spent', '세션(분)', 85), ('priority', '우선순위', 85)]:
            self.tree.heading(name, text=label)
            self.tree.column(name, width=width, stretch=name == 'title')
        scroll = ttk.Scrollbar(frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', lambda e: self.show_detail())
        self.tree.bind('<Double-1>', lambda e: self.edit_task(self.selected()))
        actions = ttk.Frame(self.today)
        actions.pack(fill='x')
        for label, command in [('시작 / 복귀', self.start_task), ('중단 메모', self.pause_task),
                               ('대기 설정', self.wait_task), ('대기 해제', self.release_task),
                               ('완료 확인', self.complete_task), ('다시 예정으로', self.reopen_task)]:
            ttk.Button(actions, text=label, command=command).pack(side='left', padx=(0, 5))
        self.session = tk.StringVar()
        ttk.Label(self.today, textvariable=self.session, style='Sub.TLabel').pack(anchor='w', pady=(10, 5))
        self.detail = tk.Text(self.today, height=6, wrap='word', font=('맑은 고딕', 10), relief='flat', padx=10, pady=10)
        self.detail.pack(fill='x')
        self.observation = tk.StringVar(value='활동 수집 꺼짐')
        bottom = ttk.Frame(self.today)
        bottom.pack(fill='x', pady=(10, 0))
        ttk.Label(bottom, textvariable=self.observation).pack(side='left')
        ttk.Button(bottom, text='최근 앱 범주를 선택 작업에 연결', command=self.confirm_link).pack(side='right')

    def _build_review(self):
        self.summary = tk.StringVar()
        ttk.Label(self.review, textvariable=self.summary, wraplength=950).pack(anchor='w', pady=(0, 10))
        ttk.Label(self.review, text='추천의 근거', font=('맑은 고딕', 13, 'bold')).pack(anchor='w')
        self.explanation = tk.Text(self.review, wrap='word', font=('맑은 고딕', 10), relief='flat', padx=12, pady=12)
        self.explanation.pack(fill='both', expand=True, pady=8)
        self.write_text(self.explanation, '계획과 실행 탭에서 “다음 작업 추천”을 누르면\n사용한 데이터, 점수 기여, 제외 사유를 볼 수 있습니다.\n\n이 버전은 명시적인 규칙 엔진입니다. 학습된 집중도 모델이 아닙니다.')
        row = ttk.Frame(self.review)
        row.pack(fill='x')
        for label, value in [('도움 됨', 'helpful'), ('도움 안 됨', 'not_helpful'), ('맥락이 틀림', 'wrong_context')]:
            ttk.Button(row, text=label, command=lambda v=value: self.give_feedback(v)).pack(side='left', padx=(0, 6))
        ttk.Button(row, text='집계 결과 내보내기', command=self.export_summary).pack(side='right')
        ttk.Button(row, text='학습 모델로 최근 맥락 확인', command=self.inspect_model).pack(side='right', padx=6)
        ttk.Label(self.review, text='피드백은 로컬 평가 기록으로 저장합니다. 추천 규칙을 자동 변경하지 않습니다.', style='Sub.TLabel').pack(anchor='w', pady=8)

    def _build_privacy(self):
        ttk.Label(self.privacy, text='필요한 정보만, 직접 선택해서', font=('맑은 고딕', 15, 'bold')).pack(anchor='w', pady=(0, 12))
        self.collection = tk.BooleanVar(value=self.store.setting('collection_enabled', False))
        ttk.Checkbutton(self.privacy, text='앱 범주와 5분 이상 입력 없음 여부 수집 (15초 간격)',
                        variable=self.collection, command=self.change_collection).pack(anchor='w')
        ttk.Label(self.privacy, text='기본값은 꺼짐입니다. 창 제목·URL·키 입력 내용·클립보드·화면·카메라는 수집하지 않습니다.\n앱 실행 경로는 메모리에서 범주로 변환하며 데이터베이스에는 저장하지 않습니다.',
                  wraplength=950, style='Sub.TLabel').pack(anchor='w', pady=10)
        row = ttk.Frame(self.privacy)
        row.pack(anchor='w', pady=5)
        ttk.Label(row, text='활동·판단 기록 보관일').pack(side='left')
        self.retention = tk.StringVar(value=str(self.store.setting('retention_days', 7)))
        ttk.Spinbox(row, from_=1, to=365, textvariable=self.retention, width=6).pack(side='left', padx=8)
        ttk.Button(row, text='적용', command=self.save_privacy).pack(side='left')
        ttk.Label(self.privacy, text='수집에서 제외할 범주 (해당 구간은 추정에 사용하지 않음)').pack(anchor='w', pady=(15, 5))
        excluded = set(self.store.setting('excluded_categories', ['meeting']))
        self.exclude_vars = {}
        group = ttk.Frame(self.privacy)
        group.pack(anchor='w')
        for key, label in CAT_NAMES.items():
            var = tk.BooleanVar(value=key in excluded)
            self.exclude_vars[key] = var
            ttk.Checkbutton(group, text=label, variable=var, command=self.save_exclusions).pack(side='left', padx=(0, 10))
        info = ('계획·메모·세션·사용자가 확인한 범주 연결은 직접 삭제할 때까지 보관합니다.\n'
                  '학습 정답(최근 5분 통계·작업 종류)은 활동 기록과 같은 보관일을 적용합니다.\n'
                  '직접 내보낸 학습 모델 파일은 기록 삭제와 별개로 파일을 직접 삭제해야 합니다.\n'
                '활동 범주는 개인정보일 수 있습니다. 로컬 DB는 암호화하지 않으며 이 Windows 계정의 파일 권한에 의존합니다.\n'
                '기본 저장 위치는 프로젝트의 local_data입니다. 클라우드 동기화 폴더라면 --data-dir로 위치를 바꾸세요.\n'
                '외부 전송·원격 API·텔레메트리 없음. 앱을 종료하면 수집·타이머 알림도 종료됩니다.')
        ttk.Label(self.privacy, text=info, wraplength=950).pack(anchor='w', pady=18)
        ttk.Label(self.privacy, text=f'저장 위치: {self.store.path.resolve()}', wraplength=950, style='Sub.TLabel').pack(anchor='w')
        controls = ttk.Frame(self.privacy)
        controls.pack(anchor='w', pady=18)
        ttk.Button(controls, text='활동 기록·범주 개인화 삭제', command=self.clear_observations).pack(side='left', padx=(0, 10))
        ttk.Button(controls, text='모든 로컬 데이터 초기화', command=self.reset).pack(side='left')

    @staticmethod
    def write_text(widget, value):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', value)
        widget.configure(state='disabled')

    def selected(self):
        items = self.tree.selection()
        return int(items[0]) if items else None

    def require_selected(self):
        task_id = self.selected()
        if task_id is None:
            messagebox.showinfo('작업 선택', '목록에서 작업을 선택하세요.', parent=self.root)
        return task_id

    def refresh(self):
        selected = self.selected()
        self.tree.delete(*self.tree.get_children())
        tasks = self.store.tasks()
        for task in tasks:
            self.tree.insert('', 'end', iid=str(task['id']), values=(STATE_NAMES[task['state']], task['title'],
                task['estimate'], f'{self.store.elapsed_minutes(task["id"]):.1f}', '★' * task['priority']))
        if selected and self.tree.exists(str(selected)):
            self.tree.selection_set(str(selected))
        active = self.store.active()
        self.session.set(f'현재 세션: {self.store.task(active["task_id"])["title"]}  ·  종료/중단 전까지 세션 시간을 기록합니다.' if active else '진행 중인 세션이 없습니다. 잠시 떠날 때는 중단 메모를 남겨주세요.')
        summary = daily_summary(self.store)
        self.summary.set(f'오늘 완료 {summary["completed_tasks"]}개  ·  기록한 세션 {summary["session_minutes"]}분\n{summary["note"]}')
        self.show_detail()

    def show_detail(self):
        task_id = self.selected()
        if task_id is None:
            self.write_text(self.detail, '작업을 선택하면 완료 조건, 다음 행동, 복귀 메모를 볼 수 있습니다.')
            return
        task = self.store.task(task_id)
        dependency = self.store.task(task['dependency'])['title'] if task['dependency'] else '없음'
        self.write_text(self.detail, f'완료 조건: {task["completion"] or "아직 입력하지 않음"}\n'
            f'다음 행동: {task["next_action"] or "아직 입력하지 않음"}\n'
            f'선행 작업: {dependency}\n복귀 메모: {self.store.return_note(task_id) or "없음"}')

    def edit_task(self, task_id=None):
        task = self.store.task(task_id) if task_id else {}
        dialog = tk.Toplevel(self.root)
        dialog.title('작업 수정' if task_id else '작업 추가')
        dialog.geometry('620x490')
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill='both', expand=True)
        variables = {}
        for index, (key, label, default) in enumerate([
            ('title', '작업 이름', ''), ('completion', '어떻게 되면 완료인가요?', ''),
            ('next_action', '다음에 할 구체적인 행동', ''), ('estimate', '예상 시간(분)', 25), ('priority', '우선순위 1~3', 2)]):
            ttk.Label(frame, text=label).grid(row=index*2, column=0, sticky='w', pady=(5, 2))
            variables[key] = tk.StringVar(value=str(task.get(key, default)))
            ttk.Entry(frame, textvariable=variables[key], width=64).grid(row=index*2+1, column=0, sticky='ew')
        kinds = list(KIND_NAMES)
        kind = ttk.Combobox(frame, values=list(KIND_NAMES.values()), state='readonly')
        kind.grid(row=10, column=0, sticky='ew', pady=8)
        kind.current(kinds.index(task.get('kind', 'general')))
        dependencies = [None] + [t['id'] for t in self.store.tasks() if t['id'] != task_id]
        dep = ttk.Combobox(frame, state='readonly', values=['선행 작업 없음'] +
                          [f'#{i} {self.store.task(i)["title"]}' for i in dependencies[1:]])
        dep.grid(row=11, column=0, sticky='ew')
        dep.current(dependencies.index(task.get('dependency')))
        def save():
            try:
                values = {k: v.get() for k, v in variables.items()}
                values['estimate'], values['priority'] = int(values['estimate']), int(values['priority'])
                saved = self.store.save_task(**values, kind=kinds[kind.current()], dependency=dependencies[dep.current()], task_id=task_id)
                dialog.destroy()
                self.refresh()
                self.tree.selection_set(str(saved))
            except ValueError as exc:
                messagebox.showerror('입력 확인', str(exc), parent=dialog)
        ttk.Button(frame, text='저장', command=save).grid(row=12, column=0, sticky='e', pady=12)
        frame.columnconfigure(0, weight=1)
        dialog.update_idletasks()
        dialog.geometry(f'{max(620, frame.winfo_reqwidth()+30)}x{frame.winfo_reqheight()+30}')

    def start_task(self):
        task_id = self.require_selected()
        if task_id is not None:
            try:
                task = self.store.task(task_id)
                if task['state'] in {'done', 'waiting'}:
                    raise ValueError('완료 또는 대기 상태입니다. 상태를 먼저 변경하세요.')
                if task['dependency'] and self.store.task(task['dependency'])['state'] != 'done':
                    raise ValueError('선행 작업이 아직 완료되지 않았습니다.')
                active = self.store.active()
                if active and active['task_id'] != task_id:
                    note = simpledialog.askstring('작업 전환', '현재 작업의 복귀 메모를 남겨주세요. 빈칸도 가능합니다.', parent=self.root)
                    if note is None:
                        return
                    self.store.pause(note)
                self.store.start(task_id)
                self.store.set_setting('last_heartbeat', time.time())
                self.banner.set('세션을 시작했습니다. 완료 여부는 직접 확인하며 집중력을 자동 단정하지 않습니다.')
                self.refresh()
            except ValueError as exc:
                messagebox.showinfo('시작할 수 없음', str(exc), parent=self.root)

    def pause_task(self):
        active = self.store.active()
        if not active:
            return
        note = simpledialog.askstring('중단 메모', '어디까지 했고 무엇이 남았나요?', parent=self.root)
        if note is None:
            return
        task = self.store.task(active['task_id'])
        action = simpledialog.askstring('다음 행동', '돌아오면 바로 할 행동', initialvalue=task['next_action'], parent=self.root)
        self.store.pause(note, action)
        self.banner.set('중단했습니다. 다음에 작업을 선택하면 복귀 메모를 볼 수 있습니다.')
        self.refresh()

    def wait_task(self):
        task_id = self.require_selected()
        if task_id is None:
            return
        mins = simpledialog.askinteger('대기 확인 타이머', '몇 분 뒤 다시 확인할까요? 실제 실행 완료를 감지하는 기능은 아닙니다.',
                                      minvalue=1, maxvalue=10080, parent=self.root)
        if mins:
            self.store.change_state(task_id, 'waiting', wait_minutes=mins)
            self.banner.set('대기 중인 작업은 추천에서 제외합니다. 다른 작업을 추천받을 수 있습니다.')
            self.refresh()

    def release_task(self):
        task_id = self.require_selected()
        if task_id and self.store.task(task_id)['state'] == 'waiting':
            self.store.change_state(task_id, 'paused')
            self.refresh()

    def complete_task(self):
        task_id = self.require_selected()
        if task_id:
            task = self.store.task(task_id)
            if messagebox.askyesno('완료 확인', f'완료 조건: {task["completion"] or "미입력"}\n\n이 작업을 완료했나요?', parent=self.root):
                self.store.change_state(task_id, 'done')
                self.refresh()

    def reopen_task(self):
        task_id = self.require_selected()
        if task_id:
            self.store.change_state(task_id, 'todo')
            self.refresh()

    def show_recommendation(self):
        try:
            budget = int(self.budget.get())
            if not 1 <= budget <= 1440:
                raise ValueError()
        except ValueError:
            messagebox.showinfo('가용 시간', '1~1440분으로 입력하세요.', parent=self.root)
            return
        self.store.set_setting('budget', budget)
        result = recommend(self.store, budget)
        candidates = result['candidates']
        selected = candidates[0] if candidates else None
        self.decision_id = self.store.trace('next_task', selected['task_id'] if selected else None, result)
        lines = [f'판단 버전: {result["version"]} / 가용 시간: {budget}분', '']
        for i, item in enumerate(candidates[:5], 1):
            lines += [f'{i}. {item["title"]} — 규칙 점수 {item["score"]}',
                      '   ' + ' / '.join(f'{t["name"]} {t["value"]:+d}' for t in item['terms']),
                      f'   예상 잔여 {item["remaining_minutes"]}분 · {item["estimate_source"]}', '']
        if not candidates:
            lines.append('지금 시작 가능한 작업이 없습니다. 대기 상태와 선행 작업을 확인하세요.')
        if selected and selected['remaining_minutes'] > budget:
            lines.append('추천 작업이 가용 시간을 넘습니다. 다음 행동을 작게 나누거나 계획을 직접 수정하세요.')
        lines += ['제외한 작업:'] + [f'  #{x["task_id"]}: {x["reason"]}' for x in result['excluded']]
        lines += ['', '판단의 한계:'] + result['limitations']
        self.write_text(self.explanation, '\n'.join(lines))
        self.banner.set(f'추천: {selected["title"]} · 이유는 “근거와 회고” 탭에서 확인하세요.' if selected else '시작 가능한 작업이 없습니다. 근거와 회고 탭에서 제외 사유를 확인하세요.')

    def give_feedback(self, value):
        if self.decision_id:
            self.store.feedback(self.decision_id, value)
            self.banner.set('피드백을 저장했습니다. 작업 정보가 틀렸다면 작업 수정에서 바로 고칠 수 있습니다.')

    def confirm_link(self):
        task_id = self.require_selected()
        if task_id is None:
            return
        if not self.collection.get() or not self.latest or time.time() - self.latest['at'] > 60:
            messagebox.showinfo('관측 없음', '수집을 켠 뒤 다른 앱을 사용하고 60초 안에 돌아오세요. 최근 관측이 있어야 연결할 수 있습니다.', parent=self.root)
            return
        category = self.latest['category']
        if messagebox.askyesno('범주 연결 확인', f'최근 {CAT_NAMES[category]} 사용이 선택한 작업과 관련 있었나요?\n최근 5분 범주 통계와 작업 종류를 로컬 학습 정답으로 저장합니다.\n브라우저의 페이지 내용 등은 알 수 없습니다.', parent=self.root):
            self.store.learn_link(task_id, category)
            saved = capture(self.store, task_id)
            self.banner.set('연결을 기록했습니다. ' + ('학습 정답도 저장했습니다.' if saved else '학습 정답은 5분에 한 번, 최근 관측이 있을 때 저장합니다.'))

    def inspect_model(self):
        if not self.collection.get():
            messagebox.showinfo('활동 수집 꺼짐', '수집을 직접 켜고 확인할 활동을 기록하세요.', parent=self.root)
            return
        x = context(self.store)
        if x is None:
            messagebox.showinfo('관측 없음', '60초 이내의 관측이 없어 판단을 보류합니다.', parent=self.root)
            return
        path = filedialog.askopenfilename(parent=self.root, title='직접 학습한 JSON 모델 선택', filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            from pathlib import Path
            if Path(path).stat().st_size > 1000000:
                raise ValueError('모델 파일이 너무 큽니다.')
            model = json.loads(Path(path).read_text(encoding='utf-8'))
            if model.get('synthetic', True):
                raise ValueError('합성 데이터 모델은 실제 앱에서 사용할 수 없습니다.')
            result = infer(model, x)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            messagebox.showerror('모델 확인 실패', str(exc), parent=self.root)
            return
        lines = ['학습된 활동 종류 모델 · 작업 자동 전환 없음',
                 '판단 보류' if result['abstain'] else '후보 종류: ' + KIND_NAMES[result['kind']],
                 f"최고 점수 종류: {KIND_NAMES[result['kind']]} / 비교 종류: {KIND_NAMES[result['runner_up']]}",
                 f"점수 {result['score']:.4f} = 절편 {result['bias']:.4f} + 아래 기여 합계",
                 f"softmax {result['softmax']:.3f} / 비교 종류와 점수 차이 {result['margin']:.4f}", result['note'], '']
        for term in result['terms']:
            lines.append(f"{term['feature']}: 입력 {term['input']:.3f}, 점수 기여 {term['contribution']:+.4f}")
        candidates = [t['title'] for t in self.store.tasks() if t['kind']==result['kind'] and t['state'] not in {'done','waiting'}
                      and (not t['dependency'] or self.store.task(t['dependency'])['state']=='done')]
        lines += ['', '해당 종류의 실행 가능한 계획 (같은 종류 안에서는 구별하지 못함):', *candidates]
        self.write_text(self.explanation, '\n'.join(lines))
        self.decision_id = self.store.trace('learned_activity_kind', None, dict(result=result, model=model))
        self.today.master.select(self.review)

    def change_collection(self):
        self.store.set_setting('collection_enabled', self.collection.get())
        self.latest = None
        self.observation.set('수집 대기 중' if self.collection.get() else '활동 수집 꺼짐')

    def save_exclusions(self):
        self.store.set_setting('excluded_categories', [k for k, v in self.exclude_vars.items() if v.get()])
        self.latest = None

    def save_privacy(self):
        try:
            days = int(self.retention.get())
            if not 1 <= days <= 365:
                raise ValueError()
            self.store.set_setting('retention_days', days)
            self.store.prune()
            self.banner.set('보관 기간을 적용했습니다. 계획·세션·개인화 기록의 보관은 별도입니다.')
        except ValueError:
            messagebox.showinfo('보관 기간', '1~365일로 입력하세요.', parent=self.root)

    def clear_observations(self):
        if messagebox.askyesno('기록 삭제', '활동·학습 정답·범주 연결·판단 기록을 삭제할까요? 계획과 세션은 남습니다. 내보낸 모델 파일은 직접 삭제하세요.', parent=self.root):
            self.store.clear_observations()
            self.latest, self.decision_id = None, None
            self.write_text(self.explanation, '활동·판단·범주 개인화 기록을 삭제했습니다.')

    def reset(self):
        if messagebox.askyesno('전체 초기화', '모든 계획·메모·세션·개인화·설정을 삭제합니다. 계속할까요?', parent=self.root):
            self.store.reset()
            self.collection.set(False)
            self.retention.set('7')
            self.budget.set('60')
            for key, var in self.exclude_vars.items():
                var.set(key == 'meeting')
            self.latest, self.decision_id = None, None
            self.observation.set('활동 수집 꺼짐')
            self.write_text(self.explanation, '모든 로컬 데이터를 초기화했습니다.')
            self.refresh()

    def export_summary(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json', initialfile='planwise-summary.json', filetypes=[('JSON', '*.json')])
        if path:
            report = {'schema': 'planwise-aggregate-v1', 'today': daily_summary(self.store),
                      'task_count': len(self.store.tasks()),
                      'feedback_counts': {r[0]: r[1] for r in self.store.db.execute('SELECT feedback,COUNT(*) FROM decisions WHERE feedback IS NOT NULL GROUP BY feedback')}}
            from pathlib import Path
            Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            self.banner.set('작업명·메모·원본 활동을 제외한 집계 결과를 저장했습니다.')

    def tick(self):
        now = time.time()
        if self.store.active():
            self.store.set_setting('last_heartbeat', now)
        if self.collection.get():
            try:
                observation = self.sensor.sample()
                if observation and observation['category'] not in self.store.setting('excluded_categories', ['meeting']):
                    self.latest = dict(observation, at=now)
                    self.store.observe(**observation, at=now)
                    suggestion = suggest_link(self.store, observation['category'])
                    label = CAT_NAMES[observation['category']]
                    self.observation.set(f'최근 관측: {label}' + (' · 5분 이상 입력 없음 (비집중 판정 아님)' if observation['idle'] else '') +
                                         (f' · 연결 후보: {suggestion["title"]}' if suggestion else ''))
                elif observation:
                    self.latest = None
                    self.observation.set('현재 범주는 수집 제외됨')
                elif not self.sensor.available:
                    self.observation.set('이 OS에서는 활동 수집 미지원 · 계획 기능은 사용 가능')
                elif self.latest and now - self.latest['at'] > 60:
                    self.latest = None
                    self.observation.set('최근 관측 없음 · 상태를 추정하지 않음')
            except OSError:
                self.latest = None
                self.observation.set('활동 조회 실패 · 상태를 추정하지 않음')
        due = [t for t in self.store.tasks() if t['state'] == 'waiting' and t['wait_until'] and t['wait_until'] <= now]
        if due:
            self.banner.set('대기 확인 시간: ' + ', '.join(t['title'] for t in due) + ' — 실제 완료 여부를 확인 후 대기를 해제하세요.')
            with self.store.db:
                self.store.db.executemany('UPDATE tasks SET wait_until=NULL WHERE id=?', [(t['id'],) for t in due])
        self.store.prune(now)
        self.refresh()
        self.tick_id = self.root.after(15000, self.tick)

    def close(self):
        if self.tick_id:
            self.root.after_cancel(self.tick_id)
        self.store.pause('앱 종료로 세션 중단')
        self.store.close()
        self.root.destroy()
