"""Scratchpad UI. Only an explicit button sends the note to local Ollama."""
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from local_agent.inbox import build_draft, apply_draft
from local_agent.note_parser import organize_with_granite


class InboxPanel:
    def __init__(self, parent, store, on_apply):
        self.parent, self.store, self.on_apply = parent, store, on_apply
        self.save_id = self.poll_id = None
        self.closed = False
        self.draft = None
        self.results = queue.Queue()
        ttk.Label(parent, text='오늘 뭐 하고 싶어요?', font=('맑은 고딕', 17, 'bold')).pack(anchor='w')
        ttk.Label(parent, text='생각나는 대로 적어 주세요. 순서나 시간은 몰라도 괜찮아요.\n예: 논문 좀 읽고 보고서도 써야 해. 메일 답장부터 끝내고 싶어.', wraplength=1000).pack(anchor='w', pady=6)
        self.notes = tk.Text(parent, height=7, wrap='word', undo=True, font=('맑은 고딕', 11))
        self.notes.pack(fill='both', expand=True)
        self.notes.insert('1.0', store.setting('inbox_text', ''))
        self.notes.edit_modified(False)
        self.notes.bind('<<Modified>>', self.changed)
        self.status = tk.StringVar(value='메모는 자동으로 저장돼요. 내일 다시 고쳐 써도 괜찮아요.')
        ttk.Label(parent, textvariable=self.status, wraplength=1000).pack(anchor='w', pady=6)
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=5)
        self.budget = tk.StringVar(value=str(store.setting('inbox_budget', 240)))
        self.ai_button = ttk.Button(row, text='할 일 정리해줘', command=self.organize)
        self.ai_button.pack(side='left', padx=4)
        self.budget_label = tk.StringVar(value=f'오늘 여유 {int(self.budget.get())/60:g}시간 기준 · 바꾸기')
        ttk.Button(row,textvariable=self.budget_label,command=self.change_budget).pack(side='right',padx=4)
        self.first = tk.StringVar(value='적어 주면, 어디부터 시작할지 함께 골라볼게요.')
        ttk.Label(parent,textvariable=self.first,font=('맑은 고딕',13,'bold'),wraplength=1000).pack(anchor='w',pady=(16,8))
        table = ttk.Frame(parent)
        table.pack(fill='both', expand=True, pady=5)
        self.tree = ttk.Treeview(table, columns=('task','duration'), show='headings', height=4)
        for key, title, width in [('task','이어서 할 일',600), ('duration','대략 이 정도',180)]:
            self.tree.heading(key, text=title)
            self.tree.column(key,width=width,stretch=key=='task')
        scroll = ttk.Scrollbar(table,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left',fill='both',expand=True)
        scroll.pack(side='right',fill='y')
        self.tree.bind('<<TreeviewSelect>>',self.details)
        self.reason = tk.StringVar(value='시간은 참고용이에요. 해 본 기록이 쌓이면 예상에 반영할게요.')
        ttk.Label(parent,textvariable=self.reason,wraplength=1000).pack(anchor='w',pady=5)
        row = ttk.Frame(parent)
        row.pack(fill='x')
        self.apply_button = ttk.Button(row,text='좋아, 시작하자',command=self.apply,state='disabled')
        self.apply_button.pack(side='left')
        ttk.Button(row,text='조금 고칠래',command=self.edit_outline).pack(side='left',padx=8)
        ttk.Button(row,text='이유·변경 내용 보기',command=self.show_reasons).pack(side='right')

    def text(self):
        return self.notes.get('1.0','end-1c')

    def changed(self, event=None):
        if not self.notes.edit_modified():
            return
        self.notes.edit_modified(False)
        self.draft = None
        self.apply_button.configure(state='disabled')
        self.status.set('메모를 바꿨어요. 다 적으면 “할 일 정리해줘”를 눌러 주세요.')
        if self.save_id:
            self.parent.after_cancel(self.save_id)
        self.save_id = self.parent.after(700,self.save)

    def save(self):
        if self.save_id:
            self.parent.after_cancel(self.save_id)
        self.save_id = None
        self.store.set_setting('inbox_text',self.text())

    def preview(self, outline=None):
        try:
            self.save()
            budget = int(self.budget.get())
            draft = build_draft(self.store,self.text() if outline is None else outline,budget)
        except ValueError as exc:
            messagebox.showinfo('메모 확인',str(exc),parent=self.parent)
            return
        self.store.set_setting('inbox_budget',budget)
        self.draft, self.source_text = draft,self.text()
        self.apply_button.configure(state='normal' if draft['items'] else 'disabled')
        self.tree.delete(*self.tree.get_children())
        ordered = sorted(enumerate(draft['items']),key=lambda pair: pair[1]['start'] if pair[1]['start'] is not None else 10000)
        for i,item in ordered:
            suffix = item['blocked'] or ('다음으로 미뤄도 돼요' if item['start'] is None else f"{item['estimate']['remaining']}분쯤")
            self.tree.insert('', 'end', iid=str(i), values=(item['title'],suffix))
        first = next((i for i in draft['items'] if i['start']==0),None)
        self.first.set(f"먼저 ‘{first['title']}’부터 해볼까요?" if first else '오늘 시작할 일을 조금 줄이거나 여유 시간을 바꿔도 좋아요.')
        changes = sum(i['operation']=='수정' for i in draft['items'])
        self.status.set(f"{len(draft['items'])}개로 정리했어요. 순서와 시간은 바꿔도 괜찮아요." + (f' 기존 작업 {changes}개 수정이 포함돼요.' if changes else ''))

    def organize(self):
        text = self.text().strip()
        lines = [x for x in text.splitlines() if x.strip()]
        if not text:
            self.status.set('할 일을 한 가지만 적어도 괜찮아요.')
        else:
            self.ask_granite()

    def change_budget(self):
        from tkinter import simpledialog
        minutes = simpledialog.askinteger('오늘 여유','오늘 쓸 수 있는 시간은 대략 몇 분인가요?',initialvalue=int(self.budget.get()),minvalue=1,maxvalue=1440,parent=self.parent)
        if minutes:
            self.budget.set(str(minutes))
            self.budget_label.set(f'오늘 여유 {minutes/60:g}시간 기준 · 바꾸기')
            self.store.set_setting('inbox_budget',minutes)
            if self.draft:
                self.preview(self.draft['text'])

    def show_reasons(self):
        if not self.draft:
            return
        dialog = tk.Toplevel(self.parent)
        dialog.title('배치 이유와 변경 내용')
        box = tk.Text(dialog,wrap='word',width=80,height=22,font=('맑은 고딕',10))
        box.pack(fill='both',expand=True,padx=12,pady=12)
        lines = ['시간은 완료 작업의 세션 기록을 참고한 예상이에요. 집중 시간이나 확정 마감은 아니에요.', '']
        for item in self.draft['items']:
            slot = item['blocked'] or (f"시작 후 {item['start']}~{item['end']}분, 여유 {item['reserve']}분" if item['start'] is not None else '오늘 시간 밖')
            lines += [f"{item['operation']}: {item['title']}",f"  {item['estimate']['source']} · {slot}"]
            if item['operation']=='수정':
                lines.append(f"  이전 이름: {item['previous_title']} · 세션 기록 유지")
        lines += ['', '메모에서 지운 작업과 세션은 삭제하지 않아요. 같은 완료 작업은 자동으로 다시 열지 않아요.']
        box.insert('1.0','\n'.join(lines))
        box.configure(state='disabled')

    def details(self,event=None):
        if not self.draft or not self.tree.selection():
            return
        item = self.draft['items'][int(self.tree.selection()[0])]
        est = item['estimate']
        observed = f" · 과거 관측 범위 {est['observed_range'][0]}~{est['observed_range'][1]}분 (신뢰구간 아님)" if est['observed_range'] else ''
        old = f" · 기존 ‘{item['previous_title']}’의 기록 유지" if item['operation']=='수정' else ''
        self.reason.set(est['source']+observed+old+' · 추정 종류: '+item['kind'])

    def edit_outline(self, outline=None):
        if outline is None and not self.draft:
            self.preview()
        if outline is None and not self.draft:
            return
        dialog = tk.Toplevel(self.parent)
        dialog.title('할 일 목록 검토 · 원본 메모는 유지됩니다')
        dialog.geometry('720x480')
        ttk.Label(dialog,text='원본: '+self.text()[:350],wraplength=680).pack(anchor='w',padx=15,pady=(10,0))
        editor = tk.Text(dialog,wrap='word',font=('맑은 고딕',11))
        editor.pack(fill='both',expand=True,padx=15,pady=15)
        editor.insert('1.0',outline if outline is not None else self.draft['text'])
        ttk.Label(dialog,text='누락·추가·우선순위를 확인하세요. 한 줄에 한 작업, 먼저 할 일은 !, 예상 시간은 [40분].',wraplength=680).pack()
        def use():
            value = editor.get('1.0','end-1c')
            self.preview(value)
            dialog.destroy()
        ttk.Button(dialog,text='이 목록으로 미리보기',command=use).pack(pady=12)
        dialog.transient(self.parent.winfo_toplevel())
        dialog.grab_set()

    def apply(self):
        if not self.draft or self.text()!=self.source_text or str(self.draft['budget'])!=self.budget.get():
            messagebox.showinfo('미리보기 필요','변경된 메모와 가용 시간으로 미리보기를 다시 만들어 주세요.',parent=self.parent)
            return
        mapping = apply_draft(self.store,self.draft)
        first_index = next((i for i,item in enumerate(self.draft['items']) if item['start']==0),None)
        first_id = mapping[first_index]['task_id'] if first_index is not None else None
        outline = self.draft['text']
        self.on_apply(first_id)
        self.preview(outline)

    def ask_granite(self):
        text = self.text()
        if not text.strip() or len(text)>20000:
            messagebox.showinfo('메모 확인','1~20,000자 메모를 입력하세요.',parent=self.parent)
            return
        self.save()
        self.ai_button.configure(state='disabled')
        self.status.set('적어 준 내용을 정리하고 있어요. 메모는 계속 써도 괜찮아요.')
        def work():
            try:
                self.results.put((text,organize_with_granite(text),None))
            except Exception as exc:
                self.results.put((text,None,str(exc)))
        threading.Thread(target=work,daemon=True).start()
        self.poll_id = self.parent.after(200,self.poll)

    def poll(self):
        self.poll_id = None
        try:
            source,result,error = self.results.get_nowait()
        except queue.Empty:
            self.poll_id = self.parent.after(200,self.poll)
            return
        self.ai_button.configure(state='normal')
        if source!=self.text():
            self.status.set('정리 중 메모가 변경되어 오래된 AI 결과를 적용하지 않았습니다. 다시 요청해 주세요.')
        elif error:
            self.preview(source)
            self.status.set('Granite 연결이 어려워 줄별로만 정리했어요. 과목 분리는 아직이에요.')
        else:
            self.preview(result['outline'])
            self.status.set('이렇게 나눠 봤어요. '+(' '.join(result['notices']) if result['notices'] else '빠진 일은 없는지만 보고 바로 시작해도 좋아요.'))

    def reset(self):
        self.notes.delete('1.0','end')
        self.draft = None
        self.tree.delete(*self.tree.get_children())

    def close(self):
        for handle in (self.save_id,self.poll_id):
            if handle:
                self.parent.after_cancel(handle)
        self.save_id = self.poll_id = None
        self.save()
