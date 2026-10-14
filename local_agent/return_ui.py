import queue
import threading
import tkinter as tk
from tkinter import ttk,messagebox,simpledialog
from local_agent.return_service import return_card,render_card,suggest,checkpoint,ProcessWatch


class ReturnPanel:
    def __init__(self,parent,store,on_resume):
        self.parent,self.store,self.on_resume=parent,store,on_resume
        self.task_id=None
        self.card=None
        self.poll_id=None
        self.queue=queue.Queue()
        self.watches=ProcessWatch()
        ttk.Label(parent,text='어디까지 했는지, 다시 찾아볼까요?',font=('맑은 고딕',16,'bold')).pack(anchor='w')
        ttk.Label(parent,text='기록은 바로 보여드려요. AI 제안은 필요할 때만 받아도 괜찮아요.').pack(anchor='w',pady=8)
        top=ttk.Frame(parent)
        top.pack(fill='x')
        self.choices=ttk.Combobox(top,state='readonly',width=55)
        self.choices.pack(side='left',fill='x',expand=True)
        self.choices.bind('<<ComboboxSelected>>',lambda e:self.show(self.ids[self.choices.current()]))
        ttk.Button(top,text='목록 새로고침',command=self.reload).pack(side='left',padx=8)
        self.facts=tk.Text(parent,wrap='word',height=13,font=('맑은 고딕',11))
        self.facts.pack(fill='both',expand=True,pady=12)
        self.help=tk.StringVar(value='아래 버튼은 실제로 작업 세션을 시작합니다. 앱이 작업을 대신 수행하지는 않아요.')
        ttk.Label(parent,textvariable=self.help,wraplength=980).pack(anchor='w',pady=8)
        bar=ttk.Frame(parent)
        bar.pack(fill='x')
        ttk.Button(bar,text='이어서 시작',command=self.resume).pack(side='left')
        ttk.Button(bar,text='복귀 메모 고치기',command=self.edit).pack(side='left',padx=6)
        self.ai=ttk.Button(bar,text='다음 행동 도움받기',command=self.ask)
        self.ai.pack(side='left',padx=6)
        ttk.Button(bar,text='실행 중인 프로세스 연결',command=self.attach).pack(side='right')
        self.reload()

    def reload(self):
        tasks=self.store.tasks()
        self.ids=[t['id'] for t in tasks]
        self.choices.configure(values=[t['title'] for t in tasks])
        if self.task_id in self.ids:
            self.choices.current(self.ids.index(self.task_id))
            current=return_card(self.store,self.task_id)
            if not self.card or current['revision']!=self.card['revision']:
                self.show(self.task_id)
        elif tasks:
            active=self.store.active()
            self.show(active['task_id'] if active else tasks[0]['id'])

    def show(self,task_id):
        self.task_id=task_id
        self.card=return_card(self.store,task_id)
        self.facts.configure(state='normal')
        self.facts.delete('1.0','end')
        self.facts.insert('1.0',render_card(self.card))
        self.facts.configure(state='disabled')
        self.help.set('저장한 기록만 표시했어요. 필요한 다음 행동을 골라 시작하면 돼요.')
        if task_id in self.ids:
            self.choices.current(self.ids.index(task_id))

    def resume(self):
        if self.task_id:
            try:
                self.on_resume(self.task_id)
            except ValueError as exc:
                messagebox.showinfo('이어가기',str(exc),parent=self.parent)

    def edit(self):
        if not self.task_id:
            return
        text=simpledialog.askstring('다음에 어디서부터 할까요?','한 줄만 적어도 괜찮아요.',parent=self.parent)
        if text is not None:
            checkpoint(self.store,self.task_id,text,text)
            with self.store.db:
                self.store.db.execute('UPDATE tasks SET next_action=? WHERE id=?',(text[:2000],self.task_id))
            self.show(self.task_id)

    def attach(self):
        if not self.task_id:
            return
        pid=simpledialog.askinteger('실행 연결','작업 관리자에서 확인한 프로세스 PID를 입력하세요.\n프로세스를 시작·종료하지 않고 종료 여부만 확인해요. 앱 종료 후에는 다시 연결해야 해요.',minvalue=1,parent=self.parent)
        if pid:
            try:
                self.watches.attach(pid,self.task_id)
                self.help.set(f'PID {pid} 종료를 확인할게요. 종료됐다고 작업 완료로 바꾸지는 않아요.')
            except ValueError as exc:
                messagebox.showinfo('연결 실패',str(exc),parent=self.parent)

    def ask(self):
        if not self.task_id:
            return
        card=return_card(self.store,self.task_id)
        self.ai.configure(state='disabled')
        self.help.set('Qwen이 제안을 준비하고 있어요. 위 기록으로 먼저 시작해도 괜찮아요.')
        def worker():
            try:
                self.queue.put((card,suggest(card),None))
            except Exception as exc:
                self.queue.put((card,None,str(exc)))
        threading.Thread(target=worker,daemon=True).start()
        self.poll_id=self.parent.after(200,self.poll)

    def poll(self):
        self.poll_id=None
        try:
            card,result,error=self.queue.get_nowait()
        except queue.Empty:
            self.poll_id=self.parent.after(200,self.poll)
            return
        self.ai.configure(state='normal')
        if card['task']['id']!=self.task_id or not self.task_id:
            return
        try:
            current=return_card(self.store,self.task_id)
        except ValueError:
            return
        if current['revision']!=card['revision']:
            self.help.set('기록이 바뀌어 예전 AI 제안은 표시하지 않았어요.')
        elif error:
            self.help.set('AI 연결이 잠시 어려워요. 저장된 복귀 카드와 시작 버튼은 그대로 쓸 수 있어요.')
        else:
            label=next(a['label'] for a in card['actions'] if a['id']==result['action_id'])
            self.help.set('AI의 제안 (사실 확인 아님): '+result['suggestion']+'\n가능한 행동: '+label+' · 근거: '+', '.join(result['evidence_ids']))
            self.store.trace('return_suggestion',self.task_id,dict(card=card,result=result))

    def close(self):
        if self.poll_id:
            self.parent.after_cancel(self.poll_id)
        self.watches.close()
