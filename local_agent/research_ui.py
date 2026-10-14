"""Opt-in observations, concept correction and explicit research sample export."""
import json
import time
import uuid
import threading
import queue
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from pathlib import Path
from local_agent.visual_capture import CaptureManager,window_list
from local_agent.model_runtime import ModelRegistry,record_concept,sequence,personalize
from local_agent.model_contract import LABELS,FEATURES,VERSION

NAMES={'camera':'카메라','screen':'화면','context':'시간 맥락'}
DISPLAY={'screen_facing':'화면 방향','away':'다른 방향','unobservable':'판별 어려움',
         'code':'코드','document':'문서','terminal':'터미널','video':'영상','other':'기타',
         'direct':'직접 작업','support':'관련 자료 확인','waiting':'대기','interrupted':'중단','unknown':'모름'}


class ResearchPanel:
    def __init__(self,parent,store,task_provider=lambda:None):
        self.parent,self.store=parent,store
        self.task_provider=task_provider
        self.registry=ModelRegistry()
        self.capture=CaptureManager()
        self.store.set_setting('screen_text_enabled',False)
        self.last_context=None
        self.vision_queue=queue.Queue()
        self.vision_poll=None
        self.vision_busy=False
        self.episode=str(uuid.uuid4())
        ttk.Label(parent,text='관측과 학습 · 필요한 때만 켜세요',font=('맑은 고딕',15,'bold')).pack(anchor='w')
        ttk.Label(parent,text='기본은 모두 꺼짐입니다. 한 번 확인할 때만 촬영하며 원본은 저장하지 않아요.\n미학습 모델은 판단하지 않습니다. 관측 없이도 복귀 카드와 계획은 사용할 수 있어요.',wraplength=980).pack(anchor='w',pady=8)
        self.status={}
        self.flags={}
        for kind in ['camera','screen','context']:
            row=ttk.Frame(parent)
            row.pack(fill='x',pady=6)
            self.status[kind]=tk.StringVar(value=NAMES[kind]+': 모델 없음 · 판단 보류')
            ttk.Label(row,textvariable=self.status[kind],width=48).pack(side='left')
            ttk.Button(row,text='학습 모델 불러오기',command=lambda k=kind:self.load_model(k)).pack(side='left',padx=6)
            if kind!='context':
                flag=tk.BooleanVar(value=False)
                self.flags[kind]=flag
                ttk.Checkbutton(row,text='이번 실행에서 허용',variable=flag,command=lambda k=kind:self.toggle(k)).pack(side='left')
                ttk.Button(row,text='한 번 확인',command=lambda k=kind:self.observe(k)).pack(side='left',padx=5)
        ttk.Button(parent,text='캡처할 창 선택',command=self.choose_window).pack(anchor='w',pady=8)
        ttk.Label(parent,text='화면 창 제목은 선택용으로만 읽습니다. 선택한 창이 바뀌면 다시 선택해야 합니다.\n카메라는 얼굴 영역만 분석합니다. 인식 실패를 비집중으로 해석하지 않습니다.',wraplength=980).pack(anchor='w')
        ttk.Separator(parent).pack(fill='x',pady=12)
        row=ttk.Frame(parent)
        row.pack(fill='x')
        ttk.Button(row,text='현재 맥락과 근거 확인',command=self.context).pack(side='left')
        self.label=ttk.Combobox(row,state='readonly',values=[DISPLAY[x] for x in LABELS['context']],width=18)
        self.label.current(4)
        self.label.pack(side='left',padx=8)
        ttk.Button(row,text='내가 확인한 상태로 교정',command=self.correct_context).pack(side='left')
        ttk.Button(row,text='맥락 학습 기록 내보내기',command=self.export).pack(side='right')
        self.text=tk.Text(parent,height=12,wrap='word',font=('맑은 고딕',10))
        self.text.pack(fill='both',expand=True,pady=12)
        self.write('카메라·화면 CNN과 GRU의 모델 파일을 각각 불러올 수 있어요.\n학습 데이터는 이 화면에서 직접 확인·저장한 샘플로 준비합니다.\n모델이 없으면 수집 기능만 사용할 수 있고 예측은 하지 않습니다.')

    def write(self,text):
        self.text.configure(state='normal')
        self.text.delete('1.0','end')
        self.text.insert('1.0',text)
        self.text.configure(state='disabled')

    def episode_key(self):
        return self.episode+'-'+time.strftime('%Y-%m-%d')

    def toggle(self,kind):
        self.capture.enabled[kind]=self.flags[kind].get()
        if kind=='screen':
            self.store.set_setting('screen_text_enabled',self.flags[kind].get())
        if not self.flags[kind].get():
            # A disabled source must not keep contributing its recent prediction.
            record_concept(self.store,kind,dict(status='disabled',probabilities=None,abstain=True))
            self.context()

    def load_model(self,kind):
        path=filedialog.askopenfilename(parent=self.parent,title=NAMES[kind]+' 모델',filetypes=[('PyTorch','*.pt')])
        if path:
            try:
                digest=self.registry.load(kind,path)
                self.status[kind].set(NAMES[kind]+': 모델 준비 · '+digest[:10])
            except Exception as exc:
                messagebox.showinfo('모델 불러오기',str(exc)+'\nPyTorch가 있는 실행 환경이 필요합니다.',parent=self.parent)

    def choose_window(self):
        try:
            windows=window_list()
        except Exception as exc:
            messagebox.showinfo('창 선택',str(exc),parent=self.parent)
            return
        dialog=tk.Toplevel(self.parent)
        dialog.title('이 창만 캡처하도록 선택')
        choices=ttk.Combobox(dialog,state='readonly',width=80,values=[w['title'] for w in windows])
        choices.pack(padx=15,pady=15)
        def choose():
            if choices.current()>=0:
                self.capture.window=windows[choices.current()]
                dialog.destroy()
        ttk.Button(dialog,text='이 창 선택',command=choose).pack(pady=10)

    def observe(self,kind):
        observed_at=time.time()
        try:
            image=self.capture.capture(kind)
            result=self.registry.predict(kind,image)
        except Exception as exc:
            messagebox.showinfo('관측 보류',str(exc),parent=self.parent)
            return
        # No raw pixels enter persistent storage until the separate save button.
        dialog=tk.Toplevel(self.parent)
        dialog.title('이번 관측 확인 · 저장 전 미리보기')
        from PIL import ImageTk
        preview=image.copy()
        preview.thumbnail((660,360))
        photo=ImageTk.PhotoImage(preview)
        view=ttk.Label(dialog,image=photo)
        view.image=photo
        view.pack(padx=12,pady=12)
        status='모델이 없어 판단하지 않았어요.' if result['status']=='untrained' else ('판단 보류 · ' if result['abstain'] else '')+DISPLAY.get(result['label'],result['label'])
        ttk.Label(dialog,text=status).pack()
        labels=ttk.Combobox(dialog,state='readonly',values=[DISPLAY[x] for x in LABELS[kind]])
        labels.pack(pady=6)
        labels.current(LABELS[kind].index(result['label']) if result.get('label') in LABELS[kind] else len(LABELS[kind])-1)
        recorded=[False]
        def use():
            if not self.capture.enabled[kind]:
                return
            label=LABELS[kind][labels.current()]
            corrected=dict(status='user_corrected',label=label,probabilities=[float(x==label) for x in LABELS[kind]],
                           abstain=False,source='user',original=result)
            record_concept(self.store,kind,corrected,at=observed_at)
            recorded[0]=True
            self.write(NAMES[kind]+' 개념을 사용자가 확인한 값으로 기록했어요. 원본 이미지는 저장하지 않았어요.\n'+json.dumps(corrected,ensure_ascii=False,indent=2))
            dialog.destroy()
            self.context()
        def save_sample():
            target=filedialog.askdirectory(parent=dialog,title='검토한 이미지와 정답을 보관할 연구 폴더')
            if not target:
                return
            if not messagebox.askyesno('연구 이미지 저장','선택한 이미지 원본과 정답을 이 폴더에 저장합니다. 개인정보가 없는지 확인했나요? 앱의 기록 삭제로 이 파일은 지워지지 않습니다.',parent=dialog):
                return
            sample_id=str(uuid.uuid4())
            directory=Path(target)
            image.save(directory/(sample_id+'.png'))
            row=dict(path=sample_id+'.png',label=LABELS[kind][labels.current()],participant='local-user',episode=self.episode_key(),at=observed_at,split='unassigned',kind=kind)
            with (directory/(kind+'.jsonl')).open('a',encoding='utf-8') as out:
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
            self.write('연구 샘플을 저장했어요. 학습 전 participant/episode 단위 split을 지정해야 합니다.')
        ttk.Button(dialog,text='이 개념으로 맥락 갱신 (이미지는 폐기)',command=use).pack(pady=5)
        if result.get('status')=='predicted':
            def accept_prediction():
                if self.capture.enabled[kind]:
                    record_concept(self.store,kind,result,at=observed_at)
                dialog.destroy()
                self.context()
            ttk.Button(dialog,text='모델 추정 그대로 사용',command=accept_prediction).pack(pady=5)
        ttk.Button(dialog,text='별도로 연구 이미지·정답 저장',command=save_sample).pack(pady=5)
        if kind=='screen':
            ttk.Button(dialog,text='이 화면 문구 읽기 (로컬 Qwen · 추가 선택)',command=lambda:self.ask_vision(image,observed_at)).pack(pady=5)
        ttk.Button(dialog,text='폐기',command=dialog.destroy).pack(pady=5)

    def ask_vision(self,image,observed_at):
        if self.vision_busy or not self.capture.enabled['screen']:
            return
        active=self.store.active()
        task_id=active['task_id'] if active else self.task_provider()
        self.vision_busy=True
        self.write('선택한 화면만 로컬 Qwen으로 읽고 있어요. 읽은 문구는 추정이며 사실 확인이 필요해요.')
        def worker():
            from local_agent.screen_text import read_screen
            try:
                self.vision_queue.put((task_id,observed_at,read_screen(image),None))
            except Exception as exc:
                self.vision_queue.put((task_id,observed_at,None,str(exc)))
        threading.Thread(target=worker,daemon=True).start()
        self.vision_poll=self.parent.after(200,self.poll_vision)

    def poll_vision(self):
        self.vision_poll=None
        try:
            task_id,observed_at,result,error=self.vision_queue.get_nowait()
        except queue.Empty:
            self.vision_poll=self.parent.after(200,self.poll_vision)
            return
        self.vision_busy=False
        if not self.capture.enabled['screen'] or time.time()-observed_at>120:
            self.write('관측 허용이 꺼졌거나 화면이 오래되어 응답을 사용하지 않았어요.')
        elif error:
            self.write('화면 문구를 읽지 못했어요. 화면 CNN과 수동 개념 확인은 별도로 사용할 수 있어요.')
        else:
            result['task_id']=task_id
            record_concept(self.store,'screen_text',result,at=observed_at)
            self.write('화면에서 읽은 후보 문구 (검증 안 됨):\n'+result['visible_text']+'\n\n불확실한 부분: '+result['uncertainties'])

    def close(self):
        if self.vision_poll:
            self.parent.after_cancel(self.vision_poll)

    def context(self):
        active=self.store.active()
        task_id=active['task_id'] if active else self.task_provider()
        inputs=sequence(self.store,task_id,enabled=self.capture.enabled)
        result=personalize(self.store,self.registry.predict('context',inputs))
        result['task_id']=task_id
        result['input_features']=inputs
        result['feature_names']=FEATURES
        result['concept_record_ids']=[r[0] for r in self.store.db.execute("SELECT id FROM concepts WHERE modality IN ('camera','screen') AND at>=? ORDER BY at,id",(time.time()-600,))]
        record_concept(self.store,'context',result)
        self.last_context=dict(features=inputs,predicted_label=result.get('base_label',result.get('label')),result=result,at=time.time())
        lines=['맥락: '+DISPLAY.get(result.get('label'),'미학습 · 판단 보류'),result.get('note','학습된 맥락 모델이 없습니다.'),'', '현재 입력 (미관측은 0과 마스크로 구분):']
        lines += [f'{name}: {value:.3f}' for name,value in zip(FEATURES,inputs[-1])]
        lines += ['', '모델/개인 보정 결과:',json.dumps({k:v for k,v in result.items() if k not in {'input_features','feature_names'}},ensure_ascii=False,indent=2)]
        self.write('\n'.join(lines))

    def correct_context(self):
        if not self.last_context or time.time()-self.last_context['at']>60:
            self.context()
        label=LABELS['context'][self.label.current()]
        data={k:v for k,v in self.last_context.items() if k!='result'}
        with self.store.db:
            last=self.store.db.execute('SELECT id,at FROM context_labels WHERE episode=? ORDER BY id DESC LIMIT 1',(self.episode_key(),)).fetchone()
            if last and time.time()-last['at']<60:
                self.store.db.execute('UPDATE context_labels SET at=?,label=?,data=? WHERE id=?',(time.time(),label,json.dumps(data),last['id']))
            else:
                self.store.db.execute('INSERT INTO context_labels(at,participant,episode,label,data) VALUES (?,?,?,?,?)',
                                     (time.time(),'local-user',self.episode_key(),label,json.dumps(data)))
        self.write('사용자가 확인한 상태를 학습 정답으로 기록했어요: '+DISPLAY[label]+'\n모델 추정만으로 정답을 만들지 않습니다.')

    def export(self):
        path=filedialog.asksaveasfilename(parent=self.parent,defaultextension='.jsonl',initialfile='context-labels.jsonl')
        if not path:
            return
        self.store.prune()
        rows=[]
        for r in self.store.db.execute('SELECT * FROM context_labels ORDER BY at'):
            data=json.loads(r['data'])
            rows.append(dict(participant=r['participant'],episode=r['episode'],at=r['at'],label=r['label'],features=data['features'],split='unassigned',version=VERSION))
        Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        self.write(f'{len(rows)}개 정답을 내보냈어요. 학습 전에 날짜/에피소드 기준으로 train/val/test를 지정하세요.')

    def reset(self):
        for kind in self.flags:
            self.flags[kind].set(False)
            self.capture.enabled[kind]=False
        self.capture.window=None
        self.store.set_setting('screen_text_enabled',False)
        self.last_context=None
        self.episode=str(uuid.uuid4())
        self.write('관측을 껐어요. 원본 연구 파일과 학습 모델은 직접 삭제해야 합니다.')
