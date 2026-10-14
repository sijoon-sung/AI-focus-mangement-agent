"""Visible controls and previews for the app's own local observation loop."""
import time
import customtkinter as ctk
from local_agent.live_observation import ObservationGate, capture_foreground, capture_webcam, default_specialists, analyze_screen


class ObservationPanel:
    def __init__(self,app):
        self.app=app;self.gate=ObservationGate();self.history=[];self.history_ids=[];self.preview=None
        self.window=app.dialog('화면과 웹캠 · 로컬 관측',760,700)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        app.label(self.window,'지금 하고 있는 일과 연결하기',22,True).pack(anchor='w',padx=24,pady=(20,8))
        app.label(self.window,'화면은 60초마다, 웹캠은 10초마다 한 장씩 확인합니다.\n원본 사진은 저장하지 않습니다. 화면 문구는 선택한 작업의 근거로 저장됩니다.',
                  color='#788397',justify='left').pack(anchor='w',padx=24)
        self.vars={};self.status={}
        for kind,title in [('screen','현재 화면 읽기'),('camera','웹캠 확인')]:
            var=ctk.BooleanVar(value=False);self.vars[kind]=var
            ctk.CTkSwitch(self.window,text=title,variable=var,command=lambda k=kind:self.set_enabled(k,self.vars[k].get())).pack(anchor='w',padx=24,pady=(16,4))
            self.status[kind]=app.label(self.window,'꺼짐',12,justify='left',wraplength=690)
            self.status[kind].pack(anchor='w',padx=24)
        self.camera_preview=app.label(self.window,'웹캠 미리보기 · 시선 CNN의 실시간 입력 보정은 아직 미연결',12)
        self.camera_preview.pack(padx=24,pady=8)
        self.output=ctk.CTkTextbox(self.window,height=160,wrap='word',font=app.font)
        self.output.pack(fill='both',expand=True,padx=24,pady=8)
        self.show('화면 읽기를 켜고 평소 작업 창으로 전환하세요. 다음 확인 결과가 여기에 표시됩니다.')
        bar=ctk.CTkFrame(self.window,fg_color='transparent');bar.pack(fill='x',padx=24,pady=12)
        app.button(bar,'화면 다시 확인 (5초 뒤)',lambda:app.later(5000,self.screen_once)).pack(side='left')
        app.button(bar,'모두 끄기',self.stop).pack(side='right')

    def show(self,text):
        self.output.configure(state='normal');self.output.delete('1.0','end');self.output.insert('1.0',text);self.output.configure(state='disabled')

    def set_enabled(self,kind,enabled):
        self.vars[kind].set(enabled)
        token=self.gate.set(kind,enabled)
        self.status[kind].configure(text='켜짐 · 첫 확인 준비 중' if enabled else '꺼짐 · 이전 요청 결과도 폐기합니다.')
        self.app.observation_button.configure(text='● 화면·웹캠' if any(self.gate.enabled.values()) else '화면·웹캠')
        if not enabled:
            if kind=='camera':
                self.camera_preview.configure(image=None,text='웹캠 꺼짐');self.preview=None
            else:self.history=[];self.history_ids=[]
            return
        def cycle():
            if not self.gate.accepts(kind,token):return
            if kind=='screen':self.screen_once()
            else:self.camera_once()
            self.app.later(60000 if kind=='screen' else 10000,cycle)
        self.app.later(5000 if kind=='screen' else 100,cycle)

    def screen_once(self):
        if not self.gate.enabled['screen'] or any(k in self.app.jobs for k in ['live_screen','screen','help','specialist_load']):return
        token=self.gate.generation['screen'];task_id=self.app.selected;privacy=self.app.privacy_revision
        from local_agent.return_service import return_card
        revision=return_card(self.app.store,task_id)['revision'] if task_id else None
        goals=[t['title'] for t in self.app.store.tasks() if t['state']!='done'][:30]
        history=list(self.history);prior_ids=list(self.history_ids);registry=self.app.specialists
        self.status['screen'].configure(text='현재 화면 읽는 중 · 로컬 Qwen + 화면 CNN + 목표 이력')
        def work():
            image=capture_foreground();at=time.time()
            try:
                selected_registry=registry
                load_error=None
                if selected_registry is None:
                    try:selected_registry=default_specialists()
                    except Exception as exc:load_error=str(exc)[:160]
                result=analyze_screen(image,goals,history,selected_registry)
                if load_error:result['specialist_errors'].append(load_error)
                return result,at,selected_registry
            finally:image.close()
        def ready(payload,error):
            if not self.gate.accepts('screen',token) or privacy!=self.app.privacy_revision:return
            if error:
                self.status['screen'].configure(text='확인 못함 · '+str(error)[:200]);return
            result,at,loaded=payload
            if task_id!=self.app.selected or (task_id and return_card(self.app.store,task_id)['revision']!=revision):
                self.history=[];self.history_ids=[];self.status['screen'].configure(text='작업이 바뀌어 이전 결과는 폐기했어요.');return
            if self.app.specialists is None:self.app.specialists=loaded
            self.history=(self.history+[result['visible_text']])[-8:]
            stamp=time.strftime('%H:%M:%S',time.localtime(at))
            details=[f'{stamp} 캡처 · 읽은 내용 (확인 필요)',result['visible_text'],result.get('uncertainties','')]
            for item in result['specialists']:
                if item['role']=='goal_history':
                    candidates=sorted(item['candidates'],key=lambda x:x['compatibility'],reverse=True)
                    details.append('목표 관련성 추정 (확률 아님): '+', '.join(f"{x['goal']} {x['compatibility']:.2f}" for x in candidates[:3]))
                else:details.append('화면 CNN 실행됨 · 8종 UI 요소 점유 추정 / 예비 모델')
            details.extend(result['specialist_errors'])
            if task_id:
                from local_agent.model_runtime import record_concept
                text_id=record_concept(self.app.store,'screen_text',dict(visible_text=result['visible_text'],uncertainties=result.get('uncertainties',''),source=result['source'],task_id=task_id,captured_at=at),at=at)
                for item in result['specialists']:
                    record_concept(self.app.store,'specialist',dict(item,task_id=task_id,captured_at=at,observation_ids=prior_ids+[text_id]),at=at)
                self.history_ids=(prior_ids+[text_id])[-8:]
                self.app.store.set_setting('screen_text_enabled',True)
                details.append('선택한 작업의 근거에 연결했어요. ‘다음 한 걸음’에서 복귀 제안을 받을 수 있어요.')
            else:
                self.history=[];self.history_ids=[]
                details.append('작업을 고르면 계획과 연결합니다. 현재는 화면 확인만 했어요.')
            self.show('\n\n'.join(x for x in details if x))
            self.status['screen'].configure(text=f'{stamp} 확인 완료 · 분석 {result["seconds"]}초 · 다음 확인 대기')
        self.app.job('live_screen',work,ready)

    def camera_once(self):
        if not self.gate.enabled['camera'] or 'live_camera' in self.app.jobs:return
        token=self.gate.generation['camera']
        def ready(result,error):
            if not self.gate.accepts('camera',token):
                if result:result['image'].close()
                return
            if error:
                self.status['camera'].configure(text='확인 못함 · '+str(error)[:200]);return
            image=result['image']
            image.thumbnail((320,180))
            self.preview=ctk.CTkImage(light_image=image,dark_image=image,size=image.size)
            self.camera_preview.configure(image=self.preview,text='')
            stamp=time.strftime('%H:%M:%S',time.localtime(result['captured_at']))
            quality=' · 영상이 매우 어두움: 렌즈 가림/조명을 확인하세요' if result.get('mean_brightness',255)<15 else ''
            self.status['camera'].configure(text=f'{stamp} 영상 수신 · 얼굴 {result["face_count"]}개 검출{quality} · 시선/집중도는 판단하지 않음')
        self.app.job('live_camera',capture_webcam,ready)

    def stop(self):
        for kind in ['screen','camera']:self.set_enabled(kind,False)

    def close(self):
        self.stop();self.window.destroy();self.app.observation_panel=None
