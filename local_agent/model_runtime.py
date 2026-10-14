"""Refuse synthetic/untrained models; missing modalities remain missing."""
import json
import math
import time
from pathlib import Path
from local_agent.model_contract import VERSION,LABELS,FEATURES,APP_CATEGORIES,TASK_KINDS


class ModelRegistry:
    def __init__(self):
        self.loaded={}

    def load(self,kind,path):
        import torch
        from research.return_models import make_model
        model_path=Path(path)
        if model_path.stat().st_size>100*1024*1024:
            raise ValueError('모델 파일이 너무 큽니다.')
        data=torch.load(model_path,map_location='cpu',weights_only=True)
        if data.get('version')!=VERSION or data.get('kind')!=kind or data.get('labels')!=LABELS[kind]:
            raise ValueError('지원하는 모델 종류/스키마가 아닙니다.')
        if data.get('synthetic',True) or not data.get('trained',False):
            raise ValueError('합성·미학습 모델은 실제 관측에 사용할 수 없습니다.')
        if kind=='context' and data.get('features')!=FEATURES:
            raise ValueError('맥락 특징 순서가 다릅니다.')
        temperature=data.get('temperature',1.)
        if not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or not .1<=temperature<=10:
            raise ValueError('유효하지 않은 보정값입니다.')
        model=make_model(kind)
        model.load_state_dict(data['state_dict'],strict=True)
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError('유효하지 않은 가중치입니다.')
        model.eval()
        import hashlib
        digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.loaded[kind]=(model,float(temperature),digest)
        return digest

    def predict(self,kind,value):
        if kind not in self.loaded:
            return dict(status='untrained',kind=kind,label='unknown',probabilities=None)
        import torch
        model,temp,digest=self.loaded[kind]
        if kind=='context':
            x=torch.tensor(value,dtype=torch.float32).unsqueeze(0)
            if x.ndim!=3 or x.shape[-1]!=len(FEATURES) or x.shape[1]>64 or x.shape[1]<1 or not torch.isfinite(x).all() or x.min()<0 or x.max()>1:
                raise ValueError('맥락 입력 형식이 다릅니다.')
            x=x[:,-16:]
            if x.shape[1]<16:
                x=torch.cat([torch.zeros(1,16-x.shape[1],len(FEATURES)),x],dim=1)
        else:
            import numpy as np
            array=np.asarray(value.convert('RGB').resize((128,128)),dtype='float32')/255
            x=torch.from_numpy(array).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            probs=torch.softmax(model(x)/temp,dim=-1)[0].tolist()
        index=max(range(len(probs)),key=probs.__getitem__)
        missing_now=kind=='context' and not any(value[-1][8:10]+value[-1][13:19])
        return dict(status='predicted',kind=kind,label=LABELS[kind][index],probabilities=probs,
                    confidence=probs[index],abstain=probs[index]<.6 or missing_now,missing_now=missing_now,model_digest=digest,
                    note='개념/상태 추정이며 집중도나 작업 완료의 증거가 아닙니다.')


def record_concept(store,kind,result,at=None):
    with store.db:
        return store.db.execute('INSERT INTO concepts(at,modality,data) VALUES (?,?,?)',
            (time.time() if at is None else at,kind,json.dumps(result,ensure_ascii=False))).lastrowid


def sequence(store,task_id=None,now=None,enabled=None):
    now=time.time() if now is None else now
    enabled=enabled or {'camera':False,'screen':False}
    rows=[dict(r) for r in store.db.execute('SELECT * FROM concepts WHERE at>=? AND at<=? ORDER BY at,id',(now-600,now))]
    # Only the past is visible at each time step. Add current time to expose age.
    instants=sorted({r['at'] for r in rows} | {now})[-16:]
    kind=store.task(task_id)['kind'] if task_id else 'general'
    seq=[]
    for instant in instants:
        values=[]
        masks=[]
        ages=[]
        for modality in ['camera','screen']:
            eligible=[r for r in rows if r['modality']==modality and r['at']<=instant]
            last=eligible[-1] if eligible else None
            age=instant-last['at'] if last else 300
            data=json.loads(last['data']) if last else {}
            available=enabled.get(modality,False) and age<=60 and data.get('probabilities') is not None and not data.get('abstain',True)
            values += data['probabilities'] if available else [0.]*len(LABELS[modality])
            masks.append(float(available))
            ages.append(min(1.,age/300))
        obs=store.db.execute('SELECT category,idle,at FROM observations WHERE at<=? ORDER BY at DESC LIMIT 1',(instant,)).fetchone()
        category=obs['category'] if store.setting('collection_enabled',False) and obs and instant-obs['at']<=60 else None
        values += masks+ages+[float(obs['idle']) if category else 0.]
        values += [float(category==c) for c in APP_CATEGORIES]
        values += [float(kind==k) for k in TASK_KINDS]
        seq.append(values)
    return seq


def personalize(store,prediction):
    if prediction.get('probabilities') is None:
        return prediction
    source=prediction['label']
    counts=[0]*len(LABELS['context'])
    for row in store.db.execute('SELECT label,data FROM context_labels'):
        data=json.loads(row['data'])
        if data.get('predicted_label')==source:
            counts[LABELS['context'].index(row['label'])]+=1
    total=sum(counts)
    if total<3:
        return dict(prediction,personal_samples=total)
    weight=min(.5,total/40)
    adjusted=[(1-weight)*p+weight*(n+1)/(total+len(counts)) for p,n in zip(prediction['probabilities'],counts)]
    top=max(range(len(adjusted)),key=adjusted.__getitem__)
    return dict(prediction,base_probabilities=prediction['probabilities'],base_label=source,
                probabilities=adjusted,label=LABELS['context'][top],confidence=adjusted[top],
                abstain=adjusted[top]<.6 or prediction.get('missing_now',False),personal_samples=total,personal_weight=weight,
                note='명시적 교정 빈도와 모델 출력을 혼합한 점수입니다. 보정된 실제 확률로 검증하지 않았습니다.')
