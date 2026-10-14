"""Train separate camera/screen CNNs or temporal GRU from approved JSONL data.

python -m research.train_return --kind screen --manifest data/screen.jsonl --output runs/screen.pt
Common fields: participant, episode, at (Unix seconds), split (train/val/test), label.
Images: path. Context: features (T x feature_count). See RETURN_IMPLEMENTATION_KO.md.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from local_agent.model_contract import VERSION,LABELS,FEATURES
from research.return_models import make_model


def validate_rows(rows,kind,mode):
    if not rows:
        raise ValueError('데이터가 없습니다.')
    seen_paths={}
    for row in rows:
        if row.get('split') not in {'train','val','test'} or row.get('label') not in LABELS[kind] or not row.get('participant') or not row.get('episode'):
            raise ValueError('정답·participant·episode·split을 확인하세요.')
        if not isinstance(row.get('at'),(int,float)) or not __import__('math').isfinite(row['at']):
            raise ValueError('at은 유한한 Unix 시각이어야 합니다.')
        if kind=='context':
            x=row.get('features')
            if not isinstance(x,list) or not 1<=len(x)<=64 or any(len(v)!=len(FEATURES) for v in x):
                raise ValueError('맥락 특징 차원이 다릅니다.')
            if any(not isinstance(n,(int,float)) or not 0<=n<=1 for v in x for n in v):
                raise ValueError('맥락 특징은 0~1의 유한한 수여야 합니다.')
        else:
            path=str(Path(row['path']).resolve())
            if not Path(path).is_file():
                raise ValueError('이미지 파일이 없습니다: '+path)
            digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest in seen_paths and seen_paths[digest]!=row['split']:
                raise ValueError('같은 이미지가 여러 split에 있습니다.')
            seen_paths[digest]=row['split']
    splits={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    if any(not v for v in splits.values()):
        raise ValueError('train/val/test가 모두 필요합니다.')
    if len({r['label'] for r in splits['train']})<2:
        raise ValueError('학습에는 최소 두 종류의 정답이 필요합니다.')
    episode_split={}
    people={}
    for r in rows:
        key=(r['participant'],r['episode'])
        if key in episode_split and episode_split[key]!=r['split']:
            raise ValueError('동일 에피소드가 여러 split에 있습니다.')
        episode_split[key]=r['split']
        if mode=='people' and r['participant'] in people and people[r['participant']]!=r['split']:
            raise ValueError('새 사용자 평가에서 참가자가 split 사이에 겹칩니다.')
        people[r['participant']]=r['split']
    if mode=='future':
        for left,right in [('train','val'),('val','test')]:
            if max(r['at'] for r in splits[left])+600 >= min(r['at'] for r in splits[right]):
                raise ValueError('미래 평가 split은 시간순이며 10분 이상의 경계 여유가 필요합니다.')
    return splits


class Records(Dataset):
    def __init__(self,rows,kind):
        self.rows,self.kind=rows,kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        if self.kind=='context':
            x=torch.tensor(row['features'],dtype=torch.float32)
            # Consistent causal left padding; same padding is applied at inference.
            x=x[-16:]
            if len(x)<16:
                x=torch.cat([torch.zeros(16-len(x),len(FEATURES)),x])
        else:
            import numpy as np
            from PIL import Image
            with Image.open(row['path']) as im:
                x=torch.from_numpy(np.array(im.convert('RGB').resize((128,128)),dtype='float32')/255).permute(2,0,1)
        return x,torch.tensor(LABELS[self.kind].index(row['label']))


def logits(model,loader,device):
    xs,ys=[],[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            xs.append(model(x.to(device)).cpu())
            ys.append(y)
    return torch.cat(xs),torch.cat(ys)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows=[json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.kind!='context':
        for row in rows:
            path=Path(row['path'])
            row['path']=str(path if path.is_absolute() else args.manifest.parent/path)
    splits=validate_rows(rows,args.kind,args.split_mode)
    device=torch.device(args.device)
    model=make_model(args.kind).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    loaders={s:DataLoader(Records(rs,args.kind),batch_size=16,shuffle=s=='train',num_workers=0) for s,rs in splits.items()}
    best_loss=float('inf')
    best=None
    for epoch in range(args.epochs):
        model.train()
        for x,y in loaders['train']:
            x,y=x.to(device),y.to(device)
            if args.kind=='context':
                # Drop either modality together with its availability flag.
                x=x.clone()
                for start,end,mask in [(0,3,8),(3,8,9)]:
                    drop=torch.rand(x.shape[0],device=device)<.15
                    x[drop,:,start:end]=0
                    x[drop,:,mask]=0
            loss=torch.nn.functional.cross_entropy(model(x),y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_x,val_y=logits(model,loaders['val'],device)
        val_loss=torch.nn.functional.cross_entropy(val_x,val_y).item()
        if val_loss<best_loss:
            best_loss=val_loss
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps(dict(epoch=epoch+1,val_loss=val_loss)),flush=True)
    model.load_state_dict(best)
    vx,vy=logits(model,loaders['val'],device)
    temperature=min([.5,.75,1.,1.5,2.,3.],key=lambda t:torch.nn.functional.cross_entropy(vx/t,vy).item())
    tx,ty=logits(model,loaders['test'],device)
    probs=torch.softmax(tx/temperature,dim=-1)
    pred=probs.argmax(-1)
    n=len(LABELS[args.kind])
    matrix=[[int(((ty==i)&(pred==j)).sum()) for j in range(n)] for i in range(n)]
    accepted=probs.max(-1).values>=.6
    f1=[2*matrix[i][i]/max(1,sum(matrix[i])+sum(row[i] for row in matrix)) for i in range(n)]
    counts={label:sum(r['label']==label for r in splits['train']) for label in LABELS[args.kind]}
    majority=LABELS[args.kind].index(max(counts,key=counts.get))
    report=dict(kind=args.kind,split_mode=args.split_mode,counts={s:len(v) for s,v in splits.items()},
                accuracy=float((pred==ty).float().mean()),macro_f1=sum(f1)/n,confusion_matrix=matrix,
                majority_accuracy=float((ty==majority).float().mean()),coverage=float(accepted.float().mean()),
                accepted_accuracy=float((pred[accepted]==ty[accepted]).float().mean()) if accepted.any() else None,
                missing_training_labels=[k for k,v in counts.items() if not v],temperature=temperature,
                synthetic=args.synthetic,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(version=VERSION,kind=args.kind,labels=LABELS[args.kind],features=FEATURES,
                    state_dict=best,temperature=temperature,trained=True,synthetic=args.synthetic,report=report),args.output)
    args.output.with_suffix('.metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=LABELS,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split-mode',choices=['people','future'],default='people')
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--synthetic',action='store_true',help='Required for synthetic fixtures; app refuses these models')
    args=p.parse_args()
    if args.epochs<1:
        p.error('epochs must be positive')
    train(args)
