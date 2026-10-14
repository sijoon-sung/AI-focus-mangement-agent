import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from research.specialist_models import LAYOUT_LABELS,OPERATIONS,TEXT_MODEL


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def audit_groups(rows,key):
    groups={};hashes={}
    for row in rows:
        if row['split'] not in {'train','val','test'}:raise ValueError('Unknown split')
        group=row[key]
        if group in groups and groups[group]!=row['split']:raise ValueError('Group leakage: '+str(group))
        groups[group]=row['split']
        digest=row.get('image_sha256')
        if digest and digest in hashes and hashes[digest]!=row['split']:raise ValueError('Duplicate image across splits')
        if digest:hashes[digest]=row['split']
    if set(groups.values())!={'train','val','test'}:raise ValueError('All splits required')


def prepare_gaze(root,out,per_person=2000):
    """Subject-disjoint internal split; not MPIIGaze official leave-one-person-out score."""
    import cv2
    from scipy.io import loadmat
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for subject in sorted((root/'MPIIGaze/Data/Normalized').glob('p*')):
        person=int(subject.name[1:]); split='train' if person<11 else 'val' if person<13 else 'test'
        mats=sorted(subject.glob('*.mat'))
        if not mats:continue
        # Sample across days, not just the first day. Deterministic per-person reservoir.
        rng=np.random.default_rng(7+person)
        candidates=[];seen=0
        for file in mats:
            raw=loadmat(file,simplify_cells=True)['data']
            for side_index,side in enumerate(['left','right']):
                part=raw[side]
                images=np.asarray(part['image']);gazes=np.atleast_2d(part['gaze']);poses=np.atleast_2d(part['pose'])
                if images.ndim==2:images=images[None]
                if images.shape[0]!=len(gazes) or images.shape[1:]!=(36,60):
                    raise ValueError(f'Unexpected official eye tensor: {images.shape} in {file}')
                for i,(gaze,pose) in enumerate(zip(gazes,poses)):
                    seen+=1
                    slot=len(candidates) if len(candidates)<per_person else int(rng.integers(seen))
                    if slot>=per_person:continue
                    rotation=cv2.Rodrigues(np.asarray(pose,dtype=np.float64))[0][:,2]
                    angles=[float(np.arcsin(np.clip(rotation[1],-1,1))),float(np.arctan2(rotation[0],rotation[2]))]
                    item=(images[i].copy(),gaze.astype(float).tolist(),angles+[float(side_index==0),float(side_index==1)],file.name,i,side)
                    if len(candidates)<per_person:candidates.append(item)
                    else:candidates[slot]=item
        for image,gaze,pose_side,day,index,side in candidates:
            relative=f'eyes/{subject.name}/{Path(day).stem}-{side}-{index}.png'
            path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(image.astype('uint8')).save(path)
            rows.append(dict(path=str(path.resolve()),gaze=gaze,pose_side=pose_side,participant=subject.name,
                             split=split,day=day,index=index,side=side,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f'{subject.name}: {len(candidates)}/{seen} eyes',flush=True)
    audit_groups(rows,'participant')
    file=out/'manifest.jsonl';file.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    return file


def eye_tensor(image):
    if image.size!=(60,36):raise ValueError('Requires a geometrically normalized 60x36 eye, not an arbitrary face crop')
    return torch.from_numpy(np.asarray(image.convert('L'),dtype='float32').copy()/255)[None]


def screen_tensor(image):
    """Letterbox, preserving screen aspect ratio and exact target geometry."""
    image=image.convert('RGB');w,h=image.size
    scale=min(256/w,256/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale));dx,dy=(256-nw)//2,(256-nh)//2
    canvas=Image.new('RGB',(256,256),(128,128,128))
    canvas.paste(image.resize((nw,nh)),(dx,dy))
    x=torch.from_numpy(np.asarray(canvas,dtype='float32').copy()/255).permute(2,0,1)
    x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return x,(scale,dx,dy,nw,nh)


def layout_target(annotations,transform):
    scale,dx,dy,nw,nh=transform
    target=torch.zeros(len(LAYOUT_LABELS),32,32)
    valid=torch.zeros(1,32,32)
    valid[:,dy//8:min(32,(dy+nh+7)//8),dx//8:min(32,(dx+nw+7)//8)]=1
    unknown=[]
    for a in annotations:
        coords=np.asarray(a['bbox'],dtype=float)
        if coords.shape!=(4,) or not np.isfinite(coords).all():raise ValueError('Invalid bounding box')
        x1,y1,x2,y2=coords
        if x2<=x1 or y2<=y1:continue
        ix1=max(0,min(31,int((x1*scale+dx)//8)));iy1=max(0,min(31,int((y1*scale+dy)//8)))
        ix2=max(ix1+1,min(32,int(np.ceil((x2*scale+dx)/8))));iy2=max(iy1+1,min(32,int(np.ceil((y2*scale+dy)/8))))
        if a.get('category') in LAYOUT_LABELS:
            target[LAYOUT_LABELS.index(a['category']),iy1:iy2,ix1:ix2]=1
        else:unknown.append((ix1,iy1,ix2,iy2))
    for x1,y1,x2,y2 in unknown:valid[:,y1:y2,x1:x2]=0
    # Known positive labels remain supervised even when unknown boxes overlap.
    valid=valid.expand_as(target).clone()
    valid[target.bool()]=1
    return target,valid


class ImageRecords(Dataset):
    def __init__(self,rows,role):self.rows,self.role=rows,role
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        with Image.open(r['path']) as image:
            if self.role=='camera_gaze':
                return eye_tensor(image),torch.tensor(r['pose_side'],dtype=torch.float32),torch.tensor(r['gaze'],dtype=torch.float32)
            x,transform=screen_tensor(image)
        annotations=json.loads(Path(r['annotation']).read_text(encoding='utf8'))
        target,valid=layout_target(annotations,transform)
        return x,target,valid


def encode_tasks(tasks,out,device='cpu',cache_folder='runs/hf-cache'):
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info
    revision=model_info(TEXT_MODEL).sha
    encoder=SentenceTransformer(TEXT_MODEL,revision=revision,device=device,cache_folder=cache_folder)
    texts=sorted({t['goal'] for t in tasks}|{a for t in tasks for a in t['observations']})
    values=encoder.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_tensor=True,show_progress_bar=True).cpu()
    mapping={s:i for i,s in enumerate(texts)}
    encoded=[]
    for task in tasks:
        if len(task['observations'])!=len(task['operations']):raise ValueError('Observation/operation length mismatch')
        if any(op not in OPERATIONS for op in task['operations']):raise ValueError('Unsupported source operation')
        encoded.append(dict(task,goal_index=mapping[task['goal']],action_indices=[mapping[a] for a in task['observations']]))
    torch.save(dict(tasks=encoded,embeddings=values,text_model=TEXT_MODEL,text_revision=revision),out)


class HistoryRecords(Dataset):
    def __init__(self,tasks,embeddings,training=False,keep_every=1):
        self.tasks,self.embeddings,self.training=tasks,embeddings,training
        self.keep_every=keep_every
        self.samples=[(i,j) for i,t in enumerate(tasks) for j in range(len(t['actions']))]
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        i,j=self.samples[index];t=self.tasks[i]
        indices=sorted(set(range(max(0,j-15),j+1,self.keep_every))|{j})
        if self.training and len(indices)>2:
            indices=[k for k in indices[:-1] if torch.rand(()).item()>.25]+[j]
        x=torch.zeros(16,386)
        for n,k in enumerate(indices):
            pos=16-len(indices)+n
            x[pos,:384]=self.embeddings[t['action_indices'][k]]
            x[pos,384]=min(16,k-indices[n-1] if n else 1)/16
            x[pos,385]=1
        if self.training and torch.rand(()).item()<.25:x[:,384]=0
        # Observation j is the page BEFORE action j. The label is never encoded.
        next_op=t['operations'][j]
        return x,self.embeddings[t['goal_index']],torch.tensor(OPERATIONS.index(next_op)),torch.tensor(t['goal_index'])
