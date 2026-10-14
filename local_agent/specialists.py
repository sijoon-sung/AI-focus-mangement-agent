"""Load public-data specialists without conflating their outputs with intent.

Training and inference share preprocessing. No network downloads during inference.
"""
import hashlib
import math
from pathlib import Path


class Specialists:
    def __init__(self):
        self.models={}
        self.encoder=None

    def load(self,path):
        import torch
        from research.specialist_models import VERSION,make_specialist
        path=Path(path)
        if path.stat().st_size>100*1024*1024:raise ValueError('Specialist checkpoint exceeds 100 MiB')
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('version')!=VERSION or data.get('trained') is not True:raise ValueError('Unsupported specialist checkpoint')
        model=make_specialist(data['role'])
        model.load_state_dict(data['state_dict'],strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite weights')
        model.eval()
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        self.models[data['role']]=(model,data,digest)
        return dict(role=data['role'],pilot=data['pilot'],digest=digest)

    def _metadata(self,role):
        _,data,digest=self.models[role]
        return dict(model_digest=digest,pilot=data['pilot'],source='public_specialist_hypothesis',role=role)

    def gaze(self,normalized_eye,head_angles,side):
        import torch
        from research.specialist_data import eye_tensor
        if side not in {'left','right'} or len(head_angles)!=2 or any(not math.isfinite(x) or abs(x)>math.pi for x in head_angles):
            raise ValueError('Head angles must be two finite radians; side is left/right')
        model,_,_=self.models['camera_gaze']
        pose=torch.tensor([head_angles+[float(side=='left'),float(side=='right')]],dtype=torch.float32)
        with torch.no_grad():v=model(eye_tensor(normalized_eye)[None],pose)[0].tolist()
        return dict(self._metadata('camera_gaze'),vector=v,pitch=math.asin(max(-1,min(1,-v[1]))),yaw=math.atan2(-v[0],-v[2]),
                    note='Normalized camera geometry; not focus, intent, or calibrated screen coordinates')

    def layout(self,image):
        import torch
        from research.specialist_data import screen_tensor
        from research.specialist_models import LAYOUT_LABELS
        model,_,_=self.models['screen_layout']
        x,transform=screen_tensor(image)
        with torch.no_grad():maps=model(x[None]).sigmoid()[0]
        _,dx,dy,nw,nh=transform
        valid=maps[:,dy//8:min(32,(dy+nh+7)//8),dx//8:min(32,(dx+nw+7)//8)]
        return dict(self._metadata('screen_layout'),occupancy={label:float((valid[i]>=.5).float().mean()) for i,label in enumerate(LAYOUT_LABELS)},
                    maps=maps.tolist(),note='Coarse 32x32 UI type occupancy; not OCR or task relevance')

    def goal_alignment(self,observations,goals,cache_folder=None):
        import torch
        import torch.nn.functional as F
        from research.specialist_models import OPERATIONS
        if not observations or not goals or len(goals)>100:raise ValueError('Need observations and 1..100 candidate goals')
        model,metadata,_=self.models['goal_history']
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            folder=cache_folder or str(Path(__file__).resolve().parents[1]/'runs/hf-cache')
            self.encoder=SentenceTransformer(metadata['text_model'],revision=metadata['text_revision'],cache_folder=folder,local_files_only=True,device='cpu')
        observations=observations[-16:]
        values=self.encoder.encode([s[:2000] for s in observations+goals],normalize_embeddings=True,convert_to_tensor=True).cpu()
        x=torch.zeros(1,16,386)
        x[0,-len(observations):,:384]=values[:len(observations)]
        # Number of unobserved actions between desktop screenshots is unknown.
        x[0,-len(observations):,384]=0;x[0,-len(observations):,385]=1
        goal=values[len(observations):]
        with torch.no_grad():
            aligned,op=model(x.expand(len(goals),-1,-1),goal)
            mean=F.normalize(values[:len(observations)].mean(0),dim=0)
            weight=metadata.get('alignment_weight')
            if weight is None:raise ValueError('Checkpoint needs validation-selected alignment weight')
            mixed=F.normalize(weight*aligned+(1-weight)*mean,dim=-1)
            score=(mixed*goal).sum(-1)
            baseline=(mean*goal).sum(-1)
            top=int(score.argmax())
            contributions=[]
            if len(observations)>1:
                for removed in range(len(observations)):
                    kept=torch.cat([values[:removed],values[removed+1:len(observations)]])
                    masked=torch.zeros(1,16,386)
                    masked[0,-len(kept):,:384]=kept;masked[0,-len(kept):,385]=1
                    prediction,_=model(masked,goal[top:top+1])
                    removed_mean=F.normalize(kept.mean(0),dim=0)
                    without=F.normalize(weight*prediction[0]+(1-weight)*removed_mean,dim=0)@goal[top]
                    contributions.append(dict(observation_index=removed,score_change=float(score[top]-without)))
        return dict(self._metadata('goal_history'),alignment_weight=weight,candidates=[dict(goal=g,compatibility=float(score[i]),
                    frozen_encoder_baseline=float(baseline[i]),next_operation=OPERATIONS[int(op[i].argmax())]) for i,g in enumerate(goals)],
                    explanation=dict(method='leave-one-observation-out recomputation',candidate_index=top,contributions=contributions,
                                     limit='Model score sensitivity, not a causal explanation of the person'),
                    note='Compatibility scores, not probabilities. Trained on before-action web page text; desktop OCR is a domain shift. Does not predict task completion.')


class ScreenCalibration:
    """Small personal calibration only, not training the base perception model."""
    def fit(self,angles,points):
        import numpy as np
        x=np.asarray(angles,dtype=float);y=np.asarray(points,dtype=float)
        if x.ndim!=2 or x.shape[1]!=2 or y.shape!=x.shape or len(x)<5 or not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any() or (y>1).any():
            raise ValueError('At least 5 gaze-angle / normalized screen-point pairs required')
        design=np.column_stack([x,np.ones(len(x))])
        if np.linalg.matrix_rank(design)<3:raise ValueError('Calibration points must span the screen')
        self.weights=np.linalg.solve(design.T@design+np.diag([1e-4,1e-4,0]),design.T@y)
        return float(np.sqrt(np.mean((design@self.weights-y)**2)))

    def predict(self,angles):
        import numpy as np
        if not hasattr(self,'weights'):raise ValueError('Calibrate before estimating screen coordinates')
        x=np.asarray(angles,dtype=float)
        if x.shape!=(2,) or not np.isfinite(x).all():raise ValueError('Two finite gaze angles required')
        point=np.r_[x,1]@self.weights
        return dict(point=point.tolist(),outside=bool((point<0).any() or (point>1).any()),note='Calibration estimate, not a concentration judgment')
