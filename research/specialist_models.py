"""Separate observable tasks, losses and input contracts; no 'distraction' proxy labels."""
import torch
from torch import nn
from torch.nn import functional as F

VERSION='public-specialists-v1'
TEXT_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
OPERATIONS=['CLICK','TYPE','SELECT']
LAYOUT_LABELS=['Button','Menu','Input Elements','Navigation','Sidebar','Visual Elements','Information Display','Others']


class EyeGazeCNN(nn.Module):
    """MPIIGaze normalized 36x60 eye + head angles + eye side -> unit gaze vector."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                   nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                   nn.Conv2d(48,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,5)),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64*3*5+4,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,3))

    def forward(self,image,pose_side):
        return F.normalize(self.head(torch.cat([self.encoder(image),pose_side],dim=-1)),dim=-1)


class ScreenLayoutCNN(nn.Module):
    """ImageNet MobileNetV3 encoder + independently learned dense GUI type head."""
    def __init__(self,pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
        self.encoder=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None).features
        # Retain stride-8 details: a coarse stride-32 encoder alone loses tiny controls.
        self.decoder=nn.Sequential(nn.Conv2d(576+24,128,3,padding=1),nn.ReLU(),nn.Conv2d(128,len(LAYOUT_LABELS),1))

    def forward(self,image):
        for i,block in enumerate(self.encoder):
            image=block(image)
            if i==3:detail=image
        coarse=F.interpolate(image,size=detail.shape[-2:],mode='bilinear',align_corners=False)
        return self.decoder(torch.cat([detail,coarse],dim=1))


class GoalHistoryGRU(nn.Module):
    """Before-action page-text embeddings -> goal compatibility + NEXT operation.

    Additional inputs are ordinal gap and observation mask, never fabricated seconds.
    Every prediction only sees the prefix; future action text is not an input.
    """
    def __init__(self,embedding_size=384):
        super().__init__()
        self.gru=nn.GRU(embedding_size+2,128,batch_first=True)
        self.alignment=nn.Linear(128,embedding_size)
        self.next_operation=nn.Sequential(nn.Linear(128+embedding_size,128),nn.ReLU(),nn.Linear(128,len(OPERATIONS)))

    def forward(self,history,goal):
        _,h=self.gru(history)
        h=h[-1]
        return F.normalize(self.alignment(h),dim=-1),self.next_operation(torch.cat([h,goal],dim=-1))


def make_specialist(role,pretrained=False):
    if role=='camera_gaze':return EyeGazeCNN()
    if role=='screen_layout':return ScreenLayoutCNN(pretrained)
    if role=='goal_history':return GoalHistoryGRU()
    raise ValueError(role)


def angular_degrees(pred,target):
    return torch.rad2deg(torch.acos((F.normalize(pred,dim=-1)*F.normalize(target,dim=-1)).sum(-1).clamp(-1,1)))


def contrastive_alignment(pred,goals,ids):
    """Multiple prefixes of the same task are positives, not false negatives."""
    scores=pred@F.normalize(goals,dim=-1).T/.1
    positive=ids[:,None]==ids[None,:]
    return (torch.logsumexp(scores,dim=1)-torch.logsumexp(scores.masked_fill(~positive,-torch.inf),dim=1)).mean()
