"""Independent camera/screen CNNs and a small temporal concept model."""
import torch
from torch import nn
from local_agent.model_contract import FEATURES,LABELS


class ConceptCNN(nn.Module):
    def __init__(self,classes):
        super().__init__()
        layers=[]
        previous=3
        for width in (16,32,64):
            layers += [nn.Conv2d(previous,width,3,stride=2,padding=1),nn.ReLU()]
            previous=width
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Linear(64,classes)

    def forward(self,x):
        return self.head(self.encoder(x))


class ContextGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(len(FEATURES),64,batch_first=True)
        self.head=nn.Linear(64,len(LABELS['context']))

    def forward(self,x):
        _,hidden=self.gru(x)
        return self.head(hidden[-1])


def make_model(kind):
    return ContextGRU() if kind=='context' else ConceptCNN(len(LABELS[kind]))
