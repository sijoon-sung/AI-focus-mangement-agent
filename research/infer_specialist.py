"""Run a trained role checkpoint on explicit inputs, offline. Never opens a camera."""
import argparse
import json
from pathlib import Path
from PIL import Image
from local_agent.specialists import Specialists


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path)
    p.add_argument('--head-angles',type=float,nargs=2)
    p.add_argument('--side',choices=['left','right'])
    p.add_argument('--observations',nargs='+')
    p.add_argument('--goals',nargs='+')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();model=Specialists();role=model.load(a.checkpoint)['role']
    if role=='goal_history':result=model.goal_alignment(a.observations or [],a.goals or [])
    else:
        if not a.image:raise ValueError('--image required')
        with Image.open(a.image) as image:
            result=model.layout(image) if role=='screen_layout' else model.gaze(image,a.head_angles or [],a.side)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='maps'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
