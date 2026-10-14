"""Offline inference with actual held-out public records; no personal capture."""
import argparse
import json
from pathlib import Path
from PIL import Image
from local_agent.specialists import Specialists
from research.specialist_data import read_rows


def main():
    import torch
    torch.set_num_threads(2)
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path,default=Path('runs/public-models'))
    p.add_argument('--data-root',type=Path,default=Path('data/public'))
    p.add_argument('--output',type=Path,default=Path('runs/public-models/inference.json'))
    a=p.parse_args();runtime=Specialists();results={}
    for role in ['camera_gaze','screen_layout','goal_history']:runtime.load(a.models/(role+'.pt'))
    manifest=next((a.data_root/'mpiigaze').glob('prepared-*/manifest.jsonl'))
    eye=next(r for r in read_rows(manifest) if r['split']=='test')
    with Image.open(eye['path']) as image:results['camera_gaze']=runtime.gaze(image,eye['pose_side'][:2],eye['side'])
    screen=next(r for r in read_rows(a.data_root/'groundcua/manifest.jsonl') if r['split']=='test')
    with Image.open(screen['path']) as image:results['screen_layout']=runtime.layout(image)
    results['screen_layout'].pop('maps')
    tasks=[r for r in read_rows(a.data_root/'mind2web/tasks.jsonl') if r['split']=='test' and len(r['observations'])>=2]
    results['goal_history']=runtime.goal_alignment(tasks[0]['observations'][:2],[t['goal'] for t in tasks[:4]])
    assert len(results['goal_history']['explanation']['contributions'])==2
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print('All three public-data checkpoints loaded and inferred; score-sensitivity explanation recomputed.',flush=True)


if __name__=='__main__':main()
