"""Synthetic-only end-to-end training validation, never deployment models."""
import json
import random
from argparse import Namespace
from pathlib import Path
import torch
from local_agent.model_contract import LABELS,FEATURES
from local_agent.model_runtime import ModelRegistry
from research.train_return import train


def main():
    from PIL import Image
    import numpy as np
    torch.set_num_threads(2)
    folder=Path('runs/return-smoke').resolve()
    folder.mkdir(parents=True,exist_ok=True)
    rng=random.Random(7)
    pixels=np.random.default_rng(7)
    for kind in ['camera','screen','context']:
        rows=[]
        for split,offset in [('train',0),('val',10000),('test',20000)]:
            for i,label in enumerate(LABELS[kind]):
                row=dict(participant=split,episode=f'{split}-{i}',at=offset+i,split=split,label=label)
                if kind=='context':
                    row['features']=[[rng.random() for _ in FEATURES] for _ in range(16)]
                else:
                    path=folder/f'{kind}-{split}-{i}.png'
                    Image.fromarray(pixels.integers(0,256,size=(128,128,3),dtype=np.uint8)).save(path)
                    row['path']=str(path)
                rows.append(row)
        manifest=folder/(kind+'.jsonl')
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        output=folder/(kind+'.pt')
        train(Namespace(kind=kind,manifest=manifest,output=output,split_mode='people',device='cpu',epochs=1,seed=7,synthetic=True))
        try:
            ModelRegistry().load(kind,output)
        except ValueError:
            print(kind+': synthetic deployment refusal confirmed')
        else:
            raise AssertionError('Synthetic model accepted')
    print('All three synthetic training/export/evaluation paths completed. No real-world accuracy claimed.')


if __name__=='__main__':
    main()
