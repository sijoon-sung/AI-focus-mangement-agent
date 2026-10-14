"""Assign whole participants or chronological episodes to train/val/test."""
import argparse
import json
import random
from pathlib import Path
from research.train_return import validate_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['camera','screen','context'],required=True)
    p.add_argument('--mode',choices=['people','future'],default='future')
    args=p.parse_args()
    rows=[json.loads(s) for s in args.input.read_text(encoding='utf-8').splitlines() if s.strip()]
    if args.kind!='context':
        for row in rows:
            path=Path(row['path'])
            row['path']=str((path if path.is_absolute() else args.input.parent/path).resolve())
    key=(lambda r:r['participant']) if args.mode=='people' else (lambda r:(r['participant'],r['episode']))
    groups=list(dict.fromkeys(key(r) for r in rows))
    if len(groups)<3:
        p.error('최소 3명의 참가자 또는 시간적으로 분리된 3개 에피소드가 필요합니다. 실험 표본 수 충분성을 보장하는 기준은 아닙니다.')
    if args.mode=='people':
        random.Random(7).shuffle(groups)
    else:
        groups.sort(key=lambda g:min(r['at'] for r in rows if key(r)==g))
    first=max(1,min(len(groups)-2,int(len(groups)*.6)))
    second=max(first+1,min(len(groups)-1,int(len(groups)*.8)))
    assignments={g:('train' if i<first else 'val' if i<second else 'test') for i,g in enumerate(groups)}
    for row in rows:
        row['split']=assignments[key(row)]
    try:
        validate_rows(rows,args.kind,args.mode)
    except ValueError as exc:
        p.error(str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print(json.dumps({s:sum(r['split']==s for r in rows) for s in ['train','val','test']}))


if __name__=='__main__':
    main()
