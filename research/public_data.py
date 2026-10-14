"""Fetch author-published training data and preserve provenance. Never fetch test corpora for training."""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from html.parser import HTMLParser

MPII_URL='https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz'
APPS={'train':['VSCode','LibreOffice Writer','Bash','VLC Media Player','GIMP','Mozilla Firefox'],
      'val':['PyCharm','OnlyOffice Document Editor','Ubuntu Terminal'],
      'test':['Geany','Joplin','Chromium']}
LAYOUT_LABELS=['Button','Menu','Input Elements','Navigation','Sidebar','Visual Elements','Information Display','Others']


def page_text(html):
    """Before-action UI text only; never use target candidate IDs or future action text."""
    class Reader(HTMLParser):
        def __init__(self):super().__init__();self.parts=[];self.hidden=0
        def handle_starttag(self,tag,attrs):
            if tag in {'script','style'}:self.hidden+=1
            if not self.hidden:
                self.parts.extend(v for k,v in attrs if k in {'aria-label','placeholder','alt'} and v)
        def handle_endtag(self,tag):
            if tag in {'script','style'}:self.hidden=max(0,self.hidden-1)
        def handle_data(self,data):
            if not self.hidden and data.strip():self.parts.append(' '.join(data.split()))
    reader=Reader();reader.feed(html)
    text=' | '.join(dict.fromkeys(reader.parts))
    return text if len(text)<=2400 else text[:1200]+' | '+text[-1200:]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def fetch(url,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size:
        return path
    temporary=path.with_suffix(path.suffix+'.part')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as f:
                total=0
                expected=response.headers.get('Content-Length')
                for block in iter(lambda:response.read(1024*1024),b''):
                    f.write(block)
                    total+=len(block)
                    if total%(128*1024*1024)==0:
                        print(f'{path.name}: {total//1048576} MiB',flush=True)
                if expected and total!=int(expected):
                    raise IOError('Incomplete download')
            temporary.replace(path)
            return path
        except Exception:
            if attempt==3:
                raise
            time.sleep(2**attempt)


def api(url):
    with urllib.request.urlopen(url,timeout=45) as r:
        return json.load(r)


def revision(repo):
    return api('https://huggingface.co/api/datasets/'+repo)['sha']


def save_provenance(root,data):
    (root/'source.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def mpiigaze(root):
    archive=fetch(MPII_URL,root/'MPIIGaze.tar.gz')
    count=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or not re.fullmatch(r'MPIIGaze/Data/Normalized/p\d{2}/[^/]+\.mat',member.name):
                continue
            target=root/Path(member.name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            count+=1
    if not count:
        raise ValueError('Official normalized MAT layout not found')
    save_provenance(root,dict(dataset='MPIIGaze',url=MPII_URL,license='CC-BY-NC-SA-4.0; non-commercial scientific use',sha256=sha(archive),mat_files=count))
    print(f'MPIIGaze: {count} normalized MAT files',flush=True)


def groundcua(root,per_app):
    repo='ServiceNow/GroundCUA'
    rev=revision(repo)
    rows=[]
    for split,apps in APPS.items():
        for app in apps:
            folder=urllib.parse.quote('data/'+app,safe='/')
            url=f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{folder}?limit=1000'
            files=[]
            while url and len(files)<per_app:
                with urllib.request.urlopen(url,timeout=45) as response:
                    files.extend(x for x in json.load(response) if x['path'].endswith('.json'))
                    match=re.search(r'<([^>]+)>; rel="next"',response.headers.get('Link',''))
                    url=match.group(1) if match else None
                    if url and not url.startswith('https://huggingface.co/'):
                        raise ValueError('Unexpected pagination host')
            def download_entry(entry):
                key=Path(entry['path']).stem
                safe_app=re.sub(r'[^a-zA-Z0-9_-]','_',app)
                ann=root/'annotations'/safe_app/(key+'.json')
                img=root/'images'/safe_app/(key+'.png')
                base=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/'
                fetch(base+urllib.parse.quote(entry['path'],safe='/'),ann)
                annotations=json.loads(ann.read_text(encoding='utf8'))
                image_path=app+'/'+key+'.png'
                if any(a.get('image_path')!=image_path for a in annotations):
                    raise ValueError('Annotation/image pairing differs')
                fetch(base+urllib.parse.quote('images/'+image_path,safe='/'),img)
                return dict(path=str(img.resolve()),annotation=str(ann.resolve()),group=app,split=split,
                            source='GroundCUA',image_sha256=sha(img),annotation_sha256=sha(ann))
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows.extend(pool.map(download_entry,files[:per_app]))
            print(f'GroundCUA {app}: {min(len(files),per_app)}',flush=True)
    (root/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    save_provenance(root,dict(dataset=repo,revision=rev,license='MIT',split='custom application-disjoint; not an official benchmark split',per_app=per_app,images=len(rows),applications=APPS))


def mind2web(root,shards):
    repo='osunlp/Mind2Web'
    rev=revision(repo)
    import ijson
    tasks=[]
    for shard in shards:
        path=f'data/train/train_{shard}.json'
        file=fetch(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}',root/Path(path).name)
        with file.open('rb') as f:
            for raw in ijson.items(f,'item'):
                tasks.append(dict(id=raw['annotation_id'],website=raw['website'],goal=raw['confirmed_task'],
                                  actions=raw['action_reprs'],observations=[page_text(a['cleaned_html']) for a in raw['actions']],
                                  operations=[a['operation']['op'] for a in raw['actions']]))
        print(f'Mind2Web shard {shard}: {len(tasks)} cumulative tasks',flush=True)
    # Split entire websites, never individual steps/prefixes from the same task.
    sites=sorted({r['website'] for r in tasks},key=lambda s:hashlib.sha256(('planwise-7:'+s).encode()).hexdigest())
    if len(sites)<3:
        raise ValueError('At least three websites required')
    ntrain=max(1,int(len(sites)*.7));nval=max(1,int(len(sites)*.15))
    if ntrain+nval>=len(sites):ntrain=len(sites)-nval-1
    splits={s:'train' if i<ntrain else 'val' if i<ntrain+nval else 'test' for i,s in enumerate(sites)}
    for t in tasks:t['split']=splits[t['website']]
    (root/'tasks.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks),encoding='utf8')
    save_provenance(root,dict(dataset=repo,revision=rev,license='CC-BY-4.0',downloaded_splits=['train'],shards=shards,
                              evaluation='internal website-disjoint split of official TRAIN ONLY; not official test performance',tasks=len(tasks),sites=splits,
                              files={f.name:sha(f) for f in root.glob('train_*.json')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=['mpiigaze','groundcua','mind2web'])
    p.add_argument('--root',type=Path,default=Path('data/public'))
    p.add_argument('--per-app',type=int,default=100)
    p.add_argument('--shards',type=int,nargs='+',default=[0])
    a=p.parse_args();root=a.root/a.source;root.mkdir(parents=True,exist_ok=True)
    if a.source=='mpiigaze':mpiigaze(root)
    elif a.source=='groundcua':
        if a.per_app<1:raise ValueError('per-app must be positive')
        groundcua(root,a.per_app)
    else:
        if any(s not in range(11) for s in a.shards):raise ValueError('Only official train shards 0..10 are supported')
        mind2web(root,a.shards)


if __name__=='__main__':main()
