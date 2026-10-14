"""Train role-specific models on author-published public data; evaluate held-out groups."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from research.specialist_models import VERSION,LAYOUT_LABELS,OPERATIONS,make_specialist,angular_degrees,contrastive_alignment
from research.specialist_data import read_rows,audit_groups,ImageRecords,HistoryRecords,prepare_gaze,encode_tasks
from research.public_data import sha


def objective(role,model,batch,operation_weights=None):
    if role=='camera_gaze':
        image,pose,target=batch
        pred=model(image,pose)
        return (1-(pred*F.normalize(target,dim=-1)).sum(-1)).mean()
    if role=='screen_layout':
        image,target,valid=batch
        pred=model(image)
        weights=1+4*target
        loss=F.binary_cross_entropy_with_logits(pred,target,reduction='none')*valid*weights
        return loss.sum()/(valid*weights).sum().clamp_min(1)
    history,goal,next_op,ids=batch
    aligned,logits=model(history,goal)
    return contrastive_alignment(aligned,goal,ids)+F.cross_entropy(logits,next_op,weight=operation_weights)


def retrieval(query,goals,ids,gallery_ids):
    scores=query@goals.T
    correct=ids[:,None]==gallery_ids[None,:]
    target=scores.masked_fill(~correct,-torch.inf).max(-1).values
    rank=(scores>target[:,None]).sum(-1)+1
    return dict(top1=float((rank==1).float().mean()),mrr=float((1/rank.float()).mean()))


def metrics(role,model,loader,device,alignment_weight=None):
    model.eval()
    errors=[];intersection=torch.zeros(8);union=torch.zeros(8);positives=torch.zeros(8)
    queries=[];goals=[];ids=[];baselines=[];right=total=0
    confusion=torch.zeros(len(OPERATIONS),len(OPERATIONS),dtype=torch.long)
    with torch.no_grad():
        for raw in loader:
            batch=[x.to(device) for x in raw]
            if role=='camera_gaze':
                image,pose,target=batch
                errors.append(angular_degrees(model(image,pose),target).cpu())
            elif role=='screen_layout':
                image,target,valid=batch
                guess=model(image).sigmoid()>=.5;gold=target.bool();mask=valid.bool()
                intersection+=((guess&gold)&mask).sum((0,2,3)).cpu()
                union+=((guess|gold)&mask).sum((0,2,3)).cpu()
                positives+=(gold&mask).sum((0,2,3)).cpu()
            else:
                history,goal,op,index=batch
                aligned,logits=model(history,goal)
                queries.append(aligned.cpu());goals.append(goal.cpu());ids.append(index.cpu())
                # Strong frozen-encoder baseline, not a random model baseline.
                mean=history[:,:,:384].sum(1)/history[:,:,385].sum(1,keepdim=True).clamp_min(1)
                baselines.append(F.normalize(mean,dim=-1).cpu())
                right+=int((logits.argmax(-1)==op).sum());total+=len(op)
                for truth,prediction in zip(op.cpu(),logits.argmax(-1).cpu()):confusion[truth,prediction]+=1
    if role=='camera_gaze':
        e=torch.cat(errors)
        return dict(mean_angular_degrees=float(e.mean()),median_angular_degrees=float(e.median()),examples=len(e))
    if role=='screen_layout':
        iou=intersection/union.clamp_min(1);supported=positives>0
        return dict(macro_iou_on_present_classes=float(iou[supported].mean()) if supported.any() else None,
                    per_class={name:dict(iou=float(iou[i]),positive_cells=int(positives[i])) for i,name in enumerate(LAYOUT_LABELS)},
                    note='32x32 multilabel occupancy IoU, not precise GUI grounding accuracy')
    q,g,index,b=torch.cat(queries),torch.cat(goals),torch.cat(ids),torch.cat(baselines)
    unique=torch.unique(index)
    gallery=torch.stack([g[(index==i).nonzero()[0,0]] for i in unique])
    mixtures={str(weight):retrieval(F.normalize(weight*q+(1-weight)*b,dim=-1),gallery,index,unique) for weight in [0.,.25,.5,.75,1.]}
    chosen=max([0.,.25,.5,.75,1.],key=lambda w:mixtures[str(w)]['mrr']) if alignment_weight is None else alignment_weight
    f1=2*confusion.diag()/(confusion.sum(0)+confusion.sum(1)).clamp_min(1)
    return dict(goal_retrieval=retrieval(q,gallery,index,unique),frozen_mean_embedding_baseline=retrieval(b,gallery,index,unique),
                alignment_weight=chosen,selected_retrieval=mixtures[str(chosen)],
                next_operation_accuracy=right/total,next_operation_macro_f1=float(f1.mean()),operation_confusion=confusion.tolist(),prefixes=total,candidate_goals=len(unique),
                note='Internal held-out websites; before-action page text. No completion/distraction labels.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['camera_gaze','screen_layout','goal_history'],required=True)
    p.add_argument('--data-root',type=Path,default=Path('data/public'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--per-person',type=int,default=2000)
    p.add_argument('--pilot',action='store_true',help='Mark bounded integration training as a pilot, not a validated release')
    p.add_argument('--no-pretrained',action='store_true',help='Screen encoder ablation only')
    p.add_argument('--freeze-encoder-epochs',type=int,default=3)
    a=p.parse_args()
    started=time.perf_counter()
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('CUDA is unavailable; install CUDA-enabled PyTorch or use --device cpu')
        torch.cuda.reset_peak_memory_stats(a.device)
    if a.epochs<1 or a.batch_size<1:raise ValueError('Positive epochs/batch-size required')
    torch.manual_seed(7);torch.set_num_threads(a.threads)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    text_metadata={}
    if a.role=='camera_gaze':
        root=a.data_root/'mpiigaze'
        manifest=prepare_gaze(root,root/f'prepared-{a.per_person}',a.per_person)
        rows=read_rows(manifest);audit_groups(rows,'participant')
        sets={s:ImageRecords([r for r in rows if r['split']==s],a.role) for s in ['train','val','test']}
    elif a.role=='screen_layout':
        root=a.data_root/'groundcua';manifest=root/'manifest.jsonl'
        rows=read_rows(manifest);audit_groups(rows,'group')
        # Category absence is unlabeled, never a made-up 'other' class.
        rows=[r for r in rows if any(x.get('category') in LAYOUT_LABELS for x in json.loads(Path(r['annotation']).read_text(encoding='utf8')))]
        audit_groups(rows,'group')
        sets={s:ImageRecords([r for r in rows if r['split']==s],a.role) for s in ['train','val','test']}
    else:
        root=a.data_root/'mind2web';manifest=root/'tasks.jsonl'
        rows=read_rows(manifest);audit_groups(rows,'website')
        encoded=root/('encoded-'+sha(manifest)[:12]+'.pt')
        if not encoded.exists():encode_tasks(rows,encoded,device=a.device)
        data=torch.load(encoded,map_location='cpu',weights_only=True)
        text_metadata={k:data[k] for k in ['text_model','text_revision']}
        sets={s:HistoryRecords([r for r in data['tasks'] if r['split']==s],data['embeddings'],training=s=='train') for s in ['train','val','test']}
    print(json.dumps(dict(role=a.role,counts={s:len(d) for s,d in sets.items()})),flush=True)
    if any(len(d)==0 for d in sets.values()):raise ValueError('Each split needs usable labeled examples')
    loaders={s:DataLoader(d,batch_size=a.batch_size,shuffle=s=='train',num_workers=0) for s,d in sets.items()}
    operation_weights=None
    if a.role=='goal_history':
        counts=torch.bincount(torch.tensor([int(sets['train'][i][2]) for i in range(len(sets['train']))]),minlength=len(OPERATIONS)).float()
        operation_weights=(counts.sum()/(len(OPERATIONS)*counts.clamp_min(1))).clamp_max(10).to(a.device)
    model=make_specialist(a.role,pretrained=a.role=='screen_layout' and not a.no_pretrained).to(a.device)
    groups=[{'params':model.encoder.parameters(),'lr':a.lr*.1},{'params':model.decoder.parameters(),'lr':a.lr}] if a.role=='screen_layout' else model.parameters()
    optimizer=torch.optim.AdamW(groups,lr=a.lr,weight_decay=.001)
    best=float('inf');best_state=None;history=[]
    for epoch in range(a.epochs):
        epoch_started=time.perf_counter()
        model.train();train_loss=0.;count=0
        if a.role=='screen_layout':
            frozen=epoch<a.freeze_encoder_epochs and not a.no_pretrained
            for parameter in model.encoder.parameters():parameter.requires_grad_(not frozen)
            if frozen:model.encoder.eval()
        for raw in loaders['train']:
            batch=[x.to(a.device) for x in raw]
            loss=objective(a.role,model,batch,operation_weights)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            train_loss+=float(loss.detach())*len(raw[0]);count+=len(raw[0])
        model.eval();val_loss=0.;val_count=0
        with torch.no_grad():
            for raw in loaders['val']:
                value=objective(a.role,model,[x.to(a.device) for x in raw],operation_weights)
                val_loss+=float(value)*len(raw[0]);val_count+=len(raw[0])
        val_loss/=val_count
        history.append(dict(epoch=epoch+1,train_loss=train_loss/count,val_loss=val_loss,seconds=round(time.perf_counter()-epoch_started,3)))
        print(json.dumps(history[-1]),flush=True)
        if val_loss<best:
            best=val_loss;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    validation=metrics(a.role,model,loaders['val'],a.device)
    # Tune fusion on validation only; test never chooses a mixing weight.
    selected_weight=validation.get('alignment_weight')
    report=dict(version=VERSION,role=a.role,seed=7,pilot=a.pilot,counts={s:len(d) for s,d in sets.items()},
                parameters=sum(p.numel() for p in model.parameters()),manifest_sha256=sha(manifest),
                source=json.loads((root/'source.json').read_text(encoding='utf8')),history=history,
                validation=validation,test=metrics(a.role,model,loaders['test'],a.device,selected_weight),
                limits='Public-domain capability training; no measured personal productivity benefit or intent/distraction labels.',**text_metadata)
    if a.role=='goal_history':
        report['missing_history_test']={}
        for every in [2,4]:
            sparse=HistoryRecords(sets['test'].tasks,sets['test'].embeddings,keep_every=every)
            report['missing_history_test'][str(every)]=metrics(a.role,model,DataLoader(sparse,batch_size=a.batch_size),a.device,selected_weight)
        labels=[int(sets['train'][i][2]) for i in range(len(sets['train']))]
        majority=max(set(labels),key=labels.count)
        test_labels=[int(sets['test'][i][2]) for i in range(len(sets['test']))]
        fraction=sum(x==majority for x in test_labels)/len(test_labels)
        report['next_operation_majority_baseline']=dict(label=OPERATIONS[majority],accuracy=fraction,macro_f1=2*fraction/(1+fraction)/len(OPERATIONS))
    torch.save(dict(version=VERSION,role=a.role,state_dict=best_state,pilot=a.pilot,trained=True,
                    labels=LAYOUT_LABELS if a.role=='screen_layout' else OPERATIONS if a.role=='goal_history' else ['x','y','z'],
                    manifest_sha256=report['manifest_sha256'],alignment_weight=selected_weight,**text_metadata),a.output)
    report['checkpoint_sha256']=sha(a.output)
    report['runtime']=dict(device=a.device,torch_version=str(torch.__version__),seconds=round(time.perf_counter()-started,3),
                           gpu=torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None,
                           peak_allocated_mib=round(torch.cuda.max_memory_allocated(a.device)/1024**2,1) if a.device.startswith('cuda') else None)
    a.output.with_suffix('.metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(saved=str(a.output),validation=report['validation'],test=report['test']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
