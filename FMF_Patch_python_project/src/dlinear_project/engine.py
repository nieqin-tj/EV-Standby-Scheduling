"""DLinear training, checkpoint continuation, fine-tuning, and evaluation."""
import argparse
import shutil
import time
from itertools import count
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fmf_project.data import StationStore,scaler,windows
from fmf_project.engine import seed_all,rng_state,restore_rng,make_loader,forward,loss_fn,evaluate,save
from .model import DLinear

ROOT=Path(__file__).resolve().parents[2]
KEYS=[f'H{h}_{name}' for h in [1,3] for name in ['MedMAE','RMSE','Pin90']]

def load(path,net,stats):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    net.load_state_dict(ck['state_dict'],strict=True)
    if not all(np.array_equal(ck[k],stats[k]) for k in ['mean','std']):raise ValueError('Checkpoint and DATA scalers differ')
    return ck

def key(metrics):
    return tuple(float(np.mean([metrics[f'H{h}_{name}'] for h in [1,3]])) for name in ['MedMAE','Pin90'])

def write_history(output,rows):
    temporary=output/'history.tmp.csv';pd.DataFrame(rows).to_csv(temporary,index=False);temporary.replace(output/'history.csv')

def snapshot(path,net,opt,stats,epoch,protocol,metrics,stage):
    tmp=path.with_suffix('.tmp.pt')
    torch.save(dict(state_dict=net.state_dict(),optimizer_state=opt.state_dict(),rng=rng_state(),mean=stats['mean'],std=stats['std'],
        epoch=epoch,stage=stage,seed=protocol['seed'],metrics=metrics,protocol=protocol),tmp)
    tmp.replace(path)

def run_training(stage):
    parser=argparse.ArgumentParser(description='DLinear '+stage)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'DATA')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'weights/DLinear/pre_finetune.pt' if stage=='finetune' else None)
    parser.add_argument('--epochs',type=int,default=3 if stage=='finetune' else None,help='Optional training cap; fine-tuning defaults to three epochs')
    parser.add_argument('--learning-rate',type=float,default=1e-5 if stage=='finetune' else None)
    parser.add_argument('--min-delta',type=float,default=.003)
    parser.add_argument('--seed',type=int,default=20260908)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--eval-batch-size',type=int,default=128)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit-train-windows',type=int,default=0)
    parser.add_argument('--limit-val-windows',type=int,default=0)
    args=parser.parse_args()
    if (args.epochs is not None and args.epochs<1) or args.min_delta<=0 or args.batch_size<1 or args.eval_batch_size<1:parser.error('Invalid training parameters')
    if args.learning_rate is not None and args.learning_rate<=0:parser.error('Learning rate must be positive')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    seed_all(args.seed)
    stats=scaler(args.data_dir);store=StationStore(args.data_dir)
    train=windows(args.data_dir,'train');validation=windows(args.data_dir,'val',True)
    counts=dict(train=len(train),pred_select=len(validation))
    if args.limit_train_windows:train=train.iloc[:args.limit_train_windows]
    if args.limit_val_windows:validation=validation.iloc[:args.limit_val_windows]
    training=make_loader(store,train,stats,args.batch_size,True)
    selection=make_loader(store,validation,stats,args.eval_batch_size)
    net=DLinear(**stats).to(args.device)
    ck=load(args.checkpoint,net,stats) if args.checkpoint else None
    origin=int(ck.get('epoch',0)) if ck else 0
    lr=args.learning_rate or 3e-4
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    if ck and stage=='train':
        if ck.get('optimizer_state') is None or ck.get('rng') is None:raise ValueError('Continuation needs optimizer and random states')
        opt.load_state_dict(ck['optimizer_state'])
        if args.learning_rate is not None:
            for group in opt.param_groups:group['lr']=args.learning_rate
    if ck and ck.get('rng') is not None:restore_rng(ck['rng'])
    protocol={**vars(args),'model':'DLinear','stage':stage,'source_epoch':origin,'station_count':len(store.stations),
        'full_window_counts':counts,'actual_window_counts':dict(train=len(train),pred_select=len(validation)),
        'optimizer_reset':stage=='finetune','optimizer_restored':bool(ck and stage=='train'),
        'loss_weights':[2/3,1/3],'train_loss_normalization':'physical pinball divided by training demand std',
        'demand_training_std':float(stats['std'][-1]),'selection_role':'pred_select','test_used_for_selection':False,
        'stopping':'both absolute H1 MedMAE and H1 Pin90 changes below min_delta' if stage=='train' else 'fixed fine-tuning budget',
        'checkpoint_selection':'six H1/H3 normalized metric ratios to epoch0' if stage=='finetune' else 'H1/H3 mean MedMAE, then H1/H3 mean Pin90',
        'smoke_only':bool(args.limit_train_windows or args.limit_val_windows),
        'learning_rate_schedule':'original six-epoch cosine 3e-4 to3e-5, then constant' if not ck and args.learning_rate is None else 'constant',
        'torch':torch.__version__}
    save(output/'protocol.json',protocol)
    history=[];previous=None;anchor=None;best=None;best_epoch=None
    if ck:
        state=rng_state();anchor=evaluate(net,selection,store,args.device);restore_rng(state)
        previous=anchor;best=(1.,) if stage=='finetune' else key(anchor);best_epoch=0
        history.append(dict(epoch=0,record_kind='validation_anchor',train_loss_modules=None,**anchor))
        snapshot(output/'epoch_00.pt',net,opt,stats,origin,protocol,anchor,stage)
        shutil.copy2(output/'epoch_00.pt',output/'checkpoint_best.pt');write_history(output,history)
    if stage=='finetune' and (anchor is None or any(anchor[k]<=0 for k in KEYS)):raise ValueError('Fine-tuning needs a valid checkpoint anchor')
    del ck
    reason='epoch_budget'
    for epoch in (count(1) if args.epochs is None else range(1,args.epochs+1)):
        if not args.checkpoint and args.learning_rate is None:
            lr=float(3e-4*(.1+.9*(1+np.cos(np.pi*min(epoch-1,5)/5))/2))
            for group in opt.param_groups:group['lr']=lr
        started=time.time();net.train();total=0.;n=0
        for step,batch in enumerate(training,1):
            forecasts,truth,*_=forward(net,batch,args.device)
            loss=loss_fn(forecasts,truth)/float(stats['std'][-1])
            if not torch.isfinite(loss).item():raise ValueError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=loss.item()*len(truth);n+=len(truth)
            if step==1 or step%200==0 or step==len(training):save(output/'status.json',dict(model='DLinear',stage=stage,epoch=epoch,batch=step,total_batches=len(training)))
        state=rng_state();metrics=evaluate(net,selection,store,args.device);restore_rng(state)
        score=(float(np.mean([metrics[k]/anchor[k] for k in KEYS])),) if stage=='finetune' else key(metrics)
        improved=best is None or score<best
        if improved:best,best_epoch=score,epoch
        mae_change=abs(metrics['H1_MedMAE']-previous['H1_MedMAE']) if previous else None
        pin_change=abs(metrics['H1_Pin90']-previous['H1_Pin90']) if previous else None
        stop=stage=='train' and mae_change is not None and mae_change<args.min_delta and pin_change<args.min_delta
        row=dict(epoch=epoch,source_phase_epoch=origin+epoch if stage=='train' else None,
            train_loss_normalized=total/n,train_loss_modules=total/n*float(stats['std'][-1]),learning_rate=float(opt.param_groups[0]['lr']),
            H1_MedMAE_absolute_change=mae_change,H1_Pin90_absolute_change=pin_change,threshold_met=stop,
            selection_score=list(score),seconds=time.time()-started,**metrics)
        history.append(row);write_history(output,history)
        file=output/f'epoch_{epoch:02d}.pt'
        snapshot(file,net,opt,stats,origin+epoch if stage=='train' else epoch,protocol,metrics,stage)
        shutil.copy2(file,output/'checkpoint_latest.pt')
        if improved:shutil.copy2(file,output/'checkpoint_best.pt')
        print(row,flush=True);previous=metrics
        if stop:reason='both_H1_absolute_changes_below_threshold';break
    report=dict(model='DLinear',stage=stage,trained_epochs=epoch,best_stage_epoch=best_epoch,stop_reason=reason,
        selected_checkpoint=str(output/'checkpoint_best.pt'),smoke_only=protocol['smoke_only'],test_used_for_selection=False)
    save(output/'completed.json',report);print(report,flush=True)

def evaluate_entry():
    parser=argparse.ArgumentParser(description='Frozen DLinear evaluation')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'DATA')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'weights/DLinear/checkpoint_best.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['train','val','test'],default='test')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--limit-windows',type=int,default=0)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);seed_all(20260908)
    stats=scaler(args.data_dir);store=StationStore(args.data_dir)
    net=DLinear(**stats).to(args.device);ck=load(args.checkpoint,net,stats)
    index=windows(args.data_dir,args.split)
    if args.limit_windows:index=index.iloc[:args.limit_windows]
    metrics=evaluate(net,make_loader(store,index,stats,args.batch_size),store,args.device,args.output/'predictions.parquet')
    metrics.update(model='DLinear',split=args.split,checkpoint=str(args.checkpoint),checkpoint_epoch=ck.get('epoch'),smoke_only=bool(args.limit_windows))
    save(args.output/'metrics.json',metrics);print(metrics)
