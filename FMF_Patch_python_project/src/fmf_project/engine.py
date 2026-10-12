"""New portable entry points. Archived historical records remain unchanged."""
import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from itertools import count
import json
import random
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model import build_model
from .data import StationStore, Windows, scaler, windows

ROOT = Path(__file__).resolve().parents[2]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def make_loader(store, index, stats, batch, shuffle=False):
    return torch.utils.data.DataLoader(Windows(store, index, stats["mean"], stats["std"]),
        batch_size=batch, shuffle=shuffle, drop_last=False, num_workers=0)


def forward(net, batch, device):
    x, y, cap, unit, last, sid, start = batch
    x, y, cap, unit, last = [a.to(device, dtype=torch.float32) for a in [x,y,cap,unit,last]]
    q = net(x, cap, unit, last)
    if set(q) != {"q50", "q90"}:
        raise ValueError("Unexpected output quantiles")
    return q, y, cap, sid, start


def loss_fn(q, y):
    return sum(weight*torch.maximum(tau*(y-q[name]), (tau-1)*(y-q[name])).mean()
               for name,tau,weight in [("q50", .5, 2/3), ("q90", .9, 1/3)])


@torch.no_grad()
def evaluate(net, loader, store, device, export=None):
    net.eval()
    sums, n, writer = {}, 0, None
    try:
        for batch in loader:
            q, y, cap, sid, start = forward(net, batch, device)
            if not all(torch.isfinite(v).all().item() for v in q.values()):
                raise ValueError("Nonfinite forecast")
            if not (q["q90"] >= q["q50"]).all().item():
                raise ValueError("Crossing quantiles")
            n += len(y)
            for h in range(3):
                e, e90 = y[:,h]-q["q50"][:,h], y[:,h]-q["q90"][:,h]
                for name, value in [("MedMAE",e.abs()), ("MSE",e.square()),
                                    ("Pin90",torch.maximum(.9*e90,-.1*e90))]:
                    key=f"H{h+1}_{name}"
                    sums[key]=sums.get(key,0.)+value.double().sum().item()
            if export is not None:
                import pyarrow as pa
                import pyarrow.parquet as pq
                frame=pd.DataFrame(dict(sid=[store.stations[int(i)].sid for i in sid for _ in range(3)],
                    win_start=np.repeat(start.numpy(),3), horizon_step=np.tile([1,2,3],len(y)),
                    target=y.cpu().numpy().reshape(-1),q50=q["q50"].cpu().numpy().reshape(-1),
                    q90=q["q90"].cpu().numpy().reshape(-1),capacity=np.repeat(cap.cpu().numpy(),3)))
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(export,table.schema,compression="snappy")
                writer.write_table(table)
    finally:
        if writer is not None:writer.close()
    if not n:raise ValueError("Empty evaluation")
    result={k:v/n for k,v in sums.items()}
    for h in [1,2,3]:result[f"H{h}_RMSE"]=result.pop(f"H{h}_MSE")**.5
    result["validation_loss"]=sum(result[f"H{h}_{name}"] for h in [1,2,3] for name in ["MedMAE","Pin90"])/9
    result["n_windows"]=n
    return result


def prediction_key(metrics, scope):
    hs=[1] if scope=="H1" else [1,3]
    return tuple(float(np.mean([metrics[f"H{h}_{name}"] for h in hs])) for name in ["MedMAE","Pin90"])


def checkpoint_score(metrics, args, anchor=None):
    if args.selection=="lexicographic":return prediction_key(metrics,args.selection_scope)
    if anchor is None:raise ValueError("Composite selection requires an anchor")
    keys=[f"H{h}_{name}" for h in [1,3] for name in ["MedMAE","RMSE","Pin90"]]
    if any(anchor[k]<=0 for k in keys):raise ValueError("Nonpositive normalization anchor")
    return (float(np.mean([metrics[k]/anchor[k] for k in keys])),)


def write_history(output, rows):
    path=Path(output)/"history.csv"
    temporary=path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temporary,index=False)
    temporary.replace(path)


def run_training(stage):
    parser=argparse.ArgumentParser(description="Train FMF-Patch from scratch" if stage=="train" else "Fine-tune an existing FMF-Patch checkpoint")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"DATA")
    parser.add_argument("--output",type=Path,required=True,help="A NEW run directory")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",type=int,default=20260908)
    parser.add_argument("--epochs",type=int,default=None if stage=="train" else 3,help="Optional maximum epochs; training otherwise runs until the stopping threshold is met")
    parser.add_argument("--learning-rate",type=float,default=3e-4 if stage=="train" else 3e-5)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--eval-batch-size",type=int,default=512)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"weights/initial_epoch7.pt" if stage=="finetune" else None)
    parser.add_argument("--selection",choices=["lexicographic","normalized_composite"],default="lexicographic" if stage=="train" else "normalized_composite")
    parser.add_argument("--selection-scope",choices=["H1","H1H3"],default="H1H3")
    parser.add_argument("--early-stop-patience",type=int,default=1 if stage=="train" else 0,help="Consecutive rounds below the change threshold; 0 disables stopping with an explicit epoch budget")
    parser.add_argument("--min-delta",type=float,default=.003,help="Absolute H1 MedMAE and H1 Pin90 epoch-to-epoch change threshold")
    parser.add_argument("--min-epochs",type=int,default=0)
    parser.add_argument("--limit-train-windows",type=int,default=0,help="Smoke test only; 0 uses all training data")
    parser.add_argument("--limit-val-windows",type=int,default=0,help="Smoke test only; 0 uses all selection data")
    args=parser.parse_args()
    if (args.epochs is not None and args.epochs<1) or args.early_stop_patience<0 or args.min_delta<=0 or args.min_epochs<0:parser.error("Invalid training budget/stopping parameters")
    if args.epochs is None and args.early_stop_patience==0:parser.error("Specify --epochs when disabling early stopping")
    if stage=="train" and args.checkpoint:parser.error("From-scratch train entry cannot load a checkpoint; use finetune.py")
    if stage=="train" and args.selection=="normalized_composite":parser.error("Composite selection is for anchored fine-tuning")
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    seed_all(args.seed)
    store=StationStore(args.data_dir)
    stats=scaler(args.data_dir)
    train=windows(args.data_dir,"train")
    selection=windows(args.data_dir,"val",prediction_selection=True)
    full_counts=dict(train=len(train),selection=len(selection))
    if args.limit_train_windows:train=train.iloc[:args.limit_train_windows]
    if args.limit_val_windows:selection=selection.iloc[:args.limit_val_windows]
    train_loader=make_loader(store,train,stats,args.batch_size,True)
    validation_loader=make_loader(store,selection,stats,args.eval_batch_size)
    device=torch.device(args.device)
    net=build_model().to(device)
    source_epoch=None
    if stage=="finetune":
        ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        net.load_state_dict(ck.get("state_dict",ck.get("model_state")),strict=True)
        if not all(np.array_equal(ck[k],stats[k]) for k in ["mean","std"]):raise ValueError("Checkpoint scaler does not match this DATA")
        source_epoch=ck.get("epoch")
        del ck
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    protocol={**vars(args),"stage":stage,"from_scratch":stage=="train","optimizer_reset":stage=="finetune",
        "source_checkpoint_epoch":source_epoch,"full_window_counts":full_counts,
        "actual_window_counts":dict(train=len(train),selection=len(selection)),
        "selection_role":"pred_select","test_used_for_selection":False,"horizon":3,"quantiles":[.5,.9],
        "loss_weights":[2/3,1/3],"loss_units":"physical modules","station_count":len(store.stations),
        "torch":torch.__version__,"cuda":torch.version.cuda,
        "smoke_only":bool(args.limit_train_windows or args.limit_val_windows),
        "note":"This portable runner records its own actual protocol. Historical archived records are separate and unchanged."}
    save(output/"protocol.json",protocol)
    history=[]
    best_score=None
    best_epoch=0
    stale=0
    anchor=None
    previous=None
    def snapshot(epoch,metrics,score):
        return dict(state_dict=net.state_dict(),optimizer_state=optimizer.state_dict(),rng=rng_state(),
            mean=stats["mean"],std=stats["std"],seed=args.seed,epoch=epoch,stage=stage,
            protocol=protocol,metrics=metrics,selection_score=list(score),history=history,
            best_epoch=best_epoch,best_score=list(best_score),stale_epochs=stale)
    if stage=="finetune":
        before=rng_state()
        anchor=evaluate(net,validation_loader,store,device)
        restore_rng(before)
        previous=anchor
        best_score=checkpoint_score(anchor,args,anchor)
        history.append(dict(epoch=0,record_kind="validation_anchor",train_loss=None,**anchor))
        write_history(output,history)
        torch.save(snapshot(0,anchor,best_score),output/"checkpoint_best.pt")
        torch.save(snapshot(0,anchor,best_score),output/"epoch_00.pt")
    reason="fixed_epoch_budget" if args.early_stop_patience==0 else "max_epoch_budget"
    for epoch in (count(1) if args.epochs is None else range(1,args.epochs+1)):
        started=time.time()
        net.train()
        total,n=0.,0
        for step,batch in enumerate(train_loader,1):
            q,y,*_=forward(net,batch,device)
            loss=loss_fn(q,y)
            if not torch.isfinite(loss).item():raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total+=loss.item()*len(y)
            n+=len(y)
            if step==1 or step%200==0 or step==len(train_loader):
                status=dict(stage=stage,epoch=epoch,batch=step,total_batches=len(train_loader),seconds=time.time()-started)
                save(output/"status.json",status)
                print(json.dumps(status),flush=True)
        before=rng_state()
        metrics=evaluate(net,validation_loader,store,device)
        restore_rng(before)
        score=checkpoint_score(metrics,args,anchor)
        improved=best_score is None or score<best_score
        if improved:best_score,best_epoch=score,epoch
        mae_drop=previous["H1_MedMAE"]-metrics["H1_MedMAE"] if previous else None
        pin_drop=previous["H1_Pin90"]-metrics["H1_Pin90"] if previous else None
        below=(mae_drop is not None and abs(mae_drop)<args.min_delta and abs(pin_drop)<args.min_delta)
        if epoch>=args.min_epochs:
            stale=stale+1 if below else 0
        else:stale=0
        row=dict(epoch=epoch,record_kind="trained_epoch",train_loss=total/n,
            seconds=time.time()-started,learning_rate=args.learning_rate,
            H1_MedMAE_drop=mae_drop,H1_Pin90_drop=pin_drop,
            H1_MedMAE_absolute_change=abs(mae_drop) if mae_drop is not None else None,
            H1_Pin90_absolute_change=abs(pin_drop) if pin_drop is not None else None,
            both_changes_below_min_delta=below,stale_epochs=stale,
            checkpoint_improved=improved,selection_score=list(score),**metrics)
        history.append(row)
        write_history(output,history)
        ck=snapshot(epoch,metrics,score)
        torch.save(ck,output/f"epoch_{epoch:02d}.pt")
        torch.save(ck,output/"checkpoint_latest.pt")
        if improved:torch.save(ck,output/"checkpoint_best.pt")
        print(json.dumps(row),flush=True)
        previous=metrics
        if args.early_stop_patience and stale>=args.early_stop_patience:
            reason="early_stopping_patience_exhausted"
            break
    result=dict(stage=stage,best_epoch=best_epoch,completed_epochs=epoch,stop_reason=reason,
        selected_checkpoint=str(output/"checkpoint_best.pt"),selection_score=list(best_score),
        test_used_for_selection=False,smoke_only=protocol["smoke_only"])
    save(output/"completed.json",result)
    print(json.dumps(result),flush=True)


def evaluate_entry():
    parser=argparse.ArgumentParser(description="Evaluate a frozen FMF-Patch checkpoint")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"weights/paper_checkpoint.pt")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"DATA")
    parser.add_argument("--split",choices=["train","val","test"],default="test")
    parser.add_argument("--selection-only",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--limit-windows",type=int,default=0,help="Smoke only; full metrics require 0")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    seed_all(20260908)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    stats={k:ck[k] for k in ["mean","std"]}
    actual=scaler(args.data_dir)
    if not all(np.array_equal(stats[k],actual[k]) for k in stats):raise ValueError("Scaler mismatch")
    store=StationStore(args.data_dir)
    index=windows(args.data_dir,args.split,args.selection_only)
    if args.limit_windows:index=index.iloc[:args.limit_windows]
    net=build_model().to(args.device)
    net.load_state_dict(ck["state_dict"],strict=True)
    metrics=evaluate(net,make_loader(store,index,stats,args.batch_size),store,args.device,args.output/"predictions.parquet")
    metrics.update(split=args.split,checkpoint=str(args.checkpoint),checkpoint_epoch=ck.get("epoch"),
                   smoke_only=bool(args.limit_windows),selection_only=args.selection_only)
    save(args.output/"metrics.json",metrics)
    print(json.dumps(metrics),flush=True)
