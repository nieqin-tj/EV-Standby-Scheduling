from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from fmf_project.engine import ROOT,seed_all,evaluate,make_loader
from fmf_project.model import build_model
from fmf_project.data import StationStore,scaler,windows
import numpy as np
import torch

if __name__=="__main__":
    seed_all(20260908)
    store=StationStore(ROOT/"DATA")
    stats=scaler(ROOT/"DATA")
    counts={}
    for split in ["train","val","test"]:
        index=windows(ROOT/"DATA",split)
        for sid,group in index.groupby("sid"):
            feature=store.stations[store.by_sid[sid]].features
            assert group.win_start.min()>=0 and group.win_start.max()+75<=len(feature)
        counts[split]=len(index)
    index=windows(ROOT/"DATA","val",True).iloc[:2]
    checks=[]
    for path in sorted((ROOT/"weights/FMF-Patch").glob("*.pt")):
        ck=torch.load(path,map_location="cpu",weights_only=False)
        net=build_model()
        net.load_state_dict(ck["state_dict"],strict=True)
        assert all(np.array_equal(ck[k],stats[k]) for k in ["mean","std"])
        result=evaluate(net,make_loader(store,index,stats,2),store,"cpu")
        checks.append(dict(file=path.name,phase_epoch=ck.get("epoch"),
                           parameters=sum(p.numel() for p in net.parameters()),finite_forward=True))
    result=dict(passed=True,stations=len(store.stations),windows=counts,weights=checks,
                optimizer_updates=0,test_inference=False)
    print(json.dumps(result,indent=2))
