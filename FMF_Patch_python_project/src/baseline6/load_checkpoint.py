"""Load one of the six locally archived baseline checkpoints."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'tslib'))

import torch
import new_models

KINDS = tuple(new_models.KINDS)
RUN = 'baseline_threshold_20261012'


def load_baseline(kind, device='cpu'):
    if kind not in KINDS:
        raise ValueError(f'Unknown baseline: {kind}')
    path = PROJECT / 'weights' / kind / RUN / 'selected_checkpoint.pt'
    record = torch.load(path, map_location='cpu', weights_only=False)
    net = new_models.build(kind, seed=20260908)
    net.load_state_dict(record['state_dict'], strict=True)
    net.to(device).eval()
    return net, record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=KINDS)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    net, record = load_baseline(args.model, args.device)
    print(json.dumps({'model': args.model,
                      'checkpoint': str(PROJECT / 'weights' / args.model / RUN / 'selected_checkpoint.pt'),
                      'parameters': sum(p.numel() for p in net.parameters()),
                      'epoch': record.get('epoch'), 'stage': record.get('stage')}, ensure_ascii=False))
