# DLinear baseline

Run from the project root after placing the complete DATA there. This implementation uses the same shared input loader and scaler as FMF-Patch. The DLinear backbone has a 25-step moving average and shared 72-to-3 seasonal/trend linear projections, followed by the archived q50/q90 heads. Forecasts are transformed to physical module units and clipped to station capacity.

## From-scratch training

```bash
python train_dlinear.py --output runs/DLinear/train
```

The archived baseline learning-rate recipe is a six-epoch cosine decline from3e-4 to3e-5, then constant3e-5. Training stops when the absolute epoch-to-epoch changes in validation H1 MedMAE and H1 Pin90 are both below0.003. An optional `--epochs` argument caps a new run.

## Continue a checkpoint

```bash
python train_dlinear.py --checkpoint weights/DLinear/pre_finetune.pt --output runs/DLinear/continue
```

Continuation restores the optimizer and random states and retains the checkpoint learning rate unless an explicit override is supplied. Each round records all three forecast horizons.

## Fine-tuning

```bash
python finetune_dlinear.py --checkpoint weights/DLinear/pre_finetune.pt --epochs 3 --learning-rate 0.00001 --output runs/DLinear/finetune
```

Fine-tuning creates a new AdamW optimizer and compares its three epochs with its epoch-0 starting checkpoint. The selection score is the mean of six H1/H3 MedMAE, RMSE, and Pin90 ratios to the starting checkpoint. Training and fine-tuning use the validation `pred_select` subset; testing is separate.

## Frozen evaluation

```bash
python evaluate_dlinear.py --checkpoint weights/DLinear/checkpoint_best.pt --output runs/DLinear/test
```

The retained `checkpoint_best.pt` is the validation-selected fine-tuning epoch2 from the completed5090 experiment. Its fine-tuning start is `pre_finetune.pt`, the earlier epoch9 checkpoint; epoch10 triggered the stopping rule but did not replace this validation-selected start. Training histories are not published in this repository.

Recorded test results: H1 MedMAE2.6874816874, Pin90 0.8316109778; H3 MedMAE4.0070032300, Pin901.0667334386. Metadata and the selection record are in `results/DLinear/` and `configs/DLinear/`.

The backbone follows DLinear (https://arxiv.org/abs/2205.13504) and the archived Time-Series-Library implementation (https://github.com/thuml/Time-Series-Library). The two quantile heads and physical output mapping are the study-specific adaptation. New runs save their own protocol and history without overwriting the archived artifacts.
