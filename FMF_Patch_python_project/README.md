# FMF-Patch Python project

Training, fine-tuning, and frozen prediction evaluation for FMF-Patch in the EV standby scheduling study.

## Files

| Path | Purpose |
| --- | --- |
| `train.py` | Train from random initialization |
| `finetune.py` | Fine-tune an existing checkpoint with a new AdamW optimizer |
| `evaluate.py` | Evaluate frozen weights and export forecasts |
| `check_project.py` | Check input windows, checkpoint loading, and finite forward outputs |
| `src/fmf_project/` | Model, data loader, loss, and training implementation |
| `weights/` | Three retained checkpoints |
| `logs/historical/` | Original training, fine-tuning, and extension histories |
| `configs/archived/` | Original experiment protocols and checkpoint selection records |
| `results/archived/` | Recorded prediction metrics and station-level test results |
| `provenance/environment.json` | Environment used for the execution checks |
| `DATA/README.md` | Required dataset layout |

## Installation and data

Use Python 3.11 and a PyTorch installation compatible with your hardware.

```bash
python -m pip install -r requirements.txt
```

The **complete DATA is hosted separately and is not included in this repository**. Its download link will be added after the external deposit. Place the data in `FMF_Patch_python_project/DATA/` as described in [DATA/README.md](DATA/README.md). All model inputs are read through project-relative paths.

Run the commands below from this project directory after placing the data.

## Training

```bash
python train.py --learning-rate 0.0003 --output runs/train_s20260908
```

Training stops when the absolute epoch-to-epoch changes in validation H1 MedMAE and H1 Pin90 are both less than 0.003. Checkpoint selection uses mean validation H1/H3 MedMAE, followed by mean H1/H3 Pin90. Training and fine-tuning use the validation `pred_select` role and do not read the test split.

## Fine-tuning

```bash
python finetune.py --checkpoint weights/initial_epoch7.pt --epochs 3 --learning-rate 0.00003 --output runs/finetune_s20260908
```

Fine-tuning resets the optimizer and runs three epochs by default. The starting checkpoint is evaluated as epoch 0 and participates in selection. The default score is the mean of six validation metric ratios relative to the starting checkpoint: H1/H3 MedMAE, RMSE, and Pin90.

Each new output directory contains the run protocol, per-epoch training loss and validation metrics, checkpoints, and completion status. The output directory must not already exist. Archived histories describe the original experiments; new runs record their own settings and results.

## Retained checkpoints

| File | Stage |
| --- | --- |
| `weights/initial_epoch7.pt` | Original training epoch 7; default fine-tuning start |
| `weights/finetune_best_epoch2.pt` | Selected checkpoint from the original three-epoch fine-tuning stage, epoch 2 |
| `weights/paper_checkpoint.pt` | Subsequent extension epoch 1; checkpoint used for the current manuscript prediction results |

The original stage and selection records are retained in `configs/archived/` and `logs/historical/`.

## Frozen evaluation

```bash
python evaluate.py --checkpoint weights/paper_checkpoint.pt --split test --output runs/paper_test_replay
```

This writes `predictions.parquet` and `metrics.json`. The recorded paper-checkpoint test metrics are:

| Horizon | MedMAE | Pin90 |
| --- | ---: | ---: |
| H1 | 2.410194 | 0.692623 |
| H3 | 3.642264 | 0.925103 |

## Input and execution checks

```bash
python check_project.py
```

The packaged implementation was checked using one small training update, one small fine-tuning update, strict loading of retained checkpoints, and a 1,024-test-window frozen prediction comparison. The sample q50 and q90 forecasts matched the archived forecasts. A complete training rerun and a complete test replay were not performed during packaging.

The data represent model-ready inputs rather than the raw charging-order database. This project covers forecasting after input preparation; it does not include the full standby scheduling evaluation pipeline. New training results may vary across hardware and dependency versions.
