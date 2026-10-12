# FMF-Patch Python项目

本仓库公开模型代码、配置、选定权重与冻结测试指标。完整313站模型输入由 [Figshare 数据页面](https://figshare.com/s/930f4c3a22bcf81b2e8f)单独提供；下载后按 `DATA/README.md` 放入本目录。训练日志不随仓库发布。数据就位后，代码和权重均通过项目相对路径读取。

## 目录

```text
FMF_Patch_python_project/
  train.py                 从随机初始化训练
  finetune.py              从已有检查点微调，重新初始化优化器
  evaluate.py              冻结权重评价，另行显式运行
  check_project.py         核验DATA、权重加载和有限值前向计算
  src/fmf_project/         模型、数据和训练实现
  DATA/README.md          数据获取地址与放置说明
  DATA/                   下载后放入模型输入
    shared/features/      313份完整float32特征数组
    shared/scaler.npz      原训练期归一化参数
    shared/stations.csv   站点容量及模块功率信息
    shared/metadata/      输入schema及来源描述
    train/                完整训练窗口清单
    val/                  完整验证窗口清单及角色
    test/                 完整测试窗口清单和窗口键
  weights/FMF-Patch/      原始权重与明确命名的阶段检查点
  results/FMF-Patch/archived/        论文版本的已冻结预测指标
  configs/FMF-Patch/archived/        各阶段原始协议和选择记录
  provenance/             历史训练源码、环境和文件校验清单
  runs/                   新运行目录，原结果不会覆盖
```

## 两个入口

建议使用Python3.11及与显卡兼容的PyTorch环境。`provenance/environment.json`记录核验时的实际版本。Python解释器和CUDA驱动不打入项目；在已有5090训练环境中可直接运行，其他机器先准备兼容环境：

```powershell
python -m pip install -r requirements.txt
python check_project.py
```

从头训练：

```powershell
python train.py --learning-rate 0.0003 --output runs/train_s20260908
```

从原第7轮检查点微调：

```powershell
python finetune.py --checkpoint weights/FMF-Patch/initial_epoch7.pt --epochs 3 --learning-rate 0.00003 --output runs/finetune_s20260908
```

每个输出目录必须是新目录。也可以给`--checkpoint`传入新训练输出的某轮检查点。微调保留网络权重，但按原微调方式重新初始化AdamW优化器，不冒充保留优化器的连续恢复训练。

新运行逐轮保存训练损失、验证损失、H1/H2/H3 MedMAE/Pin90/RMSE、学习率、检查点选择分数和停止状态；还保存每一轮权重、优化器及随机状态。微调第0轮只验证起点，不执行参数更新，且参与最佳检查点比较。

从头训练默认在验证`pred_select`角色上按H1/H3平均MedMAE，再按平均Pin90选择检查点。微调默认按相对起点的六项H1/H3 MedMAE、RMSE、Pin90比率均值选择，与原微调阶段一致。可用`--selection lexicographic --selection-scope H1`显式选择其他新运行规则；实际规则写入新运行`protocol.json`。

## 训练停止条件

训练默认在相邻轮次的验证H1 MedMAE与H1 Pin90绝对变化均小于0.003时停止，不再固定为8轮。微调默认运行3轮。

```powershell
python train.py --min-delta 0.003 --output runs/train_early_stop
```

## 权重和论文数字

- `weights/FMF-Patch/initial_epoch7.pt`：原第7轮权重，微调默认起点。
- `weights/FMF-Patch/finetune_epoch_01.pt`至`finetune_epoch_03.pt`：真实微调阶段候选；其中第2轮是原微调选择结果。
- `weights/FMF-Patch/finetune_best_epoch2.pt`：原微调阶段保留权重。
- `weights/FMF-Patch/paper_checkpoint.pt`：当前正文表B2实际使用的后续延长训练第1轮权重。它并非微调第2轮；完整选择记录保留在`configs/FMF-Patch/archived`中。

当前正文权重的原归档记录包含补充测试比较后指定这一版本的事实。项目保留该记录，不将它改写为单纯验证选择。原稿中的0.003停止表述不能由旧固定预算日志自动证实。

## 冻结评价

训练和微调入口不自动读取测试集。显式运行评价：

```powershell
python evaluate.py --checkpoint weights/FMF-Patch/paper_checkpoint.pt --split test --output runs/paper_test_replay
```

评价输出逐窗口Parquet和指标JSON。原指标是H1 MedMAE2.410194、Pin90 0.692623，H3 MedMAE3.642264、Pin90 0.925103。完整重放在验证记录中给出数值误差；不同硬件/依赖下的重新训练不保证逐位一致。

## 快速执行检查

```powershell
python train.py --epochs 1 --batch-size 2 --eval-batch-size 2 --limit-train-windows 2 --limit-val-windows 2 --output runs/train_smoke
python finetune.py --epochs 1 --batch-size 2 --eval-batch-size 2 --limit-train-windows 2 --limit-val-windows 2 --output runs/finetune_smoke
```

这些是真实输入的小批次执行检查，输出会标记`smoke_only=true`，不能作为完整训练结果。

DATA包含模型实际使用的20分钟特征输入和确切窗口，而非1400万条原始订单。项目复核范围是模型输入之后的预测，不重新运行原始订单重建流程。研究历史测试集已被使用，本项目的冻结重放不是新的未见测试实验。

## DLinear基线

训练、续训、微调及评价入口见 [DLinear_README.md](DLinear_README.md)。已选中的微调第2轮权重和微调起点分别保存在 `weights/DLinear/`，对应指标保存在 `results/DLinear/`。

## 统一归档规则

各模型使用 weights/<模型名>/、results/<模型名>/、configs/<模型名>/。模型清单、后续加入规则与回退说明见 [ARCHIVE_LAYOUT.md](ARCHIVE_LAYOUT.md)。

## 六基线最终模型与指标（2026-10-12）

六个基线的最终权重位于 `weights/<模型名>/baseline_threshold_20261012/selected_checkpoint.pt`；冻结选择、验证/测试和统一待机能耗结果位于 `results/<模型名>/baseline_threshold_20261012/`。`results/baselines/uniform_results_20261012.json` 是六基线统一测试集汇总，`src/baseline6/load_checkpoint.py` 提供检查点加载入口。本次按要求未复制这些基线的训练日志或逐窗口中间文件。

### Table B4：待机能耗与覆盖率

六个基线在统一测试集（1,897,346 个窗口、313 个站点）上评价。FMF-Patch 和 Seasonal Naive 为论文参考结果。下表照录论文表 B4；归档的机器可读结果见 [`results/baselines/summary_20261012.csv`](results/baselines/summary_20261012.csv)。

| Forecasting model | Annualized standby energy (kWh yr⁻¹) | Empirical coverage (%) | Energy increase relative to FMF-Patch (%) |
| --- | ---: | ---: | ---: |
| FMF-Patch | 242,144 | 94.69 | — |
| Informer | 246,286 | 94.72 | 1.9 |
| iTransformer | 251,115 | 94.64 | 3.7 |
| Autoformer | 260,639 | 94.52 | 7.6 |
| PatchTST | 264,691 | 94.46 | 9.3 |
| DLinear | 281,729 | 94.67 | 16.3 |
| TimesNet | 245,417 | 94.61 | 1.4 |
| Seasonal Naive | 419,924 | 94.48 | 73.4 |

### Table B2：预测精度

`Pinball₀.₉` 是 90% 分位数的 pinball loss；数值越低越好。下表照录论文表 B2。基线归档原始值见上述 CSV；FMF-Patch 的原始值见 [`results/FMF-Patch/archived/test_prediction_metrics.json`](results/FMF-Patch/archived/test_prediction_metrics.json)。

| Model | MedMAE (H=1) | Pinball₀.₉ (H=1) | MedMAE (H=3) | Pinball₀.₉ (H=3) |
| --- | ---: | ---: | ---: | ---: |
| FMF-Patch | 2.410 | 0.693 | 3.642 | 0.925 |
| Informer | 2.448 | 0.701 | 3.659 | 0.936 |
| iTransformer | 2.531 | 0.721 | 3.724 | 0.955 |
| Autoformer | 2.698 | 0.755 | 3.85 | 0.993 |
| PatchTST | 2.626 | 0.777 | 3.841 | 1.006 |
| DLinear | 2.687 | 0.832 | 4.007 | 1.067 |
| TimesNet | 2.447 | 0.704 | 3.683 | 0.943 |
