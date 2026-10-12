# FMF-Patch forecasting and standby-energy evaluation

## English

We provide the FMF-Patch implementation, six forecasting baselines, selected model checkpoints, experimental configurations, and evaluation artifacts for electric-vehicle charging-station standby scheduling. We report forecasting accuracy and operational energy outcomes on the same 313-station test set.

### Data and repository contents

We provide the model-input dataset separately on [Figshare](https://figshare.com/s/930f4c3a22bcf81b2e8f). After downloading it, place the `shared/`, `train/`, `val/`, and `test/` directories under `DATA/` as described in [DATA/README.md](DATA/README.md). The dataset contains the station features, scaling parameters, and data windows used by the evaluation code.

We organize the reproducibility materials as follows:

| Path | Contents |
| --- | --- |
| `src/fmf_project/` | FMF-Patch model, data loader, training, and evaluation code |
| `src/dlinear_project/` and `src/baseline6/` | Baseline implementations and checkpoint-loading code |
| `configs/` | Model and evaluation configurations |
| `weights/` | Selected checkpoints |
| `results/` | Forecasting, coverage, and standby-energy results |

### Table B2. Forecasting performance

We report the forecasting results below. Lower MedMAE and Pinball₀.₉ indicate better performance.

| Model | MedMAE (H=1) | Pinball₀.₉ (H=1) | MedMAE (H=3) | Pinball₀.₉ (H=3) |
| --- | ---: | ---: | ---: | ---: |
| FMF-Patch | 2.410 | 0.693 | 3.642 | 0.925 |
| Informer | 2.448 | 0.701 | 3.659 | 0.936 |
| iTransformer | 2.531 | 0.721 | 3.724 | 0.955 |
| Autoformer | 2.698 | 0.755 | 3.85 | 0.993 |
| PatchTST | 2.626 | 0.777 | 3.841 | 1.006 |
| DLinear | 2.687 | 0.832 | 4.007 | 1.067 |
| TimesNet | 2.447 | 0.704 | 3.683 | 0.943 |

### Table B4. Standby energy and empirical coverage

We obtain the operational outcomes by passing each forecasting model through the same readiness rule. The six trained baselines were evaluated on 1,897,346 test windows from 313 stations; FMF-Patch and Seasonal Naive are the reference rows reported in the manuscript.

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

### Reproducing the evaluation

We run the following commands from this directory after placing the Figshare data under `DATA/` and installing a PyTorch build compatible with the available hardware:

```bash
python -m pip install -r requirements.txt
python check_project.py
python evaluate.py --checkpoint weights/FMF-Patch/paper_checkpoint.pt --split test --output runs/fmf_patch_test
```

We provide the selected baseline checkpoints under `weights/<model>/baseline_threshold_20261012/`, their configurations under `configs/<model>/`, and their evaluation artifacts under `results/<model>/`. We summarize the unified results in the [baseline summary](results/baselines/README.md).

## 中文

我们提供 FMF-Patch 及六个预测基线的实现、选定模型权重、实验配置和评价结果，用于电动汽车充电站待机调度研究。我们在相同的 313 站点测试集上报告预测精度和运行能耗。

### 数据与项目内容

我们通过 [Figshare](https://figshare.com/s/930f4c3a22bcf81b2e8f) 单独提供模型输入数据。下载后，按照 [DATA/README.md](DATA/README.md) 将 `shared/`、`train/`、`val/` 和 `test/` 放入 `DATA/`。数据包含评价代码使用的站点特征、归一化参数和数据窗口。

我们按以下结构组织复现材料：

| 路径 | 内容 |
| --- | --- |
| `src/fmf_project/` | FMF-Patch 模型、数据读取、训练和评价代码 |
| `src/dlinear_project/` 与 `src/baseline6/` | 基线实现及权重加载代码 |
| `configs/` | 模型与评价配置 |
| `weights/` | 选定模型权重 |
| `results/` | 预测、覆盖率与待机能耗结果 |

### 表 B2：预测性能

我们报告以下预测结果。MedMAE 和 Pinball₀.₉ 越低，性能越好。

| 模型 | MedMAE (H=1) | Pinball₀.₉ (H=1) | MedMAE (H=3) | Pinball₀.₉ (H=3) |
| --- | ---: | ---: | ---: | ---: |
| FMF-Patch | 2.410 | 0.693 | 3.642 | 0.925 |
| Informer | 2.448 | 0.701 | 3.659 | 0.936 |
| iTransformer | 2.531 | 0.721 | 3.724 | 0.955 |
| Autoformer | 2.698 | 0.755 | 3.85 | 0.993 |
| PatchTST | 2.626 | 0.777 | 3.841 | 1.006 |
| DLinear | 2.687 | 0.832 | 4.007 | 1.067 |
| TimesNet | 2.447 | 0.704 | 3.683 | 0.943 |

### 表 B4：待机能耗与经验覆盖率

我们将各预测模型的输出输入相同的就绪决策规则，得到以下运行结果。六个训练基线均在 313 个站点的 1,897,346 个测试窗口上评价；FMF-Patch 和 Seasonal Naive 是论文报告的参考行。

| 预测模型 | 年待机能耗 (kWh yr⁻¹) | 经验覆盖率 (%) | 相对 FMF-Patch 的能耗增幅 (%) |
| --- | ---: | ---: | ---: |
| FMF-Patch | 242,144 | 94.69 | — |
| Informer | 246,286 | 94.72 | 1.9 |
| iTransformer | 251,115 | 94.64 | 3.7 |
| Autoformer | 260,639 | 94.52 | 7.6 |
| PatchTST | 264,691 | 94.46 | 9.3 |
| DLinear | 281,729 | 94.67 | 16.3 |
| TimesNet | 245,417 | 94.61 | 1.4 |
| Seasonal Naive | 419,924 | 94.48 | 73.4 |

### 复现评价

我们将 Figshare 数据放入 `DATA/`，并安装与运行硬件兼容的 PyTorch 后，在本目录执行：

```bash
python -m pip install -r requirements.txt
python check_project.py
python evaluate.py --checkpoint weights/FMF-Patch/paper_checkpoint.pt --split test --output runs/fmf_patch_test
```

我们在 `weights/<model>/baseline_threshold_20261012/` 保存选定基线权重，在 `configs/<model>/` 保存配置，在 `results/<model>/` 保存评价结果。我们在[基线汇总](results/baselines/README.md)中概述统一结果。
