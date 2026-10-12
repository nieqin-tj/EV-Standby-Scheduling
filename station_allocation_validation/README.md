# Station allocation validation

## English

We validate station-level module-demand estimates using ten charging stations (S01–S10). We report two complementary endpoints at different time scales; they should be interpreted separately.

| Endpoint | Resolution | Mean absolute error | 95% confidence interval | Relative error | Normalized bias |
| --- | --- | ---: | ---: | ---: | ---: |
| Station-level module demand | 20-minute windows | 0.62 modules | 0.44–0.80 | 5.44% | −2.74% |
| Cabinet equivalent occupancy | Cabinet-minute | 0.17 modules | 0.14–0.19 | 4.94% | −0.57% |

For the 20-minute endpoint, we average demand within each cabinet window, round up to an integer number of modules, sum across cabinets, and give each station equal weight in the final score. We provide the station-level breakdown in the [20-minute module-demand report](reports/module_20min.md).

For allocation confirmation, we implement the operator-declared rules for S01 and S02 and dynamic-demand allocation for S03–S10. We select numerical parameters on validation data and report confirmation results after fixing those choices. We provide station-level power and equivalent-occupancy results in the [allocation report](reports/report.md) and [metrics CSV](reports/metrics.csv).

### Reproducing the results

We include the frozen 20-minute scoring windows under `data/module_20min/windows/`, minute-level reconstructed power under `data/module_20min/minute/`, and session inputs under `data/module_20min/orders/`. From this directory, we run:

```bash
python -m pip install -r requirements.txt
python scripts/score_20min.py
python scripts/run_validation.py
python -m pytest tests -q
```

We keep the original operator curve files outside this repository. The included derived inputs and reports support reproduction of the scores reported above.

## 中文

我们使用十个充电站（S01–S10）验证站点级模块需求估计。我们分别报告两个时间尺度不同的评价指标，避免将两者混为一谈。

| 评价指标 | 时间尺度 | 平均绝对误差 | 95% 置信区间 | 相对误差 | 归一化偏差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 站点级模块需求 | 20 分钟窗口 | 0.62 个模块 | 0.44–0.80 | 5.44% | −2.74% |
| 机柜等效占用 | 机柜分钟 | 0.17 个模块 | 0.14–0.19 | 4.94% | −0.57% |

对于 20 分钟指标，我们先计算每个机柜窗口内需求的平均值，向上取整为整数模块数，再在站点内汇总；最终结果对各站点等权平均。各站点结果见 [20 分钟模块需求报告](reports/module_20min.md)。

对于分配规则验证，我们对 S01、S02 使用运营方声明的规则，对 S03–S10 使用动态需求分配。我们在验证数据上选择数值参数，并在固定参数后报告确认结果。各站点功率与等效占用结果见 [分配验证报告](reports/report.md)和[指标 CSV](reports/metrics.csv)。

### 复现结果

我们在 `data/module_20min/windows/` 提供固定的 20 分钟评分窗口，在 `data/module_20min/minute/` 提供分钟级重建功率，在 `data/module_20min/orders/` 提供订单输入。在本目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/score_20min.py
python scripts/run_validation.py
python -m pytest tests -q
```

我们未随仓库分发原始运营曲线文件；提供的衍生输入和报告可用于复现上述评分结果。
