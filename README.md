# EV-Standby-Scheduling
EV standby-power scheduling research project


## FMF-Patch

The [FMF-Patch Python project](FMF_Patch_python_project/) contains training and fine-tuning entry points, retained model weights, and frozen prediction metrics. The complete model input data is available separately on [Figshare](https://figshare.com/s/930f4c3a22bcf81b2e8f); see the [data layout instructions](FMF_Patch_python_project/DATA/README.md). Training logs are not published in this repository.

### Table B4: Annualized standby energy

The six baselines were evaluated on the same test set (1,897,346 windows from 313 stations). FMF-Patch and Seasonal Naive are paper reference results. This table reproduces manuscript Table B4; the archived machine-readable results are in the [results CSV](FMF_Patch_python_project/results/baselines/summary_20261012.csv).

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

### Table B2: Forecast accuracy

`Pinball₀.₉` is the pinball loss at the 90th percentile; lower values indicate better performance. This table reproduces manuscript Table B2. Detailed metrics are in the [project README](FMF_Patch_python_project/README.md) and [results CSV](FMF_Patch_python_project/results/baselines/summary_20261012.csv).

| Model | MedMAE (H=1) | Pinball₀.₉ (H=1) | MedMAE (H=3) | Pinball₀.₉ (H=3) |
| --- | ---: | ---: | ---: | ---: |
| FMF-Patch | 2.410 | 0.693 | 3.642 | 0.925 |
| Informer | 2.448 | 0.701 | 3.659 | 0.936 |
| iTransformer | 2.531 | 0.721 | 3.724 | 0.955 |
| Autoformer | 2.698 | 0.755 | 3.85 | 0.993 |
| PatchTST | 2.626 | 0.777 | 3.841 | 1.006 |
| DLinear | 2.687 | 0.832 | 4.007 | 1.067 |
| TimesNet | 2.447 | 0.704 | 3.683 | 0.943 |

**Manuscript vs. archive:** Table B4 reports a 1.9% Informer energy increase, while 246,286 versus 242,144 kWh/yr gives about 1.7% (the archived value is 1.710%). Table B2 reports Informer H=1 MedMAE as 2.448, while the archived value 2.448833 rounds to 2.449 at three decimal places. The tables above retain the manuscript values.

## Station allocation validation

The [station allocation validation project](station_allocation_validation/) reconstructs delivered charging power at ten stations and scores 20-min station-level module demand. Frozen windows and minute predictions needed to reproduce the 0.62-module result are included. Raw operator curve files remain outside this repository; see the project README.
