# EV-Standby-Scheduling

This repository contains two research subprojects on electric-vehicle charging-station standby scheduling.

## FMF-Patch forecasting

The [FMF-Patch Python project](FMF_Patch_python_project/) provides forecasting and baseline-model code, selected checkpoints, experimental protocols, and evaluation results. Its README contains the forecasting and standby-energy comparison tables for reviewers. The model input data is available separately on [Figshare](https://figshare.com/s/930f4c3a22bcf81b2e8f).

## Station allocation validation

The [station allocation validation project](station_allocation_validation/) reconstructs delivered charging power at ten stations under operator-published allocation rules and scores 20-min station-level module demand. Frozen windows in that project reproduce the 0.62-module result. Internal station layout source files are not included. High-resolution charging-process curves and the vehicle-series library will be released on Figshare.
