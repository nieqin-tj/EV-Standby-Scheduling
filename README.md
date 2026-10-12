# EV-Standby-Scheduling
EV standby-power scheduling research project


## FMF-Patch

The [FMF-Patch Python project](FMF_Patch_python_project/) contains training and fine-tuning entry points, retained weights, historical logs, and prediction metrics. Complete model input data will be distributed through an external platform; see the project README for the required layout.

## Station allocation validation

The [station allocation validation project](station_allocation_validation/) reconstructs delivered charging power at ten stations and scores 20-min station-level module demand. Frozen windows and minute predictions needed to reproduce the 0.62-module result are included. Raw operator curve files remain outside this repository; see the project README.
