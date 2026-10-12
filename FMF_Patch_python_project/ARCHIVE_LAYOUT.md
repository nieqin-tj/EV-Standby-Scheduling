# 模型与结果归档

模型统一放在 `weights/<模型名>/`，冻结结果在 `results/<模型名>/`，协议配置在 `configs/<模型名>/`。模型包括 FMF-Patch、DLinear、Informer、iTransformer、Autoformer、PatchTST 和 TimesNet。机器可读清单见 `archive_models.json`。源码位于 `src/`；第三方许可见 `THIRD_PARTY_LICENSES/`。

六个基线的选定检查点位于 `weights/<模型名>/baseline_threshold_20261012/`，冻结选择、验证、测试和统一能耗指标位于 `results/<模型名>/baseline_threshold_20261012/`。总体比较见 `results/baselines/summary_20261012.csv` 和 `uniform_results_20261012.json`。

完整模型输入数据从 [Figshare](https://figshare.com/s/930f4c3a22bcf81b2e8f) 获取；放置说明见 `DATA/README.md`。本仓库不发布训练日志和逐窗口预测 Parquet。
