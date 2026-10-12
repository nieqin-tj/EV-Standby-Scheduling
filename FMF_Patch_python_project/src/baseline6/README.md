# 六个预测基线的模型定义

本目录保存 5090 最终实验使用的 `new_models.py` 和依赖的时序模型源码。在项目根目录安装 `requirements.txt` 和本目录的 `requirements.txt` 后，可运行：

```powershell
python src/baseline6/load_checkpoint.py --model TimesNet
```

检查点默认取 `weights/<模型名>/baseline_threshold_20261012/selected_checkpoint.pt`。推理输入和量纲以各模型的 `configs/<模型名>/baseline_threshold_20261012/protocol.json`、`results/<模型名>/baseline_threshold_20261012/frozen_selection.json` 为准。TimesNet 的加速训练使用 BF16 和 batch 2048；最终选中的是历史第 6 轮检查点。归档不包含训练日志。
