# 数据获取与放置

本项目的完整313站模型输入数据单独存放于 [Figshare](https://figshare.com/s/930f4c3a22bcf81b2e8f)。GitHub 仓库不包含数据文件。下载后将数据放置在本目录，预期结构如下：

```text
DATA/
  shared/features/     313 个站点的特征数组
  shared/scaler.npz   训练期归一化参数
  shared/stations.csv 站点容量与功率
  shared/metadata/   输入说明
  train/windows.parquet
  val/windows.parquet
  test/windows.parquet
  test/test_keys.parquet
```

数据放置完成后，从项目根目录运行 `python check_project.py` 检查数据、权重和前向计算。
