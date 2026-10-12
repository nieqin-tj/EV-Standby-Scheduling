# Model input data

The complete dataset is distributed separately and is not included in this GitHub repository. The external download link will be added when the data deposit is available.

Place the dataset in this directory with the following layout before training or evaluation:

```text
DATA/
  shared/
    features/<station_id>.npy
    scaler.npz
    stations.csv
    metadata/schema.json
  train/windows.parquet
  val/windows.parquet
  test/windows.parquet
```

The feature arrays contain the complete 20-minute model inputs for 313 stations. `windows.parquet` files define the archived split membership; validation membership also records selection roles. Keep the archived scaler and station metadata with the corresponding feature arrays.
