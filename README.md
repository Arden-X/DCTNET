# DCTNET

MT-DCTNet experiments for underwater acoustic preamble classification and synchronization.

## Files

- `models_cao_ch3.py`: model definitions, including `MT-DCTNet`.
- `dataset_cao_ch3.py`: MAT-file dataset loader.
- `train_classifier.py`: classification training for `MT-DCTNet-IQ`, `MT-DCTNet-Corr`, and `MT-DCTNet-Dual`.
- `train_multitask_sync.py`: synchronization-only and multi-task training for `MT-DCTNet`.
- `split_cao_ch3.py`: regenerate `train.csv`, `val.csv`, `test.csv` from `index.xlsx`.
- `run_mtdctnet_matrix.ps1`: Windows PowerShell batch runner.
- `run_mtdctnet_matrix.sh`: Linux/macOS shell batch runner.
- `outputs/splits/train.csv`, `outputs/splits/val.csv`: train/validation split metadata.

## Expected Data

The MAT files are not included. Set `DataRoot` / `DATA_ROOT` to the folder containing the files referenced by `outputs/splits/*.csv`.

Required MAT variables:

- `I_obs`
- `Q_obs`
- `A_corr`

To regenerate the split files:

```bash
python DCTNET/split_cao_ch3.py \
  --index-path /path/to/20260405_NN_10/index.xlsx \
  --output-dir DCTNET/outputs/splits
```

The default split is grouped by `h_id`: for each `case`, 15 channel IDs are assigned to train, 5 to validation, and 5 to test. This prevents windows from the same channel ID appearing in different splits.

## Windows

```powershell
.\DCTNET\run_mtdctnet_matrix.ps1 `
  -Python "D:\Tools\conda\envs\Test\python.exe" `
  -DataRoot "F:\gen_data_202601\20260405_NN_10" `
  -Epochs 80 `
  -BatchSize 256 `
  -Device cuda
```

Small smoke test:

```powershell
.\DCTNET\run_mtdctnet_matrix.ps1 `
  -Python "D:\Tools\conda\envs\Test\python.exe" `
  -DataRoot "F:\gen_data_202601\20260405_NN_10" `
  -Epochs 2 `
  -BatchSize 64 `
  -MaxRows 512
```

## Linux

```bash
chmod +x DCTNET/run_mtdctnet_matrix.sh

DATA_ROOT=/path/to/20260405_NN_10 \
PYTHON=/path/to/python \
EPOCHS=80 \
BATCH_SIZE=256 \
DEVICE=cuda \
./DCTNET/run_mtdctnet_matrix.sh
```

## Experiment Matrix

- `MT-DCTNet-IQ`: IQ-only classification.
- `MT-DCTNet-Corr`: correlation-only classification.
- `MT-DCTNet-Dual`: IQ + correlation classification.
- `MT-DCTNet` with `lambda_cls=0`: synchronization-only.
- `MT-DCTNet` with `lambda_cls=1`, `lambda_loc=3`: multi-task classification + synchronization.
