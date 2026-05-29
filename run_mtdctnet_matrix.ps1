param(
  [string]$Python = "python",
  [string]$DataRoot = "F:\gen_data_202601\20260405_NN_10",
  [string]$TrainCsv = "DCTNET\outputs\splits\train.csv",
  [string]$ValCsv = "DCTNET\outputs\splits\val.csv",
  [string]$OutputDir = "DCTNET\outputs\runs",
  [string]$Device = "cuda",
  [int]$Epochs = 80,
  [int]$BatchSize = 256,
  [double]$Lr = 0.001,
  [int]$Patience = 15,
  [int]$HitTol = 10,
  [int]$MaxRows = 0
)

$ErrorActionPreference = "Stop"

function Add-OptionalMaxRows {
  param([object[]]$ArgsIn)
  if ($MaxRows -gt 0) {
    return $ArgsIn + @("--max-rows", "$MaxRows")
  }
  return $ArgsIn
}

$common = @(
  "--data-root", $DataRoot,
  "--train-csv", $TrainCsv,
  "--val-csv", $ValCsv,
  "--output-dir", $OutputDir,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--lr", "$Lr",
  "--device", $Device
)

$clsModels = @(
  "MT-DCTNet-IQ",
  "MT-DCTNet-Corr",
  "MT-DCTNet-Dual"
)

foreach ($m in $clsModels) {
  Write-Host "===== Classification: $m ====="
  $baseArgs = @(
    "DCTNET\train_classifier.py",
    "--task", "multiclass6",
    "--model-name", $m
  ) + $common
  $args = Add-OptionalMaxRows -ArgsIn $baseArgs
  & $Python @args
}

Write-Host "===== Sync only: MT-DCTNet ====="
$baseSyncOnlyArgs = @(
  "DCTNET\train_multitask_sync.py",
  "--model-name", "MT-DCTNet",
  "--data-root", $DataRoot,
  "--train-csv", $TrainCsv,
  "--val-csv", $ValCsv,
  "--output-dir", $OutputDir,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--lr", "$Lr",
  "--device", $Device,
  "--lambda-cls", "0",
  "--lambda-loc", "1",
  "--select-by", "loc_mae",
  "--hit-tol", "$HitTol",
  "--early-stop-patience", "$Patience"
)
$syncOnlyArgs = Add-OptionalMaxRows -ArgsIn $baseSyncOnlyArgs
& $Python @syncOnlyArgs

Write-Host "===== Multi-task: MT-DCTNet ====="
$baseMtlArgs = @(
  "DCTNET\train_multitask_sync.py",
  "--model-name", "MT-DCTNet",
  "--data-root", $DataRoot,
  "--train-csv", $TrainCsv,
  "--val-csv", $ValCsv,
  "--output-dir", $OutputDir,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--lr", "$Lr",
  "--device", $Device,
  "--lambda-cls", "1",
  "--lambda-loc", "3",
  "--select-by", "joint",
  "--hit-tol", "$HitTol",
  "--early-stop-patience", "$Patience"
)
$mtlArgs = Add-OptionalMaxRows -ArgsIn $baseMtlArgs
& $Python @mtlArgs
