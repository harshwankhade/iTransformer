# Train + Quantize script
# Usage: .\train_and_quantize.ps1

Write-Host "Starting Training + Quantization Pipeline..." -ForegroundColor Green
Write-Host ""

# Run with default ETTh1 dataset
conda run -n itransformer python run_with_quantization.py `
    --data ETTh1 `
    --root_path ./dataset/ETT-small/ `
    --data_path ETTh1.csv `
    --batch_size 32 `
    --train_epochs 10 `
    --learning_rate 0.0001 `
    --use_gpu 1 `
    --apply_quantization 1 `
    --quantization_type dynamic `
    --profile_iterations 10

Write-Host ""
Write-Host "Training + Quantization Pipeline Complete!" -ForegroundColor Green
