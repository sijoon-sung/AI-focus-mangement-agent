param(
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [ValidateRange(1,1000)][int]$Epochs = 15,
    [ValidateRange(1,100000)][int]$PerPerson = 1000,
    [ValidateRange(1,256)][int]$BatchSize = 16,
    [string]$OutputDirectory = 'runs/local-gpu'
)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $python = Join-Path (Get-Location) '.venv-research/Scripts/python.exe'
    if (-not (Test-Path $python)) { throw 'Create .venv-research and install requirements-specialists.txt first.' }
    $env:HF_HOME = Join-Path (Get-Location) 'runs/hf-cache'
    $env:TORCH_HOME = Join-Path (Get-Location) 'runs/torch-cache'
    $env:PYTHONIOENCODING = 'utf-8'
    # Use already downloaded public data. No webcam or desktop capture.
    foreach ($path in @('data/public/mpiigaze/source.json','data/public/groundcua/manifest.jsonl','data/public/mind2web/tasks.jsonl')) {
        if (-not (Test-Path $path)) { throw "Missing $path. Download public data as described in docs/PUBLIC_SPECIALISTS_KO.md." }
    }
    New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
    foreach ($role in @('camera_gaze','screen_layout','goal_history')) {
        Write-Host "Training $role on $Device"
        # Windows PowerShell treats native stderr warnings as ErrorRecords.
        # Preserve the log, and use the process exit code to detect failures.
        $ErrorActionPreference = 'Continue'
        & $python -u -m research.train_specialists --role $role --device $Device --epochs $Epochs --per-person $PerPerson --batch-size $BatchSize --pilot --output "$OutputDirectory/$role.pt" 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath "$OutputDirectory/$role.log"
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -ne 0) { throw "$role failed; see $OutputDirectory/$role.log" }
    }
    & $python -m research.verify_public_models --models $OutputDirectory --output "$OutputDirectory/inference.json"
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint inference verification failed.' }
    Write-Host "Finished. Check evaluation reports in $OutputDirectory before connecting models in the app."
} finally { Pop-Location }
