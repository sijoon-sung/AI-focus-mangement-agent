param([switch]$Demo)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$planwiseArgs = @('-m', 'local_agent')
if ($Demo) { $planwiseArgs += '--demo' }
$planwisePython = Join-Path $PSScriptRoot '.venv-research\Scripts\python.exe'
if (Test-Path -LiteralPath $planwisePython) { & $planwisePython @planwiseArgs }
else { python @planwiseArgs }
