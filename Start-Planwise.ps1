param([switch]$Demo)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$planwiseArgs = @('-m', 'local_agent')
if ($Demo) { $planwiseArgs += '--demo' }
python @planwiseArgs
