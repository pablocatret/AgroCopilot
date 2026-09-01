$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$output = Join-Path $root "evaluation\results\benchmark_final"
$logDir = Join-Path $output "overnight_logs"
$null = New-Item -ItemType Directory -Force -Path $logDir
$masterLog = Join-Path $logDir ("overnight_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$env:EVALUATION_ENABLE_LLM = "1"

function Invoke-EvaluationStep {
    param(
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeLabel = $Label -replace "[^A-Za-z0-9_.-]", "_"
    $stdout = Join-Path $logDir ("{0}_{1}.out.log" -f $stamp, $safeLabel)
    $stderr = Join-Path $logDir ("{0}_{1}.err.log" -f $stamp, $safeLabel)
    Add-Content -Path $masterLog -Value ("[{0}] START {1} {2}" -f (Get-Date), $Label, ($Arguments -join " "))

    $process = Start-Process -FilePath $python -WorkingDirectory $root `
        -ArgumentList (@("-m", "evaluation.cli") + $Arguments) `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -WindowStyle Hidden -Wait -PassThru

    Get-Content $stdout, $stderr -ErrorAction SilentlyContinue | Add-Content -Path $masterLog
    Add-Content -Path $masterLog -Value ("[{0}] END {1} exit={2}" -f (Get-Date), $Label, $process.ExitCode)
    if ($process.ExitCode -ne 0) {
        throw "El paso '$Label' terminó con código $($process.ExitCode). Revisar $stdout y $stderr"
    }
}

try {
    Add-Content -Path $masterLog -Value ("[{0}] Final benchmark started" -f (Get-Date))

    # Use stable, resumable ranges. Do not retain batch identifiers from a
    # previous checkout: compare creates a new batch and remains resumable.
    foreach ($range in @("1:5", "6:12", "13:20", "21:30", "31:40", "41:50", "51:57")) {
        Invoke-EvaluationStep ("compare_{0}" -f $range.Replace(":", "_")) @(
            "-v", "compare",
            "--config", "evaluation/configs/benchmark_final.json",
            "--case-range", $range,
            "--output", "evaluation/results/benchmark_final",
            "--max-concurrent", "2"
        )
    }

    Add-Content -Path $masterLog -Value ("[{0}] Final benchmark completed successfully" -f (Get-Date))
}
catch {
    Add-Content -Path $masterLog -Value ("[{0}] STOP ERROR: {1}" -f (Get-Date), $_.Exception.Message)
    Write-Error $_.Exception.Message
    exit 1
}
