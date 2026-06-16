$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$logDir = Join-Path $root "build\local"

foreach ($name in @("proxy", "mock_vllm")) {
    $pidFile = Join-Path $logDir "$name.pid"
    if (Test-Path $pidFile) {
        $processId = Get-Content $pidFile | Select-Object -First 1
        if ($processId) {
            Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped $name pid=$processId"
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}
