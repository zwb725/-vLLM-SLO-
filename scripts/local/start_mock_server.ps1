param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [double]$FirstTokenDelaySeconds = 0.05,
    [double]$TokenDelaySeconds = 0.05,
    [string]$Mode = "ok",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$logDir = Join-Path $root "build\local"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:MOCK_VLLM_FIRST_TOKEN_DELAY_SECONDS = "$FirstTokenDelaySeconds"
$env:MOCK_VLLM_TOKEN_DELAY_SECONDS = "$TokenDelaySeconds"
$env:MOCK_VLLM_MODE = $Mode

$script = Join-Path $root "tools\mock_vllm\mock_vllm_server.py"
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if ($env:PYTHON_EXE) {
        $PythonExe = $env:PYTHON_EXE
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExe = "python"
    } else {
        $codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
        if (Test-Path $codexPython) {
            $PythonExe = $codexPython
        } else {
            throw "Python executable not found. Set -PythonExe or PYTHON_EXE."
        }
    }
}

$stdoutLog = Join-Path $logDir "mock_vllm.out.log"
$stderrLog = Join-Path $logDir "mock_vllm.err.log"
$process = Start-Process -FilePath $PythonExe -ArgumentList @($script, "--host", $HostAddress, "--port", "$Port") -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
$process.Id | Set-Content -Path (Join-Path $logDir "mock_vllm.pid")
Write-Host "Mock vLLM server started: pid=$($process.Id), stdout=$stdoutLog, stderr=$stderrLog"
