param(
    [string]$Executable = "",
    [string]$ListenHost = "127.0.0.1",
    [int]$ListenPort = 8080,
    [string]$UpstreamHost = "127.0.0.1",
    [int]$UpstreamPort = 8000,
    [int]$MaxInflight = 2,
    [int]$MaxQueueSize = 8,
    [int]$WorkerThreads = 1
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$logDir = Join-Path $root "build\local"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $root "build\windows-debug\controller\vllm_slo_proxy.exe"
}

if (!(Test-Path $Executable)) {
    throw "Proxy executable not found: $Executable"
}

$env:VLLM_SLO_LISTEN_HOST = $ListenHost
$env:VLLM_SLO_LISTEN_PORT = "$ListenPort"
$env:VLLM_SLO_UPSTREAM_HOST = $UpstreamHost
$env:VLLM_SLO_UPSTREAM_PORT = "$UpstreamPort"
$env:VLLM_SLO_MAX_INFLIGHT = "$MaxInflight"
$env:VLLM_SLO_MAX_QUEUE_SIZE = "$MaxQueueSize"
$env:VLLM_SLO_WORKER_THREADS = "$WorkerThreads"

$stdoutLog = Join-Path $logDir "proxy.out.log"
$stderrLog = Join-Path $logDir "proxy.err.log"
$process = Start-Process -FilePath $Executable -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
$process.Id | Set-Content -Path (Join-Path $logDir "proxy.pid")
Write-Host "Proxy started: pid=$($process.Id), stdout=$stdoutLog, stderr=$stderrLog"
