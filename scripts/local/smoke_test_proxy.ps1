param(
    [string]$ProxyBaseUrl = "http://127.0.0.1:8080",
    [switch]$CheckOverload429
)

$ErrorActionPreference = "Stop"

Write-Host "Checking proxy /health"
$health = Invoke-WebRequest -UseBasicParsing -Uri "$ProxyBaseUrl/health" -Method GET
if ($health.StatusCode -ne 200) {
    throw "health failed with status $($health.StatusCode)"
}

Write-Host "Checking streaming chat completion through proxy"
$body = '{"model":"mock","stream":true,"messages":[{"role":"user","content":"hello"}]}'
$response = Invoke-WebRequest -UseBasicParsing -Uri "$ProxyBaseUrl/v1/chat/completions" -Method POST -ContentType "application/json" -Body $body
$content = $response.Content
if ($content -notmatch 'Hello' -or $content -notmatch '\[DONE\]') {
    throw "stream response did not contain expected SSE events"
}
$eventCount = ([regex]::Matches($content, '(?m)^data:')).Count
if ($eventCount -lt 4) {
    throw "expected at least 4 SSE data events, got $eventCount"
}

if ($CheckOverload429) {
    Write-Host "Checking overload 429. Start proxy with MaxInflight=1 MaxQueueSize=0 for this check."
    $job1 = Start-Job -ScriptBlock {
        param($Url, $Body)
        Invoke-WebRequest -UseBasicParsing -Uri "$Url/v1/chat/completions" -Method POST -ContentType "application/json" -Body $Body | Out-Null
    } -ArgumentList $ProxyBaseUrl, $body

    Start-Sleep -Milliseconds 50
    $statusCode = $null
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$ProxyBaseUrl/v1/chat/completions" -Method POST -ContentType "application/json" -Body $body | Out-Null
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
    } finally {
        Wait-Job $job1 -Timeout 10 | Out-Null
        Remove-Job $job1 -Force -ErrorAction SilentlyContinue
    }

    if ($statusCode -ne 429) {
        throw "expected overload HTTP 429, got $statusCode"
    }
}

Write-Host "Smoke test passed"
