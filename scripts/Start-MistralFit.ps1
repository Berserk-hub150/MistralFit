[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonCommand = $null
$pythonPrefix = @()

foreach ($candidate in @('py', 'python', 'python3')) {
    $found = Get-Command $candidate -ErrorAction SilentlyContinue
    if (-not $found) { continue }
    $prefix = @()
    if ($candidate -eq 'py') { $prefix = @('-3') }
    try {
        & $found.Source @prefix -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = $found.Source
            $pythonPrefix = $prefix
            break
        }
    } catch { continue }
}

if (-not $pythonCommand) {
    Write-Host 'Python 3.10 or newer was not found. Install Python, reopen PowerShell, then rerun this script.' -ForegroundColor Yellow
    Write-Host 'winget install -e --id Python.Python.3.13 --accept-source-agreements --accept-package-agreements'
    exit 1
}

Write-Host 'MistralFit: inspecting this computer...' -ForegroundColor Cyan
& $pythonCommand @pythonPrefix -m mistralfit doctor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ''
& $pythonCommand @pythonPrefix -m mistralfit demo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ''
Write-Host 'Read LEGGIMI.md for a real GGUF experiment. The demo above is an illustrative plan, not inference.' -ForegroundColor Yellow
