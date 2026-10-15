[CmdletBinding()]
param(
    [switch]$Publish,
    [ValidatePattern('^[A-Za-z0-9_.-]+$')][string]$RepositoryName = 'mistralfit'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

if (-not $Publish) {
    Write-Host 'This creates a new PUBLIC GitHub repository in your authenticated gh account.'
    Write-Host 'To publish: powershell -ExecutionPolicy Bypass -File .\scripts\Publish-GitHub.ps1 -Publish'
    exit 0
}
foreach ($program in @('git', 'gh')) {
    if (-not (Get-Command $program -ErrorAction SilentlyContinue)) { throw "$program is required. Install it and reopen PowerShell." }
}

$account = & gh api user --jq '.login'
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($account)) { throw 'Authenticate GitHub CLI first: gh auth login' }
$account = $account.Trim()
if ($account -notmatch '^[A-Za-z0-9-]+$') { throw 'Unexpected authenticated account name.' }
$repo = "$account/$RepositoryName"
Write-Host "Publishing a NEW public repository: $repo" -ForegroundColor Cyan

# Refuse to work inside any existing checkout, including a parent repository.
& git rev-parse --show-toplevel 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { throw 'This folder is already inside a Git checkout. Extract a fresh copy elsewhere before publishing.' }
& gh repo view $repo --json name 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { throw "Repository $repo already exists. Nothing was changed." }

& git init -b main
if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
& git add -- .
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
& git commit -m 'Initial MistralFit experimental release'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed. Configure git user.name/user.email and retry the Git commands manually.' }
& gh repo create $repo --public --source . --remote origin --push --description 'Measured local Mistral configuration search: GGUF inspection, MoE placement candidates, reproducible llama.cpp reports.'
if ($LASTEXITCODE -ne 0) { throw 'GitHub publication failed. The local commit is retained; inspect the error before retrying.' }
& gh repo edit $repo --add-topic mistral --add-topic llama-cpp --add-topic gguf --add-topic local-llm --add-topic benchmarking
if ($LASTEXITCODE -ne 0) { Write-Warning 'Repository was published, but topics could not be set.' }
Write-Host "https://github.com/$repo" -ForegroundColor Green
