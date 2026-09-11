[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServerArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$installedCli = Join-Path $repoRoot '.venv\Scripts\isaaclab-mcp-server.exe'

if (-not $env:ISAACLAB_PATH) {
    $env:ISAACLAB_PATH = 'D:\IsaacLab'
}

if (Test-Path -LiteralPath $installedCli) {
    & $installedCli @ServerArgs
    exit $LASTEXITCODE
}

if (Test-Path -LiteralPath $python) {
    Push-Location $repoRoot
    try {
        & $python -m isaaclab_mcp.server @ServerArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

throw 'isaaclab-mcp-server was not found. Run: uv sync --dev'
