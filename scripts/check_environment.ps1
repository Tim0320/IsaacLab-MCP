[CmdletBinding()]
param(
    [string]$IsaacLabPath = $(if ($env:ISAACLAB_PATH) { $env:ISAACLAB_PATH } else { 'D:\IsaacLab' })
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($IsaacLabPath)
$launcher = Join-Path $root 'isaaclab.bat'
$versionFile = Join-Path $root 'VERSION'
$sourceRoot = Join-Path $root 'source'
$simLink = Join-Path $root '_isaac_sim'
$runtimePython = Join-Path $simLink 'python.bat'

$result = [ordered]@{
    isaaclab_path = $root
    isaaclab_exists = Test-Path -LiteralPath $root -PathType Container
    version = if (Test-Path -LiteralPath $versionFile -PathType Leaf) { (Get-Content -LiteralPath $versionFile -Raw).Trim() } else { $null }
    launcher_exists = Test-Path -LiteralPath $launcher -PathType Leaf
    source_exists = Test-Path -LiteralPath $sourceRoot -PathType Container
    isaac_sim_link_exists = Test-Path -LiteralPath $simLink
    runtime_python_exists = Test-Path -LiteralPath $runtimePython -PathType Leaf
}

$result | ConvertTo-Json

if (-not ($result.isaaclab_exists -and $result.launcher_exists -and $result.source_exists -and $result.runtime_python_exists)) {
    exit 1
}
