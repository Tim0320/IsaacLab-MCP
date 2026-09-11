[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Checkpoint,

    [ValidateRange(500, 1000)]
    [int]$VideoLength = 750,

    [switch]$Headless,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArgs
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$isaacPython = 'C:\isaacsim\python.bat'
$playScript = 'D:\IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py'
$experience = Join-Path $projectRoot 'apps\isaaclab.dofbot.demo.kit'
$logRoot = Join-Path $projectRoot 'logs\rsl_rl\g1_people_chase'

foreach ($requiredPath in @($isaacPython, $playScript, $experience)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

if ($Headless -or ($AdditionalArgs | Where-Object { $_ -match '^--headless(?:=|$)' })) {
    throw 'This launcher records a visible Isaac Sim window and rejects --headless.'
}

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Get-ChildItem -LiteralPath $logRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike '_*' } |
        ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Filter 'model_*.pt' -File } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}

if ([string]::IsNullOrWhiteSpace($Checkpoint) -or -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "G1 chase checkpoint does not exist: $Checkpoint"
}
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path

$playArgs = @(
    $playScript,
    '--task', 'Isaac-Chase-Person-G1-Play-v0',
    '--num_envs', 1,
    '--checkpoint', $Checkpoint,
    '--device', 'cuda:0',
    '--video',
    '--video_length', $VideoLength,
    '--real-time',
    '--viz', 'kit',
    '--experience', $experience,
    '--external_callback', 'isaaclab_mcp.runtime_tasks.register_tasks',
    '--kit_args', '--/renderer/multiGpu/enabled=false --/physics/cudaDevice=0'
)

if ($AdditionalArgs) {
    $playArgs += $AdditionalArgs
}

Push-Location $projectRoot
try {
    & $isaacPython @playArgs
    $playExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($playExitCode -ne 0) {
    exit $playExitCode
}

$videoDirectory = Join-Path (Split-Path -Parent $Checkpoint) 'videos\play'
$videos = @(Get-ChildItem -LiteralPath $videoDirectory -Filter '*.mp4' -File -ErrorAction SilentlyContinue)
if ($videos.Count -eq 0) {
    throw "Evaluation completed but Isaac Lab did not produce an MP4 in: $videoDirectory"
}

Write-Host "Evaluation checkpoint: $Checkpoint"
Write-Host "Isaac Lab video directory: $videoDirectory"
$videos | ForEach-Object { Write-Host "Recorded video: $($_.FullName)" }

exit $playExitCode
