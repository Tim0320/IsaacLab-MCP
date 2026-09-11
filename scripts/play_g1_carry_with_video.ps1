[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('Isaac-Carry-Box-G1-Play-v0', 'Isaac-Run-Carry-Box-G1-Play-v0')]
    [string]$Task = 'Isaac-Carry-Box-G1-Play-v0',

    [Parameter(Mandatory = $true)]
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

foreach ($requiredPath in @($isaacPython, $playScript, $experience, $Checkpoint)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

if ($Headless -or ($AdditionalArgs | Where-Object { $_ -match '^--headless(?:=|$)' })) {
    throw 'This launcher records a visible Isaac Sim window and rejects --headless.'
}

$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
$playArgs = @(
    $playScript,
    '--task', $Task,
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

Write-Host "Evaluation task: $Task"
Write-Host "Evaluation checkpoint: $Checkpoint"
Write-Host "Isaac Lab video directory: $videoDirectory"
$videos | ForEach-Object { Write-Host "Recorded video: $($_.FullName)" }

exit $playExitCode
