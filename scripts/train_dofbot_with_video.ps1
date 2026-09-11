[CmdletBinding()]
param(
    [ValidateSet('Isaac-Grasp-Cube-Dofbot-v0', 'Isaac-Lift-Cube-Dofbot-v0')]
    [string]$Task = 'Isaac-Grasp-Cube-Dofbot-v0',

    [ValidateRange(1, 16384)]
    [int]$NumEnvs = 32,

    [ValidateRange(1, 1000000)]
    [int]$MaxIterations = 40,

    [ValidateNotNullOrEmpty()]
    [string]$RunName = 'dofbot_visible_recorded',

    [int]$Seed = 42,

    # Isaac Lab's training recorder writes at 50 FPS in the verified runtime.
    # Keep every clip between 10 and 20 seconds, defaulting to 15 seconds.
    [ValidateRange(500, 1000)]
    [int]$VideoLength = 750,

    [ValidateRange(500, 1000000)]
    [int]$VideoInterval = 1000,

    [string]$ResumeRun = '',

    [string]$Checkpoint = '',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArgs
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$isaacPython = 'C:\isaacsim\python.bat'
$trainScript = 'D:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py'
$experience = Join-Path $projectRoot 'apps\isaaclab.dofbot.demo.kit'

foreach ($requiredPath in @($isaacPython, $trainScript, $experience)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

if (($ResumeRun -eq '') -xor ($Checkpoint -eq '')) {
    throw 'ResumeRun and Checkpoint must be provided together.'
}

if ($VideoInterval -lt $VideoLength) {
    throw 'VideoInterval must be greater than or equal to VideoLength.'
}

$stepsPerIteration = 32
$availableRecordingSteps = $MaxIterations * $stepsPerIteration
if ($availableRecordingSteps -lt $VideoLength) {
    throw "MaxIterations provides only $availableRecordingSteps recording steps. Increase it to at least $([math]::Ceiling($VideoLength / $stepsPerIteration)) iterations for a complete clip."
}

if ($AdditionalArgs | Where-Object { $_ -match '^--headless(?:=|$)' }) {
    throw 'This launcher records a visible Isaac Sim window and rejects --headless.'
}

$trainArgs = @(
    $trainScript,
    '--task', $Task,
    '--external_callback', 'isaaclab_mcp.runtime_tasks.register_tasks',
    '--num_envs', $NumEnvs,
    '--max_iterations', $MaxIterations,
    '--run_name', $RunName,
    '--device', 'cuda:0',
    '--seed', $Seed,
    '--video',
    '--video_length', $VideoLength,
    '--video_interval', $VideoInterval,
    '--viz', 'kit',
    '--experience', $experience,
    '--kit_args', '--/renderer/multiGpu/enabled=false --/physics/cudaDevice=0'
)

if ($ResumeRun -ne '') {
    $trainArgs += @('--resume', '--load_run', $ResumeRun, '--checkpoint', $Checkpoint)
}

if ($AdditionalArgs) {
    $trainArgs += $AdditionalArgs
}

$startedAt = Get-Date
Push-Location $projectRoot
try {
    & $isaacPython @trainArgs
    $trainingExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

$experimentName = if ($Task -eq 'Isaac-Grasp-Cube-Dofbot-v0') {
    'dofbot_cube_grasp_pretrain'
}
else {
    'dofbot_cube_lift'
}
$logRoot = Join-Path $projectRoot "logs\rsl_rl\$experimentName"
$runDirectory = Get-ChildItem -LiteralPath $logRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -like "*_$RunName" -and $_.LastWriteTime -ge $startedAt.AddMinutes(-1)
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -ne $runDirectory) {
    $videoDirectory = Join-Path $runDirectory.FullName 'videos\train'
    $videos = @(Get-ChildItem -LiteralPath $videoDirectory -Filter '*.mp4' -File -ErrorAction SilentlyContinue)
    Write-Host "Training run: $($runDirectory.FullName)"
    Write-Host "Isaac Lab video directory: $videoDirectory"
    if ($videos.Count -gt 0) {
        $videos | ForEach-Object { Write-Host "Recorded video: $($_.FullName)" }
    }
    elseif ($trainingExitCode -eq 0) {
        throw "Training completed but Isaac Lab did not produce an MP4 in: $videoDirectory"
    }
}
elseif ($trainingExitCode -eq 0) {
    throw "Training completed but its run directory could not be resolved under: $logRoot"
}

exit $trainingExitCode
