[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(
        'Isaac-Velocity-Flat-G1-v0',
        'Isaac-Chase-Person-G1-v0',
        'Isaac-Carry-Box-G1-v0',
        'Isaac-Run-Carry-Box-G1-v0'
    )]
    [string]$Task = 'Isaac-Velocity-Flat-G1-v0',

    [ValidateRange(1, 16384)]
    [int]$NumEnvs = 64,

    [ValidateRange(1, 1000000)]
    [int]$MaxIterations = 32,

    [ValidateNotNullOrEmpty()]
    [string]$RunName = 'g1_run_visible_recorded',

    [int]$Seed = 42,

    # The verified Isaac Lab recorder writes at 50 FPS.
    # Keep every clip between 10 and 20 seconds, defaulting to 15 seconds.
    [ValidateRange(500, 1000)]
    [int]$VideoLength = 750,

    [ValidateRange(500, 1000000)]
    [int]$VideoInterval = 1000000,

    [string]$WarmStartCheckpoint,

    [switch]$FromScratch,

    [switch]$Headless,

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

if ($VideoInterval -lt $VideoLength) {
    throw 'VideoInterval must be greater than or equal to VideoLength.'
}

$stepsPerIteration = 24
$availableRecordingSteps = $MaxIterations * $stepsPerIteration
if ($availableRecordingSteps -lt $VideoLength) {
    throw "MaxIterations provides only $availableRecordingSteps recording steps. Increase it to at least $([math]::Ceiling($VideoLength / $stepsPerIteration)) iterations for a complete clip."
}
$lastRecordingStart = [math]::Floor(($availableRecordingSteps - 1) / $VideoInterval) * $VideoInterval
if (($availableRecordingSteps - $lastRecordingStart) -lt $VideoLength) {
    throw (
        "VideoInterval would start an incomplete trailing clip at step $lastRecordingStart. " +
        "Choose an interval whose final trigger leaves at least $VideoLength steps, or set it above " +
        "$($availableRecordingSteps - 1) to record only the clip at step 0."
    )
}

if ($Headless -or ($AdditionalArgs | Where-Object { $_ -match '^--headless(?:=|$)' })) {
    throw 'This launcher records a visible Isaac Sim window and rejects --headless.'
}

$trainArgs = @(
    $trainScript,
    '--task', $Task,
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

$isCarryTask = $Task -in @('Isaac-Carry-Box-G1-v0', 'Isaac-Run-Carry-Box-G1-v0')
$isProjectTask = $Task -eq 'Isaac-Chase-Person-G1-v0' -or $isCarryTask
if ($isProjectTask) {
    $trainArgs += @('--external_callback', 'isaaclab_mcp.runtime_tasks.register_tasks')

    if (-not $FromScratch) {
        if ([string]::IsNullOrWhiteSpace($WarmStartCheckpoint)) {
            if ($Task -eq 'Isaac-Run-Carry-Box-G1-v0') {
                $WarmStartCheckpoint = Get-ChildItem (
                    Join-Path $projectRoot 'logs\rsl_rl\g1_box_carry'
                ) -Recurse -Filter 'model_*.pt' -File -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1 -ExpandProperty FullName
            }
            elseif ($Task -eq 'Isaac-Carry-Box-G1-v0') {
                $WarmStartCheckpoint = Get-ChildItem (
                    Join-Path $projectRoot 'logs\rsl_rl\g1_people_chase'
                ) -Recurse -Filter 'model_*.pt' -File -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1 -ExpandProperty FullName
            }
            if ([string]::IsNullOrWhiteSpace($WarmStartCheckpoint)) {
                $WarmStartCheckpoint = Join-Path $projectRoot (
                    '.pretrained_checkpoints\rsl_rl\Isaac-Velocity-Flat-G1-v0\Assets\Isaac\6.0\Isaac\' +
                    'IsaacLab\PretrainedCheckpoints\rsl_rl\Isaac-Velocity-Flat-G1-v0\checkpoint.pt'
                )
            }
        }
        if (-not (Test-Path -LiteralPath $WarmStartCheckpoint -PathType Leaf)) {
            throw "G1 locomotion warm-start checkpoint does not exist: $WarmStartCheckpoint"
        }

        $experimentName = switch ($Task) {
            'Isaac-Run-Carry-Box-G1-v0' { 'g1_box_carry_run' }
            'Isaac-Carry-Box-G1-v0' { 'g1_box_carry' }
            default { 'g1_people_chase' }
        }
        $bootstrapRunName = switch ($Task) {
            'Isaac-Run-Carry-Box-G1-v0' { '_g1_carry_run_bootstrap' }
            'Isaac-Carry-Box-G1-v0' { '_g1_carry_bootstrap' }
            default { '_g1_flat_bootstrap' }
        }
        $bootstrapDirectory = Join-Path $projectRoot "logs\rsl_rl\$experimentName\$bootstrapRunName"
        $bootstrapCheckpoint = Join-Path $bootstrapDirectory 'checkpoint.pt'
        New-Item -ItemType Directory -Path $bootstrapDirectory -Force | Out-Null
        if (
            -not (Test-Path -LiteralPath $bootstrapCheckpoint -PathType Leaf) -or
            (Get-FileHash -LiteralPath $WarmStartCheckpoint).Hash -ne
            (Get-FileHash -LiteralPath $bootstrapCheckpoint).Hash
        ) {
            Copy-Item -LiteralPath $WarmStartCheckpoint -Destination $bootstrapCheckpoint -Force
        }
        Write-Host "Warm-start checkpoint: $WarmStartCheckpoint"
        $trainArgs += @(
            '--resume',
            '--load_run', $bootstrapRunName,
            '--checkpoint', 'checkpoint\.pt'
        )
    }
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

if ($trainingExitCode -ne 0) {
    exit $trainingExitCode
}

$experimentName = switch ($Task) {
    'Isaac-Chase-Person-G1-v0' { 'g1_people_chase' }
    'Isaac-Carry-Box-G1-v0' { 'g1_box_carry' }
    'Isaac-Run-Carry-Box-G1-v0' { 'g1_box_carry_run' }
    default { 'g1_flat' }
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
