param(
    [string]$Session = "20260622_remote_1080ti_wall_worker",
    [int]$Cycles = 96,
    [int]$PollSeconds = 900,
    [int]$Rocks = 96,
    [int]$Trials = 1,
    [int]$Candidates = 6,
    [int]$CandidateProbeSteps = 50,
    [int]$StepsPerRock = 240,
    [int]$HoldSteps = 900,
    [string]$Targets = "single_face_wall_3course_v1,single_face_wall_4course_v1",
    [string]$Gravities = "moon",
    [string]$RockProfile = "high_wall",
    [double]$ReleaseSearchStepM = 0.004,
    [double]$ReleaseExtraClearanceM = 0.003,
    [double]$BaseSupportPriorWeight = 1.0,
    [double]$BaseContinuityPriorWeight = 0.35
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$BatchRoot = Join-Path $Repo "batch_runs"
$LocalSessionDir = Join-Path $BatchRoot ("remote_1080ti_scheduler\" + $Session)
New-Item -ItemType Directory -Force -Path $LocalSessionDir | Out-Null

$LogPath = Join-Path $LocalSessionDir "scheduler.log"
$SshExe = "C:\Program Files\Git\usr\bin\ssh.exe"
$ScpExe = "C:\Program Files\Git\usr\bin\scp.exe"
$KeyPath = "D:/MoonStack/deploy/desktop_m57fdie/moonstack_desktop_m57fdie_ed25519_all"
$KnownHosts = "D:/MoonStack/deploy/desktop_m57fdie/known_hosts_git"
$RemoteHost = "all@desktop-m57fdie.tail83f520.ts.net"
$RemoteIncoming = "D:/ResearchManagerData/incoming"
$RemoteRunRoot = "D:/ResearchManagerData/runs/$Session"
$RemoteScriptName = "run_$Session.ps1"
$RemoteScriptPath = "$RemoteIncoming/$RemoteScriptName"

function LogLine {
    param([string]$Text)
    $Line = "[$((Get-Date).ToString('s'))] $Text"
    Write-Host $Line
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

function Invoke-Remote {
    param(
        [string]$Command,
        [int]$TimeoutSeconds = 60
    )
    $OutFile = Join-Path $LocalSessionDir ("ssh_out_" + [guid]::NewGuid().ToString("N") + ".txt")
    $ErrFile = Join-Path $LocalSessionDir ("ssh_err_" + [guid]::NewGuid().ToString("N") + ".txt")
    $Args = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-i", $KeyPath,
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=$KnownHosts",
        $RemoteHost,
        $Command
    )
    $Process = Start-Process -FilePath $SshExe -ArgumentList $Args -NoNewWindow -PassThru -Wait -RedirectStandardOutput $OutFile -RedirectStandardError $ErrFile
    $Stdout = if (Test-Path -LiteralPath $OutFile) { Get-Content -LiteralPath $OutFile -Raw -ErrorAction SilentlyContinue } else { "" }
    $Stderr = if (Test-Path -LiteralPath $ErrFile) { Get-Content -LiteralPath $ErrFile -Raw -ErrorAction SilentlyContinue } else { "" }
    return [PSCustomObject]@{
        ExitCode = $Process.ExitCode
        Stdout = $Stdout
        Stderr = $Stderr
    }
}

function Upload-RemoteWorker {
    param([string]$LocalWorkerPath)
    $Args = @(
        "-i", $KeyPath,
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=$KnownHosts",
        $LocalWorkerPath,
        "$RemoteHost`:$RemoteScriptPath"
    )
    & $ScpExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "scp upload failed with exit code $LASTEXITCODE"
    }
}

function Download-RemoteResult {
    $DownloadRoot = Join-Path $LocalSessionDir "downloaded_remote_run"
    New-Item -ItemType Directory -Force -Path $DownloadRoot | Out-Null
    $Args = @(
        "-r",
        "-i", $KeyPath,
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=$KnownHosts",
        "$RemoteHost`:$RemoteRunRoot",
        $DownloadRoot
    )
    & $ScpExe @Args
    if ($LASTEXITCODE -ne 0) {
        LogLine "scp_download_failed exit=$LASTEXITCODE"
    } else {
        LogLine "remote_result_downloaded local=$DownloadRoot"
    }
}

function Write-RemoteWorker {
    $LocalWorkerPath = Join-Path $LocalSessionDir $RemoteScriptName
    $Worker = @"
`$ErrorActionPreference = "Stop"
`$Session = "$Session"
`$RunRoot = "$RemoteRunRoot"
`$OutDir = "`$RunRoot/structured_3to4_moon"
`$Py = "D:/ResearchManagerData/conda_envs/moon-rock-stack/python.exe"
`$PreferredCode = "D:/ResearchManager/Codex_Stack"
`$LegacyCode = "D:/ResearchManager/moon_rock_stack"
`$GitExe = "git"

New-Item -ItemType Directory -Force -Path `$RunRoot | Out-Null
"started `$((Get-Date).ToString('s'))" | Set-Content -LiteralPath "`$RunRoot/started_at.txt" -Encoding UTF8
Start-Transcript -Path "`$RunRoot/transcript.txt" -Append | Out-Null

try {
    `$CodeRoot = `$LegacyCode
    try {
        if (Test-Path "`$PreferredCode/.git") {
            & `$GitExe -C `$PreferredCode pull --ff-only
            if (`$LASTEXITCODE -eq 0) { `$CodeRoot = `$PreferredCode }
        } elseif (-not (Test-Path `$PreferredCode)) {
            & `$GitExe clone https://github.com/George3215/Codex_Stack.git `$PreferredCode
            if (`$LASTEXITCODE -eq 0) { `$CodeRoot = `$PreferredCode }
        }
    } catch {
        "git_update_failed `$(`$_.Exception.Message)" | Add-Content -LiteralPath "`$RunRoot/events.log" -Encoding UTF8
        `$CodeRoot = `$LegacyCode
    }

    `$env:PYTHONPATH = `$CodeRoot
    `$env:PYTHONUTF8 = "1"
    `$env:KMP_DUPLICATE_LIB_OK = "TRUE"
    Set-Location -LiteralPath `$CodeRoot

    & nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv | Tee-Object -FilePath "`$RunRoot/gpu_before.csv"
    & `$Py -c "import numpy as np, torch, mujoco, moon_rock_stack; print('remote_probe', np.__version__, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, mujoco.__version__); print('polyfit', np.polyfit([0,1,2],[1,3,5],1))" | Tee-Object -FilePath "`$RunRoot/python_probe.txt"

    & `$Py -m moon_rock_stack.run_structured_experiment `
        --rocks $Rocks `
        --rock-profile $RockProfile `
        --clusters 10 `
        --trials $Trials `
        --targets $Targets `
        --strategies statics_wall `
        --gravities $Gravities `
        --candidates $Candidates `
        --steps-per-rock $StepsPerRock `
        --hold-steps $HoldSteps `
        --candidate-probe-steps $CandidateProbeSteps `
        --workers 1 `
        --seed 306220108 `
        --output `$OutDir `
        --low-release-search `
        --release-search-step-m $ReleaseSearchStepM `
        --release-extra-clearance-m $ReleaseExtraClearanceM `
        --base-support-prior `
        --base-support-prior-weight $BaseSupportPriorWeight `
        --base-continuity-prior `
        --base-continuity-prior-weight $BaseContinuityPriorWeight
    `$ExitCode = `$LASTEXITCODE
    & nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv | Tee-Object -FilePath "`$RunRoot/gpu_after.csv"
    "`$ExitCode" | Set-Content -LiteralPath "`$RunRoot/exit_code.txt" -Encoding UTF8
    "finished `$((Get-Date).ToString('s'))" | Set-Content -LiteralPath "`$RunRoot/finished_at.txt" -Encoding UTF8
    Stop-Transcript | Out-Null
    exit `$ExitCode
} catch {
    "failed `$((Get-Date).ToString('s')) `$(`$_.Exception.Message)" | Set-Content -LiteralPath "`$RunRoot/failed_at.txt" -Encoding UTF8
    "1" | Set-Content -LiteralPath "`$RunRoot/exit_code.txt" -Encoding UTF8
    Stop-Transcript | Out-Null
    exit 1
}
"@
    Set-Content -LiteralPath $LocalWorkerPath -Value $Worker -Encoding UTF8
    return $LocalWorkerPath
}

LogLine "scheduler_start session=$Session cycles=$Cycles poll_seconds=$PollSeconds no_delete=true"
$LaunchedMarker = Join-Path $LocalSessionDir "launched_at.txt"

for ($Cycle = 0; $Cycle -lt $Cycles; $Cycle++) {
    LogLine "cycle=$Cycle begin"
    $Probe = Invoke-Remote -Command "whoami" -TimeoutSeconds 30
    if ($Probe.ExitCode -ne 0) {
        LogLine "remote_unreachable exit=$($Probe.ExitCode) stderr=$($Probe.Stderr.Trim())"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    LogLine "remote_reachable whoami=$($Probe.Stdout.Trim())"
    $Gpu = Invoke-Remote -Command "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv" -TimeoutSeconds 60
    LogLine "gpu_status stdout=$($Gpu.Stdout.Trim()) stderr=$($Gpu.Stderr.Trim())"

    $RemoteStarted = Invoke-Remote -Command "powershell -NoProfile -Command `"if (Test-Path '$RemoteRunRoot/started_at.txt') { 'started' } else { 'not_started' }`"" -TimeoutSeconds 60
    if ($RemoteStarted.Stdout -match "started") {
        LogLine "remote_job_already_started run_root=$RemoteRunRoot"
    } elseif (-not (Test-Path -LiteralPath $LaunchedMarker)) {
        $WorkerPath = Write-RemoteWorker
        Upload-RemoteWorker -LocalWorkerPath $WorkerPath
        LogLine "worker_uploaded remote_script=$RemoteScriptPath"
        $Launch = Invoke-Remote -Command "cmd /c start `"`" /B powershell -NoProfile -ExecutionPolicy Bypass -File $RemoteScriptPath" -TimeoutSeconds 60
        if ($Launch.ExitCode -eq 0) {
            "launched $((Get-Date).ToString('s'))" | Set-Content -LiteralPath $LaunchedMarker -Encoding UTF8
            LogLine "remote_worker_launched run_root=$RemoteRunRoot"
        } else {
            LogLine "remote_launch_failed exit=$($Launch.ExitCode) stderr=$($Launch.Stderr.Trim())"
        }
    } else {
        LogLine "local_launch_marker_present"
    }

    $RemoteFinished = Invoke-Remote -Command "powershell -NoProfile -Command `"if (Test-Path '$RemoteRunRoot/finished_at.txt') { Get-Content '$RemoteRunRoot/finished_at.txt'; if (Test-Path '$RemoteRunRoot/exit_code.txt') { Get-Content '$RemoteRunRoot/exit_code.txt' } } else { 'not_finished' }`"" -TimeoutSeconds 60
    LogLine "remote_finish_status stdout=$($RemoteFinished.Stdout.Trim()) stderr=$($RemoteFinished.Stderr.Trim())"
    if ($RemoteFinished.Stdout -notmatch "not_finished" -and $RemoteFinished.Stdout.Trim()) {
        Download-RemoteResult
        break
    }

    Start-Sleep -Seconds $PollSeconds
}

LogLine "scheduler_exit"
