param(
    [string]$Repo = "D:\MoonStack\experiments\moon_rock_stack",
    [string]$Python = "C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

function Start-CmdJob {
    param(
        [string]$Name,
        [string[]]$PythonArgs
    )

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $JobDir = Join-Path $Repo "batch_runs\async_jobs\${Stamp}_cmd_$Name"
    New-Item -ItemType Directory -Force -Path $JobDir | Out-Null
    $RunCmd = Join-Path $JobDir "run.cmd"
    $Stdout = Join-Path $JobDir "stdout.txt"
    $Stderr = Join-Path $JobDir "stderr.txt"
    $StartedAt = Join-Path $JobDir "started_at.txt"
    $FinishedAt = Join-Path $JobDir "finished_at.txt"
    $ExitCode = Join-Path $JobDir "exit_code.txt"

    $Lines = New-Object System.Collections.Generic.List[string]
    $Lines.Add("@echo off")
    $Lines.Add("setlocal")
    $Lines.Add("set `"PYTHONUTF8=1`"")
    $Lines.Add("set `"KMP_DUPLICATE_LIB_OK=TRUE`"")
    $Lines.Add("echo %date% %time% > `"$StartedAt`"")
    $Lines.Add("cd /d `"$Repo`"")
    $CommandLine = "`"$Python`""
    foreach ($Arg in $PythonArgs) {
        $CommandLine += " `"$Arg`""
    }
    $Lines.Add("$CommandLine > `"$Stdout`" 2> `"$Stderr`"")
    $Lines.Add("set `"EC=%ERRORLEVEL%`"")
    $Lines.Add("echo %EC% > `"$ExitCode`"")
    $Lines.Add("echo %date% %time% > `"$FinishedAt`"")
    $Lines.Add("exit /b %EC%")
    Set-Content -LiteralPath $RunCmd -Value $Lines -Encoding ASCII

    $Meta = [ordered]@{
        job_name = $Name
        job_dir = $JobDir
        created_at = (Get-Date).ToString("s")
        launcher = "cmd_start_b"
        run_cmd = $RunCmd
        stdout = $Stdout
        stderr = $Stderr
        status = "launched"
        argv = @($Python) + $PythonArgs
    }
    $Meta | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $JobDir "job.json") -Encoding UTF8
    & C:\Windows\System32\cmd.exe /c start "" /B "$RunCmd"
    Write-Output ($Meta | ConvertTo-Json -Depth 6)
}

$CommonArgs = @(
    "-m", "moon_rock_stack.run_structured_experiment",
    "--rocks", "140",
    "--rock-profile", "high_wall",
    "--clusters", "10",
    "--trials", "2",
    "--targets", "single_face_wall_4course_v1",
    "--gravities", "moon",
    "--candidates", "14",
    "--steps-per-rock", "720",
    "--hold-steps", "3456",
    "--candidate-probe-steps", "120",
    "--workers", "1",
    "--stone-fit-ranker-dir", "batch_runs\20260622_autonomous_wall_flywheel_master_v2_c17_flywheel_3to4_stone_slot_net",
    "--stone-fit-top-k", "22",
    "--stone-fit-ranker-max-course", "3",
    "--candidate-pose-ranker-dir", "batch_runs\20260624_merged_c12_c16_supportmap_drift_guarded_train_20260624_190945",
    "--candidate-pose-top-k", "10",
    "--candidate-pose-ranker-max-course", "3",
    "--pose-risk-ranker-dir", "batch_runs\20260622_poserisk_v18b_recent_3to4_train",
    "--pose-risk-weight", "0.55",
    "--pose-risk-ranker-max-course", "3"
)

Start-CmdJob -Name "eval_merged_driftguard_c17stone_linelock_4course_moon" -PythonArgs (
    $CommonArgs + @(
        "--strategies", "statics_wall_line_lock",
        "--seed", "206270101",
        "--output", "batch_runs\20260625_merged_driftguard_c17stone_linelock_4course_moon_seed206270101"
    )
)

Start-CmdJob -Name "eval_merged_driftguard_c17stone_statics_4course_moon" -PythonArgs (
    $CommonArgs + @(
        "--strategies", "statics_wall",
        "--seed", "206270102",
        "--output", "batch_runs\20260625_merged_driftguard_c17stone_statics_4course_moon_seed206270102"
    )
)
