param(
    [string]$Python = "C:\Users\all\miniconda3\envs\moon-rock-stack\python.exe",
    [string]$Repo = "D:\MoonStack\experiments\moon_rock_stack",
    [string]$BatchRoot = "D:\MoonStack\experiments\moon_rock_stack\batch_runs",
    [int]$MaxGroups = 2200,
    [int]$Epochs = 110
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $Repo
$Session = "20260624_merged_c12_c16_supportmap_drift_guarded"
$RunRoot = Join-Path $BatchRoot $Session
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$LogPath = Join-Path $RunRoot "pipeline.log"

function LogLine {
    param([string]$Text)
    $Line = "[$((Get-Date).ToString('s'))] $Text"
    Write-Host $Line
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

function Invoke-PythonPhase {
    param(
        [string]$Phase,
        [string[]]$PythonArgs
    )
    LogLine "phase_start name=$Phase"
    LogLine "phase_args name=$Phase args=$($PythonArgs -join ' ')"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Output = & $Python @PythonArgs 2>&1
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    foreach ($Line in $Output) {
        LogLine "$Phase $($Line.ToString())"
    }
    if ($ExitCode -ne 0) {
        throw "phase_failed name=$Phase exit_code=$ExitCode"
    }
    $PathLine = $Output | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ -match '^[A-Za-z]:\\' } | Select-Object -Last 1
    if (-not $PathLine) {
        throw "phase_missing_output_path name=$Phase"
    }
    LogLine "phase_done name=$Phase output=$PathLine"
    return $PathLine
}

$Runs = @(
    "20260619_closed_loop_wall4_baseline_full6_v1",
    "20260619_closed_loop_wall4_neural_presim_top3_v1",
    "20260620_highcourse_4course_presim_pose_top3_seed97001_v1",
    "20260620_highcourse_4course_expanded_quality_top3_seed97002_v1",
    "20260620_highcourse_4course_mujoco_depth_structure_proxyonline_top3_seed97007_v1",
    "20260620_highcourse_4course_stoneslot_structure_top3_seed97008_v1",
    "20260620_highcourse_4course_stoneslot_structure_strictline_seed97009_v2",
    "20260620_highcourse_4course_stoneslot_structure_physicsgate_seed97011_v4",
    "20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_collect_exploit_00_seed206252708",
    "20260622_autonomous_wall_flywheel_master_v2_c12_flywheel_3to4_closed_loop_eval",
    "20260622_autonomous_wall_flywheel_master_v2_c12_strict_4course",
    "20260622_autonomous_wall_flywheel_master_v2_c13_flywheel_3to4_collect_exploit_00_seed206253717",
    "20260622_autonomous_wall_flywheel_master_v2_c13_flywheel_3to4_closed_loop_eval",
    "20260622_autonomous_wall_flywheel_master_v2_c13_strict_4course",
    "20260622_autonomous_wall_flywheel_master_v2_c14_flywheel_3to4_collect_exploit_00_seed206254726",
    "20260622_autonomous_wall_flywheel_master_v2_c14_flywheel_3to4_closed_loop_eval",
    "20260622_autonomous_wall_flywheel_master_v2_c14_strict_4course",
    "20260622_autonomous_wall_flywheel_master_v2_c15_flywheel_3to4_collect_exploit_00_seed206255735",
    "20260622_autonomous_wall_flywheel_master_v2_c15_flywheel_3to4_closed_loop_eval",
    "20260622_autonomous_wall_flywheel_master_v2_c15_strict_4course",
    "20260622_autonomous_wall_flywheel_master_v2_c16_flywheel_3to4_collect_exploit_00_seed206256744",
    "20260622_autonomous_wall_flywheel_master_v2_c16_strict_4course",
    "20260623_c12_latest_models_4course_parallel_eval_seed206258901",
    "20260623_v19_baseline_4course_parallel_eval_seed206258901"
)

LogLine "pipeline_start session=$Session"
LogLine "purpose=merge_c12_to_c16_positive_negative_examples_for_drift_guarded_support_map"
LogLine "input_policy=geometry_and_observation_features_only exclude_postsim_features=true"
LogLine "delete_policy=no_delete_append_only"
LogLine "run_count_requested=$($Runs.Count)"

$BuildArgs = @(
    "-m", "scripts.build_learning_dataset",
    "--batch-root", $BatchRoot,
    "--output", (Join-Path $BatchRoot "20260624_merged_c12_c16_wall_dataset")
)
foreach ($RunName in $Runs) {
    $BuildArgs += @("--run", $RunName)
}
$DatasetDir = Invoke-PythonPhase -Phase "build_learning_dataset" -PythonArgs $BuildArgs

$ExportArgs = @(
    "-m", "scripts.export_mujoco_depth_observation_maps",
    "--dataset", $DatasetDir,
    "--output", (Join-Path $BatchRoot "20260624_merged_c12_c16_mujoco_depth_maps"),
    "--source", "candidate",
    "--grid-size", "64",
    "--window-m", "0.9",
    "--front-height-m", "0.60",
    "--shard-size", "1000",
    "--max-groups", "$MaxGroups",
    "--sample-mode", "candidate-groups",
    "--sample-seed", "206262001",
    "--dtype", "float16",
    "--target-contains", "single_face_wall",
    "--strategy-contains", "statics_wall"
)
$TensorDir = Invoke-PythonPhase -Phase "export_mujoco_depth_observation_maps" -PythonArgs $ExportArgs

$TrainArgs = @(
    "-m", "scripts.train_torch_support_map_ranker",
    "--tensor-dir", $TensorDir,
    "--output", (Join-Path $BatchRoot "20260624_merged_c12_c16_supportmap_drift_guarded_train"),
    "--epochs", "$Epochs",
    "--batch-size", "96",
    "--hidden", "224",
    "--dropout", "0.20",
    "--lr", "0.0007",
    "--weight-decay", "0.00045",
    "--test-fraction", "0.2",
    "--split-by-run",
    "--seed", "206262101",
    "--device", "auto",
    "--amp",
    "--target-mode", "drift_guarded",
    "--quality-temperature", "32",
    "--exclude-postsim-features",
    "--group-role-weight", "middle=1.35",
    "--group-role-weight", "cap=2.05",
    "--group-course-weight", "2=1.45",
    "--group-course-weight", "3=2.20"
)
$ModelDir = Invoke-PythonPhase -Phase "train_support_map_ranker" -PythonArgs $TrainArgs

$Summary = [ordered]@{
    session = $Session
    created_at = (Get-Date).ToString("s")
    dataset_dir = $DatasetDir
    tensor_dir = $TensorDir
    model_dir = $ModelDir
    requested_runs = $Runs
    max_groups = $MaxGroups
    epochs = $Epochs
    target_mode = "drift_guarded"
    input_policy = "exclude_postsim_features"
    next_step = "Use this support-map model with c12 stone-slot model in 4-course line-lock evaluation."
}
$SummaryPath = Join-Path $RunRoot "summary.json"
$Summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
LogLine "pipeline_done summary=$SummaryPath model_dir=$ModelDir"
