param(
    [string]$RemoteUrl = "https://github.com/George3215/Codex_Stack.git",
    [string]$RemoteName = "origin",
    [string]$Branch = "main",
    [int64]$MaxFileBytes = 90000000
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Repo

$LogDir = Join-Path $Repo "git_sync_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "git_sync_$Stamp.log"

function LogLine {
    param([string]$Text)
    $Line = "[$((Get-Date).ToString('s'))] $Text"
    Write-Host $Line
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

function RunGit {
    param([string[]]$GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

try {
    LogLine "sync_start repo=$Repo"
    LogLine "policy=no_delete no_deletion_staging max_file_bytes=$MaxFileBytes"

    $inside = (& git rev-parse --is-inside-work-tree 2>$null)
    if ($inside -ne "true") {
        throw "Not inside a git work tree: $Repo"
    }

    $currentBranchRaw = & git branch --show-current
    $currentBranch = if ($null -eq $currentBranchRaw) { "" } else { ($currentBranchRaw -join "`n").Trim() }
    if (-not $currentBranch) {
        throw "Detached HEAD is not supported by the safe snapshot script."
    }
    if ($currentBranch -ne $Branch) {
        LogLine "branch_switch_requested current=$currentBranch target=$Branch"
        RunGit @("branch", "-M", $Branch)
    }

    $userNameRaw = & git config user.name 2>$null
    $userName = if ($null -eq $userNameRaw) { "" } else { ($userNameRaw -join "`n").Trim() }
    if (-not $userName) {
        RunGit @("config", "user.name", "Codex Snapshot Bot")
        LogLine "configured_local_user_name"
    }
    $userEmailRaw = & git config user.email 2>$null
    $userEmail = if ($null -eq $userEmailRaw) { "" } else { ($userEmailRaw -join "`n").Trim() }
    if (-not $userEmail) {
        RunGit @("config", "user.email", "codex-snapshot@local")
        LogLine "configured_local_user_email"
    }

    $remote = (& git remote 2>$null | Where-Object { $_ -eq $RemoteName })
    if (-not $remote) {
        RunGit @("remote", "add", $RemoteName, $RemoteUrl)
        LogLine "remote_added $RemoteName=$RemoteUrl"
    } else {
        $existingUrlRaw = & git remote get-url $RemoteName
        $existingUrl = if ($null -eq $existingUrlRaw) { "" } else { ($existingUrlRaw -join "`n").Trim() }
        if ($existingUrl -ne $RemoteUrl) {
            RunGit @("remote", "set-url", $RemoteName, $RemoteUrl)
            LogLine "remote_updated $RemoteName=$RemoteUrl"
        }
    }

    $modified = @(& git ls-files -m)
    $untracked = @(& git ls-files -o --exclude-standard)
    $candidates = @($modified + $untracked | Where-Object { $_ } | Sort-Object -Unique)
    LogLine "candidate_count=$($candidates.Count)"

    $stagedCount = 0
    $skippedLarge = 0
    foreach ($relativePath in $candidates) {
        $fullPath = Join-Path $Repo $relativePath
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            LogLine "skip_missing_or_deleted $relativePath"
            continue
        }
        $item = Get-Item -LiteralPath $fullPath
        if ($item.Length -gt $MaxFileBytes) {
            LogLine "skip_large bytes=$($item.Length) path=$relativePath"
            $skippedLarge += 1
            continue
        }
        RunGit @("add", "--", $relativePath)
        $stagedCount += 1
    }
    LogLine "staged_count=$stagedCount skipped_large=$skippedLarge"

    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        LogLine "no_staged_changes"
        exit 0
    }

    $commitMessage = "auto snapshot $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    RunGit @("commit", "-m", $commitMessage)
    LogLine "commit_created message=$commitMessage"

    RunGit @("push", "-u", $RemoteName, $Branch)
    LogLine "push_complete remote=$RemoteName branch=$Branch"
    exit 0
} catch {
    LogLine "sync_failed line=$($_.InvocationInfo.ScriptLineNumber) command=$($_.InvocationInfo.Line) message=$($_.Exception.Message)"
    exit 1
}
