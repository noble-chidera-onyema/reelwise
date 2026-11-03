# Creates github.com/noble-chidera-onyema/reelwise if needed, commits and pushes.
# Uses the GitHub sign-in Git already has on this computer; the token is never printed.
$ErrorActionPreference = 'Continue'
$env:GCM_INTERACTIVE = 'never'
$env:GIT_TERMINAL_PROMPT = '0'
$owner = 'noble-chidera-onyema'
$repo = 'reelwise'
Set-Location (Split-Path $PSScriptRoot -Parent)

$cred = "protocol=https`nhost=github.com`n`n" | git credential fill 2>$null
$token = (($cred | Where-Object { $_ -like 'password=*' }) -replace '^password=', '')
if (-not $token) { 'NO_STORED_GITHUB_CREDENTIAL'; exit 2 }
'credential: found'
$h = @{ Authorization = "token $token"; Accept = 'application/vnd.github+json'; 'User-Agent' = 'reelwise-setup' }
try {
    $r = Invoke-RestMethod -Headers $h -Uri "https://api.github.com/repos/$owner/$repo" -Method Get
    "repo exists: $($r.html_url)"
} catch {
    $body = @{ name = $repo; private = $false; has_wiki = $false;
               description = 'Free studio for short true-story videos: scene-by-scene video analysis, 3D motion from still images, host voices, TV push-through, and quality checks before download.' } | ConvertTo-Json
    try {
        $r = Invoke-RestMethod -Headers $h -Uri 'https://api.github.com/user/repos' -Method Post -Body $body -ContentType 'application/json'
        "repo created: $($r.html_url)"
    } catch { "create failed: $($_.Exception.Message)"; exit 3 }
}
if (-not (Test-Path .git)) { git init -b main 2>&1 }
git add -A 2>&1
$msg = @"
reelwise: video breakdown, channel outliers and a free episode Studio

Scene detection measured on synthetic ground truth (held-out F1 0.963),
YouTube outlier analysis, and the Studio: Edge host voices, depth-based 3D
motion, TV push-through, host lip flaps from audio, captions, -14 LUFS mix
and a quality gate before download.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E8fay8MDCpJz1yy9qBrnST
"@
$mf = Join-Path (Get-Location) '.local\commit_message.txt'
if (Test-Path $mf) { git commit -F $mf 2>&1; Remove-Item $mf } else { git commit -m $msg 2>&1 }
git branch -M main 2>&1
git remote remove origin 2>$null
git remote add origin "https://github.com/$owner/$repo.git"
git push -u origin main 2>&1
"push exit: $LASTEXITCODE"
git log --oneline -1 2>&1
