# Fetch the free ffmpeg 8.1 release build into %USERPROFILE%\.cache\reelwise\ffmpeg-8.1.
# reelwise puts this folder first on PATH. Skipped when it is already there.
# (A development build, ffmpeg N-128293 of 2 November 2026, hangs on the loudness pass
# of the final encode, so a release build is used.)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes downloads many times slower
$d = Join-Path $env:USERPROFILE '.cache\reelwise'
$dst = Join-Path $d 'ffmpeg-8.1'
$bin = Join-Path $dst 'bin\ffmpeg.exe'
if (Test-Path $bin) { "ffmpeg already present: $(& $bin -version | Select-Object -First 1)"; exit 0 }
New-Item -ItemType Directory -Force -Path $d | Out-Null
$zip = Join-Path $d 'ffmpeg-8.1.zip'
$url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip'
if (-not (Test-Path $zip) -or (Get-Item $zip).Length -lt 100MB) {
  "downloading $url"
  Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
}
"zip $((Get-Item $zip).Length) bytes"
$tmp = Join-Path $d 'ffmpeg_tmp'
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive -Path $zip -DestinationPath $tmp
$src = Get-ChildItem $tmp -Directory | Select-Object -First 1
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Move-Item $src.FullName $dst
Remove-Item $zip
Remove-Item -Recurse -Force $tmp
& $bin -version | Select-Object -First 1
