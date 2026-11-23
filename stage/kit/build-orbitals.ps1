# build-orbitals.ps1 - run the NX2navi build kit against the Orbitals .nup.
#
# Handles the two things that broke the first attempt:
#   1. MSYS2 is not at C:\msys64\ucrt64  -> auto-detects and passes --msys
#   2. PowerShell has no bash brace expansion -> package names are explicit
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\build-orbitals.ps1
#   powershell -ExecutionPolicy Bypass -File .\build-orbitals.ps1 -Until lift
#   powershell -ExecutionPolicy Bypass -File .\build-orbitals.ps1 -Jobs 8

param(
    [string]$Nup  = 'D:\Games\Orbitals [040032B023CA4000].nup',
    [string]$Exefs = '',
    [string]$Romfs = '',
    [switch]$Rebuild,
    [string]$Out  = 'D:\Games\Orbitals-Build',
    [string]$Msys = '',
    [int]$Jobs    = 0,
    [string]$Until = ''
)

$ErrorActionPreference = 'Stop'
$kit = $PSScriptRoot

# ---- locate MSYS2 UCRT64 ----------------------------------------------------
if (-not $Msys) {
    foreach ($c in @('C:\msys64', 'D:\msys64', "$env:LOCALAPPDATA\Programs\msys64", "$env:USERPROFILE\msys64")) {
        if (Test-Path (Join-Path $c 'ucrt64\bin\clang.exe')) { $Msys = Join-Path $c 'ucrt64'; break }
    }
}
if (-not $Msys -or -not (Test-Path (Join-Path $Msys 'bin\clang.exe'))) {
    Write-Host 'MSYS2 UCRT64 not found. Run .\setup-msys2.ps1 first.' -ForegroundColor Red
    exit 1
}
Write-Host "MSYS2 : $Msys" -ForegroundColor DarkGray

# ---- preflight: the five paths make_game.py probes --------------------------
$missing = @()
foreach ($r in @('bin\clang.exe','bin\clang++.exe','bin\SDL3.dll','lib\libSDL3.dll.a','include\vma')) {
    if (-not (Test-Path (Join-Path $Msys $r))) { $missing += $r }
}
if ($missing.Count) {
    Write-Host 'MSYS2 is missing required files:' -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $Msys\$_" }
    Write-Host 'Run .\setup-msys2.ps1 to install them.'
    exit 1
}

# ---- preflight: python packages --------------------------------------------
foreach ($m in @('numpy','lz4','zstandard','zbic')) {
    python -c "import $m" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python package '$m' is missing. Run: pip install numpy lz4 zstandard zbic" -ForegroundColor Red
        exit 1
    }
}

# -LiteralPath: the filename contains [ ], which Test-Path would otherwise
# treat as a wildcard character class and never match.
if (($Exefs -or $Romfs) -and (-not $Exefs -or -not $Romfs)) {
    throw 'Supply both -Exefs and -Romfs.'
}
if ($Rebuild -and ($Exefs -or $Romfs -or $PSBoundParameters.ContainsKey('Nup'))) {
    throw '-Rebuild cannot be combined with input paths.'
}
if ($Exefs -and $PSBoundParameters.ContainsKey('Nup')) {
    throw 'Supply either -Nup or -Exefs/-Romfs.'
}
if (-not $Rebuild -and -not $Exefs -and -not (Test-Path -LiteralPath $Nup)) {
    Write-Host "Game file not found: $Nup" -ForegroundColor Red
    exit 1
}

# ---- build ------------------------------------------------------------------
$buildArgs = @(
    (Join-Path $kit 'tools\make_game.py')
    '--out', $Out
    '--msys', $Msys
)
if ($Rebuild) { $buildArgs += '--rebuild' }
elseif ($Exefs) { $buildArgs += @('--exefs', $Exefs, '--romfs', $Romfs) }
else { $buildArgs += @('--nup', $Nup) }
if ($Jobs -gt 0) { $buildArgs += @('--jobs', $Jobs) }
if ($Until)      { $buildArgs += @('--until', $Until) }

Write-Host ''
Write-Host "Building Orbitals" -ForegroundColor Cyan
Write-Host "Output  : $Out" -ForegroundColor Cyan
Write-Host "Steps   : exefs -> romfs -> lift -> compile -> shaders" -ForegroundColor DarkGray
Write-Host ''

$sw = [Diagnostics.Stopwatch]::StartNew()
python @buildArgs
$code = $LASTEXITCODE
$sw.Stop()

Write-Host ''
if ($code -eq 0) {
    Write-Host ("Build finished in {0:N1} min." -f $sw.Elapsed.TotalMinutes) -ForegroundColor Green
    $exe = Join-Path $Out 'orbitals.exe'
    if (Test-Path $exe) {
        Write-Host "Run: $exe" -ForegroundColor Cyan
    }
} else {
    Write-Host ("Build failed (exit $code) after {0:N1} min." -f $sw.Elapsed.TotalMinutes) -ForegroundColor Red
    Write-Host 'Send the last ~40 lines above; the failing step names the next wall.'
}
exit $code
