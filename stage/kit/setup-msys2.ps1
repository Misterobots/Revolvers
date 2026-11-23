# setup-msys2.ps1 - install the MSYS2 UCRT64 packages the NX2navi build kit needs.
#
# PowerShell does NOT do bash brace expansion, so
#   pacman -S mingw-w64-ucrt-x86_64-{clang,gcc,...}
# is a syntax error. The package names are spelled out below instead.
#
# Run from an elevated PowerShell if MSYS2 is not already installed:
#   powershell -ExecutionPolicy Bypass -File .\setup-msys2.ps1

$ErrorActionPreference = 'Stop'

# ---- locate MSYS2 -----------------------------------------------------------
$candidates = @(
    'C:\msys64',
    'D:\msys64',
    "$env:LOCALAPPDATA\Programs\msys64",
    "$env:USERPROFILE\msys64"
)
$msysRoot = $null
foreach ($c in $candidates) {
    if (Test-Path (Join-Path $c 'usr\bin\pacman.exe')) { $msysRoot = $c; break }
}

if (-not $msysRoot) {
    Write-Host 'MSYS2 not found. Install it first:' -ForegroundColor Yellow
    Write-Host '  winget install --id MSYS2.MSYS2 -e'
    Write-Host 'or download from https://www.msys2.org/'
    Write-Host ''
    Write-Host 'Then re-run this script.'
    exit 1
}

$ucrt = Join-Path $msysRoot 'ucrt64'
$pacman = Join-Path $msysRoot 'usr\bin\pacman.exe'
Write-Host "MSYS2 root : $msysRoot"
Write-Host "UCRT64 dir : $ucrt"
Write-Host ''

# ---- packages ---------------------------------------------------------------
# Roots from toolchain.lock.json. pacman resolves the transitive deps
# (clang-libs, llvm-libs, binutils, crt, headers, libgcc, libstdc++, ...).
$packages = @(
    'mingw-w64-ucrt-x86_64-clang'
    'mingw-w64-ucrt-x86_64-gcc'
    'mingw-w64-ucrt-x86_64-sdl3'
    'mingw-w64-ucrt-x86_64-opus'
    'mingw-w64-ucrt-x86_64-libvpx'
    'mingw-w64-ucrt-x86_64-dav1d'
    'mingw-w64-ucrt-x86_64-zstd'
    'mingw-w64-ucrt-x86_64-vulkan-headers'
    'mingw-w64-ucrt-x86_64-vulkan-loader'
    'mingw-w64-ucrt-x86_64-vulkan-memory-allocator'
)

Write-Host "Installing $($packages.Count) packages..." -ForegroundColor Cyan
& $pacman -S --needed --noconfirm @packages
if ($LASTEXITCODE -ne 0) {
    Write-Host "pacman failed (exit $LASTEXITCODE)." -ForegroundColor Red
    Write-Host 'If the database is stale, run: & "' + $pacman + '" -Syuu'
    exit $LASTEXITCODE
}

# ---- verify the exact files make_game.py probes for -------------------------
# tools/make_game.py checks these five paths before it will build.
$required = @(
    'bin\clang.exe'
    'bin\clang++.exe'
    'bin\SDL3.dll'
    'lib\libSDL3.dll.a'
    'include\vma'
)
$missing = @()
foreach ($r in $required) {
    if (-not (Test-Path (Join-Path $ucrt $r))) { $missing += $r }
}

Write-Host ''
if ($missing.Count -eq 0) {
    Write-Host 'All required files present.' -ForegroundColor Green
    Write-Host ''
    Write-Host 'Build with:' -ForegroundColor Cyan
    Write-Host "  python tools\make_game.py --nup `"D:\Games\Orbitals [040032B023CA4000].nup`" --out `"D:\Games\Orbitals-Build`" --msys `"$ucrt`""
} else {
    Write-Host 'Still missing:' -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $ucrt\$_" }
    Write-Host ''
    Write-Host 'include\vma comes from vulkan-memory-allocator; if it is absent, check:'
    Write-Host "  & `"$pacman`" -Ql mingw-w64-ucrt-x86_64-vulkan-memory-allocator"
    exit 1
}
