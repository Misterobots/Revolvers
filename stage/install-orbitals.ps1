param(
    [Parameter(Mandatory=$true)][string]$KitDir,
    [Parameter(Mandatory=$true)][string]$SidecarDir,
    [Parameter(Mandatory=$true)][string]$IniFile,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][ValidateSet('nup','folder')][string]$SourceType,
    [Parameter(Mandatory=$true)][string]$SourcePath
)

$ErrorActionPreference = 'Stop'

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Fail-Pause([string]$Message) {
    Write-Host ''
    Write-Host "ERROR: $Message" -ForegroundColor Red
    Write-Host ''
    Read-Host 'Press Enter to close this window'
    exit 1
}

try {
    Write-Host '============================================================' -ForegroundColor Cyan
    Write-Host '  Orbitals PC - Build From Switch 2 Copy' -ForegroundColor Cyan
    Write-Host '============================================================' -ForegroundColor Cyan
    Write-Host "Source ($SourceType): $SourcePath"
    Write-Host "Install folder      : $OutDir"
    Write-Host ''

    # ---- 1. Ensure Python 3 is available ------------------------------------
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Write-Host 'Python 3 not found. Installing Python 3.12 via winget...' -ForegroundColor Yellow
        winget install --id Python.Python.3.12 --exact --silent --accept-source-agreements --accept-package-agreements
        Refresh-Path
        $pyCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pyCmd) {
            Fail-Pause 'Python 3 was not found in PATH. Please install Python 3 from https://www.python.org/ and re-run setup.'
        }
    }
    Write-Host ("Python : " + (& python --version 2>&1)) -ForegroundColor DarkGray

    # ---- 2. Ensure required Python packages are installed -------------------
    $missingPy = @()
    foreach ($m in @('numpy','lz4','zstandard','zbic')) {
        & python -c "import $m" 2>$null
        if ($LASTEXITCODE -ne 0) { $missingPy += $m }
    }
    if ($missingPy.Count -gt 0) {
        Write-Host ("Installing missing Python packages: " + ($missingPy -join ', ')) -ForegroundColor Yellow
        & python -m pip install --user @missingPy
        if ($LASTEXITCODE -ne 0) {
            Fail-Pause ("Failed to install Python packages: " + ($missingPy -join ', '))
        }
    }

    # ---- 3. Ensure MSYS2 UCRT64 + required toolchain packages ---------------
    function Find-MsysRoot {
        foreach ($c in @('C:\msys64', 'D:\msys64', "$env:LOCALAPPDATA\Programs\msys64", "$env:USERPROFILE\msys64")) {
            if (Test-Path (Join-Path $c 'usr\bin\pacman.exe')) { return $c }
        }
        return $null
    }

    $msysRoot = Find-MsysRoot
    if (-not $msysRoot) {
        Write-Host 'MSYS2 not found. Installing MSYS2 via winget...' -ForegroundColor Yellow
        winget install --id MSYS2.MSYS2 --exact --silent --accept-source-agreements --accept-package-agreements
        $msysRoot = Find-MsysRoot
        if (-not $msysRoot) {
            Fail-Pause 'MSYS2 was not found after installation. Please install MSYS2 from https://www.msys2.org/ and re-run setup.'
        }
    }

    $ucrt = Join-Path $msysRoot 'ucrt64'
    $missingMsys = @()
    foreach ($r in @('bin\clang.exe','bin\clang++.exe','bin\SDL3.dll','lib\libSDL3.dll.a','include\vma')) {
        if (-not (Test-Path (Join-Path $ucrt $r))) { $missingMsys += $r }
    }
    if ($missingMsys.Count -gt 0) {
        Write-Host 'Installing required MSYS2 UCRT64 toolchain packages...' -ForegroundColor Yellow
        & powershell -ExecutionPolicy Bypass -File (Join-Path $KitDir 'setup-msys2.ps1')
        if ($LASTEXITCODE -ne 0) {
            Fail-Pause 'MSYS2 toolchain setup failed. Check your internet connection and re-run setup.'
        }
    }

    # ---- 4. Run build-orbitals.ps1 against user's Switch 2 copy -------------
    if (-not (Test-Path -LiteralPath $OutDir)) {
        New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    }

    $buildScript = Join-Path $KitDir 'build-orbitals.ps1'
    $buildArgs = @('-ExecutionPolicy', 'Bypass', '-File', $buildScript, '-Out', $OutDir, '-Msys', $ucrt)

    if ($SourceType -eq 'nup') {
        if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
            Fail-Pause "Selected .nup file does not exist: $SourcePath"
        }
        $buildArgs += @('-Nup', $SourcePath)
    } else {
        $exefsDir = Join-Path $SourcePath 'exefs'
        $romfsDir = Join-Path $SourcePath 'romfs'
        if (-not (Test-Path -LiteralPath $exefsDir -PathType Container)) {
            Fail-Pause "Selected folder does not contain an 'exefs' directory: $SourcePath"
        }
        if (-not (Test-Path -LiteralPath $romfsDir -PathType Container)) {
            Fail-Pause "Selected folder does not contain a 'romfs' directory: $SourcePath"
        }
        $buildArgs += @('-Exefs', $exefsDir, '-Romfs', $romfsDir)
    }

    & powershell @buildArgs
    if ($LASTEXITCODE -ne 0) {
        Fail-Pause "Build failed with exit code $LASTEXITCODE. Make sure you selected a valid Orbitals Switch 2 copy."
    }

    $exePath = Join-Path $OutDir 'orbitals.exe'
    if (-not (Test-Path -LiteralPath $exePath)) {
        Fail-Pause "Build reported success, but orbitals.exe was not found at $exePath"
    }

    # ---- 5. Copy open-source shader translator sidecars & default config ----
    Write-Host 'Installing shader translator sidecars and default config...' -ForegroundColor Cyan
    Get-ChildItem -LiteralPath $SidecarDir -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $OutDir $_.Name) -Force
    }

    $destIni = Join-Path $OutDir 'nx2navi.ini'
    if (-not (Test-Path -LiteralPath $destIni)) {
        Copy-Item -LiteralPath $IniFile -Destination $destIni -Force
    }

    # ---- 6. Clean up temporary build-cache to save ~2+ GB -------------------
    $buildCache = Join-Path $OutDir 'build-cache'
    if (Test-Path -LiteralPath $buildCache) {
        Write-Host 'Cleaning up temporary build-cache...' -ForegroundColor DarkGray
        Remove-Item -LiteralPath $buildCache -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host ''
    Write-Host 'Orbitals PC built and installed successfully!' -ForegroundColor Green
    exit 0
}
catch {
    Fail-Pause $_.Exception.Message
}
