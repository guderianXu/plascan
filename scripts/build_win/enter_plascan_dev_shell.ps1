[CmdletBinding()]
param(
    [string] $SourceDir = "",
    [string] $BuildDir = "",
    [string] $VcpkgRoot = "C:\BuildTools\VC\vcpkg",
    [string] $VsDevCmd = "",
    [string] $CMakeExe = "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    [string] $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [switch] $HeadlessQt,
    [switch] $NoLaunch,
    [switch] $SkipVsDevCmd,
    [switch] $Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Set-Utf8ConsoleEnvironment
{
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $global:OutputEncoding = $utf8NoBom
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    if ($env:OS -eq "Windows_NT" -and (Get-Command chcp.com -ErrorAction SilentlyContinue))
    {
        & chcp.com 65001 > $null
    }
}

function Resolve-FullPath
{
    param([Parameter(Mandatory = $true)][string] $Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Convert-ToCMakePath
{
    param([Parameter(Mandatory = $true)][string] $Path)
    return ($Path -replace '\\', '/')
}

function Assert-ExistingPath
{
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Label
    )

    if (-not (Test-Path -LiteralPath $Path))
    {
        throw "$Label not found: $Path"
    }
}

function Get-UniquePathList
{
    param([string[]] $Entries)

    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    $result = New-Object 'System.Collections.Generic.List[string]'
    foreach ($entry in $Entries)
    {
        if ([string]::IsNullOrWhiteSpace($entry))
        {
            continue
        }

        $trimmed = $entry.Trim()
        if ($seen.Add($trimmed))
        {
            [void] $result.Add($trimmed)
        }
    }
    return $result.ToArray()
}

function Resolve-DefaultVsDevCmd
{
    $candidates = @(
        "C:\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )

    foreach ($candidate in $candidates)
    {
        if (Test-Path -LiteralPath $candidate)
        {
            return $candidate
        }
    }

    return $candidates[0]
}

function Import-VsDeveloperEnvironment
{
    param([Parameter(Mandatory = $true)][string] $BatchPath)

    Assert-ExistingPath $BatchPath "Visual Studio developer environment"

    $batchName = [System.IO.Path]::GetFileName($BatchPath)
    $quoted = '"' + $BatchPath + '"'
    if ($batchName -ieq "vcvars64.bat")
    {
        $lines = & cmd.exe /d /s /c "call $quoted >nul && set"
    }
    elseif ($batchName -ieq "vcvarsall.bat")
    {
        $lines = & cmd.exe /d /s /c "call $quoted x64 >nul && set"
    }
    else
    {
        $lines = & cmd.exe /d /s /c "call $quoted -arch=x64 -host_arch=x64 >nul && set"
    }

    if ($LASTEXITCODE -ne 0)
    {
        throw "Visual Studio developer environment failed with exit code $LASTEXITCODE"
    }

    $vsDevPathValue = $null
    foreach ($line in $lines)
    {
        if ($line -match '^([^=]+)=(.*)$')
        {
            $name = $matches[1]
            $value = $matches[2]
            if ($name -ieq "Path")
            {
                if ($null -eq $vsDevPathValue -or $value -match '\\VC\\Tools\\MSVC\\')
                {
                    $vsDevPathValue = $value
                }
                continue
            }

            Set-Item -Path ("Env:" + $name) -Value $value
        }
    }

    if ($null -ne $vsDevPathValue)
    {
        $env:PATH = $vsDevPathValue
    }
}

function Capture-VsDevPathEntries
{
    if (-not $env:PATH)
    {
        return @()
    }

    return @($env:PATH -split ';' | Where-Object {
        $_ -match '\\VC\\Tools\\MSVC\\' -or
        $_ -match '\\Windows Kits\\10\\bin\\' -or
        $_ -match '\\Common7\\IDE\\' -or
        $_ -match '\\Common7\\Tools\\'
    })
}

function Resolve-NinjaDirectory
{
    param([Parameter(Mandatory = $true)][string] $CMakePath)

    $cmakeBin = Split-Path -Parent $CMakePath
    $candidateDirs = New-Object 'System.Collections.Generic.List[string]'
    [void] $candidateDirs.Add((Join-Path (Split-Path -Parent (Split-Path -Parent $cmakeBin)) "Ninja"))
    [void] $candidateDirs.Add((Join-Path (Split-Path -Parent $cmakeBin) "Ninja"))
    [void] $candidateDirs.Add((Join-Path $cmakeBin "Ninja"))

    $existingNinja = Get-Command ninja -ErrorAction SilentlyContinue
    if ($existingNinja)
    {
        [void] $candidateDirs.Add((Split-Path -Parent $existingNinja.Source))
    }

    foreach ($candidate in (Get-UniquePathList $candidateDirs.ToArray()))
    {
        if (Test-Path -LiteralPath (Join-Path $candidate "ninja.exe"))
        {
            return (Resolve-FullPath $candidate)
        }
    }

    return ""
}

function Set-PlascanWindowsBuildEnvironment
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $VcpkgPath,
        [Parameter(Mandatory = $true)][string] $CudaPath,
        [Parameter(Mandatory = $true)][string] $CMakePath,
        [Parameter(Mandatory = $true)][string] $ProjectRoot,
        [string[]] $VsDevPathEntries = @()
    )

    $vcpkgInstalled = Join-Path $BuildPath "vcpkg_installed"
    $tripletRoot = Join-Path $vcpkgInstalled "x64-windows"
    $qtPluginsRoot = Join-Path $tripletRoot "Qt6\plugins"
    $qtPlatformsRoot = Join-Path $qtPluginsRoot "platforms"
    $cmakeBin = Split-Path -Parent $CMakePath
    $ninjaDir = Resolve-NinjaDirectory $CMakePath
    $pythonRuntimeRoot = Join-Path $ProjectRoot ".venv"
    $legacyPythonRuntimeRoot = Join-Path $ProjectRoot "build\env\python-runtime"
    $legacyRuntimePython = Join-Path $legacyPythonRuntimeRoot "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath (Join-Path $pythonRuntimeRoot "Scripts\python.exe")) -and
        (Test-Path -LiteralPath $legacyRuntimePython))
    {
        $pythonRuntimeRoot = $legacyPythonRuntimeRoot
    }
    $pythonRuntimeScripts = Join-Path $pythonRuntimeRoot "Scripts"
    $runtimePython = Join-Path $pythonRuntimeScripts "python.exe"

    $env:PLASCAN_SOURCE_DIR = $ProjectRoot
    $env:PLASCAN_BUILD_DIR = $BuildPath
    $env:PLASCAN_MODEL_DIR = Resolve-FullPath (Join-Path $ProjectRoot "resources\models")
    $env:PLASCAN_SCRIPT_DIR = Resolve-FullPath (Join-Path $ProjectRoot "scripts")
    $env:PLASCAN_PYTHON_RUNTIME_DIR = Resolve-FullPath $pythonRuntimeRoot
    $env:VCPKG_ROOT = $VcpkgPath
    $env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
    $env:VCPKG_INSTALLED_DIR = Convert-ToCMakePath $vcpkgInstalled
    $env:CUDAToolkit_ROOT = Convert-ToCMakePath $CudaPath
    $env:CUDA_PATH = Convert-ToCMakePath $CudaPath
    $env:CUDA_HOME = Convert-ToCMakePath $CudaPath
    $env:CUDA_BIN_PATH = (Convert-ToCMakePath (Join-Path $CudaPath "bin"))
    $env:CMAKE_PREFIX_PATH = Convert-ToCMakePath $tripletRoot
    $env:QT_PLUGIN_PATH = Convert-ToCMakePath $qtPluginsRoot
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = Convert-ToCMakePath $qtPlatformsRoot

    if ([string]::IsNullOrWhiteSpace($env:CTEST_PARALLEL_LEVEL))
    {
        $env:CTEST_PARALLEL_LEVEL = [Math]::Max(
            1,
            [int] [Math]::Floor([Environment]::ProcessorCount / 2)
        ).ToString()
    }

    if ($HeadlessQt)
    {
        $env:QT_QPA_PLATFORM = "offscreen"
    }

    $runtimePythonPathEntries = @()
    if (Test-Path -LiteralPath $runtimePython)
    {
        $env:PLASCAN_PYTHON_EXECUTABLE = Resolve-FullPath $runtimePython
        $env:PLASCAN_PYTHON = $env:PLASCAN_PYTHON_EXECUTABLE
        $runtimePythonPathEntries = @(
            (Resolve-FullPath $pythonRuntimeScripts),
            (Resolve-FullPath $pythonRuntimeRoot)
        )
    }

    $rejectRoots = @(
        (Join-Path $ProjectRoot "build\windows-vcpkg-release"),
        "E:\code\sat_sim_cuda\build\vcpkg"
    )

    $oldPath = @()
    if ($env:PATH)
    {
        $oldPath = $env:PATH -split ';'
    }

    $filtered = foreach ($entry in $oldPath)
    {
        if ([string]::IsNullOrWhiteSpace($entry))
        {
            continue
        }

        $skip = $false
        foreach ($reject in $rejectRoots)
        {
            if ($entry.StartsWith($reject, [System.StringComparison]::OrdinalIgnoreCase))
            {
                $skip = $true
                break
            }
        }
        if (-not $skip)
        {
            $entry
        }
    }

    $prepend = @(
        (Join-Path $BuildPath "bin"),
        (Join-Path $BuildPath "tests"),
        $runtimePythonPathEntries,
        (Join-Path $tripletRoot "bin"),
        (Join-Path $tripletRoot "tools\Qt6\bin"),
        (Join-Path $CudaPath "bin"),
        (Join-Path $CudaPath "nvvm\bin"),
        $cmakeBin,
        $ninjaDir
    )

    $env:PATH = (Get-UniquePathList ($prepend + $VsDevPathEntries + $filtered)) -join ';'
}

Set-Utf8ConsoleEnvironment

if ([string]::IsNullOrWhiteSpace($SourceDir))
{
    $SourceDir = Resolve-FullPath (Join-Path $PSScriptRoot "..\..")
}
else
{
    $SourceDir = Resolve-FullPath $SourceDir
}

if ([string]::IsNullOrWhiteSpace($BuildDir))
{
    $BuildDir = Join-Path $SourceDir "build\windows-vcpkg-cuda-release"
}
$BuildDir = Resolve-FullPath $BuildDir

$VcpkgRoot = Resolve-FullPath $VcpkgRoot
$CudaRoot = Resolve-FullPath $CudaRoot
$CMakeExe = Resolve-FullPath $CMakeExe
if ([string]::IsNullOrWhiteSpace($VsDevCmd))
{
    $VsDevCmd = Resolve-DefaultVsDevCmd
}
$VsDevCmd = Resolve-FullPath $VsDevCmd

$vcpkgToolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
$cudaNvcc = Join-Path $CudaRoot "bin\nvcc.exe"

Assert-ExistingPath $SourceDir "SourceDir"
Assert-ExistingPath (Join-Path $SourceDir "CMakeLists.txt") "PlaScan CMakeLists.txt"
Assert-ExistingPath $VcpkgRoot "VcpkgRoot"
Assert-ExistingPath $vcpkgToolchain "vcpkg toolchain"
Assert-ExistingPath $CudaRoot "CudaRoot"
Assert-ExistingPath $cudaNvcc "CUDA nvcc"
Assert-ExistingPath $CMakeExe "CMake"

$vsDevPathEntries = @()
if (-not $SkipVsDevCmd)
{
    Import-VsDeveloperEnvironment $VsDevCmd
    $vsDevPathEntries = Capture-VsDevPathEntries
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Set-PlascanWindowsBuildEnvironment `
    -BuildPath $BuildDir `
    -VcpkgPath $VcpkgRoot `
    -CudaPath $CudaRoot `
    -CMakePath $CMakeExe `
    -ProjectRoot $SourceDir `
    -VsDevPathEntries $vsDevPathEntries

$ninjaCommand = Get-Command ninja -ErrorAction SilentlyContinue
if (-not $ninjaCommand)
{
    throw "ninja.exe was not found after loading the PlaScan Windows environment."
}

$clCommand = Get-Command cl -ErrorAction SilentlyContinue
if (-not $SkipVsDevCmd -and -not $clCommand)
{
    throw "cl.exe was not found after loading the Visual Studio developer environment."
}

if (-not $Quiet)
{
    Write-Host "PlaScan Windows development environment is ready."
    Write-Host "  SourceDir: $SourceDir"
    Write-Host "  BuildDir:  $BuildDir"
    Write-Host "  CMake:     $((Get-Command cmake).Source)"
    Write-Host "  Ninja:     $($ninjaCommand.Source)"
    if ($clCommand)
    {
        Write-Host "  MSVC:      $($clCommand.Source)"
    }
    Write-Host "  CUDA:      $CudaRoot"
    if ($env:PLASCAN_PYTHON_EXECUTABLE)
    {
        Write-Host "  Python:    $env:PLASCAN_PYTHON_EXECUTABLE"
    }
    else
    {
        Write-Host "  Python:    not initialized (run scripts\env\setup_python_runtime.py)"
    }
}

$dotSourced = $MyInvocation.InvocationName -eq "."
if (-not $dotSourced -and -not $NoLaunch)
{
    $shell = (Get-Command powershell.exe -ErrorAction Stop).Source
    $escapedSourceDir = $SourceDir -replace "'", "''"
    $message = "Set-Location -LiteralPath '$escapedSourceDir'; " +
        "Write-Host 'PlaScan Windows dev shell is active.'; " +
        "Write-Host 'Try: ninja -C build\windows-vcpkg-cuda-release plascan_gui'"
    & $shell -NoLogo -NoExit -NoProfile -ExecutionPolicy Bypass -Command $message
}
