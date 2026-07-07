[CmdletBinding()]
param(
    [string] $SourceDir = "",
    [string] $BuildDir = "",
    [string] $VcpkgRoot = "C:\BuildTools\VC\vcpkg",
    [string] $VsDevCmd = "C:\BuildTools\Common7\Tools\VsDevCmd.bat",
    [string] $CMakeExe = "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    [string] $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [string] $CudnnRoot = "",
    [string] $TorchRoot = "",
    [string] $Target = "",
    [string] $CTestRegex = "",
    [int] $Jobs = 0,
    [switch] $ConfigureOnly,
    [switch] $BuildOnly,
    [switch] $RunTests,
    [switch] $CleanConfigure,
    [switch] $CleanRootCache,
    [switch] $InstallDeps,
    [switch] $EnableOpenCvDnnCuda,
    [switch] $SkipVsDevCmd
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

Set-Utf8ConsoleEnvironment

function Resolve-FullPath
{
    param([Parameter(Mandatory = $true)][string] $Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-PathUnder
{
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Root
    )

    $fullPath = (Resolve-FullPath $Path).TrimEnd('\', '/')
    $fullRoot = (Resolve-FullPath $Root).TrimEnd('\', '/')
    return $fullPath.Equals($fullRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $fullPath.StartsWith($fullRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)
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

function Remove-SafeItem
{
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $AllowedRoot
    )

    if (-not (Test-Path -LiteralPath $Path))
    {
        return
    }

    $fullPath = Resolve-FullPath $Path
    $fullRoot = Resolve-FullPath $AllowedRoot
    if (-not (Test-PathUnder $fullPath $fullRoot))
    {
        throw "Refusing to remove path outside allowed root: $fullPath"
    }

    Write-Host "remove: $fullPath"
    Remove-Item -LiteralPath $fullPath -Recurse -Force
}

function Import-VsDevCmd
{
    param([Parameter(Mandatory = $true)][string] $BatchPath)

    Assert-ExistingPath $BatchPath "VsDevCmd"
    $quoted = '"' + $BatchPath + '"'
    $lines = & cmd.exe /d /s /c "call $quoted -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0)
    {
        throw "VsDevCmd failed with exit code $LASTEXITCODE"
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

function Get-UniqueExistingPathList
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

    foreach ($candidate in (Get-UniqueExistingPathList $candidateDirs.ToArray()))
    {
        if (Test-Path -LiteralPath (Join-Path $candidate "ninja.exe"))
        {
            return (Resolve-FullPath $candidate)
        }
    }

    return ""
}

function Set-IsolatedBuildEnvironment
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $VcpkgPath,
        [Parameter(Mandatory = $true)][string] $TorchPath,
        [Parameter(Mandatory = $true)][string] $CudaPath,
        [Parameter(Mandatory = $true)][string] $CMakePath,
        [Parameter(Mandatory = $true)][string] $ProjectRoot,
        [string[]] $VsDevPathEntries = @()
    )

    $vcpkgInstalled = Join-Path $BuildPath "vcpkg_installed"
    $tripletRoot = Join-Path $vcpkgInstalled "x64-windows"
    $torchConfig = Join-Path $TorchPath "share\cmake\Torch"
    $vcpkgInstalledCMake = Convert-ToCMakePath $vcpkgInstalled
    $tripletRootCMake = Convert-ToCMakePath $tripletRoot
    $torchPathCMake = Convert-ToCMakePath $TorchPath
    $torchConfigCMake = Convert-ToCMakePath $torchConfig
    $cudaPathCMake = Convert-ToCMakePath $CudaPath
    $qtPluginsRoot = Join-Path $tripletRoot "Qt6\plugins"
    $qtPlatformsRoot = Join-Path $qtPluginsRoot "platforms"
    $qtPluginsRootCMake = Convert-ToCMakePath $qtPluginsRoot
    $qtPlatformsRootCMake = Convert-ToCMakePath $qtPlatformsRoot
    $cmakeBin = Split-Path -Parent $CMakePath
    $ninjaDir = Resolve-NinjaDirectory $CMakePath

    $env:VCPKG_ROOT = $VcpkgPath
    $env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
    $env:VCPKG_INSTALLED_DIR = $vcpkgInstalledCMake
    $env:PLASCAN_TORCH_DIR = $torchConfigCMake
    $env:Torch_DIR = $torchConfigCMake
    $env:CUDAToolkit_ROOT = $cudaPathCMake
    $env:CUDA_PATH = $cudaPathCMake
    $env:CUDA_HOME = $cudaPathCMake
    $env:CUDA_BIN_PATH = "$cudaPathCMake/bin"
    $env:CMAKE_PREFIX_PATH = ($tripletRootCMake, $torchPathCMake) -join ';'
    $env:QT_PLUGIN_PATH = $qtPluginsRootCMake
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = $qtPlatformsRootCMake
    $env:QT_QPA_PLATFORM = "offscreen"

    $rejectRoots = @(
        (Join-Path $ProjectRoot "build\windows-vcpkg-release"),
        (Join-Path $ProjectRoot "build\env\libtorch\libtorch"),
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
        (Join-Path $tripletRoot "bin"),
        (Join-Path $tripletRoot "tools\Qt6\bin"),
        (Join-Path $TorchPath "lib"),
        (Join-Path $TorchPath "bin"),
        (Join-Path $CudaPath "bin"),
        (Join-Path $CudaPath "nvvm\bin"),
        $cmakeBin,
        $ninjaDir
    )

    $env:PATH = (Get-UniqueExistingPathList ($prepend + $vsDevPathEntries + $filtered)) -join ';'
}

function Sync-QtRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $TripletRoot
    )

    $qtPlatformsRoot = Join-Path $TripletRoot "Qt6\plugins\platforms"
    $qtImageFormatsRoot = Join-Path $TripletRoot "Qt6\plugins\imageformats"
    if (-not (Test-Path -LiteralPath $qtPlatformsRoot))
    {
        return
    }

    foreach ($dir in @((Join-Path $BuildPath "bin"), (Join-Path $BuildPath "tests")))
    {
        if (-not (Test-Path -LiteralPath $dir))
        {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }

        $platformDir = Join-Path $dir "platforms"
        New-Item -ItemType Directory -Force -Path $platformDir | Out-Null
        Copy-Item -LiteralPath (Join-Path $qtPlatformsRoot "qwindows.dll") -Destination $platformDir -Force
        Copy-Item -LiteralPath (Join-Path $qtPlatformsRoot "qoffscreen.dll") -Destination $platformDir -Force
        $minimalPlugin = Join-Path $qtPlatformsRoot "qminimal.dll"
        if (Test-Path -LiteralPath $minimalPlugin)
        {
            Copy-Item -LiteralPath $minimalPlugin -Destination $platformDir -Force
        }

        if (Test-Path -LiteralPath $qtImageFormatsRoot)
        {
            $imageFormatsDir = Join-Path $dir "imageformats"
            New-Item -ItemType Directory -Force -Path $imageFormatsDir | Out-Null
            Get-ChildItem -LiteralPath $qtImageFormatsRoot -Force -File -Filter "q*.dll" -ErrorAction SilentlyContinue |
                ForEach-Object {
                    Copy-Item -LiteralPath $_.FullName -Destination $imageFormatsDir -Force
                }

            $qjpegPlugin = Join-Path $qtImageFormatsRoot "qjpeg.dll"
            if (Test-Path -LiteralPath $qjpegPlugin)
            {
                Copy-Item -LiteralPath $qjpegPlugin -Destination $imageFormatsDir -Force
            }
        }
    }
}

function Sync-TorchRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $TorchPath
    )

    if (-not (Test-Path -LiteralPath $BuildPath))
    {
        return
    }

    $torchDllRoots = @(
        (Join-Path $TorchPath "bin"),
        (Join-Path $TorchPath "lib")
    ) | Where-Object { Test-Path -LiteralPath $_ }

    $sources = @{}
    foreach ($root in $torchDllRoots)
    {
        Get-ChildItem -LiteralPath $root -Force -File -Filter "*.dll" -ErrorAction SilentlyContinue |
            ForEach-Object {
                if (-not $sources.ContainsKey($_.Name))
                {
                    $sources[$_.Name] = $_.FullName
                }
            }
    }

    if ($sources.Count -eq 0)
    {
        return
    }

    function Copy-RuntimeDllIfNeeded
    {
        param(
            [Parameter(Mandatory = $true)][string] $Source,
            [Parameter(Mandatory = $true)][string] $Destination
        )

        $copy = $true
        if (Test-Path -LiteralPath $Destination)
        {
            $src = Get-Item -LiteralPath $Source
            $dst = Get-Item -LiteralPath $Destination
            $copy = ($src.Length -ne $dst.Length) -or ($src.LastWriteTimeUtc -gt $dst.LastWriteTimeUtc)
        }

        if ($copy)
        {
            Copy-Item -LiteralPath $Source -Destination $Destination -Force
        }
    }

    $runtimeDirs = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    $primaryBinDir = Join-Path $BuildPath "bin"
    New-Item -ItemType Directory -Path $primaryBinDir -Force | Out-Null
    [void] $runtimeDirs.Add((Resolve-FullPath $primaryBinDir))

    $testBinDir = Join-Path $BuildPath "tests"
    if (Test-Path -LiteralPath $testBinDir)
    {
        [void] $runtimeDirs.Add((Resolve-FullPath $testBinDir))
    }

    foreach ($marker in @("c10.dll", "torch.dll", "torch_cpu.dll", "torch_cuda.dll"))
    {
        Get-ChildItem -LiteralPath $BuildPath -Recurse -Force -File -Filter $marker -ErrorAction SilentlyContinue |
            ForEach-Object {
                $dir = Resolve-FullPath $_.DirectoryName
                [void] $runtimeDirs.Add($dir)
            }
    }

    $sourceEntries = $sources.GetEnumerator() | Sort-Object Name
    foreach ($dir in $runtimeDirs)
    {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        foreach ($entry in $sourceEntries)
        {
            $dst = Join-Path $dir $entry.Name
            Copy-RuntimeDllIfNeeded -Source $entry.Value -Destination $dst
        }
    }

    Write-Host ("Synced {0} LibTorch runtime DLLs into {1} runtime director{2}." -f `
        $sources.Count, $runtimeDirs.Count, $(if ($runtimeDirs.Count -eq 1) { "y" } else { "ies" }))
}

function Assert-VcpkgInstalledPackages
{
    param([Parameter(Mandatory = $true)][string] $TripletRoot)

    $requiredShareDirs = @(
        "Qt6",
        "opencv4",
        "gdal",
        "gtest",
        "libzip",
        "tiff"
    )

    $missing = @()
    foreach ($name in $requiredShareDirs)
    {
        $candidate = Join-Path $TripletRoot ("share\" + $name)
        if (-not (Test-Path -LiteralPath $candidate))
        {
            $missing += $name
        }
    }

    if ($missing.Count -gt 0)
    {
        $joined = $missing -join ", "
        throw "Missing vcpkg packages in ${TripletRoot}: $joined. Seed this directory or rerun with -InstallDeps."
    }
}

function Assert-OpenCvDnnCudaFeatures
{
    param([Parameter(Mandatory = $true)][string] $TripletRoot)

    $abiInfo = Join-Path $TripletRoot "share\opencv4\vcpkg_abi_info.txt"
    if (-not (Test-Path -LiteralPath $abiInfo))
    {
        throw "OpenCV ABI info not found: $abiInfo. Rerun with -InstallDeps -EnableOpenCvDnnCuda."
    }

    $text = Get-Content -LiteralPath $abiInfo -Raw
    if ($text -notmatch "(?m)^features .*cuda" -or
        $text -notmatch "(?m)^features .*dnn-cuda")
    {
        throw "OpenCV DNN CUDA is not installed in $TripletRoot. Rerun this script with -InstallDeps -EnableOpenCvDnnCuda."
    }
}

function Test-CudnnDevRoot
{
    param([Parameter(Mandatory = $true)][string] $Root)

    if ([string]::IsNullOrWhiteSpace($Root))
    {
        return $false
    }

    $include = Join-Path $Root "include\cudnn.h"
    $libX64 = Join-Path $Root "lib\x64\cudnn.lib"
    $lib = Join-Path $Root "lib\cudnn.lib"
    return (Test-Path -LiteralPath $include) -and
           ((Test-Path -LiteralPath $libX64) -or (Test-Path -LiteralPath $lib))
}

function Resolve-CudnnDevRoot
{
    param([Parameter(Mandatory = $true)][string] $SourceRoot)

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($CudnnRoot))
    {
        $candidates += $CudnnRoot
    }

    foreach ($envName in @("CUDNN_ROOT_DIR", "CUDNN", "cudnn"))
    {
        $value = [Environment]::GetEnvironmentVariable($envName)
        if (-not [string]::IsNullOrWhiteSpace($value))
        {
            $candidates += $value
        }
    }

    $candidates += (Join-Path $SourceRoot "build\env\cudnn-cu13")

    foreach ($candidate in $candidates)
    {
        $resolved = Resolve-FullPath $candidate
        if (Test-CudnnDevRoot $resolved)
        {
            return $resolved
        }
    }

    $joined = ($candidates | Select-Object -Unique) -join ", "
    throw "OpenCV DNN CUDA requires cuDNN developer files. Expected include\cudnn.h and lib\x64\cudnn.lib under one of: $joined"
}

function Ensure-CudnnOverlayTriplet
{
    param([Parameter(Mandatory = $true)][string] $SourceRoot)

    $tripletDir = Join-Path $SourceRoot "build\env\vcpkg-triplets"
    New-Item -ItemType Directory -Force -Path $tripletDir | Out-Null

    $tripletFile = Join-Path $tripletDir "x64-windows.cmake"
    @(
        "set(VCPKG_TARGET_ARCHITECTURE x64)",
        "set(VCPKG_CRT_LINKAGE dynamic)",
        "set(VCPKG_LIBRARY_LINKAGE dynamic)",
        "set(VCPKG_ENV_PASSTHROUGH CUDNN_ROOT_DIR CUDNN cudnn)"
    ) | Set-Content -Encoding ASCII -Path $tripletFile

    return (Resolve-FullPath $tripletDir)
}

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

if ([string]::IsNullOrWhiteSpace($TorchRoot))
{
    $TorchRoot = Join-Path $SourceDir "build\env\libtorch-cu130\libtorch"
}
$TorchRoot = Resolve-FullPath $TorchRoot
$VcpkgRoot = Resolve-FullPath $VcpkgRoot
$CudaRoot = Resolve-FullPath $CudaRoot
$CMakeExe = Resolve-FullPath $CMakeExe
$VsDevCmd = Resolve-FullPath $VsDevCmd

$buildRoot = Join-Path $SourceDir "build"
$vcpkgInstalled = Join-Path $BuildDir "vcpkg_installed"
$vcpkgTripletRoot = Join-Path $vcpkgInstalled "x64-windows"
$torchConfigDir = Join-Path $TorchRoot "share\cmake\Torch"
$torchConfig = Join-Path $torchConfigDir "TorchConfig.cmake"
$cudaNvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$vcpkgToolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
$sourceDirCMake = Convert-ToCMakePath $SourceDir
$buildDirCMake = Convert-ToCMakePath $BuildDir
$vcpkgInstalledCMake = Convert-ToCMakePath $vcpkgInstalled
$vcpkgOverlayTripletsCMake = ""
$torchConfigDirCMake = Convert-ToCMakePath $torchConfigDir
$cudaRootCMake = Convert-ToCMakePath $CudaRoot
$cudaNvccCMake = Convert-ToCMakePath $cudaNvcc
$vcpkgToolchainCMake = Convert-ToCMakePath $vcpkgToolchain

if (-not (Test-PathUnder $BuildDir $buildRoot))
{
    throw "BuildDir must stay under project build root. BuildDir=$BuildDir BuildRoot=$buildRoot"
}

Assert-ExistingPath $SourceDir "SourceDir"
Assert-ExistingPath (Join-Path $SourceDir "CMakeLists.txt") "PlaScan CMakeLists.txt"
Assert-ExistingPath $VcpkgRoot "VcpkgRoot"
Assert-ExistingPath $vcpkgToolchain "vcpkg toolchain"
Assert-ExistingPath $CudaRoot "CudaRoot"
Assert-ExistingPath $cudaNvcc "CUDA nvcc"
Assert-ExistingPath $TorchRoot "TorchRoot"
Assert-ExistingPath $torchConfig "TorchConfig.cmake"
Assert-ExistingPath $CMakeExe "CMake"

if ($Jobs -le 0)
{
    $Jobs = [Math]::Max(1, [Environment]::ProcessorCount)
}

if ($CleanRootCache)
{
    $rootCacheItems = @(
        ".cmake",
        ".ninja_deps",
        ".ninja_log",
        ".qt",
        "3rdparty",
        "bin",
        "build.ninja",
        "CMakeCache.txt",
        "CMakeFiles",
        "cmake_install.cmake",
        "compile_commands.json",
        "CPackConfig.cmake",
        "CPackSourceConfig.cmake",
        "CTestTestfile.cmake",
        "detect_cuda_version.cc",
        "install_manifest.txt",
        "src",
        "test",
        "Testing",
        "tests"
    )

    foreach ($item in $rootCacheItems)
    {
        Remove-SafeItem -Path (Join-Path $buildRoot $item) -AllowedRoot $buildRoot
    }
}

if ($CleanConfigure -and (Test-Path -LiteralPath $BuildDir))
{
    Get-ChildItem -LiteralPath $BuildDir -Force | Where-Object { $_.Name -ne "vcpkg_installed" } | ForEach-Object {
        Remove-SafeItem -Path $_.FullName -AllowedRoot $BuildDir
    }
}

$vsDevPathEntries = @()
if (-not $SkipVsDevCmd)
{
    Import-VsDevCmd $VsDevCmd
    $vsDevPathEntries = Capture-VsDevPathEntries
}

Set-IsolatedBuildEnvironment `
    -BuildPath $BuildDir `
    -VcpkgPath $VcpkgRoot `
    -TorchPath $TorchRoot `
    -CudaPath $CudaRoot `
    -CMakePath $CMakeExe `
    -ProjectRoot $SourceDir `
    -VsDevPathEntries $vsDevPathEntries

if ($EnableOpenCvDnnCuda)
{
    $resolvedCudnnRoot = Resolve-CudnnDevRoot $SourceDir
    $vcpkgOverlayTripletsCMake = Convert-ToCMakePath (Ensure-CudnnOverlayTriplet $SourceDir)
    $env:CUDNN_ROOT_DIR = $resolvedCudnnRoot
    $env:CUDNN = $resolvedCudnnRoot
    $env:PATH = (Get-UniqueExistingPathList @(
        (Join-Path $resolvedCudnnRoot "bin"),
        (Join-Path $resolvedCudnnRoot "lib\x64"),
        ($env:PATH -split ';')
    )) -join ';'
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Sync-QtRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
Sync-TorchRuntime -BuildPath $BuildDir -TorchPath $TorchRoot
if (-not $InstallDeps)
{
    Assert-VcpkgInstalledPackages $vcpkgTripletRoot
    if ($EnableOpenCvDnnCuda)
    {
        Assert-OpenCvDnnCudaFeatures $vcpkgTripletRoot
    }
}

Write-Host "PlaScan Windows CUDA build"
Write-Host "  SourceDir: $SourceDir"
Write-Host "  BuildDir:  $BuildDir"
Write-Host "  Vcpkg:     $VcpkgRoot"
Write-Host "  Installed: $vcpkgInstalled"
Write-Host "  CUDA:      $CudaRoot"
if ($EnableOpenCvDnnCuda)
{
    Write-Host "  cuDNN:     $env:CUDNN_ROOT_DIR"
}
Write-Host "  Torch:     $TorchRoot"
Write-Host "  Prefix:    $env:CMAKE_PREFIX_PATH"
Write-Host "  OpenCV DNN CUDA: $(if ($EnableOpenCvDnnCuda) { 'enabled' } else { 'disabled' })"

if (-not $BuildOnly)
{
    $configureArgs = @(
        "-S", $sourceDirCMake,
        "-B", $buildDirCMake,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchainCMake",
        "-DVCPKG_TARGET_TRIPLET=x64-windows",
        "-DVCPKG_INSTALLED_DIR=$vcpkgInstalledCMake",
        "-DVCPKG_MANIFEST_DIR=$sourceDirCMake",
        "-DVCPKG_MANIFEST_INSTALL=$(if ($InstallDeps) { 'ON' } else { 'OFF' })",
        "-DPLASCAN_ENABLE_CONDA=OFF",
        "-DPLASCAN_CONDA_PREFIX=",
        "-DPLASCAN_ENABLE_VCPKG=ON",
        "-DPLASCAN_BUNDLE_RUNTIME=ON",
        "-DBUILD_TESTS=ON",
        "-DPLASCAN_TORCH_DIR=$torchConfigDirCMake",
        "-DTorch_DIR=$torchConfigDirCMake",
        "-DCUDAToolkit_ROOT=$cudaRootCMake",
        "-DCUDA_TOOLKIT_ROOT_DIR=$cudaRootCMake",
        "-DCMAKE_CUDA_COMPILER=$cudaNvccCMake",
        "-DPLASCAN_CUDA_ARCHITECTURES=75;86;89",
        "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON",
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE",
        "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE"
    )
    if ($EnableOpenCvDnnCuda)
    {
        $configureArgs += "-DVCPKG_MANIFEST_FEATURES=opencv-dnn-cuda"
        $configureArgs += "-DVCPKG_OVERLAY_TRIPLETS=$vcpkgOverlayTripletsCMake"
    }

    & $CMakeExe @configureArgs
    if ($LASTEXITCODE -ne 0)
    {
        throw "CMake configure failed with exit code $LASTEXITCODE"
    }
    if ($EnableOpenCvDnnCuda)
    {
        Assert-OpenCvDnnCudaFeatures $vcpkgTripletRoot
    }
}

if (-not $ConfigureOnly)
{
    Sync-TorchRuntime -BuildPath $BuildDir -TorchPath $TorchRoot
    Sync-QtRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot

    $buildArgs = @("--build", $BuildDir, "--config", "Release", "--parallel", "$Jobs")
    if (-not [string]::IsNullOrWhiteSpace($Target))
    {
        $buildArgs += @("--target", $Target)
    }

    & $CMakeExe @buildArgs
    if ($LASTEXITCODE -ne 0)
    {
        throw "CMake build failed with exit code $LASTEXITCODE"
    }

    Sync-TorchRuntime -BuildPath $BuildDir -TorchPath $TorchRoot
    Sync-QtRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
}

if ($RunTests)
{
    $ctestExe = Join-Path (Split-Path -Parent $CMakeExe) "ctest.exe"
    Assert-ExistingPath $ctestExe "CTest"

    $ctestArgs = @("--test-dir", $BuildDir, "-C", "Release", "--output-on-failure")
    if (-not [string]::IsNullOrWhiteSpace($CTestRegex))
    {
        $ctestArgs += @("-R", $CTestRegex)
    }

    & $ctestExe @ctestArgs
    if ($LASTEXITCODE -ne 0)
    {
        throw "CTest failed with exit code $LASTEXITCODE"
    }
}

Write-Host "PlaScan Windows CUDA build script completed."
