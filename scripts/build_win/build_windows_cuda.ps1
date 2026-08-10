[CmdletBinding()]
param(
    [string] $SourceDir = "",
    [string] $BuildDir = "",
    [string] $VcpkgRoot = "C:\BuildTools\VC\vcpkg",
    [string] $VcpkgBuildtreesRoot = "",
    [string] $VcpkgPackagesRoot = "",
    [string] $VcpkgDownloadsRoot = "",
    [string] $VsDevCmd = "C:\BuildTools\Common7\Tools\VsDevCmd.bat",
    [string] $CMakeExe = "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    [string] $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [string] $TensorRtRoot = "",
    [string] $Target = "",
    [string] $CTestRegex = "",
    [int] $Jobs = 0,
    [int] $CTestJobs = 0,
    [switch] $ConfigureOnly,
    [switch] $BuildOnly,
    [switch] $RunTests,
    [switch] $RunU2NetTensorRtDeploymentTest,
    [switch] $RunU2NetCudaDeploymentTest,
    [switch] $CleanConfigure,
    [switch] $CleanRootCache,
    [switch] $InstallDeps,
    [bool] $EnableCeresCudaBa = $true,
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

function Invoke-NativeCommand
{
    param(
        [Parameter(Mandatory = $true)][string] $FilePath,
        [string[]] $Arguments = @(),
        [Parameter(Mandatory = $true)][ref] $ExitCode
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try
    {
        $ErrorActionPreference = "Continue"
        & $FilePath @Arguments
        $ExitCode.Value = $LASTEXITCODE
    }
    finally
    {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Resolve-FullPath
{
    param([Parameter(Mandatory = $true)][string] $Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Resolve-ReparseTargetPath
{
    param([Parameter(Mandatory = $true)][string] $Path)

    $fullPath = Resolve-FullPath $Path
    $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction SilentlyContinue
    if ($null -eq $item -or
        ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0 -or
        $null -eq $item.Target)
    {
        return $fullPath
    }

    $target = @($item.Target)[0]
    if (-not [System.IO.Path]::IsPathRooted($target))
    {
        $target = Join-Path $item.Parent.FullName $target
    }
    return Resolve-FullPath $target
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

function Resolve-MsvcCompilerPathEntries
{
    param([Parameter(Mandatory = $true)][string] $VcpkgPath)

    $candidates = New-Object 'System.Collections.Generic.List[string]'
    if (-not [string]::IsNullOrWhiteSpace($env:VCToolsInstallDir))
    {
        [void] $candidates.Add((Join-Path $env:VCToolsInstallDir "bin\Hostx64\x64"))
    }

    $vcpkgParent = Split-Path -Parent $VcpkgPath
    $toolsRoot = Join-Path $vcpkgParent "Tools\MSVC"
    if (Test-Path -LiteralPath $toolsRoot)
    {
        Get-ChildItem -LiteralPath $toolsRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object {
                [void] $candidates.Add((Join-Path $_.FullName "bin\Hostx64\x64"))
            }
    }

    $result = New-Object 'System.Collections.Generic.List[string]'
    foreach ($candidate in (Get-UniqueExistingPathList $candidates.ToArray()))
    {
        if (Test-Path -LiteralPath (Join-Path $candidate "cl.exe"))
        {
            [void] $result.Add((Resolve-FullPath $candidate))
        }
    }
    return $result.ToArray()
}

function Resolve-MsvcRedistRuntimeDlls
{
    param([Parameter(Mandatory = $true)][string] $VcpkgPath)

    $redistRoots = New-Object 'System.Collections.Generic.List[string]'
    if (-not [string]::IsNullOrWhiteSpace($env:VCToolsRedistDir))
    {
        [void] $redistRoots.Add($env:VCToolsRedistDir)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:VCINSTALLDIR))
    {
        [void] $redistRoots.Add((Join-Path $env:VCINSTALLDIR "Redist\MSVC"))
    }

    $vcpkgParent = Split-Path -Parent $VcpkgPath
    [void] $redistRoots.Add((Join-Path $vcpkgParent "Redist\MSVC"))

    $runtimeNames = @(
        "concrt140.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "vcomp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll"
    )

    $result = [ordered]@{}
    foreach ($root in (Get-UniqueExistingPathList $redistRoots.ToArray()))
    {
        if (-not (Test-Path -LiteralPath $root))
        {
            continue
        }

        foreach ($name in $runtimeNames)
        {
            if ($result.Contains($name))
            {
                continue
            }

            $candidate = Get-ChildItem -LiteralPath $root -Recurse -Force -File -Filter $name -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match '\\x64\\' } |
                Sort-Object @{ Expression = { if ($_.FullName -match '\\onecore\\') { 1 } else { 0 } } }, FullName |
                Select-Object -First 1
            if ($candidate)
            {
                $result[$name] = $candidate.FullName
            }
        }
    }

    return @($result.Values)
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

function Resolve-TensorRtSdkRoot
{
    param(
        [string] $RequestedRoot,
        [string] $BuildPath
    )

    $cachedRoot = ""
    if (-not [string]::IsNullOrWhiteSpace($BuildPath))
    {
        $cachePath = Join-Path $BuildPath "CMakeCache.txt"
        if (Test-Path -LiteralPath $cachePath)
        {
            $cacheEntry = @(Get-Content -LiteralPath $cachePath |
                Where-Object { $_ -match '^TensorRT_ROOT:PATH=' }) | Select-Object -First 1
            if (-not [string]::IsNullOrWhiteSpace($cacheEntry))
            {
                $cachedRoot = $cacheEntry -replace '^TensorRT_ROOT:PATH=', ''
            }
        }
    }

    $candidates = @(
        $RequestedRoot,
        [Environment]::GetEnvironmentVariable("TENSORRT_ROOT"),
        [Environment]::GetEnvironmentVariable("TensorRT_ROOT"),
        $cachedRoot
    )
    foreach ($candidate in (Get-UniqueExistingPathList $candidates))
    {
        if (-not (Test-Path -LiteralPath $candidate))
        {
            continue
        }

        $resolved = Resolve-FullPath $candidate
        $includeDir = Join-Path $resolved "include"
        $libraryRoots = @(
            (Join-Path $resolved "lib"),
            (Join-Path $resolved "lib\x64")
        ) | Where-Object { Test-Path -LiteralPath $_ }
        $hasNvInfer = @($libraryRoots | ForEach-Object {
            Get-ChildItem -LiteralPath $_ -File -Filter "nvinfer*.lib" -ErrorAction SilentlyContinue
        }).Count -gt 0
        $hasOnnxParser = @($libraryRoots | ForEach-Object {
            Get-ChildItem -LiteralPath $_ -File -Filter "nvonnxparser*.lib" -ErrorAction SilentlyContinue
        }).Count -gt 0
        if ((Test-Path -LiteralPath (Join-Path $includeDir "NvInferRuntime.h")) -and
            (Test-Path -LiteralPath (Join-Path $includeDir "NvOnnxParser.h")) -and
            $hasNvInfer -and $hasOnnxParser)
        {
            return $resolved
        }
    }

    throw "TensorRT SDK root is required. Pass -TensorRtRoot or set TENSORRT_ROOT to a directory " +
        "containing include, import libraries, runtime DLLs, ONNX parser, and builder resources."
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
        [Parameter(Mandatory = $true)][string] $CudaPath,
        [Parameter(Mandatory = $true)][string] $CMakePath,
        [Parameter(Mandatory = $true)][string] $ProjectRoot,
        [string[]] $VsDevPathEntries = @()
    )

    $vcpkgInstalled = Join-Path $BuildPath "vcpkg_installed"
    $tripletRoot = Join-Path $vcpkgInstalled "x64-windows"
    $vcpkgInstalledCMake = Convert-ToCMakePath $vcpkgInstalled
    $tripletRootCMake = Convert-ToCMakePath $tripletRoot
    $cudaPathCMake = Convert-ToCMakePath $CudaPath
    $qtPluginsRoot = Join-Path $tripletRoot "Qt6\plugins"
    $qtPlatformsRoot = Join-Path $qtPluginsRoot "platforms"
    $qtPluginsRootCMake = Convert-ToCMakePath $qtPluginsRoot
    $qtPlatformsRootCMake = Convert-ToCMakePath $qtPlatformsRoot
    $cmakeBin = Split-Path -Parent $CMakePath
    $ninjaDir = Resolve-NinjaDirectory $CMakePath
    $msvcCompilerPathEntries = Resolve-MsvcCompilerPathEntries $VcpkgPath
    $systemRoot = if (-not [string]::IsNullOrWhiteSpace($env:SystemRoot))
    {
        $env:SystemRoot
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:WINDIR))
    {
        $env:WINDIR
    }
    else
    {
        throw "Windows system root is unavailable; cannot construct an isolated build PATH."
    }
    $system32Dir = Join-Path $systemRoot "System32"
    $tensorRtRuntimeDir = if (-not [string]::IsNullOrWhiteSpace($env:TENSORRT_ROOT))
    {
        Join-Path $env:TENSORRT_ROOT "bin"
    }
    else
    {
        ""
    }

    $env:VCPKG_ROOT = $VcpkgPath
    $env:VCPKG_DEFAULT_TRIPLET = "x64-windows"
    $env:VCPKG_INSTALLED_DIR = $vcpkgInstalledCMake
    $env:CUDAToolkit_ROOT = $cudaPathCMake
    $env:CUDA_PATH = $cudaPathCMake
    $env:CUDA_HOME = $cudaPathCMake
    $env:CUDA_BIN_PATH = "$cudaPathCMake/bin"
    $env:CMAKE_PREFIX_PATH = $tripletRootCMake
    $env:QT_PLUGIN_PATH = $qtPluginsRootCMake
    $env:QT_QPA_PLATFORM_PLUGIN_PATH = $qtPlatformsRootCMake
    $env:QT_QPA_PLATFORM = "offscreen"

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
        $system32Dir,
        $systemRoot,
        (Join-Path $tripletRoot "bin"),
        (Join-Path $tripletRoot "tools\Qt6\bin"),
        (Join-Path $CudaPath "bin"),
        (Join-Path $CudaPath "bin\x64"),
        (Join-Path $CudaPath "nvvm\bin"),
        $tensorRtRuntimeDir,
        $cmakeBin,
        $ninjaDir
    )

    $env:PATH = (Get-UniqueExistingPathList ($prepend + $vsDevPathEntries + $filtered)) -join ';'
    $env:PATH = (Get-UniqueExistingPathList (@($msvcCompilerPathEntries) + @($env:PATH -split ';'))) -join ';'

    $chcpProbe = & (Join-Path $system32Dir "cmd.exe") /d /c "where chcp"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($chcpProbe -join "")))
    {
        throw "The isolated build PATH cannot resolve chcp.com: $env:PATH"
    }
}

function Sync-QtRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $TripletRoot
    )

    $qtPlatformsRoot = Join-Path $TripletRoot "Qt6\plugins\platforms"
    $qtImageFormatsRoot = Join-Path $TripletRoot "Qt6\plugins\imageformats"
    $qtTlsRoot = Join-Path $TripletRoot "Qt6\plugins\tls"
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
        foreach ($platformPlugin in @("qwindows.dll", "qoffscreen.dll", "qminimal.dll"))
        {
            $pluginPath = Join-Path $qtPlatformsRoot $platformPlugin
            if (Test-Path -LiteralPath $pluginPath)
            {
                Copy-Item -LiteralPath $pluginPath -Destination $platformDir -Force
            }
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

        # Qt6 的 HTTPS 实现是运行时插件。没有 tls 目录时 Qt6Network.dll 仍可加载，
        # 但所有 HTTPS 请求都会在握手前以 `TLS initialization failed` 结束。
        if (Test-Path -LiteralPath $qtTlsRoot)
        {
            $tlsDir = Join-Path $dir "tls"
            New-Item -ItemType Directory -Force -Path $tlsDir | Out-Null
            Get-ChildItem -LiteralPath $qtTlsRoot -Force -File -Filter "q*backend.dll" -ErrorAction SilentlyContinue |
                ForEach-Object {
                    Copy-Item -LiteralPath $_.FullName -Destination $tlsDir -Force
                }
        }
    }
}

function Sync-VcpkgRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $TripletRoot
    )

    $vcpkgBin = Join-Path $TripletRoot "bin"
    if (-not (Test-Path -LiteralPath $BuildPath) -or -not (Test-Path -LiteralPath $vcpkgBin))
    {
        return
    }

    $sources = @(Get-ChildItem -LiteralPath $vcpkgBin -Force -File -Filter "*.dll" -ErrorAction SilentlyContinue)
    if ($sources.Count -eq 0)
    {
        return
    }

    function Copy-VcpkgDllIfNeeded
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
    foreach ($relativeDir in @("bin", "tests"))
    {
        $dir = Join-Path $BuildPath $relativeDir
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        [void] $runtimeDirs.Add((Resolve-FullPath $dir))
    }

    foreach ($dir in $runtimeDirs)
    {
        foreach ($source in $sources)
        {
            Copy-VcpkgDllIfNeeded -Source $source.FullName -Destination (Join-Path $dir $source.Name)
        }
    }

    Write-Host ("Synced {0} vcpkg runtime DLLs into {1} runtime director{2}." -f `
        $sources.Count, $runtimeDirs.Count, $(if ($runtimeDirs.Count -eq 1) { "y" } else { "ies" }))
}

function Sync-MsvcRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $VcpkgPath
    )

    if (-not (Test-Path -LiteralPath $BuildPath))
    {
        return
    }

    $sources = @(Resolve-MsvcRedistRuntimeDlls $VcpkgPath)
    if ($sources.Count -eq 0)
    {
        return
    }

    function Copy-MsvcDllIfNeeded
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
    foreach ($relativeDir in @("bin", "tests"))
    {
        $dir = Join-Path $BuildPath $relativeDir
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        [void] $runtimeDirs.Add((Resolve-FullPath $dir))
    }

    foreach ($dir in $runtimeDirs)
    {
        foreach ($source in $sources)
        {
            Copy-MsvcDllIfNeeded -Source $source -Destination (Join-Path $dir (Split-Path -Leaf $source))
        }
    }

    Write-Host ("Synced {0} MSVC runtime DLLs into {1} runtime director{2}." -f `
        $sources.Count, $runtimeDirs.Count, $(if ($runtimeDirs.Count -eq 1) { "y" } else { "ies" }))
}

function Sync-CudaRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $CudaPath
    )

    if (-not (Test-Path -LiteralPath $BuildPath) -or -not (Test-Path -LiteralPath $CudaPath))
    {
        return
    }

    $cudaDllRoots = @(
        (Join-Path $CudaPath "bin"),
        (Join-Path $CudaPath "bin\x64")
    ) | Where-Object { Test-Path -LiteralPath $_ }

    $sources = @{}
    foreach ($root in $cudaDllRoots)
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

    function Copy-CudaDllIfNeeded
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
    foreach ($relativeDir in @("bin", "tests"))
    {
        $dir = Join-Path $BuildPath $relativeDir
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        [void] $runtimeDirs.Add((Resolve-FullPath $dir))
    }

    $sourceEntries = $sources.GetEnumerator() | Sort-Object Name
    foreach ($dir in $runtimeDirs)
    {
        foreach ($entry in $sourceEntries)
        {
            Copy-CudaDllIfNeeded -Source $entry.Value -Destination (Join-Path $dir $entry.Name)
        }
    }

    Write-Host ("Synced {0} CUDA runtime DLLs into {1} runtime director{2}." -f `
        $sources.Count, $runtimeDirs.Count, $(if ($runtimeDirs.Count -eq 1) { "y" } else { "ies" }))
}

function Sync-OpenCvRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $TripletRoot
    )

    $opencvBin = Join-Path $TripletRoot "bin"
    if (-not (Test-Path -LiteralPath $BuildPath) -or -not (Test-Path -LiteralPath $opencvBin))
    {
        return
    }

    $sources = @(Get-ChildItem -LiteralPath $opencvBin -Force -File -Filter "opencv*.dll" -ErrorAction SilentlyContinue)
    if ($sources.Count -eq 0)
    {
        return
    }

    function Copy-OpenCvDllIfNeeded
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
    foreach ($relativeDir in @("bin", "tests"))
    {
        $dir = Join-Path $BuildPath $relativeDir
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        [void] $runtimeDirs.Add((Resolve-FullPath $dir))
    }

    foreach ($dir in $runtimeDirs)
    {
        foreach ($source in $sources)
        {
            Copy-OpenCvDllIfNeeded -Source $source.FullName -Destination (Join-Path $dir $source.Name)
        }
    }

    Write-Host ("Synced {0} OpenCV runtime DLLs into {1} runtime director{2}." -f `
        $sources.Count, $runtimeDirs.Count, $(if ($runtimeDirs.Count -eq 1) { "y" } else { "ies" }))
}

function Sync-TensorRtRuntime
{
    param(
        [Parameter(Mandatory = $true)][string] $BuildPath,
        [Parameter(Mandatory = $true)][string] $TensorRtPath
    )

    $tensorRtDllRoots = @(@(
        (Join-Path $TensorRtPath "bin"),
        (Join-Path $TensorRtPath "lib"),
        (Join-Path $TensorRtPath "lib\x64")
    ) | Where-Object { Test-Path -LiteralPath $_ })
    if ($tensorRtDllRoots.Count -eq 0)
    {
        throw "TensorRT runtime directory not found under: $TensorRtPath"
    }

    $patterns = @(
        "nvinfer.dll",
        "nvinfer_*.dll",
        "nvinfer_plugin.dll",
        "nvinfer_plugin_*.dll",
        "nvinfer_builder_resource_*.dll",
        "nvonnxparser.dll",
        "nvonnxparser_*.dll"
    )
    $sources = [ordered]@{}
    foreach ($root in $tensorRtDllRoots)
    {
        foreach ($pattern in $patterns)
        {
            Get-ChildItem -LiteralPath $root -Force -File -Filter $pattern -ErrorAction SilentlyContinue |
                ForEach-Object {
                    if (-not $sources.Contains($_.Name))
                    {
                        $sources[$_.Name] = $_.FullName
                    }
                }
        }
    }

    $requiredPatterns = @(
        @{ Label = "TensorRT runtime"; Pattern = "^nvinfer(_[0-9]+)?\.dll$" },
        @{ Label = "TensorRT ONNX parser"; Pattern = "^nvonnxparser(_[0-9]+)?\.dll$" },
        @{ Label = "TensorRT plugin runtime"; Pattern = "^nvinfer_plugin(_[0-9]+)?\.dll$" },
        @{ Label = "TensorRT builder resources"; Pattern = "^nvinfer_builder_resource_.+\.dll$" }
    )
    $missing = New-Object 'System.Collections.Generic.List[string]'
    foreach ($requirement in $requiredPatterns)
    {
        $matches = @($sources.Keys | Where-Object { $_ -match $requirement.Pattern })
        if ($matches.Count -eq 0)
        {
            [void] $missing.Add($requirement.Label)
        }
    }
    if ($missing.Count -gt 0)
    {
        throw "TensorRT SDK is incomplete under $TensorRtPath. Missing: $($missing -join ', ')"
    }

    function Copy-TensorRtDllIfNeeded
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
            $copy = ($src.Length -ne $dst.Length) -or
                ($src.LastWriteTimeUtc -gt $dst.LastWriteTimeUtc)
        }
        if ($copy)
        {
            Copy-Item -LiteralPath $Source -Destination $Destination -Force
        }
    }

    $runtimeDirs = @(
        (Join-Path $BuildPath "bin"),
        (Join-Path $BuildPath "tests")
    )
    foreach ($dir in $runtimeDirs)
    {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        foreach ($entry in $sources.GetEnumerator())
        {
            $destination = Join-Path $dir $entry.Name
            Copy-TensorRtDllIfNeeded -Source $entry.Value -Destination $destination
            if (-not (Test-Path -LiteralPath $destination))
            {
                throw "Failed to deploy TensorRT runtime DLL: $destination"
            }
        }
    }

    Write-Host ("Synced {0} TensorRT runtime DLLs into bin and tests." -f $sources.Count)
}

function Remove-StaleCudnnRuntime
{
    param([Parameter(Mandatory = $true)][string] $BuildPath)

    $removed = 0
    foreach ($relativeDir in @("bin", "tests"))
    {
        $runtimeDir = Join-Path $BuildPath $relativeDir
        if (-not (Test-Path -LiteralPath $runtimeDir))
        {
            continue
        }

        Get-ChildItem -LiteralPath $runtimeDir -Recurse -Force -File -Filter "cudnn*.dll" `
            -ErrorAction SilentlyContinue | ForEach-Object {
                if (-not (Test-PathUnder $_.FullName $BuildPath))
                {
                    throw "Refusing to remove stale cuDNN DLL outside build directory: $($_.FullName)"
                }
                Remove-Item -LiteralPath $_.FullName -Force
                $removed++
            }
    }
    if ($removed -gt 0)
    {
        Write-Host "Removed $removed stale cuDNN runtime DLL(s); OpenCV is CPU-only."
    }
}

function Assert-U2NetTensorRtDeployment
{
    param([Parameter(Mandatory = $true)][string] $BuildPath)

    $runtimeDir = Join-Path $BuildPath "bin"
    Assert-ExistingPath $runtimeDir "PlaScan runtime directory"

    $requirements = @(
        @{ Label = "OpenCV core"; Pattern = "opencv_core*.dll" },
        @{ Label = "OpenCV DNN CPU"; Pattern = "opencv_dnn*.dll" },
        @{ Label = "TensorRT runtime"; Pattern = "nvinfer_*.dll"; Regex = "^nvinfer_[0-9]+\.dll$" },
        @{ Label = "TensorRT ONNX parser"; Pattern = "nvonnxparser_*.dll"; Regex = "^nvonnxparser_[0-9]+\.dll$" },
        @{ Label = "TensorRT plugin runtime"; Pattern = "nvinfer_plugin_*.dll" },
        @{ Label = "TensorRT builder resource"; Pattern = "nvinfer_builder_resource_*.dll" },
        @{ Label = "CUDA runtime"; Pattern = "cudart64_*.dll" },
        @{ Label = "cuBLAS"; Pattern = "cublas64_*.dll" },
        @{ Label = "cuBLAS Lt"; Pattern = "cublasLt64_*.dll" },
        @{ Label = "NVRTC"; Pattern = "nvrtc64_*.dll" },
        @{ Label = "NVRTC builtins"; Pattern = "nvrtc-builtins64_*.dll" },
        @{ Label = "nvFatbin"; Pattern = "nvfatbin_*.dll" }
    )

    $missing = New-Object 'System.Collections.Generic.List[string]'
    foreach ($requirement in $requirements)
    {
        $matches = @(Get-ChildItem -LiteralPath $runtimeDir -File -Filter $requirement.Pattern `
            -ErrorAction SilentlyContinue)
        if ($requirement.ContainsKey("Regex"))
        {
            $matches = @($matches | Where-Object { $_.Name -match $requirement.Regex })
        }
        if ($matches.Count -eq 0)
        {
            [void] $missing.Add("$($requirement.Label) ($($requirement.Pattern))")
        }
    }
    if ($missing.Count -gt 0)
    {
        throw "U2Net TensorRT deployment is incomplete in $runtimeDir. Missing: $($missing -join ', ')"
    }

    $cudnnDlls = @(Get-ChildItem -LiteralPath $runtimeDir -Recurse -File -Filter "cudnn*.dll" `
        -ErrorAction SilentlyContinue)
    if ($cudnnDlls.Count -gt 0)
    {
        throw "U2Net TensorRT deployment must not contain cuDNN DLLs: $($cudnnDlls.FullName -join ', ')"
    }

    Write-Host "Validated CPU-only OpenCV + U2Net TensorRT runtime deployment in $runtimeDir."
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

function Assert-OpenCvCpuOnlyFeatures
{
    param([Parameter(Mandatory = $true)][string] $TripletRoot)

    $abiInfo = Join-Path $TripletRoot "share\opencv4\vcpkg_abi_info.txt"
    if (-not (Test-Path -LiteralPath $abiInfo))
    {
        throw "OpenCV ABI info not found: $abiInfo. Rerun this script with -InstallDeps."
    }

    $featureLine = @(Get-Content -LiteralPath $abiInfo | Where-Object { $_ -match '^features\s+' }) |
        Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($featureLine))
    {
        throw "OpenCV ABI feature list is missing from: $abiInfo"
    }

    $features = @((($featureLine -replace '^features\s+', '') -split ';') |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($features -notcontains "dnn")
    {
        throw "OpenCV DNN CPU support is missing in $TripletRoot. Rerun this script with -InstallDeps."
    }

    $forbidden = @(@("cuda", "cudnn", "dnn-cuda") |
        Where-Object { $features -contains $_ })
    if ($forbidden.Count -gt 0)
    {
        throw "OpenCV must remain CPU-only, but its vcpkg ABI enables: $($forbidden -join ', '). " +
            "Remove the stale vcpkg_installed directory or rerun this script with -InstallDeps."
    }
}

function Assert-CeresCudaFeatures
{
    param([Parameter(Mandatory = $true)][string] $TripletRoot)

    $abiInfo = Join-Path $TripletRoot "share\ceres\vcpkg_abi_info.txt"
    if (-not (Test-Path -LiteralPath $abiInfo))
    {
        throw "Ceres ABI info not found: $abiInfo. Rerun with -InstallDeps -EnableCeresCudaBa:`$true."
    }

    $text = Get-Content -LiteralPath $abiInfo -Raw
    if ($text -notmatch "(?m)^features .*cuda" -or
        $text -notmatch "(?m)^features .*lapack" -or
        $text -notmatch "(?m)^features .*suitesparse")
    {
        throw "Ceres CUDA/LAPACK/SuiteSparse is not installed in $TripletRoot. Rerun this script with -InstallDeps -EnableCeresCudaBa:`$true."
    }
}

function Find-VcpkgPortDirectory
{
    param(
        [Parameter(Mandatory = $true)][string] $VcpkgPath,
        [Parameter(Mandatory = $true)][string] $PortName
    )

    $candidates = New-Object 'System.Collections.Generic.List[string]'
    $builtinPort = Join-Path $VcpkgPath ("ports\" + $PortName)
    if (Test-Path -LiteralPath (Join-Path $builtinPort "portfile.cmake"))
    {
        [void] $candidates.Add($builtinPort)
    }

    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    if (-not [string]::IsNullOrWhiteSpace($localAppData))
    {
        $registryRoot = Join-Path $localAppData "vcpkg\registries\git-trees"
        if (Test-Path -LiteralPath $registryRoot)
        {
            Get-ChildItem -LiteralPath $registryRoot -Directory | ForEach-Object {
                $manifestPath = Join-Path $_.FullName "vcpkg.json"
                $portfilePath = Join-Path $_.FullName "portfile.cmake"
                if (-not (Test-Path -LiteralPath $manifestPath) -or
                    -not (Test-Path -LiteralPath $portfilePath))
                {
                    return
                }

                try
                {
                    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
                    if ($manifest.name -eq $PortName)
                    {
                        [void] $candidates.Add($_.FullName)
                    }
                }
                catch
                {
                }
            }
        }
    }

    foreach ($candidate in (Get-UniqueExistingPathList $candidates.ToArray()))
    {
        if ((Test-Path -LiteralPath (Join-Path $candidate "portfile.cmake")) -and
            (Test-Path -LiteralPath (Join-Path $candidate "vcpkg.json")))
        {
            return (Resolve-FullPath $candidate)
        }
    }

    throw "Unable to locate vcpkg port '$PortName'. Run vcpkg manifest install once, or seed the vcpkg registry cache."
}

function Ensure-CeresCuda13OverlayPort
{
    param(
        [Parameter(Mandatory = $true)][string] $SourceRoot,
        [Parameter(Mandatory = $true)][string] $VcpkgPath
    )

    # Keep the Ceres overlay isolated. Reusing the historical shared overlay root
    # could make a stale OpenCV CUDA port override the CPU-only manifest port.
    $overlayRoot = Join-Path $SourceRoot "build\env\vcpkg-overlay-ports-ceres"
    New-Item -ItemType Directory -Force -Path $overlayRoot | Out-Null

    $overlayPort = Join-Path $overlayRoot "ceres"
    Remove-SafeItem -Path $overlayPort -AllowedRoot $overlayRoot

    $portSource = Find-VcpkgPortDirectory -VcpkgPath $VcpkgPath -PortName "ceres"
    Copy-Item -LiteralPath $portSource -Destination $overlayPort -Recurse -Force

    $portfile = Join-Path $overlayPort "portfile.cmake"
    $portfileText = Get-Content -LiteralPath $portfile -Raw
    if ($portfileText -notmatch "CMAKE_CUDA_STANDARD=17")
    {
        $marker = '        "-DCUDAToolkit_ROOT=${cuda_toolkit_root}"'
        if (-not $portfileText.Contains($marker))
        {
            throw "Unable to inject Ceres CUDA 13 build flags into generated overlay port: $portfile"
        }

        $cudaFlagsBlock = @'
        "-DCMAKE_CUDA_STANDARD=17"
        "-DCMAKE_CUDA_STANDARD_REQUIRED=ON"
        "-DCMAKE_CUDA_FLAGS=--std=c++17"
        "-DCMAKE_CUDA_ARCHITECTURES=75\;86\;89\;120"
'@
        $portfileText = $portfileText.Replace($marker, "$marker`r`n$cudaFlagsBlock")
        Set-Content -LiteralPath $portfile -Encoding ASCII -Value $portfileText
    }

    return (Resolve-FullPath $overlayRoot)
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

$vcpkgWorkDrive = Split-Path -Qualifier $BuildDir
if ([string]::IsNullOrWhiteSpace($vcpkgWorkDrive))
{
    throw "BuildDir must have a Windows drive qualifier so short vcpkg work roots can be derived: $BuildDir"
}
if ([string]::IsNullOrWhiteSpace($VcpkgBuildtreesRoot))
{
    $VcpkgBuildtreesRoot = Join-Path $vcpkgWorkDrive "vbt"
}
if ([string]::IsNullOrWhiteSpace($VcpkgPackagesRoot))
{
    $VcpkgPackagesRoot = Join-Path $vcpkgWorkDrive "vpk"
}
if ([string]::IsNullOrWhiteSpace($VcpkgDownloadsRoot))
{
    $VcpkgDownloadsRoot = Join-Path $vcpkgWorkDrive "vdl"
}

$VcpkgRoot = Resolve-FullPath $VcpkgRoot
$VcpkgBuildtreesRoot = Resolve-ReparseTargetPath $VcpkgBuildtreesRoot
$VcpkgPackagesRoot = Resolve-ReparseTargetPath $VcpkgPackagesRoot
$VcpkgDownloadsRoot = Resolve-ReparseTargetPath $VcpkgDownloadsRoot
$CudaRoot = Resolve-FullPath $CudaRoot
$TensorRtRoot = Resolve-TensorRtSdkRoot -RequestedRoot $TensorRtRoot -BuildPath $BuildDir
$env:TENSORRT_ROOT = $TensorRtRoot
$CMakeExe = Resolve-FullPath $CMakeExe
$VsDevCmd = Resolve-FullPath $VsDevCmd
$ninjaDir = Resolve-NinjaDirectory $CMakeExe
if ([string]::IsNullOrWhiteSpace($ninjaDir))
{
    throw "Ninja executable was not found beside CMake or on PATH."
}
$ninjaExe = Join-Path $ninjaDir "ninja.exe"

$buildRoot = Join-Path $SourceDir "build"
$vcpkgInstalled = Join-Path $BuildDir "vcpkg_installed"
$vcpkgTripletRoot = Join-Path $vcpkgInstalled "x64-windows"
$cudaNvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$vcpkgToolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
$sourceDirCMake = Convert-ToCMakePath $SourceDir
$buildDirCMake = Convert-ToCMakePath $BuildDir
$vcpkgInstalledCMake = Convert-ToCMakePath $vcpkgInstalled
$vcpkgBuildtreesRootCMake = Convert-ToCMakePath $VcpkgBuildtreesRoot
$vcpkgPackagesRootCMake = Convert-ToCMakePath $VcpkgPackagesRoot
$vcpkgDownloadsRootCMake = Convert-ToCMakePath $VcpkgDownloadsRoot
$vcpkgOverlayPortsCMake = ""
$cudaRootCMake = Convert-ToCMakePath $CudaRoot
$cudaNvccCMake = Convert-ToCMakePath $cudaNvcc
$tensorRtRootCMake = Convert-ToCMakePath $TensorRtRoot
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
Assert-ExistingPath $CMakeExe "CMake"

foreach ($vcpkgWorkRoot in @($VcpkgBuildtreesRoot, $VcpkgPackagesRoot, $VcpkgDownloadsRoot))
{
    New-Item -ItemType Directory -Force -Path $vcpkgWorkRoot | Out-Null
}

if ($Jobs -le 0)
{
    $Jobs = [Math]::Max(1, [Environment]::ProcessorCount)
}

if ($CTestJobs -le 0)
{
    $configuredCTestJobs = 0
    if (-not [string]::IsNullOrWhiteSpace($env:CTEST_PARALLEL_LEVEL))
    {
        [void] [int]::TryParse($env:CTEST_PARALLEL_LEVEL, [ref] $configuredCTestJobs)
    }
    if ($configuredCTestJobs -gt 0)
    {
        $CTestJobs = $configuredCTestJobs
    }
    else
    {
        $CTestJobs = [Math]::Max(1, [int] [Math]::Floor([Environment]::ProcessorCount / 2))
    }
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

$windowsSdkRc = (Get-Command rc.exe -ErrorAction SilentlyContinue).Source
$windowsSdkMt = (Get-Command mt.exe -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($windowsSdkRc) -or [string]::IsNullOrWhiteSpace($windowsSdkMt))
{
    throw "Windows SDK rc.exe and mt.exe must be available after loading VsDevCmd."
}

Set-IsolatedBuildEnvironment `
    -BuildPath $BuildDir `
    -VcpkgPath $VcpkgRoot `
    -CudaPath $CudaRoot `
    -CMakePath $CMakeExe `
    -ProjectRoot $SourceDir `
    -VsDevPathEntries $vsDevPathEntries

$msvcCompilerPathEntries = @(Resolve-MsvcCompilerPathEntries $VcpkgRoot)
$msvcCudaHostCompiler = ""
$msvcCudaHostCompilerDir = ""
if ($msvcCompilerPathEntries.Count -gt 0)
{
    $candidateCompiler = Join-Path $msvcCompilerPathEntries[0] "cl.exe"
    if (Test-Path -LiteralPath $candidateCompiler)
    {
        $msvcCudaHostCompiler = Convert-ToCMakePath (Resolve-FullPath $candidateCompiler)
        $msvcCudaHostCompilerDir = Convert-ToCMakePath (Resolve-FullPath $msvcCompilerPathEntries[0])
        $env:CUDAHOSTCXX = $msvcCudaHostCompiler
    }
}

if ($EnableCeresCudaBa)
{
    $vcpkgOverlayPortsCMake = Convert-ToCMakePath (Ensure-CeresCuda13OverlayPort -SourceRoot $SourceDir -VcpkgPath $VcpkgRoot)
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Sync-VcpkgRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
Sync-MsvcRuntime -BuildPath $BuildDir -VcpkgPath $VcpkgRoot
Sync-CudaRuntime -BuildPath $BuildDir -CudaPath $CudaRoot
Sync-TensorRtRuntime -BuildPath $BuildDir -TensorRtPath $TensorRtRoot
Sync-QtRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
Sync-OpenCvRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
Remove-StaleCudnnRuntime -BuildPath $BuildDir
if (-not $InstallDeps)
{
    Assert-VcpkgInstalledPackages $vcpkgTripletRoot
    Assert-OpenCvCpuOnlyFeatures $vcpkgTripletRoot
    if ($EnableCeresCudaBa)
    {
        Assert-CeresCudaFeatures $vcpkgTripletRoot
    }
}

Write-Host "PlaScan Windows CUDA build"
Write-Host "  SourceDir: $SourceDir"
Write-Host "  BuildDir:  $BuildDir"
Write-Host "  Vcpkg:     $VcpkgRoot"
Write-Host "  Installed: $vcpkgInstalled"
Write-Host "  Buildtrees: $VcpkgBuildtreesRoot"
Write-Host "  Packages:   $VcpkgPackagesRoot"
Write-Host "  Downloads:  $VcpkgDownloadsRoot"
Write-Host "  CUDA:      $CudaRoot"
Write-Host "  TensorRT:  $TensorRtRoot"
if (-not [string]::IsNullOrWhiteSpace($vcpkgOverlayPortsCMake))
{
    Write-Host "  Overlay:   $vcpkgOverlayPortsCMake"
}
if (-not [string]::IsNullOrWhiteSpace($msvcCudaHostCompiler))
{
    Write-Host "  CUDA host: $msvcCudaHostCompiler"
}
Write-Host "  Prefix:    $env:CMAKE_PREFIX_PATH"
Write-Host "  Ceres BA CUDA: $(if ($EnableCeresCudaBa) { 'enabled' } else { 'disabled' })"
Write-Host "  OpenCV DNN: CPU-only"
Write-Host "  U2Net GPU: TensorRT"

if (-not $BuildOnly)
{
    $vcpkgInstallOptions = @(
        "--x-buildtrees-root=$vcpkgBuildtreesRootCMake",
        "--x-packages-root=$vcpkgPackagesRootCMake",
        "--downloads-root=$vcpkgDownloadsRootCMake"
    ) -join ";"
    $configureArgs = @(
        "-S", $sourceDirCMake,
        "-B", $buildDirCMake,
        "-G", "Ninja",
        "-DCMAKE_MAKE_PROGRAM=$(Convert-ToCMakePath $ninjaExe)",
        "-DCMAKE_RC_COMPILER=$(Convert-ToCMakePath $windowsSdkRc)",
        "-DCMAKE_MT=$(Convert-ToCMakePath $windowsSdkMt)",
        "-UVCPKG_MANIFEST_FEATURES",
        "-UVCPKG_OVERLAY_PORTS",
        "-UQt6*_DIR",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchainCMake",
        "-DVCPKG_TARGET_TRIPLET=x64-windows",
        "-DVCPKG_INSTALLED_DIR=$vcpkgInstalledCMake",
        "-DVCPKG_MANIFEST_DIR=$sourceDirCMake",
        "-DVCPKG_MANIFEST_INSTALL=$(if ($InstallDeps) { 'ON' } else { 'OFF' })",
        "-DVCPKG_INSTALL_OPTIONS=$vcpkgInstallOptions",
        "-DCMAKE_PREFIX_PATH=$env:CMAKE_PREFIX_PATH",
        "-DVCPKG_APPLOCAL_DEPS=OFF",
        "-DPLASCAN_ENABLE_CONDA=OFF",
        "-DPLASCAN_CONDA_PREFIX=",
        "-DPLASCAN_ENABLE_VCPKG=ON",
        "-DPLASCAN_BUNDLE_RUNTIME=ON",
        "-DBUILD_TESTS=ON",
        "-DPLASCAN_ENABLE_TENSORRT=ON",
        "-DTensorRT_ROOT=$tensorRtRootCMake",
        "-DCUDAToolkit_ROOT=$cudaRootCMake",
        "-DCUDA_TOOLKIT_ROOT_DIR=$cudaRootCMake",
        "-DCMAKE_CUDA_COMPILER=$cudaNvccCMake",
        "-DPLASCAN_CUDA_ARCHITECTURES=75;86;89;120",
        "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON",
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE",
        "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE"
    )
    if (-not [string]::IsNullOrWhiteSpace($msvcCudaHostCompiler))
    {
        $configureArgs += "-DCMAKE_CXX_COMPILER=$msvcCudaHostCompiler"
        $configureArgs += "-DCMAKE_CUDA_HOST_COMPILER=$msvcCudaHostCompiler"
    }
    if (-not [string]::IsNullOrWhiteSpace($msvcCudaHostCompilerDir))
    {
        $configureArgs += "-DCMAKE_CUDA_FLAGS=--compiler-bindir=$msvcCudaHostCompilerDir"
    }
    $manifestFeaturesValue = ""
    if ($EnableCeresCudaBa)
    {
        $manifestFeaturesValue = "ceres-cuda"
    }
    if (-not [string]::IsNullOrWhiteSpace($vcpkgOverlayPortsCMake))
    {
        $configureArgs += "-DVCPKG_OVERLAY_PORTS=$vcpkgOverlayPortsCMake"
    }
    if (-not [string]::IsNullOrWhiteSpace($manifestFeaturesValue))
    {
        $configureArgs += "-DVCPKG_MANIFEST_FEATURES=$manifestFeaturesValue"
    }

    $configureExitCode = 0
    Invoke-NativeCommand -FilePath $CMakeExe -Arguments $configureArgs -ExitCode ([ref]$configureExitCode)
    if ($configureExitCode -ne 0)
    {
        throw "CMake configure failed with exit code $configureExitCode"
    }
    Assert-OpenCvCpuOnlyFeatures $vcpkgTripletRoot
    if ($EnableCeresCudaBa)
    {
        Assert-CeresCudaFeatures $vcpkgTripletRoot
    }
}

if (-not $ConfigureOnly)
{
    Sync-VcpkgRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
    Sync-MsvcRuntime -BuildPath $BuildDir -VcpkgPath $VcpkgRoot
    Sync-CudaRuntime -BuildPath $BuildDir -CudaPath $CudaRoot
    Sync-TensorRtRuntime -BuildPath $BuildDir -TensorRtPath $TensorRtRoot
    Sync-QtRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
    Sync-OpenCvRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
    Remove-StaleCudnnRuntime -BuildPath $BuildDir
    Assert-OpenCvCpuOnlyFeatures $vcpkgTripletRoot

    $buildArgs = @("--build", $BuildDir, "--config", "Release", "--parallel", "$Jobs")
    if (-not [string]::IsNullOrWhiteSpace($Target))
    {
        $buildArgs += @("--target", $Target)
    }

    $buildExitCode = 0
    Invoke-NativeCommand -FilePath $CMakeExe -Arguments $buildArgs -ExitCode ([ref]$buildExitCode)
    if ($buildExitCode -ne 0)
    {
        throw "CMake build failed with exit code $buildExitCode"
    }

    Sync-VcpkgRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
    Sync-MsvcRuntime -BuildPath $BuildDir -VcpkgPath $VcpkgRoot
    Sync-CudaRuntime -BuildPath $BuildDir -CudaPath $CudaRoot
    Sync-TensorRtRuntime -BuildPath $BuildDir -TensorRtPath $TensorRtRoot
    Sync-QtRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
    Sync-OpenCvRuntime -BuildPath $BuildDir -TripletRoot $vcpkgTripletRoot
    Remove-StaleCudnnRuntime -BuildPath $BuildDir
    Assert-U2NetTensorRtDeployment -BuildPath $BuildDir
}

if ($RunU2NetCudaDeploymentTest)
{
    Write-Warning "RunU2NetCudaDeploymentTest is deprecated; using the TensorRT deployment test."
    $RunU2NetTensorRtDeploymentTest = $true
}

if ($RunU2NetTensorRtDeploymentTest)
{
    if ($ConfigureOnly)
    {
        throw "RunU2NetTensorRtDeploymentTest requires a build; do not combine it with ConfigureOnly."
    }

    $deploymentTestScript = Join-Path $SourceDir "scripts\build_win\test_u2net_tensorrt_deployment.ps1"
    Assert-ExistingPath $deploymentTestScript "U2Net TensorRT deployment test script"
    & $deploymentTestScript -BuildDir $BuildDir `
        -ModelPath (Join-Path $SourceDir "resources\models\U2Net_v1.onnx")
}

if ($RunTests)
{
    $ctestExe = Join-Path (Split-Path -Parent $CMakeExe) "ctest.exe"
    Assert-ExistingPath $ctestExe "CTest"

    $ctestArgs = @(
        "--test-dir", $BuildDir,
        "-C", "Release",
        "--output-on-failure",
        "--parallel", "$CTestJobs"
    )
    if (-not [string]::IsNullOrWhiteSpace($CTestRegex))
    {
        $ctestArgs += @("-R", $CTestRegex)
    }

    $ctestExitCode = 0
    Invoke-NativeCommand -FilePath $ctestExe -Arguments $ctestArgs -ExitCode ([ref]$ctestExitCode)
    if ($ctestExitCode -ne 0)
    {
        throw "CTest failed with exit code $ctestExitCode"
    }
}

Write-Host "PlaScan Windows CUDA build script completed."
