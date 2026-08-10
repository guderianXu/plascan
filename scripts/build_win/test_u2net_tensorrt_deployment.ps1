[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $BuildDir,
    [string] $ModelPath = "",
    [string] $TestFilter = "U2NetMaskGeneratorIntegrationTest.OnnxModelRunsOnTensorRtWhenAvailable"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-FullPath
{
    param([Parameter(Mandatory = $true)][string] $Path)
    return [System.IO.Path]::GetFullPath($Path)
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

function Assert-RuntimePattern
{
    param(
        [Parameter(Mandatory = $true)][string] $RuntimeDir,
        [Parameter(Mandatory = $true)][string] $Label,
        [Parameter(Mandatory = $true)][string] $Pattern,
        [string] $NameRegex = ""
    )

    $matches = @(Get-ChildItem -LiteralPath $RuntimeDir -File -Filter $Pattern `
        -ErrorAction SilentlyContinue)
    if (-not [string]::IsNullOrWhiteSpace($NameRegex))
    {
        $matches = @($matches | Where-Object { $_.Name -match $NameRegex })
    }
    if ($matches.Count -eq 0)
    {
        throw "U2Net TensorRT deployment is missing $Label ($Pattern) in: $RuntimeDir"
    }
}

function Assert-OpenCvCpuOnly
{
    param([Parameter(Mandatory = $true)][string] $BuildPath)

    $abiInfo = Join-Path $BuildPath "vcpkg_installed\x64-windows\share\opencv4\vcpkg_abi_info.txt"
    Assert-ExistingPath $abiInfo "OpenCV vcpkg ABI info"
    $featureLine = @(Get-Content -LiteralPath $abiInfo | Where-Object { $_ -match '^features\s+' }) |
        Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($featureLine))
    {
        throw "OpenCV ABI feature list is missing from: $abiInfo"
    }

    $features = @((($featureLine -replace '^features\s+', '') -split ';') |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $forbidden = @(@("cuda", "cudnn", "dnn-cuda") |
        Where-Object { $features -contains $_ })
    if ($forbidden.Count -gt 0)
    {
        throw "OpenCV must be CPU-only, but its vcpkg ABI enables: $($forbidden -join ', ')"
    }
    if ($features -notcontains "dnn")
    {
        throw "OpenCV CPU DNN support is missing from: $abiInfo"
    }
}

$BuildDir = Resolve-FullPath $BuildDir
$runtimeDir = Join-Path $BuildDir "bin"
$testCandidates = @(
    (Join-Path $BuildDir "src\core\mask\tests\test_mask_generation.exe"),
    (Join-Path $BuildDir "tests\test_mask_generation.exe"),
    (Join-Path $runtimeDir "test_mask_generation.exe")
)
$testExe = $testCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($ModelPath))
{
    $sourceRoot = Resolve-FullPath (Join-Path $PSScriptRoot "..\..")
    $ModelPath = Join-Path $sourceRoot "resources\models\U2Net_v1.onnx"
}
$ModelPath = Resolve-FullPath $ModelPath

Assert-ExistingPath $runtimeDir "PlaScan deployment runtime directory"
if ([string]::IsNullOrWhiteSpace($testExe))
{
    throw "U2Net mask test executable was not found; build target test_mask_generation. Checked: " +
        ($testCandidates -join ", ")
}
Assert-ExistingPath $ModelPath "U2Net ONNX model"
Assert-OpenCvCpuOnly $BuildDir

$requirements = @(
    @{ Label = "OpenCV DNN CPU runtime"; Pattern = "opencv_dnn*.dll" },
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
foreach ($requirement in $requirements)
{
    $regex = if ($requirement.ContainsKey("Regex")) { $requirement.Regex } else { "" }
    Assert-RuntimePattern -RuntimeDir $runtimeDir -Label $requirement.Label `
        -Pattern $requirement.Pattern -NameRegex $regex
}

$forbiddenRuntime = @(Get-ChildItem -LiteralPath $runtimeDir -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "cudnn*.dll" -or $_.Extension -in @(".engine", ".plan") })
if ($forbiddenRuntime.Count -gt 0)
{
    throw "Portable U2Net deployment contains forbidden cuDNN or machine-specific engine files: " +
        ($forbiddenRuntime.FullName -join ", ")
}

$systemRoot = [Environment]::GetEnvironmentVariable("SystemRoot")
$tempRoot = [Environment]::GetEnvironmentVariable("TEMP")
if ([string]::IsNullOrWhiteSpace($systemRoot))
{
    throw "SystemRoot is required for the clean deployment environment."
}
if ([string]::IsNullOrWhiteSpace($tempRoot))
{
    $tempRoot = [System.IO.Path]::GetTempPath()
}

$cleanProfile = Join-Path $tempRoot ("plascan-u2net-tensorrt-deployment-" + [Guid]::NewGuid().ToString("N"))
$localAppData = Join-Path $cleanProfile "LocalAppData"
$roamingAppData = Join-Path $cleanProfile "AppData"
New-Item -ItemType Directory -Force -Path $localAppData, $roamingAppData | Out-Null

try
{
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $testExe
    $startInfo.Arguments = "--gtest_color=no --gtest_filter=$TestFilter"
    $startInfo.WorkingDirectory = $BuildDir
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    $startInfo.EnvironmentVariables.Clear()
    $startInfo.EnvironmentVariables["PATH"] = "$runtimeDir;$systemRoot\System32;$systemRoot"
    $startInfo.EnvironmentVariables["SystemRoot"] = $systemRoot
    $startInfo.EnvironmentVariables["WINDIR"] = $systemRoot
    $startInfo.EnvironmentVariables["TEMP"] = $cleanProfile
    $startInfo.EnvironmentVariables["TMP"] = $cleanProfile
    $startInfo.EnvironmentVariables["LOCALAPPDATA"] = $localAppData
    $startInfo.EnvironmentVariables["APPDATA"] = $roamingAppData
    $startInfo.EnvironmentVariables["USERPROFILE"] = $cleanProfile
    $startInfo.EnvironmentVariables["PLASCAN_U2NET_MODEL"] = $ModelPath

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start())
    {
        throw "Failed to start U2Net TensorRT deployment test: $testExe"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $timeoutMilliseconds = 20 * 60 * 1000
    if (-not $process.WaitForExit($timeoutMilliseconds))
    {
        try
        {
            $process.Kill($true)
        }
        catch
        {
            $process.Kill()
        }
        throw "U2Net TensorRT deployment test exceeded the 20-minute timeout."
    }
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()

    $output = ($stdout + [Environment]::NewLine + $stderr).Trim()
    if (-not [string]::IsNullOrWhiteSpace($output))
    {
        Write-Host $output
    }
    if ($process.ExitCode -ne 0)
    {
        throw "U2Net TensorRT deployment test failed with exit code $($process.ExitCode)."
    }

    $passedMarker = "[       OK ] $TestFilter"
    if (-not $output.Contains($passedMarker) -or $output.Contains("[  SKIPPED ]"))
    {
        throw "U2Net TensorRT deployment test did not execute successfully; it may have been skipped."
    }
    if ($output -notmatch '(?i)TensorRT available')
    {
        throw "U2Net deployment test did not report 'TensorRT available'."
    }
    if ($output -notmatch '(?i)(backend|device).{0,40}TensorRT')
    {
        throw "U2Net deployment test did not report an actual TensorRT backend/device."
    }

    Write-Host "U2Net TensorRT clean deployment test passed with CPU-only OpenCV and no external SDK PATH."
}
finally
{
    if ((Test-Path -LiteralPath $cleanProfile) -and
        $cleanProfile.StartsWith((Resolve-FullPath $tempRoot), [System.StringComparison]::OrdinalIgnoreCase))
    {
        Remove-Item -LiteralPath $cleanProfile -Recurse -Force -ErrorAction SilentlyContinue
    }
}
