[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $BuildDir,
    [string] $InstallRoot = "",
    [string] $TestFilter =
        "BiRefNetMaskGeneratorIntegrationTest.OnnxModelRunsOnTensorRtWhenExplicitlyEnabled",
    [ValidateRange(1, 180)][int] $TimeoutMinutes = 60
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

function Test-PathUnder
{
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Parent
    )

    $resolvedPath = (Resolve-FullPath $Path).TrimEnd('\', '/')
    $resolvedParent = (Resolve-FullPath $Parent).TrimEnd('\', '/')
    return $resolvedPath.StartsWith(
        $resolvedParent + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)
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
        throw "BiRefNet deployment is missing $Label ($Pattern) in: $RuntimeDir"
    }
}

function Assert-NoEngineArtifacts
{
    param(
        [Parameter(Mandatory = $true)][string] $Root,
        [Parameter(Mandatory = $true)][string] $Label
    )

    $artifacts = @(Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".engine", ".plan") })
    if ($artifacts.Count -gt 0)
    {
        throw "$Label contains machine-specific TensorRT artifacts: $($artifacts.FullName -join ', ')"
    }
}

function Invoke-BiRefNetIntegrationTest
{
    param(
        [Parameter(Mandatory = $true)][string] $TestExe,
        [Parameter(Mandatory = $true)][string] $WorkingDirectory,
        [Parameter(Mandatory = $true)][string] $TestHarnessDir,
        [Parameter(Mandatory = $true)][string] $RuntimeDir,
        [Parameter(Mandatory = $true)][string] $ModelPath,
        [Parameter(Mandatory = $true)][string] $EngineCache,
        [Parameter(Mandatory = $true)][string] $CleanProfile,
        [Parameter(Mandatory = $true)][string] $SystemRoot,
        [Parameter(Mandatory = $true)][string] $ExpectedReuse,
        [Parameter(Mandatory = $true)][string] $PassLabel
    )

    $localAppData = Join-Path $CleanProfile "LocalAppData"
    $roamingAppData = Join-Path $CleanProfile "AppData"
    New-Item -ItemType Directory -Force -Path $localAppData, $roamingAppData, $EngineCache |
        Out-Null

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $TestExe
    $startInfo.Arguments = "--gtest_color=no --gtest_filter=$TestFilter"
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    $startInfo.EnvironmentVariables.Clear()
    $startInfo.EnvironmentVariables["PATH"] =
        "$TestHarnessDir;$RuntimeDir;$SystemRoot\System32;$SystemRoot"
    $startInfo.EnvironmentVariables["SystemRoot"] = $SystemRoot
    $startInfo.EnvironmentVariables["WINDIR"] = $SystemRoot
    $startInfo.EnvironmentVariables["TEMP"] = $CleanProfile
    $startInfo.EnvironmentVariables["TMP"] = $CleanProfile
    $startInfo.EnvironmentVariables["LOCALAPPDATA"] = $localAppData
    $startInfo.EnvironmentVariables["APPDATA"] = $roamingAppData
    $startInfo.EnvironmentVariables["USERPROFILE"] = $CleanProfile
    $startInfo.EnvironmentVariables["PLASCAN_BIREFNET_INTEGRATION"] = "1"
    $startInfo.EnvironmentVariables["PLASCAN_BIREFNET_MODEL"] = $ModelPath
    $startInfo.EnvironmentVariables["PLASCAN_BIREFNET_ENGINE_CACHE"] = $EngineCache

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start())
    {
        throw "Failed to start BiRefNet TensorRT deployment test: $TestExe"
    }

    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($TimeoutMinutes * 60 * 1000))
    {
        try
        {
            $process.Kill($true)
        }
        catch
        {
            $process.Kill()
        }
        throw "BiRefNet TensorRT $PassLabel exceeded the $TimeoutMinutes-minute timeout."
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
        throw "BiRefNet TensorRT $PassLabel failed with exit code $($process.ExitCode)."
    }

    $passedMarker = "[       OK ] $TestFilter"
    if (-not $output.Contains($passedMarker) -or $output.Contains("[  SKIPPED ]"))
    {
        throw "BiRefNet TensorRT $PassLabel did not execute successfully; it may have been skipped."
    }
    if ($output -notmatch "(?i)engine_reused\s*=\s*$ExpectedReuse")
    {
        throw "BiRefNet TensorRT $PassLabel did not report engine_reused=$ExpectedReuse."
    }

    return $output
}

$BuildDir = Resolve-FullPath $BuildDir
if ([string]::IsNullOrWhiteSpace($InstallRoot))
{
    $InstallRoot = Join-Path $BuildDir "package-smoke\PlaScan"
}
$InstallRoot = Resolve-FullPath $InstallRoot
$runtimeDir = Join-Path $InstallRoot "bin"
$modelPath = Join-Path $InstallRoot `
    "resources\models\birefnet_dynamic\BiRefNet_dynamic_1024.onnx"
$provenancePath = Join-Path $InstallRoot `
    "resources\models\birefnet_dynamic\BiRefNet_dynamic_1024.provenance.json"

if (-not (Test-PathUnder -Path $InstallRoot -Parent $BuildDir))
{
    throw "BiRefNet deployment test install root must stay under the build directory: $InstallRoot"
}
Assert-ExistingPath $runtimeDir "PlaScan package-smoke runtime directory"
Assert-ExistingPath $modelPath "Installed BiRefNet Dynamic ONNX model"
Assert-ExistingPath $provenancePath "Installed BiRefNet Dynamic provenance"
Assert-NoEngineArtifacts $InstallRoot "PlaScan package-smoke tree before inference"

$testCandidates = @(
    (Join-Path $BuildDir "src\core\mask\tests\test_mask_generation.exe"),
    (Join-Path $BuildDir "tests\test_mask_generation.exe"),
    (Join-Path $BuildDir "bin\test_mask_generation.exe")
)
$builtTestExe = $testCandidates | Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($builtTestExe))
{
    throw "BiRefNet mask test executable was not found; build target test_mask_generation. Checked: " +
        ($testCandidates -join ", ")
}

$requirements = @(
    @{ Label = "OpenCV core runtime"; Pattern = "opencv_core*.dll" },
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

$forbiddenRuntime = @(Get-ChildItem -LiteralPath $InstallRoot -Recurse -File `
    -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "cudnn*.dll" })
if ($forbiddenRuntime.Count -gt 0)
{
    throw "Portable BiRefNet deployment contains forbidden cuDNN DLLs: " +
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

$cleanProfile = Join-Path $tempRoot `
    ("plascan-birefnet-tensorrt-deployment-" + [Guid]::NewGuid().ToString("N"))
$engineCache = Join-Path $cleanProfile "BiRefNetEngineCache"
$testHarnessDir = Join-Path $cleanProfile "TestHarness"
$harnessTestExe = Join-Path $testHarnessDir "test_mask_generation.exe"
$buildRuntimeDir = Join-Path $BuildDir "bin"
$testRuntimeDlls = @("gtest.dll", "gtest_main.dll")

try
{
    New-Item -ItemType Directory -Force -Path $cleanProfile, $testHarnessDir | Out-Null
    Copy-Item -LiteralPath $builtTestExe -Destination $harnessTestExe -Force
    foreach ($testRuntimeDll in $testRuntimeDlls)
    {
        $testRuntimePath = Join-Path $buildRuntimeDir $testRuntimeDll
        Assert-ExistingPath $testRuntimePath "BiRefNet test-only runtime DLL"
        Copy-Item -LiteralPath $testRuntimePath -Destination $testHarnessDir -Force
    }

    [void] (Invoke-BiRefNetIntegrationTest `
        -TestExe $harnessTestExe `
        -WorkingDirectory $InstallRoot `
        -TestHarnessDir $testHarnessDir `
        -RuntimeDir $runtimeDir `
        -ModelPath $modelPath `
        -EngineCache $engineCache `
        -CleanProfile $cleanProfile `
        -SystemRoot $systemRoot `
        -ExpectedReuse "no" `
        -PassLabel "first-build inference")

    $firstEngines = @(Get-ChildItem -LiteralPath $engineCache -Recurse -File -Filter "*.engine" `
        -ErrorAction SilentlyContinue)
    if ($firstEngines.Count -eq 0)
    {
        throw "BiRefNet first inference did not create an engine in the isolated user cache: $engineCache"
    }
    Assert-NoEngineArtifacts $InstallRoot "PlaScan package-smoke tree after first inference"

    $firstEngineState = @{}
    foreach ($engine in $firstEngines)
    {
        $firstEngineState[$engine.FullName] = "$($engine.Length):$($engine.LastWriteTimeUtc.Ticks)"
    }

    [void] (Invoke-BiRefNetIntegrationTest `
        -TestExe $harnessTestExe `
        -WorkingDirectory $InstallRoot `
        -TestHarnessDir $testHarnessDir `
        -RuntimeDir $runtimeDir `
        -ModelPath $modelPath `
        -EngineCache $engineCache `
        -CleanProfile $cleanProfile `
        -SystemRoot $systemRoot `
        -ExpectedReuse "yes" `
        -PassLabel "cache-reuse inference")

    $secondEngines = @(Get-ChildItem -LiteralPath $engineCache -Recurse -File -Filter "*.engine" `
        -ErrorAction SilentlyContinue)
    if ($secondEngines.Count -ne $firstEngines.Count)
    {
        throw "BiRefNet cache reuse changed the engine count: $($firstEngines.Count) -> $($secondEngines.Count)"
    }
    foreach ($engine in $secondEngines)
    {
        if (-not $firstEngineState.ContainsKey($engine.FullName) -or
            $firstEngineState[$engine.FullName] -ne "$($engine.Length):$($engine.LastWriteTimeUtc.Ticks)")
        {
            throw "BiRefNet cache reuse unexpectedly replaced or modified an engine: $($engine.FullName)"
        }
    }
    Assert-NoEngineArtifacts $InstallRoot "PlaScan package-smoke tree after cache reuse"

    Write-Host "BiRefNet TensorRT clean package-smoke deployment test passed: first build and second-run cache reuse."
}
finally
{
    if ((Test-Path -LiteralPath $cleanProfile) -and
        (Test-PathUnder -Path $cleanProfile -Parent $tempRoot))
    {
        Remove-Item -LiteralPath $cleanProfile -Recurse -Force -ErrorAction SilentlyContinue
    }
}
