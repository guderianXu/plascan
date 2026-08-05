[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $BuildDir,
    [string] $ModelPath = ""
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

$BuildDir = Resolve-FullPath $BuildDir
$runtimeDir = Join-Path $BuildDir "bin"
$testExe = Join-Path $BuildDir "src\core\mask\tests\test_mask_generation.exe"
if ([string]::IsNullOrWhiteSpace($ModelPath))
{
    $sourceRoot = Resolve-FullPath (Join-Path $PSScriptRoot "..\..")
    $ModelPath = Join-Path $sourceRoot "resources\models\U2Net_v1.onnx"
}
$ModelPath = Resolve-FullPath $ModelPath

Assert-ExistingPath $runtimeDir "PlaScan deployment runtime directory"
Assert-ExistingPath $testExe "U2Net mask test executable; build target test_mask_generation"
Assert-ExistingPath $ModelPath "U2Net ONNX model"

$systemRoot = [Environment]::GetEnvironmentVariable("SystemRoot")
$tempRoot = [Environment]::GetEnvironmentVariable("TEMP")
if ([string]::IsNullOrWhiteSpace($systemRoot))
{
    throw "SystemRoot is required for the clean deployment environment."
}

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $testExe
$startInfo.Arguments = "--gtest_color=no --gtest_filter=U2NetMaskGeneratorIntegrationTest.OnnxModelRunsOnCudaWhenBackendAvailable"
$startInfo.WorkingDirectory = $BuildDir
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.CreateNoWindow = $true
$startInfo.EnvironmentVariables.Clear()
$startInfo.EnvironmentVariables["PATH"] = "$runtimeDir;$systemRoot\System32;$systemRoot"
$startInfo.EnvironmentVariables["SystemRoot"] = $systemRoot
$startInfo.EnvironmentVariables["WINDIR"] = $systemRoot
$startInfo.EnvironmentVariables["PLASCAN_U2NET_MODEL"] = $ModelPath
if (-not [string]::IsNullOrWhiteSpace($tempRoot))
{
    $startInfo.EnvironmentVariables["TEMP"] = $tempRoot
    $startInfo.EnvironmentVariables["TMP"] = $tempRoot
}

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo
if (-not $process.Start())
{
    throw "Failed to start U2Net CUDA deployment test: $testExe"
}
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
$process.WaitForExit()

$output = ($stdout + [Environment]::NewLine + $stderr).Trim()
if (-not [string]::IsNullOrWhiteSpace($output))
{
    Write-Host $output
}
if ($process.ExitCode -ne 0)
{
    throw "U2Net CUDA deployment test failed with exit code $($process.ExitCode)."
}

$passedMarker = "[       OK ] U2NetMaskGeneratorIntegrationTest.OnnxModelRunsOnCudaWhenBackendAvailable"
if (-not $output.Contains($passedMarker))
{
    throw "U2Net CUDA deployment test did not execute on CUDA; the test may have been skipped."
}

Write-Host "U2Net CUDA clean deployment test passed using only $runtimeDir and Windows system DLL paths."
