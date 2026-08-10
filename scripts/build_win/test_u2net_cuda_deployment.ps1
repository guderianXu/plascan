[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $BuildDir,
    [string] $ModelPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$replacement = Join-Path $PSScriptRoot "test_u2net_tensorrt_deployment.ps1"
if (-not (Test-Path -LiteralPath $replacement))
{
    throw "Replacement U2Net TensorRT deployment test script not found: $replacement"
}

Write-Warning "test_u2net_cuda_deployment.ps1 is deprecated; U2Net GPU inference now uses TensorRT."
& $replacement -BuildDir $BuildDir -ModelPath $ModelPath
