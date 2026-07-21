param(
    [ValidateSet("Dino", "Uav9", "All")]
    [string]$Dataset = "All",

    [ValidateSet("highest", "high", "medium", "low", "lowest")]
    [string]$Quality = "medium",

    [ValidateSet("auto", "orbital_object", "aerial_terrain")]
    [string]$SceneProfile = "auto",

    [ValidateSet("auto", "mild", "moderate", "aggressive")]
    [string]$DepthFilter = "auto",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",

    [int]$MaxFrames = 0,
    [string]$BuildDir = "",
    [string]$OutputRoot = "",
    [switch]$SaveLevels,
    [switch]$SkipQualityCheck
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if ([string]::IsNullOrWhiteSpace($BuildDir))
{
    $BuildDir = Join-Path $repoRoot "build\windows-vcpkg-cuda-release"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot))
{
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputRoot = Join-Path $BuildDir "regression\depth-pyramid-$stamp"
}

$reconstructCli = Join-Path $BuildDir "bin\reconstruct_pipeline_cli.exe"
$qualityCli = Join-Path $BuildDir "bin\model_quality_cli.exe"
if (-not (Test-Path -LiteralPath $reconstructCli -PathType Leaf))
{
    throw "Missing reconstruction CLI: $reconstructCli"
}
if (-not $SkipQualityCheck -and -not (Test-Path -LiteralPath $qualityCli -PathType Leaf))
{
    throw "Missing model quality CLI: $qualityCli"
}

$datasetConfigs = @{
    Dino = @{
        ListFile = Join-Path $repoRoot (
            "testData\photogrammetry_benchmarks\middlebury_dino_sparse_ring\" +
            "prepared\plascan\image_camera.lis")
        QualityScene = "dino"
    }
    Uav9 = @{
        ListFile = Join-Path $repoRoot (
            "testData\photogrammetry_benchmarks\agisoft_aerial_gcps_small\" +
            "extracted\aerial_images_with_gcps\image_camera.lis")
        QualityScene = "aerial"
    }
}

function Invoke-DepthPyramidRegression
{
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $config = $datasetConfigs[$Name]
    $listFile = [string]$config.ListFile
    if (-not (Test-Path -LiteralPath $listFile -PathType Leaf))
    {
        throw "Missing dataset list: $listFile"
    }

    $outputDir = Join-Path $OutputRoot $Name.ToLowerInvariant()
    New-Item -ItemType Directory -Force $outputDir | Out-Null
    $logPath = Join-Path $outputDir "reconstruct.log"

    $arguments = @(
        $listFile,
        "--output-dir", $outputDir,
        "--device", $Device,
        "--sfm-feature-algorithm", "sift",
        "--sfm-match-algorithm", "lightglue",
        "--sfm-guided-rematching",
        "--mvs-quality", $Quality,
        "--mvs-scene-profile", $SceneProfile,
        "--mvs-depth-filter", $DepthFilter,
        "--mvs-max-frames", [string][Math]::Max(0, $MaxFrames),
        "--skip-terrain",
        "--force"
    )
    if ($SaveLevels)
    {
        $arguments += "--mvs-save-levels"
    }

    Write-Host "[$Name] reconstruct_pipeline_cli.exe $($arguments -join ' ')"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $reconstructCli @arguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $logPath
    $reconstructExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($reconstructExitCode -ne 0)
    {
        throw "[$Name] reconstruction failed with exit code $reconstructExitCode. See $logPath"
    }

    $reportPath = Join-Path $outputDir "report.json"
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf))
    {
        throw "[$Name] report not found: $reportPath"
    }
    if ($SkipQualityCheck)
    {
        return
    }

    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $modelPath = [string]$report.model.model_ply
    if ([string]::IsNullOrWhiteSpace($modelPath))
    {
        $modelPath = [string]$report.model.mesh_ply
    }
    if ([string]::IsNullOrWhiteSpace($modelPath) -or
        -not (Test-Path -LiteralPath $modelPath -PathType Leaf))
    {
        throw "[$Name] model path is missing from $reportPath"
    }

    $qualityDir = Join-Path $outputDir "model-quality"
    $qualityArguments = @(
        "--mesh", $modelPath,
        "--image-camera-list", $listFile,
        "--scene-type", [string]$config.QualityScene,
        "--validation-split", "auto",
        "--output-dir", $qualityDir,
        "--max-render-dim", "1024"
    )
    $qualityLog = Join-Path $outputDir "model-quality.log"
    Write-Host "[$Name] model_quality_cli.exe $($qualityArguments -join ' ')"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $qualityCli @qualityArguments 2>&1 |
        ForEach-Object { "$_" } |
        Tee-Object -FilePath $qualityLog
    $qualityExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($qualityExitCode -ne 0)
    {
        throw "[$Name] quality validation failed with exit code $qualityExitCode. See $qualityLog"
    }
}

New-Item -ItemType Directory -Force $OutputRoot | Out-Null
$selectedDatasets = if ($Dataset -eq "All") { @("Dino", "Uav9") } else { @($Dataset) }
foreach ($selectedDataset in $selectedDatasets)
{
    Invoke-DepthPyramidRegression -Name $selectedDataset
}

Write-Host "Depth pyramid regression outputs: $OutputRoot"
