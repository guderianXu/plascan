param(
    [Parameter(Mandatory = $true)]
    [string]$Project,

    [string]$ListFile = "",
    [string]$OutputDirectory = "",

    [ValidateSet("auto", "orbital_object", "aerial_terrain")]
    [string]$SceneProfile = "auto",

    [ValidateSet("highest", "high", "medium", "low", "lowest")]
    [string]$Quality = "medium",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",

    [int]$MaximumDimension = 1024,
    [int]$MaximumFrames = 0,
    [string]$BuildDirectory = "",
    [switch]$TwoSourceGrowth,
    [int]$TwoSourceGrowthDistance = 3,
    [double]$TwoSourceGrowthSpread = 0.01,
    [double]$TwoSourceGrowthNormalAngle = 15.0,
    [int]$TwoSourceGrowthMaximumArea = 64,
    [switch]$SkipImageQuality
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Resolve-InputPath
{
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path))
    {
        throw "$Description not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-ImageCameraListRecord
{
    param([Parameter(Mandatory = $true)][string]$Path)

    $line = Get-Content -LiteralPath $Path -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and $_ -notmatch '^\s*#' } |
        Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($line))
    {
        return $null
    }
    $tokens = @([regex]::Matches($line, '"([^"]+)"|(\S+)') | ForEach-Object {
        if ($_.Groups[1].Success) { $_.Groups[1].Value } else { $_.Groups[2].Value }
    })
    if ($tokens.Count -lt 2)
    {
        return $null
    }
    $directory = Split-Path -Parent $Path
    $imagePath = if ([System.IO.Path]::IsPathRooted($tokens[0])) {
        $tokens[0]
    } else {
        Join-Path $directory $tokens[0]
    }
    $cameraPath = if ([System.IO.Path]::IsPathRooted($tokens[1])) {
        $tokens[1]
    } else {
        Join-Path $directory $tokens[1]
    }
    if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $cameraPath -PathType Leaf))
    {
        return $null
    }
    return [ordered]@{ image = $imagePath; camera = $cameraPath }
}

function Test-ImageCameraListFile
{
    param([Parameter(Mandatory = $true)][string]$Path)
    return $null -ne (Get-ImageCameraListRecord -Path $Path)
}

function Resolve-ProjectListFile
{
    param(
        [Parameter(Mandatory = $true)]
        [string]$ResolvedProject,
        [string]$RequestedListFile
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedListFile))
    {
        $resolvedRequested = Resolve-InputPath -Path $RequestedListFile -Description "Image/camera list"
        if (-not (Test-ImageCameraListFile -Path $resolvedRequested))
        {
            throw "Image/camera list must contain existing '<image> <camera.tsai>' pairs: $resolvedRequested"
        }
        return $resolvedRequested
    }

    $projectItem = Get-Item -LiteralPath $ResolvedProject
    $projectDirectory = if ($projectItem.PSIsContainer) { $projectItem.FullName } else { $projectItem.DirectoryName }
    $projectStem = if ($projectItem.PSIsContainer) { $projectItem.Name } else { $projectItem.BaseName }
    $preferredCandidates = @(
        (Join-Path $projectDirectory "image_camera.lis"),
        (Join-Path $projectDirectory ($projectStem + "_images_for_probe.lis")),
        (Join-Path $projectDirectory ($projectStem + ".lis"))
    )
    foreach ($candidate in $preferredCandidates)
    {
        if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and
            (Test-ImageCameraListFile -Path $candidate))
        {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $recursiveCandidates = @(Get-ChildItem -LiteralPath $projectDirectory -Recurse -Filter *.lis -File |
        Where-Object { Test-ImageCameraListFile -Path $_.FullName } |
        Sort-Object FullName)
    $canonicalCandidates = @($recursiveCandidates | Where-Object { $_.Name -eq "image_camera.lis" })
    if ($canonicalCandidates.Count -eq 1)
    {
        return $canonicalCandidates[0].FullName
    }
    if ($recursiveCandidates.Count -eq 1)
    {
        return $recursiveCandidates[0].FullName
    }
    throw "Cannot infer one valid image/camera .lis file below $projectDirectory; pass -ListFile explicitly."
}

function Quote-ProcessArgument
{
    param([string]$Value)
    if ($Value -notmatch '[\s"]')
    {
        return $Value
    }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Invoke-MonitoredProcess
{
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    $stdoutPath = $LogPath + ".stdout"
    $stderrPath = $LogPath + ".stderr"
    $argumentLine = ($Arguments | ForEach-Object { Quote-ProcessArgument $_ }) -join " "
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $Executable -ArgumentList $argumentLine -PassThru -NoNewWindow `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    # Start-Process does not retain a usable exit code on every PowerShell version unless
    # the native process handle is materialized before the process exits.
    $processHandle = $process.Handle
    [int64]$peakMemoryBytes = 0
    while (-not $process.HasExited)
    {
        try
        {
            $process.Refresh()
            $peakMemoryBytes = [Math]::Max($peakMemoryBytes, [int64]$process.PeakWorkingSet64)
        }
        catch
        {
        }
        Start-Sleep -Milliseconds 200
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = [int]$process.ExitCode
    $stopwatch.Stop()
    $stdout = if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath -Raw -Encoding UTF8
    } else { "" }
    $stderr = if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath -Raw -Encoding UTF8
    } else { "" }
    @($stdout, $stderr) | Set-Content -LiteralPath $LogPath -Encoding UTF8
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    return [ordered]@{
        exit_code = $exitCode
        elapsed_ms = [Math]::Round($stopwatch.Elapsed.TotalMilliseconds, 3)
        peak_memory_bytes = $peakMemoryBytes
        stdout = $stdout
        stderr = $stderr
    }
}

function Property-Value
{
    param($Object, [string]$Name, $Default = $null)
    if ($null -eq $Object)
    {
        return $Default
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property)
    {
        return $Default
    }
    return $property.Value
}

$resolvedProject = Resolve-InputPath -Path $Project -Description "Project"
$resolvedListFile = Resolve-ProjectListFile -ResolvedProject $resolvedProject -RequestedListFile $ListFile
if ([string]::IsNullOrWhiteSpace($BuildDirectory))
{
    $BuildDirectory = Join-Path $repoRoot "build\windows-vcpkg-cuda-release"
}
$BuildDirectory = Resolve-InputPath -Path $BuildDirectory -Description "Build directory"
$reconstructCli = Resolve-InputPath -Path (Join-Path $BuildDirectory "bin\reconstruct_pipeline_cli.exe") `
    -Description "Reconstruction CLI"
$meshCli = Resolve-InputPath -Path (Join-Path $BuildDirectory "bin\mesh_reconstruct_cli.exe") `
    -Description "Direct depth mesh CLI"
$qualityCli = Join-Path $BuildDirectory "bin\model_quality_cli.exe"
if (-not $SkipImageQuality)
{
    $qualityCli = Resolve-InputPath -Path $qualityCli -Description "Model quality CLI"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory))
{
    $OutputDirectory = Join-Path $BuildDirectory "regression\depth-overlay"
}
$projectName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedProject)
if ((Get-Item -LiteralPath $resolvedProject).PSIsContainer)
{
    $projectName = (Get-Item -LiteralPath $resolvedProject).Name
}
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDirectory = Join-Path $OutputDirectory ($projectName + "-" + $timestamp)
New-Item -ItemType Directory -Force $runDirectory | Out-Null
$pipelineDirectory = Join-Path $runDirectory "pipeline"
New-Item -ItemType Directory -Force $pipelineDirectory | Out-Null

$reconstructArguments = @(
    $resolvedListFile,
    "--output-dir", $pipelineDirectory,
    "--device", $Device,
    "--sfm-feature-algorithm", "sift",
    "--sfm-match-algorithm", "lightglue",
    "--sfm-guided-rematching",
    "--mvs-quality", $Quality,
    "--mvs-scene-profile", $SceneProfile,
    "--mvs-depth-filter", "auto",
    "--mvs-save-levels",
    "--mvs-max-frames", [string][Math]::Max(0, $MaximumFrames),
    "--mvs-fusion-max-image-dim", [string][Math]::Max(0, $MaximumDimension),
    "--mvs-depth-only",
    "--skip-terrain",
    "--force"
)
if ($TwoSourceGrowth)
{
    $reconstructArguments += @(
        "--mvs-two-source-growth",
        "--mvs-two-source-growth-distance", [string]$TwoSourceGrowthDistance,
        "--mvs-two-source-growth-spread", [string]$TwoSourceGrowthSpread,
        "--mvs-two-source-growth-normal-angle", [string]$TwoSourceGrowthNormalAngle,
        "--mvs-two-source-growth-maximum-area", [string]$TwoSourceGrowthMaximumArea
    )
}
$projectItem = Get-Item -LiteralPath $resolvedProject
$projectDirectory = if ($projectItem.PSIsContainer) { $projectItem.FullName } else { $projectItem.DirectoryName }
$projectMaskDirectory = Join-Path $projectDirectory "assets\masks"
if (Test-Path -LiteralPath $projectMaskDirectory -PathType Container)
{
    $reconstructArguments += @("--mvs-mask-dir", $projectMaskDirectory)
}
$reconstructRun = Invoke-MonitoredProcess -Executable $reconstructCli `
    -Arguments $reconstructArguments -LogPath (Join-Path $runDirectory "reconstruct.log")
$pipelineReportPath = Join-Path $pipelineDirectory "report.json"
if (-not (Test-Path -LiteralPath $pipelineReportPath -PathType Leaf))
{
    throw "Reconstruction report not found after exit code $($reconstructRun.exit_code): $pipelineReportPath"
}
$pipelineReport = Get-Content -LiteralPath $pipelineReportPath -Raw -Encoding UTF8 | ConvertFrom-Json

$modelSettingsPath = Join-Path $runDirectory "model-settings.json"
$modelSettings = [ordered]@{
    generate_model = [ordered]@{
        source_data = "depth_maps"
        surface_type = "arbitrary_3d"
        method = "Depth TSDF"
        reconstruction_mode = "depth_tsdf"
        meshResolution = 320
        calculateVertexColors = $true
        depthFiltering = "moderate"
    }
}
$modelSettings | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $modelSettingsPath -Encoding UTF8
$modelArguments = @(
    "--source-data", "depth_maps",
    "--depth-map-dir", (Join-Path $pipelineDirectory "mvs"),
    "--output-dir", (Join-Path $pipelineDirectory "model"),
    "--settings-json", $modelSettingsPath,
    "--settings-key", "generate_model"
)
$meshRun = Invoke-MonitoredProcess -Executable $meshCli `
    -Arguments $modelArguments -LogPath (Join-Path $runDirectory "mesh.log")
$meshJsonText = if (-not [string]::IsNullOrWhiteSpace([string]$meshRun.stdout)) {
    [string]$meshRun.stdout
} else {
    [string]$meshRun.stderr
}
$modelReport = $null
if (-not [string]::IsNullOrWhiteSpace($meshJsonText))
{
    try
    {
        $modelReport = $meshJsonText | ConvertFrom-Json
    }
    catch
    {
        $modelReport = [ordered]@{ parse_error = $_.Exception.Message }
    }
}
$actualMeshAlgorithm = [string](Property-Value $modelReport "actual_mesh_algorithm" "")
$meshContractError = ""
if ($meshRun.exit_code -eq 0 -and $actualMeshAlgorithm -ne "depth_tsdf")
{
    $meshContractError = "Direct depth reconstruction returned '$actualMeshAlgorithm' instead of depth_tsdf."
}

$depthMapRecords = @(
    Property-Value -Object (Property-Value $pipelineReport "dense") -Name "depth_maps" -Default @()
)
$depthMaps = @(
    $depthMapRecords |
        Group-Object -Property { [string](Property-Value $_ "ref_image" "") } |
        ForEach-Object {
            $filteredRecords = @(
                $_.Group | Where-Object {
                    $postprocess = Property-Value $_ "depth_postprocess"
                    [int64](Property-Value $postprocess "valid_after" 0) -gt 0
                }
            )
            if ($filteredRecords.Count -gt 0)
            {
                $filteredRecords[-1]
            }
            else
            {
                $_.Group[-1]
            }
        }
)
$selectedLevels = @($depthMaps | ForEach-Object { Property-Value $_ "selected_level" 0 })
$maskSources = @($depthMaps | ForEach-Object { Property-Value $_ "mask_source" "none" } | Sort-Object -Unique)
$coverageValues = @($depthMaps | ForEach-Object {
    $qualityObject = Property-Value $_ "depth_quality"
    [double](Property-Value $qualityObject "valid_coverage" 0.0)
})
$meanCoverage = if ($coverageValues.Count -gt 0) {
    ($coverageValues | Measure-Object -Average).Average
} else { 0.0 }
$completenessRecords = @($depthMaps | ForEach-Object { Property-Value $_ "depth_completeness" })
$maskNormalizedCoverageValues = @($completenessRecords | ForEach-Object {
    [double](Property-Value $_ "valid_within_mask_ratio" -1.0)
} | Where-Object { $_ -ge 0.0 })
$outputRetentionValues = @($completenessRecords | ForEach-Object {
    [double](Property-Value $_ "output_filter_retention_ratio" -1.0)
} | Where-Object { $_ -ge 0.0 })
$consistencyRetentionValues = @($completenessRecords | ForEach-Object {
    [double](Property-Value $_ "consistency_retention_ratio" -1.0)
} | Where-Object { $_ -ge 0.0 })
$meanMaskNormalizedCoverage = if ($maskNormalizedCoverageValues.Count -gt 0) {
    ($maskNormalizedCoverageValues | Measure-Object -Average).Average
} else { -1.0 }
$minimumMaskNormalizedCoverage = if ($maskNormalizedCoverageValues.Count -gt 0) {
    ($maskNormalizedCoverageValues | Measure-Object -Minimum).Minimum
} else { -1.0 }
$minimumOutputRetention = if ($outputRetentionValues.Count -gt 0) {
    ($outputRetentionValues | Measure-Object -Minimum).Minimum
} else { -1.0 }
$minimumConsistencyRetention = if ($consistencyRetentionValues.Count -gt 0) {
    ($consistencyRetentionValues | Measure-Object -Minimum).Minimum
} else { -1.0 }
$smallHolePixels = ($completenessRecords | ForEach-Object {
    [int64](Property-Value $_ "small_internal_hole_pixel_count" 0)
} | Measure-Object -Sum).Sum
$largeOpeningPixels = ($completenessRecords | ForEach-Object {
    [int64](Property-Value $_ "large_internal_opening_pixel_count" 0)
} | Measure-Object -Sum).Sum
$boundaryInvalidPixels = ($completenessRecords | ForEach-Object {
    [int64](Property-Value $_ "boundary_connected_invalid_pixel_count" 0)
} | Measure-Object -Sum).Sum
$acceptanceCounts = [ordered]@{}
$depthMaps | Group-Object -Property { [string](Property-Value $_ "acceptance" "unknown") } |
    ForEach-Object { $acceptanceCounts[$_.Name] = $_.Count }
$denseReport = Property-Value $pipelineReport "dense"
$modelPath = [string](Property-Value $modelReport "model_ply" "")
if ([string]::IsNullOrWhiteSpace($modelPath))
{
    $modelPath = [string](Property-Value $modelReport "mesh_ply" "")
}

$qualityRun = $null
$qualityResult = $null
if (-not $SkipImageQuality -and $meshRun.exit_code -eq 0 -and
    [string]::IsNullOrWhiteSpace($meshContractError) -and
    -not [string]::IsNullOrWhiteSpace($modelPath) -and
    (Test-Path -LiteralPath $modelPath -PathType Leaf))
{
    $qualityDirectory = Join-Path $runDirectory "model-quality"
    New-Item -ItemType Directory -Force $qualityDirectory | Out-Null
    $sceneType = if ($SceneProfile -eq "aerial_terrain") { "aerial" } else { "dino" }
    $qualityArguments = @(
        "--mesh", $modelPath,
        "--mvs-workspace", (Join-Path $pipelineDirectory "mvs"),
        "--scene-type", $sceneType,
        "--validation-split", "auto",
        "--output-dir", $qualityDirectory,
        "--max-render-dim", [string][Math]::Max(256, $MaximumDimension)
    )
    $qualityRun = Invoke-MonitoredProcess -Executable $qualityCli `
        -Arguments $qualityArguments -LogPath (Join-Path $runDirectory "model-quality.log")
    $qualityJsonText = if (-not [string]::IsNullOrWhiteSpace([string]$qualityRun.stdout))
    {
        [string]$qualityRun.stdout
    }
    else
    {
        [string]$qualityRun.stderr
    }
    if (-not [string]::IsNullOrWhiteSpace($qualityJsonText))
    {
        try
        {
            $qualityResult = $qualityJsonText | ConvertFrom-Json
        }
        catch
        {
            $qualityResult = [ordered]@{ parse_error = $_.Exception.Message }
        }
    }
}

$rejectionCounts = Property-Value $denseReport "fusion_rejection_counts" ([ordered]@{})
$densePoints = [int64](Property-Value $denseReport "points" 0)
$hasNormals = [bool](Property-Value $denseReport "has_normals" $false)
$comparisonReport = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    project = $resolvedProject
    list_file = $resolvedListFile
    output_directory = $runDirectory
    scene_profile = $SceneProfile
    quality = $Quality
    maximum_dimension = $MaximumDimension
    pipeline_status = [string](Property-Value $pipelineReport "status" "unknown")
    selected_level = $selectedLevels
    mask_source = $maskSources
    mean_valid_coverage = [Math]::Round([double]$meanCoverage, 6)
    depth_completeness = [ordered]@{
        mean_mask_normalized_coverage = [Math]::Round([double]$meanMaskNormalizedCoverage, 6)
        minimum_mask_normalized_coverage = [Math]::Round([double]$minimumMaskNormalizedCoverage, 6)
        minimum_output_filter_retention = [Math]::Round([double]$minimumOutputRetention, 6)
        minimum_consistency_retention = [Math]::Round([double]$minimumConsistencyRetention, 6)
        small_internal_hole_pixel_count = [int64]$smallHolePixels
        large_internal_opening_pixel_count = [int64]$largeOpeningPixels
        boundary_connected_invalid_pixel_count = [int64]$boundaryInvalidPixels
        acceptance_counts = $acceptanceCounts
    }
    depth_frame_count = $depthMaps.Count
    accepted_fused_points = $densePoints
    finite_normals = if ($hasNormals) { $densePoints } else { 0 }
    rejection_counts = $rejectionCounts
    mesh = [ordered]@{
        path = $modelPath
        vertices = [int64](Property-Value $modelReport "vertex_count" 0)
        faces = [int64](Property-Value $modelReport "face_count" 0)
        algorithm = $actualMeshAlgorithm
        component_count = [int](Property-Value $modelReport "component_count" 0)
        largest_component_face_ratio = [double](Property-Value $modelReport "largest_component_face_ratio" 0.0)
        component_face_counts = @(Property-Value $modelReport "component_face_counts" @())
        single_view_supported_samples = [int64](Property-Value $modelReport "single_view_supported_sample_count" 0)
        multi_view_supported_samples = [int64](Property-Value $modelReport "multi_view_supported_sample_count" 0)
        boundary_edges_before = [int](Property-Value $modelReport "boundary_edge_count_before" 0)
        boundary_edges_after = [int](Property-Value $modelReport "boundary_edge_count_after" 0)
        filled_boundary_holes = [int](Property-Value $modelReport "filled_boundary_hole_count" 0)
    }
    elapsed_ms = [ordered]@{
        reconstruction = $reconstructRun.elapsed_ms
        mesh = $meshRun.elapsed_ms
        image_quality = if ($null -eq $qualityRun) { 0 } else { $qualityRun.elapsed_ms }
    }
    peak_memory_bytes = [ordered]@{
        reconstruction = $reconstructRun.peak_memory_bytes
        mesh = $meshRun.peak_memory_bytes
        image_quality = if ($null -eq $qualityRun) { 0 } else { $qualityRun.peak_memory_bytes }
    }
    image_quality = $qualityResult
    exit_codes = [ordered]@{
        reconstruction = $reconstructRun.exit_code
        mesh = $meshRun.exit_code
        image_quality = if ($null -eq $qualityRun) { $null } else { $qualityRun.exit_code }
    }
    mesh_contract_error = $meshContractError
}
$comparisonReportPath = Join-Path $runDirectory "comparison_report.json"
$comparisonReport | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $comparisonReportPath -Encoding UTF8
Write-Host "Depth overlay regression report: $comparisonReportPath"

if ($reconstructRun.exit_code -ne 0)
{
    throw "Reconstruction failed with exit code $($reconstructRun.exit_code). See $($runDirectory)\reconstruct.log"
}
if ($meshRun.exit_code -ne 0)
{
    throw "Direct depth mesh failed with exit code $($meshRun.exit_code). See $($runDirectory)\mesh.log"
}
if (-not [string]::IsNullOrWhiteSpace($meshContractError))
{
    throw $meshContractError
}
if ($null -ne $qualityRun -and $qualityRun.exit_code -ne 0)
{
    throw "Image quality validation failed with exit code $($qualityRun.exit_code). See $($runDirectory)\model-quality.log"
}
