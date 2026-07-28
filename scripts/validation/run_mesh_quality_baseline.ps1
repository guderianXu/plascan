[CmdletBinding()]
param(
    [string]$RepositoryRoot = "",
    [string]$BuildDirectory = "",
    [string]$SceneManifest = "",
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    [string]$CandidateName,
    [string[]]$Scene = @(),
    [switch]$ValidateOnly,
    [switch]$ReuseExistingModel
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-FullPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$BaseDirectory
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDirectory $Path))
}

function Expand-PathTokens {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [hashtable]$Tokens
    )

    $expanded = $Value
    foreach ($name in $Tokens.Keys) {
        $expanded = $expanded.Replace(
            '${' + $name + '}',
            [string]$Tokens[$name])
    }
    $matches = [regex]::Matches($expanded, '\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
    foreach ($match in $matches) {
        $name = $match.Groups[1].Value
        $environmentValue = [Environment]::GetEnvironmentVariable($name)
        if ([string]::IsNullOrWhiteSpace($environmentValue)) {
            throw "Environment path token is not set: $name"
        }
        $expanded = $expanded.Replace($match.Value, $environmentValue)
    }
    return $expanded
}

function Get-FileRecord {
    param([Parameter(Mandatory = $true)][string]$Path)

    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer) {
        throw "Expected a file but found a directory: $Path"
    }
    return [ordered]@{
        path = $item.FullName.Replace('\', '/')
        length = [long]$item.Length
        last_write_utc = $item.LastWriteTimeUtc.ToString("o")
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString(
            $sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Get-DirectoryRecord {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [string[]]$Extensions = @(".bin", ".json", ".png", ".tif", ".tiff")
    )

    $root = Get-Item -LiteralPath $Path
    if (-not $root.PSIsContainer) {
        throw "Expected a directory but found a file: $Path"
    }
    $extensionSet = @{}
    foreach ($extension in $Extensions) {
        $extensionSet[$extension.ToLowerInvariant()] = $true
    }
    $files = @(
        Get-ChildItem -LiteralPath $root.FullName -File -Recurse |
            Where-Object {
                $extensionSet.ContainsKey($_.Extension.ToLowerInvariant())
            } |
            Sort-Object FullName
    )
    if ($files.Count -eq 0) {
        throw "No depth artifacts were found for the baseline: $Path"
    }

    $records = New-Object System.Collections.Generic.List[object]
    $aggregateLines = New-Object System.Collections.Generic.List[string]
    $rootUri = [Uri]($root.FullName.TrimEnd('\') + '\')
    foreach ($file in $files) {
        $relative = [Uri]::UnescapeDataString(
            $rootUri.MakeRelativeUri([Uri]$file.FullName).ToString())
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $record = [ordered]@{
            relative_path = $relative
            length = [long]$file.Length
            last_write_utc = $file.LastWriteTimeUtc.ToString("o")
            sha256 = $hash
        }
        $records.Add($record)
        $aggregateLines.Add("$relative|$($file.Length)|$hash")
    }
    return [ordered]@{
        path = $root.FullName.Replace('\', '/')
        file_count = $records.Count
        aggregate_sha256 = Get-TextSha256 ($aggregateLines -join "`n")
        files = $records
    }
}

function Invoke-CheckedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string]$StandardOutputPath,
        [Parameter(Mandatory = $true)]
        [string]$StandardErrorPath
    )

    # Windows PowerShell 5.1 joins ArgumentList entries into one command line
    # without preserving an entry that contains spaces. Quote those entries
    # explicitly so labels and paths arrive as one argv value.
    $quotedArguments = @(
        foreach ($argument in $ArgumentList) {
            if ($argument -match '[\s"]') {
                '"' + ($argument -replace '(\\*)"', '$1$1\"') + '"'
            }
            else {
                $argument
            }
        }
    )

    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $quotedArguments `
        -RedirectStandardOutput $StandardOutputPath `
        -RedirectStandardError $StandardErrorPath `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        $tail = Get-Content -LiteralPath $StandardErrorPath -Tail 8
        throw "Command failed with exit code $($process.ExitCode): $FilePath`n$($tail -join "`n")"
    }
}

function Test-QualityGate {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Baseline,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Candidate,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Gate,
        [AllowNull()]
        [Nullable[double]]$Chamfer
    )

    $failures = New-Object System.Collections.Generic.List[string]
    if ([double]$Candidate.face_count -gt
        [double]$Baseline.face_count * [double]$Gate.maximum_face_count_ratio) {
        $failures.Add("Face count exceeds the baseline ratio limit")
    }
    if ([double]$Candidate.topology_quality_boundary_edge_count -gt
        [double]$Baseline.boundary_edge_count *
        [double]$Gate.maximum_boundary_edge_count_ratio) {
        $failures.Add("Boundary edge count exceeds the baseline ratio limit")
    }
    if ([int]$Candidate.topology_quality_component_count -gt
        [int]$Baseline.component_count +
        [int]$Gate.maximum_component_count_increase) {
        $failures.Add("Connected component count increased")
    }
    if ([double]$Candidate.topology_quality_high_aspect_face_ratio -gt
        [double]$Baseline.high_aspect_face_ratio *
        [double]$Gate.maximum_high_aspect_ratio_multiplier) {
        $failures.Add("High-aspect face ratio exceeds the limit")
    }
    if ([double]$Candidate.topology_quality_adjacent_normal_angle_median_degrees -gt
        [double]$Baseline.normal_median_degrees +
        [double]$Gate.maximum_normal_median_increase_degrees) {
        $failures.Add("Adjacent-normal median exceeds the limit")
    }
    if ($null -ne $Chamfer -and
        $Baseline.PSObject.Properties.Name -contains "chamfer_l1" -and
        [double]$Chamfer -gt
        [double]$Baseline.chamfer_l1 *
        [double]$Gate.maximum_chamfer_l1_ratio) {
        $failures.Add("Chamfer-L1 exceeds the baseline ratio limit")
    }
    return [ordered]@{
        passed = $failures.Count -eq 0
        failures = $failures
    }
}

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = [System.IO.Path]::GetFullPath(
        (Join-Path $PSScriptRoot "..\.."))
}
else {
    $RepositoryRoot = Resolve-FullPath $RepositoryRoot (Get-Location).Path
}
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $RepositoryRoot "build\windows-vcpkg-cuda-release"
}
$BuildDirectory = Resolve-FullPath $BuildDirectory $RepositoryRoot
if ([string]::IsNullOrWhiteSpace($SceneManifest)) {
    $SceneManifest = Join-Path $PSScriptRoot "mesh_quality_scenes.json"
}
$SceneManifest = Resolve-FullPath $SceneManifest $RepositoryRoot

$meshCli = Join-Path $BuildDirectory "bin\mesh_reconstruct_cli.exe"
$python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$compareScript = Join-Path $PSScriptRoot "compare_mesh_geometry.py"
$renderScript = Join-Path $PSScriptRoot "render_mesh_comparison.py"
foreach ($requiredFile in @(
    $meshCli,
    $python,
    $compareScript,
    $renderScript,
    $SceneManifest)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file is missing: $requiredFile"
    }
}

$manifest = Get-Content -LiteralPath $SceneManifest -Raw |
    ConvertFrom-Json
if ($manifest.schema -ne "plascan.mesh_quality_scenes.v1") {
    throw "Unsupported scene manifest schema: $($manifest.schema)"
}
$tokens = @{
    REPO_ROOT = $RepositoryRoot
    BUILD_DIR = $BuildDirectory
}
$selectedScenes = @(
    $manifest.scenes | Where-Object {
        $Scene.Count -eq 0 -or $Scene -contains $_.name
    }
)
if ($selectedScenes.Count -eq 0) {
    throw "No matching scenes. Available scenes: $($manifest.scenes.name -join ', ')"
}

$runRoot = Join-Path $BuildDirectory "analysis\quality_baselines\$CandidateName"
if ((Test-Path -LiteralPath $runRoot) -and
    -not $ReuseExistingModel) {
    throw "Candidate output already exists; refusing to overwrite: $runRoot"
}
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$runSummaries = New-Object System.Collections.Generic.List[object]
foreach ($sceneDefinition in $selectedScenes) {
    $sceneName = [string]$sceneDefinition.name
    Write-Host "[$sceneName] Capturing the input snapshot..."
    $depthDirectory = Resolve-FullPath (
        Expand-PathTokens $sceneDefinition.depth_map_dir $tokens) $RepositoryRoot
    $settingsJson = Resolve-FullPath (
        Expand-PathTokens $sceneDefinition.settings_json $tokens) $RepositoryRoot
    $baselineMesh = Resolve-FullPath (
        Expand-PathTokens $sceneDefinition.baseline_mesh $tokens) $RepositoryRoot
    $referenceMesh = $null
    if ($sceneDefinition.PSObject.Properties.Name -contains "reference_mesh") {
        $referenceMesh = Resolve-FullPath (
            Expand-PathTokens $sceneDefinition.reference_mesh $tokens) $RepositoryRoot
    }
    foreach ($path in @($depthDirectory, $settingsJson, $baselineMesh)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "[$sceneName] Input does not exist: $path"
        }
    }
    if ($null -ne $referenceMesh -and
        -not (Test-Path -LiteralPath $referenceMesh -PathType Leaf)) {
        throw "[$sceneName] Reference mesh does not exist: $referenceMesh"
    }

    $sceneOutput = Join-Path $runRoot $sceneName
    New-Item -ItemType Directory -Force -Path $sceneOutput | Out-Null
    $snapshot = [ordered]@{
        schema = "plascan.mesh_quality_input_snapshot.v1"
        scene = $sceneName
        depth_artifacts = Get-DirectoryRecord $depthDirectory
        settings = Get-FileRecord $settingsJson
        mesh_cli = Get-FileRecord $meshCli
        baseline_mesh = Get-FileRecord $baselineMesh
        reference_mesh = if ($null -ne $referenceMesh) {
            Get-FileRecord $referenceMesh
        } else {
            $null
        }
    }
    if ($sceneDefinition.PSObject.Properties.Name -contains
        "expected_depth_file_count") {
        $expectedFileCount = [int]$sceneDefinition.expected_depth_file_count
        if ([int]$snapshot.depth_artifacts.file_count -ne $expectedFileCount) {
            throw "[$sceneName] Depth artifact drift: expected $expectedFileCount files, found $($snapshot.depth_artifacts.file_count)."
        }
    }
    if ($sceneDefinition.PSObject.Properties.Name -contains
        "expected_depth_artifacts_sha256") {
        $expectedDepthHash = [string]$sceneDefinition.expected_depth_artifacts_sha256
        if ([string]$snapshot.depth_artifacts.aggregate_sha256 -ne
            $expectedDepthHash) {
            throw "[$sceneName] Depth artifact drift: expected SHA-256 $expectedDepthHash, found $($snapshot.depth_artifacts.aggregate_sha256)."
        }
    }
    if ($sceneDefinition.PSObject.Properties.Name -contains
        "expected_settings_sha256") {
        $expectedSettingsHash = [string]$sceneDefinition.expected_settings_sha256
        if ([string]$snapshot.settings.sha256 -ne $expectedSettingsHash) {
            throw "[$sceneName] Settings drift: expected SHA-256 $expectedSettingsHash, found $($snapshot.settings.sha256)."
        }
    }
    $snapshotJson = $snapshot | ConvertTo-Json -Depth 12
    $snapshotPath = Join-Path $sceneOutput "input_snapshot.json"
    [System.IO.File]::WriteAllText(
        $snapshotPath,
        $snapshotJson,
        [System.Text.UTF8Encoding]::new($false))
    $snapshotFingerprint = Get-TextSha256 (
        $snapshot | ConvertTo-Json -Depth 12 -Compress)
    if ($ValidateOnly) {
        $runSummaries.Add([ordered]@{
            scene = $sceneName
            snapshot_fingerprint_sha256 = $snapshotFingerprint
            input_snapshot = $snapshotPath.Replace('\', '/')
            validated_only = $true
        })
        continue
    }

    $stdoutPath = Join-Path $sceneOutput "mesh_stdout.json"
    $stderrPath = Join-Path $sceneOutput "mesh_progress.log"
    $candidateMesh = Join-Path $sceneOutput "products\model_from_mesh.ply"
    if (-not ($ReuseExistingModel -and
        (Test-Path -LiteralPath $candidateMesh -PathType Leaf))) {
        Write-Host "[$sceneName] Reconstructing the candidate mesh..."
        Invoke-CheckedProcess `
            -FilePath $meshCli `
            -ArgumentList @(
                "--source-data", "depth_maps",
                "--depth-map-dir", $depthDirectory,
                "--output-dir", $sceneOutput,
                "--settings-json", $settingsJson,
                "--settings-key", [string]$sceneDefinition.settings_key) `
            -StandardOutputPath $stdoutPath `
            -StandardErrorPath $stderrPath
    }
    if (-not (Test-Path -LiteralPath $candidateMesh -PathType Leaf)) {
        throw "[$sceneName] Candidate mesh was not produced: $candidateMesh"
    }
    $meshStatistics = Get-Content -LiteralPath $stdoutPath -Raw |
        ConvertFrom-Json

    $baselineComparison = Join-Path $sceneOutput "vs_baseline.json"
    Invoke-CheckedProcess `
        -FilePath $python `
        -ArgumentList @(
            $compareScript,
            "--reference", $baselineMesh,
            "--candidate", $candidateMesh,
            "--output", $baselineComparison,
            "--samples", "5000") `
        -StandardOutputPath (Join-Path $sceneOutput "compare_baseline.log") `
        -StandardErrorPath (Join-Path $sceneOutput "compare_baseline_error.log")
    $contactSheet = Join-Path $sceneOutput "contact_sheet.png"
    Invoke-CheckedProcess `
        -FilePath $python `
        -ArgumentList @(
            $renderScript,
            "--reference", $baselineMesh,
            "--candidate", $candidateMesh,
            "--output", $contactSheet,
            "--reference-label", "$sceneName baseline",
            "--candidate-label", $CandidateName,
            "--max-faces", "180000") `
        -StandardOutputPath (Join-Path $sceneOutput "render.log") `
        -StandardErrorPath (Join-Path $sceneOutput "render_error.log")

    $referenceComparison = $null
    $chamfer = $null
    if ($null -ne $referenceMesh) {
        $referenceComparison = Join-Path $sceneOutput "vs_reference.json"
        Invoke-CheckedProcess `
            -FilePath $python `
            -ArgumentList @(
                $compareScript,
                "--reference", $referenceMesh,
                "--candidate", $candidateMesh,
                "--output", $referenceComparison,
                "--samples", "5000") `
            -StandardOutputPath (Join-Path $sceneOutput "compare_reference.log") `
            -StandardErrorPath (Join-Path $sceneOutput "compare_reference_error.log")
        $comparison = Get-Content -LiteralPath $referenceComparison -Raw |
            ConvertFrom-Json
        $chamfer = [double]$comparison.symmetric.chamfer_l1_mean
    }
    $gateResult = Test-QualityGate `
        -Baseline $sceneDefinition.baseline_metrics `
        -Candidate $meshStatistics `
        -Gate $manifest.gate `
        -Chamfer $chamfer
    $runSummaries.Add([ordered]@{
        scene = $sceneName
        validated_only = $false
        snapshot_fingerprint_sha256 = $snapshotFingerprint
        input_snapshot = $snapshotPath.Replace('\', '/')
        candidate_mesh = $candidateMesh.Replace('\', '/')
        mesh_statistics = $stdoutPath.Replace('\', '/')
        baseline_comparison = $baselineComparison.Replace('\', '/')
        reference_comparison = if ($null -ne $referenceComparison) {
            $referenceComparison.Replace('\', '/')
        } else {
            $null
        }
        contact_sheet = $contactSheet.Replace('\', '/')
        gate = $gateResult
    })
}

$summary = [ordered]@{
    schema = "plascan.mesh_quality_baseline_run.v1"
    candidate_name = $CandidateName
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    scene_manifest = (Get-FileRecord $SceneManifest)
    validate_only = [bool]$ValidateOnly
    scenes = $runSummaries
    passed = @($runSummaries | Where-Object {
        -not $_.validated_only -and -not $_.gate.passed
    }).Count -eq 0
}
$summaryPath = Join-Path $runRoot "baseline_run.json"
[System.IO.File]::WriteAllText(
    $summaryPath,
    ($summary | ConvertTo-Json -Depth 12),
    [System.Text.UTF8Encoding]::new($false))
Write-Host "Baseline run report: $summaryPath"
if (-not $summary.passed) {
    exit 2
}
