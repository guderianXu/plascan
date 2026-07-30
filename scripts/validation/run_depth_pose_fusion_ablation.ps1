param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [string]$OutputDirectory = "",
    [string]$Variant = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildDirectory = Join-Path $repoRoot "build\windows-vcpkg-cuda-release"

function Expand-PathTokens
{
    param([Parameter(Mandatory = $true)][string]$Value)

    return $Value.
        Replace('${REPO_ROOT}', $repoRoot).
        Replace('${BUILD_DIR}', $buildDirectory)
}

function Resolve-RequiredPath
{
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $expanded = Expand-PathTokens -Value $Value
    if (-not [System.IO.Path]::IsPathRooted($expanded))
    {
        $expanded = Join-Path $repoRoot $expanded
    }
    if (-not (Test-Path -LiteralPath $expanded))
    {
        throw "$Description not found: $expanded"
    }
    return (Resolve-Path -LiteralPath $expanded).Path
}

function Get-ArtifactFingerprint
{
    param([Parameter(Mandatory = $true)][string]$Path)

    $item = Get-Item -LiteralPath $Path
    $files = if ($item.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $item.FullName -Recurse -File |
            Sort-Object FullName)
    } else {
        @($item)
    }
    $records = foreach ($file in $files)
    {
        [ordered]@{
            relative_path = if ($item.PSIsContainer) {
                [System.IO.Path]::GetRelativePath($item.FullName, $file.FullName)
            } else {
                $file.Name
            }
            size_bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $canonical = $records | ConvertTo-Json -Depth 5 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($canonical)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try
    {
        $fingerprint = [System.Convert]::ToHexString(
            $hasher.ComputeHash($bytes)).ToLowerInvariant()
    }
    finally
    {
        $hasher.Dispose()
    }
    return [ordered]@{
        path = $item.FullName
        file_count = $files.Count
        sha256 = $fingerprint
        files = @($records)
    }
}

function Get-StageFingerprints
{
    param([Parameter(Mandatory = $true)][pscustomobject]$Stages)

    $result = [ordered]@{}
    foreach ($stageProperty in $Stages.PSObject.Properties)
    {
        $artifacts = @()
        foreach ($configuredPath in @($stageProperty.Value))
        {
            $resolved = Resolve-RequiredPath `
                -Value ([string]$configuredPath) `
                -Description "Stage '$($stageProperty.Name)' artifact"
            $artifacts += Get-ArtifactFingerprint -Path $resolved
        }
        $result[$stageProperty.Name] = @($artifacts)
    }
    return $result
}

$configPath = Resolve-RequiredPath -Value $Config -Description "Ablation config"
$definition = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 |
    ConvertFrom-Json
if ($null -eq $definition.scenes -or @($definition.scenes).Count -eq 0)
{
    throw "Ablation config must define at least one scene."
}

$outputRoot = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path (Split-Path -Parent $configPath) (
        [System.IO.Path]::GetFileNameWithoutExtension($configPath) + "_results")
} else {
    Expand-PathTokens -Value $OutputDirectory
}
if (-not [System.IO.Path]::IsPathRooted($outputRoot))
{
    $outputRoot = Join-Path $repoRoot $outputRoot
}
if ((Test-Path -LiteralPath $outputRoot) -and -not $DryRun)
{
    throw "Output directory already exists; choose a new run directory: $outputRoot"
}

$manifest = [ordered]@{
    schema = "plascan.depth_pose_fusion_ablation.v1"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    config = $configPath
    repository_root = $repoRoot
    build_directory = $buildDirectory
    dry_run = [bool]$DryRun
    scenes = @()
}

foreach ($scene in @($definition.scenes))
{
    if ([string]::IsNullOrWhiteSpace([string]$scene.name))
    {
        throw "Every scene must have a non-empty name."
    }
    $sceneRecord = [ordered]@{
        name = [string]$scene.name
        stage_fingerprints = Get-StageFingerprints -Stages $scene.stages
        variants = @()
    }
    foreach ($variantDefinition in @($scene.variants))
    {
        $variantName = [string]$variantDefinition.name
        if (-not [string]::IsNullOrWhiteSpace($Variant) -and
            $variantName -ne $Variant)
        {
            continue
        }
        $executable = Resolve-RequiredPath `
            -Value ([string]$variantDefinition.executable) `
            -Description "Variant '$variantName' executable"
        $workingDirectory = if (
            [string]::IsNullOrWhiteSpace([string]$variantDefinition.working_directory)) {
            $repoRoot
        } else {
            Resolve-RequiredPath `
                -Value ([string]$variantDefinition.working_directory) `
                -Description "Variant '$variantName' working directory"
        }
        $arguments = @($variantDefinition.arguments | ForEach-Object {
            Expand-PathTokens -Value ([string]$_)
        })
        $variantOutput = Join-Path (Join-Path $outputRoot $scene.name) $variantName
        $arguments = @($arguments | ForEach-Object {
            $_.Replace('${VARIANT_OUTPUT}', $variantOutput)
        })
        $variantRecord = [ordered]@{
            name = $variantName
            executable = $executable
            arguments = $arguments
            output_directory = $variantOutput
            executed = $false
            exit_code = $null
            elapsed_seconds = 0.0
            expected_outputs = @()
        }
        if (-not $DryRun)
        {
            New-Item -ItemType Directory -Path $variantOutput -Force |
                Out-Null
            $logPath = Join-Path $variantOutput "command.log"
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            & $executable @arguments *> $logPath
            $variantRecord.exit_code = $LASTEXITCODE
            $stopwatch.Stop()
            $variantRecord.elapsed_seconds =
                [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
            $variantRecord.executed = $true
            if ($LASTEXITCODE -ne 0)
            {
                throw "Variant '$variantName' failed with exit code $LASTEXITCODE. See $logPath"
            }
            foreach ($expectedOutput in @($variantDefinition.expected_outputs))
            {
                $expandedOutput = (Expand-PathTokens -Value ([string]$expectedOutput)).
                    Replace('${VARIANT_OUTPUT}', $variantOutput)
                $resolvedOutput = Resolve-RequiredPath `
                    -Value $expandedOutput `
                    -Description "Variant '$variantName' output"
                $variantRecord.expected_outputs +=
                    Get-ArtifactFingerprint -Path $resolvedOutput
            }
        }
        $sceneRecord.variants += $variantRecord
    }
    $manifest.scenes += $sceneRecord
}

if ($DryRun)
{
    $manifest | ConvertTo-Json -Depth 12
    exit 0
}

$manifestPath = Join-Path $outputRoot "ablation_manifest.json"
$manifest | ConvertTo-Json -Depth 12 |
    Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "Ablation manifest: $manifestPath"
