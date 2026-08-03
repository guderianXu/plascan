param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeDir,

    [Parameter(Mandatory = $true)]
    [string]$SetupScript,

    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",

    [string]$CudaWheel = "cu130"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Step([string]$Message)
{
    Write-Output "[PlaScan] $Message"
}

function Resolve-UsablePython
{
    param([string]$ManagedPython)

    if (Test-Path -LiteralPath $ManagedPython -PathType Leaf)
    {
        return (Resolve-Path -LiteralPath $ManagedPython).Path
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher)
    {
        $resolved = & $launcher.Source -3.12 -c "import sys; print(sys.executable); raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved)
        {
            return $resolved.Trim()
        }
    }

    foreach ($commandName in @("python3.exe", "python.exe", "python3", "python"))
    {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if (-not $command)
        {
            continue
        }

        $resolved = & $command.Source -c "import sys; print(sys.executable); raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved)
        {
            return $resolved.Trim()
        }
    }

    return ""
}

function Install-BootstrapPython
{
    $version = "3.12.10"
    $downloadUrl = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"
    $cacheDir = Join-Path $env:LOCALAPPDATA "PlaScan\downloads"
    $installerPath = Join-Path $cacheDir "python-$version-amd64.exe"
    $targetDir = Join-Path $env:LOCALAPPDATA "PlaScan\bootstrap-python\$version"
    $pythonPath = Join-Path $targetDir "python.exe"

    if (Test-Path -LiteralPath $pythonPath -PathType Leaf)
    {
        return $pythonPath
    }

    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf))
    {
        Write-Step "Downloading Python $version from python.org..."
        Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
    }

    $signature = Get-AuthenticodeSignature -FilePath $installerPath
    if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
        $signature.SignerCertificate.Subject -notmatch "Python Software Foundation")
    {
        Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
        throw "The downloaded Python installer has an invalid signature and will not be executed."
    }

    Write-Step "Installing the per-user PlaScan bootstrap Python..."
    $arguments = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=$targetDir",
        "Include_pip=1",
        "Include_launcher=0",
        "Include_test=0",
        "PrependPath=0",
        "Shortcuts=0"
    )
    $process = Start-Process -FilePath $installerPath -ArgumentList $arguments -PassThru -Wait
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf))
    {
        throw "Bootstrap Python installation failed with exit code $($process.ExitCode)."
    }

    return $pythonPath
}

if (-not (Test-Path -LiteralPath $SetupScript -PathType Leaf))
{
    throw "PlaScan Python setup script was not found: $SetupScript"
}

$managedPython = Join-Path $RuntimeDir "Scripts\python.exe"
$bootstrapPython = Resolve-UsablePython -ManagedPython $managedPython
if (-not $bootstrapPython)
{
    $bootstrapPython = Install-BootstrapPython
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Step "Creating or updating the PlaScan Python runtime..."
$setupArguments = @(
    $SetupScript,
    "--source-dir", $SourceDir,
    "--runtime-dir", $RuntimeDir,
    "--output-dir", $OutputDir,
    "--python", $bootstrapPython,
    "--device", $Device,
    "--cuda-wheel", $CudaWheel
)
if (Test-Path -LiteralPath $managedPython -PathType Leaf)
{
    $setupArguments += "--skip-create"
}

& $bootstrapPython @setupArguments
if ($LASTEXITCODE -ne 0)
{
    throw "PlaScan Python dependency installation failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path -LiteralPath $managedPython -PathType Leaf))
{
    throw "Python setup completed but the managed runtime was not found: $managedPython"
}

Write-Step "Python runtime is ready: $managedPython"
