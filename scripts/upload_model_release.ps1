<#
PowerShell helper: create a GitHub release (or upload asset to existing release) and upload model.pth
Prerequisites:
 - GitHub CLI (gh) installed and authenticated: https://cli.github.com/
 - model.pth present in repository root
Usage:
 .\scripts\upload_model_release.ps1 -Tag v1.0.0
#>
param(
    [string]$Tag = "v1.0.0"
)

# Ensure gh exists
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "GitHub CLI 'gh' is not installed. Install from https://cli.github.com/ and authenticate (gh auth login)."
    exit 1
}

# Ensure model file exists
$modelPath = Join-Path (Get-Location) 'model.pth'
if (-not (Test-Path $modelPath)) {
    Write-Error "model.pth not found at $modelPath. Place the model file in the repository root before running this script."
    exit 1
}

# Get repo owner/name from git remote
$remote = git remote get-url origin 2>$null
if (-not $remote) {
    Write-Error "No git remote 'origin' found. Make sure this repo has an origin pointing to GitHub."
    exit 1
}

# Parse owner/repo
if ($remote -match 'github.com[:/](.+?)/(.+?)(?:\.git)?$') {
    $owner = $Matches[1]
    $repo = $Matches[2]
} else {
    Write-Error "Could not parse GitHub repo from remote URL: $remote"
    exit 1
}

$fullRepo = "$owner/$repo"
Write-Output "Repository: $fullRepo"

# Check if release exists
$releaseExists = $false
try {
    gh release view $Tag --repo $fullRepo > $null 2>&1
    if ($LASTEXITCODE -eq 0) { $releaseExists = $true }
} catch {
    $releaseExists = $false
}

if ($releaseExists) {
    Write-Output "Release $Tag exists — uploading asset (will overwrite if exists)..."
    # --clobber overwrites existing asset with same name
    gh release upload $Tag $modelPath --repo $fullRepo --clobber
} else {
    Write-Output "Creating release $Tag and uploading asset..."
    gh release create $Tag --notes "Model weights $Tag" $modelPath --repo $fullRepo
}

$fileName = Split-Path $modelPath -Leaf
$publicUrl = "https://github.com/$owner/$repo/releases/download/$Tag/$fileName"
Write-Output "Upload finished. Public URL: $publicUrl"
Write-Output "Copy this URL and set it as MODEL_URL in Render (or other host)."
