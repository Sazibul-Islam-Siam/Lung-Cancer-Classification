<#
Simple PowerShell helper to upload model.pth as a GitHub Release asset.

Prerequisites:
 - GitHub CLI (gh) installed and authenticated (gh auth login)
 - model.pth present in the repository root

Usage:
 .\scripts\upload_model_release.ps1 -Tag v1.0.0
#>

param(
    [string]$Tag = 'v1.0.0'
)

function Fail($msg) { Write-Host "ERROR: $msg"; exit 1 }

# Check gh
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Fail 'GitHub CLI "gh" is not installed. Install from https://cli.github.com/' }

# Check model file
$modelPath = Join-Path (Get-Location) 'model.pth'
if (-not (Test-Path $modelPath)) { Fail "model.pth not found at $modelPath. Place the file in the repository root." }

# Get origin remote URL
$remote = git remote get-url origin 2>$null
if (-not $remote) { Fail "No git remote 'origin' found. Make sure this repo has an origin pointing to GitHub." }

# Parse owner and repo (simple parse)
$parts = $remote -split '[:/]'
# The last two path parts should be owner and repo.git (or repo)
if ($parts.Length -lt 2) { Fail "Could not parse remote URL: $remote" }
$owner = $parts[-2]
$repoWithExt = $parts[-1]
$repo = if ($repoWithExt.EndsWith('.git')) { $repoWithExt.Substring(0, $repoWithExt.Length - 4) } else { $repoWithExt }

Write-Host "Repository: $owner/$repo"

# Check for existing release
gh release view $Tag --repo "${owner}/${repo}" > $null 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Release $Tag exists. Uploading asset (will overwrite if exists)..."
    gh release upload $Tag $modelPath --repo "${owner}/${repo}" --clobber
} else {
    Write-Host "Creating release $Tag and uploading asset..."
    gh release create $Tag --notes "Model weights $Tag" $modelPath --repo "${owner}/${repo}"
}

$fileName = Split-Path $modelPath -Leaf
$publicUrl = "https://github.com/$owner/$repo/releases/download/$Tag/$fileName"
Write-Host "Upload finished. Public URL: $publicUrl"
Write-Host "Copy this URL and set it as MODEL_URL in Render (or other host)."
