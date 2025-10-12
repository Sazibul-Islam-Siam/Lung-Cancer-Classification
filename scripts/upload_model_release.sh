#!/usr/bin/env bash
# Bash helper: create a GitHub release (or upload asset to existing release) and upload model.pth
# Prerequisites:
#  - GitHub CLI (gh) installed and authenticated
#  - model.pth present in repository root
# Usage: ./scripts/upload_model_release.sh v1.0.0

TAG=${1:-v1.0.0}
MODEL=model.pth

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install from https://cli.github.com/ and authenticate (gh auth login)."
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "model.pth not found in repo root. Place model.pth in the repo root before running."
  exit 1
fi

REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [ -z "$REMOTE_URL" ]; then
  echo "No git remote 'origin' found. Make sure this repo has an origin pointing to GitHub."
  exit 1
fi

# Determine owner/repo
if [[ "$REMOTE_URL" =~ github.com[:/](.+?)/(.+?)(\.git)?$ ]]; then
  OWNER=${BASH_REMATCH[1]}
  REPO=${BASH_REMATCH[2]}
else
  echo "Could not parse GitHub repo from remote URL: $REMOTE_URL"
  exit 1
fi

FULL_REPO="$OWNER/$REPO"

echo "Repository: $FULL_REPO"

# Check release existence
if gh release view "$TAG" --repo "$FULL_REPO" >/dev/null 2>&1; then
  echo "Release $TAG exists — uploading asset (will overwrite if exists)..."
  gh release upload "$TAG" "$MODEL" --repo "$FULL_REPO" --clobber
else
  echo "Creating release $TAG and uploading asset..."
  gh release create "$TAG" --notes "Model weights $TAG" "$MODEL" --repo "$FULL_REPO"
fi

PUBLIC_URL="https://github.com/$OWNER/$REPO/releases/download/$TAG/$MODEL"

echo "Upload finished. Public URL: $PUBLIC_URL"
echo "Copy this URL and set it as MODEL_URL in Render (or other host)."