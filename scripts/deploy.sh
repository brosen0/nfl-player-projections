#!/usr/bin/env bash
# Deploy the draft advisor to the gh-pages branch.
# GitHub Pages auto-serves whatever is in gh-pages root.
#
# Usage:
#   bash scripts/deploy.sh              # current season
#   bash scripts/deploy.sh --season 2027

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Generating HTML..."
python scripts/generate_dashboard_html.py "$@"

REMOTE=$(git remote get-url origin)
DEPLOY_DIR=$(mktemp -d)
trap 'rm -rf "$DEPLOY_DIR"' EXIT

echo "Deploying to gh-pages..."
cd "$DEPLOY_DIR"
git init -b gh-pages
git remote add origin "$REMOTE"

# Pull existing gh-pages history if branch exists (keeps commit log readable).
git fetch origin gh-pages 2>/dev/null && git reset --hard origin/gh-pages 2>/dev/null || true

# Copy fresh HTML after any reset.
cp "$REPO_ROOT/_site/index.html" ./index.html
git add index.html
git commit -m "Deploy $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push origin gh-pages --force

echo "Done. GitHub Pages will update in ~30 seconds."
