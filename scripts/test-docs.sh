#!/usr/bin/env bash
# Build and serve the Jupyter Book documentation locally for testing.
#
# Usage:
#   ./scripts/test-docs.sh          # Build and serve on port 8000
#   ./scripts/test-docs.sh --build  # Build only (no serve)
#   ./scripts/test-docs.sh --serve  # Serve existing build (skip rebuild)
#   ./scripts/test-docs.sh --port=3000
#
# Open http://localhost:8000 in your browser to view the docs.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUILD_ONLY=false
SERVE_ONLY=false
PORT=8000

for arg in "$@"; do
  case $arg in
    --build) BUILD_ONLY=true ;;
    --serve) SERVE_ONLY=true ;;
    --port=*) PORT="${arg#*=}" ;;
  esac
done

# Ensure docs dependencies are installed (docutils 0.17.1 is pinned to avoid Sphinx anchorname compatibility issues)
echo "Installing docs dependencies ([docs] extra)..."
uv pip install ".[docs]"

if [ "$SERVE_ONLY" = false ]; then
  echo "Building Jupyter Book..."
  uv run jupyter-book build . --config docs/_config.yml --toc docs/_toc.yml
fi

if [ "$BUILD_ONLY" = true ]; then
  echo "Build complete. Output: _build/html/"
  exit 0
fi

if [ ! -d "_build/html" ]; then
  echo "Error: _build/html not found. Run without --serve to build first."
  exit 1
fi

echo "Serving docs at http://localhost:$PORT"
echo "Press Ctrl+C to stop."
exec uv run python -m http.server "$PORT" --directory _build/html
