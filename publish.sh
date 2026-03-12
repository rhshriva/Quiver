#!/usr/bin/env bash
# publish.sh — build cross-platform wheels and upload to PyPI
#
# Usage:
#   ./publish.sh              # build all platforms + upload to PyPI
#   ./publish.sh --test       # upload to TestPyPI instead
#   ./publish.sh --build-only # build wheels without uploading
#   ./publish.sh --help       # show this message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[quiver]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[ok]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[warn]${RESET} $*"; }
die()     { echo -e "${RED}${BOLD}[error]${RESET} $*" >&2; exit 1; }

OPT_TEST=0
OPT_BUILD_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --test)       OPT_TEST=1       ;;
    --build-only) OPT_BUILD_ONLY=1 ;;
    --help|-h)
      sed -n '2,7p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) die "unknown option: $arg  (run with --help)" ;;
  esac
done

# ── virtual environment ──────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at .venv/ ..."
  python3 -m venv "$VENV_DIR"
fi

info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# ── prerequisites ────────────────────────────────────────────────────────────
command -v cargo &>/dev/null || die "cargo not found"

info "Installing Python build dependencies..."
pip install --quiet --upgrade maturin ziglang twine

# ── ensure rust targets are installed ────────────────────────────────────────
TARGETS=(
  "aarch64-apple-darwin"
  "x86_64-apple-darwin"
  "x86_64-unknown-linux-gnu"
  "aarch64-unknown-linux-gnu"
)

info "Checking Rust cross-compilation targets..."
INSTALLED=$(rustup target list --installed)
for target in "${TARGETS[@]}"; do
  if ! echo "$INSTALLED" | grep -q "$target"; then
    info "Adding target: $target"
    rustup target add "$target"
  fi
done

# ── clean old wheels ─────────────────────────────────────────────────────────
WHEEL_DIR="target/wheels"
rm -f "$WHEEL_DIR"/quiver_vector_db-*.whl 2>/dev/null || true

# ── build wheels ─────────────────────────────────────────────────────────────
MANIFEST=crates/quiver-python/Cargo.toml

for target in "${TARGETS[@]}"; do
  info "Building wheel for $target ..."
  maturin build --release --zig -m "$MANIFEST" --target "$target" || {
    warn "Failed to build for $target -- skipping"
    continue
  }
  success "Built wheel for $target"
done

echo ""
info "Built wheels:"
ls -lh "$WHEEL_DIR"/quiver_vector_db-*.whl
echo ""

if [[ $OPT_BUILD_ONLY -eq 1 ]]; then
  success "Build complete. Wheels are in $WHEEL_DIR/"
  exit 0
fi

# ── upload ───────────────────────────────────────────────────────────────────
if [[ $OPT_TEST -eq 1 ]]; then
  info "Uploading to TestPyPI..."
  twine upload --repository testpypi "$WHEEL_DIR"/quiver_vector_db-*.whl
  echo ""
  success "Published to TestPyPI!"
  echo "  Install: pip install --index-url https://test.pypi.org/simple/ quiver-vector-db"
else
  info "Uploading to PyPI..."
  twine upload "$WHEEL_DIR"/quiver_vector_db-*.whl
  echo ""
  success "Published to PyPI!"
  echo "  Install: pip install quiver-vector-db"
fi
