#!/bin/bash
# Script to update the Homebrew formula with SHA256 hashes from a release
# and push to the homebrew-invers tap
#
# Usage: ./scripts/update-formula.sh v0.1.0

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <version-tag>"
    echo "Example: $0 v0.1.0"
    exit 1
fi

VERSION_TAG="$1"
VERSION="${VERSION_TAG#v}"  # Remove 'v' prefix
REPO="narrowstacks/invers"
TAP_REPO_PATH="${TAP_REPO_PATH:-$HOME/GitHub/homebrew-invers}"
FORMULA_NAME="invers.rb"

echo "============================================"
echo "Updating Homebrew formula for invers $VERSION"
echo "============================================"
echo ""

# Check if tap repo exists
if [ ! -d "$TAP_REPO_PATH" ]; then
    echo "Error: Tap repo not found at $TAP_REPO_PATH"
    echo "Clone it with: git clone https://github.com/narrowstacks/homebrew-invers.git $TAP_REPO_PATH"
    echo "Or set TAP_REPO_PATH environment variable to the correct path."
    exit 1
fi

# Download SHA256 files from the release
BASE_URL="https://github.com/$REPO/releases/download/$VERSION_TAG"

echo "Fetching SHA256 checksums from release..."
echo ""

X86_64_DARWIN_SHA=$(curl -sfL "$BASE_URL/invers-x86_64-apple-darwin.tar.gz.sha256" | awk '{print $1}')
AARCH64_DARWIN_SHA=$(curl -sfL "$BASE_URL/invers-aarch64-apple-darwin.tar.gz.sha256" | awk '{print $1}')
X86_64_LINUX_SHA=$(curl -sfL "$BASE_URL/invers-x86_64-unknown-linux-gnu.tar.gz.sha256" | awk '{print $1}')

# Validate we got actual hashes
if [ -z "$X86_64_DARWIN_SHA" ] || [ -z "$AARCH64_DARWIN_SHA" ] || [ -z "$X86_64_LINUX_SHA" ]; then
    echo "Error: Failed to download SHA256 checksums."
    echo "Make sure the release $VERSION_TAG exists and has all artifacts."
    echo ""
    echo "Expected files at $BASE_URL:"
    echo "  - invers-x86_64-apple-darwin.tar.gz.sha256"
    echo "  - invers-aarch64-apple-darwin.tar.gz.sha256"
    echo "  - invers-x86_64-unknown-linux-gnu.tar.gz.sha256"
    exit 1
fi

echo "SHA256 checksums:"
echo "  x86_64-apple-darwin:      $X86_64_DARWIN_SHA"
echo "  aarch64-apple-darwin:     $AARCH64_DARWIN_SHA"
echo "  x86_64-unknown-linux-gnu: $X86_64_LINUX_SHA"
echo ""

# Create the updated formula
FORMULA_PATH="$TAP_REPO_PATH/Formula/$FORMULA_NAME"
mkdir -p "$TAP_REPO_PATH/Formula"

cat > "$FORMULA_PATH" << EOF
class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/$REPO"
  version "$VERSION"
  license any_of: ["MIT"]

  on_macos do
    on_intel do
      url "https://github.com/$REPO/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "$X86_64_DARWIN_SHA"
    end

    on_arm do
      url "https://github.com/$REPO/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "$AARCH64_DARWIN_SHA"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/$REPO/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "$X86_64_LINUX_SHA"
    end
  end

  def install
    bin.install "invers"
    pkgshare.install "config"
    pkgshare.install "profiles"
  end

  def caveats
    <<~EOS
      To set up your configuration directory with default presets, run:
        invers init

      This will create ~/invers/ with:
        ~/invers/pipeline_defaults.yml  - Pipeline processing defaults
        ~/invers/presets/film/          - Film preset profiles
        ~/invers/presets/scan/          - Scanner profiles

      Default presets are also available in:
        #{pkgshare}/
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/invers --version")
  end
end
EOF

echo "Formula written to: $FORMULA_PATH"
echo ""

# Also update local copy for reference
LOCAL_FORMULA="Formula/$FORMULA_NAME"
mkdir -p "Formula"
cp "$FORMULA_PATH" "$LOCAL_FORMULA"
echo "Local copy updated: $LOCAL_FORMULA"
echo ""

# Commit and push to tap repo
echo "Pushing to homebrew-invers tap..."
cd "$TAP_REPO_PATH"
git add "Formula/$FORMULA_NAME"
git commit -m "Update invers to $VERSION"
git push origin main

echo ""
echo "============================================"
echo "SUCCESS!"
echo "============================================"
echo ""
echo "Formula published to: https://github.com/narrowstacks/homebrew-invers"
echo ""
echo "Users can now install with:"
echo "  brew tap narrowstacks/invers"
echo "  brew install invers"
echo ""
echo "Or directly:"
echo "  brew install narrowstacks/invers/invers"
