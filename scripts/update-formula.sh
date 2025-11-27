#!/bin/bash
# Script to update the Homebrew formula with SHA256 hashes from a release
# Usage: ./scripts/update-formula.sh v0.1.0

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <version-tag>"
    echo "Example: $0 v0.1.0"
    exit 1
fi

VERSION_TAG="$1"
VERSION="${VERSION_TAG#v}"  # Remove 'v' prefix
FORMULA_PATH="Formula/invers.rb"
REPO="narrowstacks/invers"

echo "Updating formula for version: $VERSION"

# Download SHA256 files from the release
BASE_URL="https://github.com/$REPO/releases/download/$VERSION_TAG"

echo "Downloading SHA256 files..."

X86_64_DARWIN_SHA=$(curl -sL "$BASE_URL/invers-x86_64-apple-darwin.tar.gz.sha256" | awk '{print $1}')
AARCH64_DARWIN_SHA=$(curl -sL "$BASE_URL/invers-aarch64-apple-darwin.tar.gz.sha256" | awk '{print $1}')
X86_64_LINUX_SHA=$(curl -sL "$BASE_URL/invers-x86_64-unknown-linux-gnu.tar.gz.sha256" | awk '{print $1}')

echo "SHA256 hashes:"
echo "  x86_64-apple-darwin:      $X86_64_DARWIN_SHA"
echo "  aarch64-apple-darwin:     $AARCH64_DARWIN_SHA"
echo "  x86_64-unknown-linux-gnu: $X86_64_LINUX_SHA"

# Create the updated formula
cat > "$FORMULA_PATH" << EOF
class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/$REPO"
  version "$VERSION"
  license any_of: ["MIT", "Apache-2.0"]

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
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/invers --version")
  end
end
EOF

echo ""
echo "Formula updated at: $FORMULA_PATH"
echo ""
echo "Next steps:"
echo "1. Review the formula: cat $FORMULA_PATH"
echo "2. Test locally: brew install --build-from-source $FORMULA_PATH"
echo "3. Commit and push to your homebrew-invers tap"
