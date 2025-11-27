class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/narrowstacks/invers"
  version "0.1.0"
  license any_of: ["MIT", "Apache-2.0"]

  on_macos do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "Not"
    end

    on_arm do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "Not"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "Not"
    end
  end

  def install
    bin.install "invers"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/invers --version")
  end
end
