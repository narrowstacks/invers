class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/narrowstacks/invers"
  version "0.2.0"
  license any_of: ["MIT"]

  on_macos do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "e80208109990139ed7c90d723ce34469969ad766c949aeafef851b53dff6d2e4"
    end

    on_arm do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "fcce8cd620ed6df9d6e407a939d3ca304072bf787fa3b6ff85312eeb0db129b6"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "e94dc8ca2ddaf1d4231ef8e6346d940a4ccfd46bfcfb3389ec81154ad0aacf24"
    end
  end

  def install
    bin.install "invers"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/invers --version")
  end
end
