class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/narrowstacks/invers"
  version "0.2.2"
  license any_of: ["MIT"]

  on_macos do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "4abf53abb62c81467d86a21af608150cb5dfb408e70b1acff1b8d3d279796553"
    end

    on_arm do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "fa962865b6784a6dc2203d1435593d3791d08008d5b46fdece4fb11277882b0e"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "0d91ce46e037e13630dbdeffdac36669c97d875bcd3db3568d18fe7af1657ef0"
    end
  end

  def install
    bin.install "invers"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/invers --version")
  end
end
