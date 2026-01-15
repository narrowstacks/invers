class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/narrowstacks/invers"
  version "0.5.0"
  license any_of: ["MIT"]

  on_macos do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "6e9b7ad6b19750de406651914957ee596de985a3ab01ab83e85458cbebfe79b5"
    end

    on_arm do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "ff2edc5e7ee478fd1c9d0d50858351bb9ffa0836eac3383d5f84dd8f5d492be6"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "6cd7c831182f91c7a090cbcc3c07baaf84dde608a6fde723cc2e6cb38c108a1e"
    end
  end

  def install
    bin.install "invers"
    pkgshare.install "config"
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
