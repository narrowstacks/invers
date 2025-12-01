class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/narrowstacks/invers"
  version "0.2.4"
  license any_of: ["MIT"]

  on_macos do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "97d406681b9009ae488c37f820eb813d22adaca5bedff7ab8725fb70c8a04f71"
    end

    on_arm do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "693f9e1d5261de5bc504346431baeb1eab48d32e70babc03c28e227452e2ccf4"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "50663501b34e415ca6b6e2b141725e991e9f883927c5267eac0b6d5a5af1d33e"
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
