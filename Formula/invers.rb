class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/narrowstacks/invers"
  version "0.3.0"
  license any_of: ["MIT"]

  on_macos do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "d8ff425872b6ce2e902d716ad273d3e3224b72915a40e3532b9491c957e55269"
    end

    on_arm do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "34c5bd9b772aff3c67d6cd49dfb45b19821dc347781829e66c8a9483e216a2ee"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "800d6cee8f7026e50bbdaa6b84fc07639c224da39fbae45c90db14fe03b47ec1"
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
