class Invers < Formula
  desc "Professional-grade film negative to positive conversion tool"
  homepage "https://github.com/narrowstacks/invers"
  version "0.2.5"
  license any_of: ["MIT"]

  on_macos do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-apple-darwin.tar.gz"
      sha256 "5bc76f09d449b6098db541518a5fefc95f44cc50f2d441e1c2c3ee153d044e49"
    end

    on_arm do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-aarch64-apple-darwin.tar.gz"
      sha256 "dc52c87a23d59ce9280ec0be10d96506cc569025cdbb6d18037b36f6ec11ce53"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/narrowstacks/invers/releases/download/v#{version}/invers-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "77d03068e24805fd9f3bca92438f64e3e7f1590294aecba03ec943f08f2302b1"
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
