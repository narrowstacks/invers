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
    pkgshare.install "config"
    pkgshare.install "profiles"
  end

  def post_install
    invers_dir = Pathname.new(Dir.home)/"invers"

    # Create ~/invers directory structure if it doesn't exist
    (invers_dir/"presets").mkpath

    # Copy config file if it doesn't exist
    config_file = invers_dir/"pipeline_defaults.yml"
    unless config_file.exist?
      cp pkgshare/"config/pipeline_defaults.yml", config_file
    end

    # Copy film presets if directory is empty
    presets_film_dir = invers_dir/"presets/film"
    presets_film_dir.mkpath
    if presets_film_dir.children.empty?
      cp_r pkgshare/"profiles/film/.", presets_film_dir
    end

    # Copy scan profiles if directory is empty
    presets_scan_dir = invers_dir/"presets/scan"
    presets_scan_dir.mkpath
    if presets_scan_dir.children.empty?
      cp_r pkgshare/"profiles/scan/.", presets_scan_dir
    end
  end

  def caveats
    <<~EOS
      Configuration files have been installed to ~/invers/

      You can customize these files:
        ~/invers/pipeline_defaults.yml  - Pipeline processing defaults
        ~/invers/presets/film/          - Film preset profiles
        ~/invers/presets/scan/          - Scanner profiles

      Default presets are available in:
        #{pkgshare}/
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/invers --version")
  end
end
