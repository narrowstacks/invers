use std::path::PathBuf;

/// Initialize user configuration directory with default presets.
///
/// Copies default configuration and preset files from the Homebrew share
/// directory to `~/invers/`. Safe to run multiple times - won't overwrite
/// existing files unless `force` is true.
pub fn cmd_init(force: bool) -> Result<(), String> {
    let home = std::env::var("HOME").map_err(|_| "Could not determine home directory")?;
    let invers_dir = PathBuf::from(&home).join("invers");

    println!(
        "Initializing invers configuration in: {}",
        invers_dir.display()
    );
    println!();

    // Look for default presets in common Homebrew locations
    let share_locations = [
        PathBuf::from("/opt/homebrew/opt/invers/share/invers"), // Apple Silicon
        PathBuf::from("/usr/local/opt/invers/share/invers"),    // Intel Mac
        PathBuf::from("/home/linuxbrew/.linuxbrew/opt/invers/share/invers"), // Linux
    ];

    let share_dir = share_locations.iter().find(|p| p.exists()).ok_or_else(|| {
        "Could not find invers share directory. Make sure invers is installed via Homebrew."
            .to_string()
    })?;

    println!("Found default presets in: {}", share_dir.display());
    println!();

    // Create directory structure
    let presets_film_dir = invers_dir.join("presets/film");
    let presets_scan_dir = invers_dir.join("presets/scan");

    std::fs::create_dir_all(&presets_film_dir)
        .map_err(|e| format!("Failed to create film presets directory: {}", e))?;
    std::fs::create_dir_all(&presets_scan_dir)
        .map_err(|e| format!("Failed to create scan presets directory: {}", e))?;

    // Copy pipeline_defaults.yml
    let src_config = share_dir.join("config/pipeline_defaults.yml");
    let dst_config = invers_dir.join("pipeline_defaults.yml");

    if src_config.exists() {
        if !dst_config.exists() || force {
            std::fs::copy(&src_config, &dst_config)
                .map_err(|e| format!("Failed to copy pipeline_defaults.yml: {}", e))?;
            println!("  Copied: pipeline_defaults.yml");
        } else {
            println!("  Skipped: pipeline_defaults.yml (already exists, use --force to overwrite)");
        }
    }

    // Copy film presets
    let src_film = share_dir.join("profiles/film");
    if src_film.exists() {
        copy_dir_contents(&src_film, &presets_film_dir, force, "  ")?;
    }

    // Copy scan profiles
    let src_scan = share_dir.join("profiles/scan");
    if src_scan.exists() {
        copy_dir_contents(&src_scan, &presets_scan_dir, force, "  ")?;
    }

    println!();
    println!("Initialization complete!");
    println!();
    println!("Configuration files are now in:");
    println!("  ~/invers/pipeline_defaults.yml  - Pipeline processing defaults");
    println!("  ~/invers/presets/film/          - Film preset profiles");
    println!("  ~/invers/presets/scan/          - Scanner profiles");

    Ok(())
}

/// Recursively copy directory contents, skipping existing files unless forced.
fn copy_dir_contents(
    src: &std::path::Path,
    dst: &std::path::Path,
    force: bool,
    indent: &str,
) -> Result<(), String> {
    let entries = std::fs::read_dir(src)
        .map_err(|e| format!("Failed to read directory {}: {}", src.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let src_path = entry.path();
        let file_name = src_path
            .file_name()
            .ok_or_else(|| format!("Invalid path (no filename): {}", src_path.display()))?;
        let dst_path = dst.join(file_name);

        if src_path.is_dir() {
            std::fs::create_dir_all(&dst_path)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
            copy_dir_contents(&src_path, &dst_path, force, indent)?;
        } else if !dst_path.exists() || force {
            std::fs::copy(&src_path, &dst_path)
                .map_err(|e| format!("Failed to copy {}: {}", src_path.display(), e))?;
            println!("{}Copied: {}", indent, file_name.to_string_lossy());
        } else {
            println!(
                "{}Skipped: {} (exists)",
                indent,
                file_name.to_string_lossy()
            );
        }
    }

    Ok(())
}
