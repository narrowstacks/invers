//! Preset and profile management
//!
//! Load, save, and manage film presets and scan profiles.

use crate::models::{FilmPreset, ScanProfile};
use std::path::Path;

/// Validate a preset name to prevent path traversal attacks.
/// Rejects names containing path separators, "..", or other dangerous patterns.
pub fn validate_preset_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Preset name cannot be empty".to_string());
    }

    // Reject path separators
    if name.contains('/') || name.contains('\\') {
        return Err("Preset name cannot contain path separators".to_string());
    }

    // Reject parent directory references
    if name.contains("..") {
        return Err("Preset name cannot contain '..'".to_string());
    }

    // Reject names that start with a dot (hidden files)
    if name.starts_with('.') {
        return Err("Preset name cannot start with '.'".to_string());
    }

    // Reject null bytes
    if name.contains('\0') {
        return Err("Preset name cannot contain null bytes".to_string());
    }

    Ok(())
}

/// Load a film preset from a YAML file
pub fn load_film_preset<P: AsRef<Path>>(path: P) -> Result<FilmPreset, String> {
    let path = path.as_ref();
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read preset file: {}", e))?;

    serde_yaml::from_str(&contents).map_err(|e| format!("Failed to parse preset YAML: {}", e))
}

/// Save a film preset to a YAML file
pub fn save_film_preset<P: AsRef<Path>>(preset: &FilmPreset, path: P) -> Result<(), String> {
    let path = path.as_ref();
    let yaml =
        serde_yaml::to_string(preset).map_err(|e| format!("Failed to serialize preset: {}", e))?;

    std::fs::write(path, yaml).map_err(|e| format!("Failed to write preset file: {}", e))
}

/// Load a scan profile from a YAML file
pub fn load_scan_profile<P: AsRef<Path>>(path: P) -> Result<ScanProfile, String> {
    let path = path.as_ref();
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read profile file: {}", e))?;

    serde_yaml::from_str(&contents).map_err(|e| format!("Failed to parse profile YAML: {}", e))
}

/// Save a scan profile to a YAML file
pub fn save_scan_profile<P: AsRef<Path>>(profile: &ScanProfile, path: P) -> Result<(), String> {
    let path = path.as_ref();
    let yaml = serde_yaml::to_string(profile)
        .map_err(|e| format!("Failed to serialize profile: {}", e))?;

    std::fs::write(path, yaml).map_err(|e| format!("Failed to write profile file: {}", e))
}

/// List all available film presets in a directory
pub fn list_film_presets<P: AsRef<Path>>(dir: P) -> Result<Vec<String>, String> {
    let dir = dir.as_ref();
    let mut presets = Vec::new();

    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("Failed to read presets directory: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) == Some("yml")
            || path.extension().and_then(|e| e.to_str()) == Some("yaml")
        {
            if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                presets.push(name.to_string());
            }
        }
    }

    Ok(presets)
}

/// Get the default presets directory
pub fn get_presets_dir() -> Result<std::path::PathBuf, String> {
    let home_dir =
        dirs::home_dir().ok_or_else(|| "Could not determine home directory".to_string())?;

    let presets_dir = home_dir.join("invers").join("presets");

    // Create directory if it doesn't exist
    if !presets_dir.exists() {
        std::fs::create_dir_all(&presets_dir)
            .map_err(|e| format!("Failed to create presets directory: {}", e))?;
    }

    Ok(presets_dir)
}
