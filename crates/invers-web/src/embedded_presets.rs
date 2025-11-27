//! Compile-time embedded presets for WASM builds
//!
//! This module embeds film presets from the profiles directory at compile time,
//! allowing the WASM build to ship with presets without file system access.

use invers_core::models::FilmPreset;
use invers_core::presets::load_film_preset_from_str;
use once_cell::sync::Lazy;
use std::collections::HashMap;

// Embed preset YAML files at compile time
const GENERIC_COLOR_NEGATIVE: &str =
    include_str!("../../../profiles/film/generic-color-negative.yml");
const GENERIC_BW: &str = include_str!("../../../profiles/film/generic-bw.yml");
const OPTIMIZED_STANDARD: &str = include_str!("../../../profiles/film/optimized-standard.yml");
const DARK_BASE_NEGATIVE: &str = include_str!("../../../profiles/film/dark-base-negative.yml");
const LIGHT_BASE_NEGATIVE: &str = include_str!("../../../profiles/film/light-base-negative.yml");
const FUJI_SUPERIA_400: &str = include_str!("../../../profiles/film/fuji-superia-400.yml");
const GENERIC_COLOR_NEGATIVE_ASYMMETRIC: &str =
    include_str!("../../../profiles/film/generic-color-negative-asymmetric.yml");

/// All embedded film presets, keyed by slug name
pub static FILM_PRESETS: Lazy<HashMap<&'static str, FilmPreset>> = Lazy::new(|| {
    let mut presets = HashMap::new();

    // Parse each embedded preset and add to the map
    let preset_sources = [
        ("generic-color-negative", GENERIC_COLOR_NEGATIVE),
        ("generic-bw", GENERIC_BW),
        ("optimized-standard", OPTIMIZED_STANDARD),
        ("dark-base-negative", DARK_BASE_NEGATIVE),
        ("light-base-negative", LIGHT_BASE_NEGATIVE),
        ("fuji-superia-400", FUJI_SUPERIA_400),
        (
            "generic-color-negative-asymmetric",
            GENERIC_COLOR_NEGATIVE_ASYMMETRIC,
        ),
    ];

    for (slug, yaml) in preset_sources {
        match load_film_preset_from_str(yaml) {
            Ok(preset) => {
                presets.insert(slug, preset);
            }
            Err(e) => {
                log::warn!("Failed to parse embedded preset '{}': {}", slug, e);
            }
        }
    }

    presets
});

/// List of all available preset names (slugs)
pub static PRESET_NAMES: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "optimized-standard",
        "generic-color-negative",
        "generic-color-negative-asymmetric",
        "generic-bw",
        "dark-base-negative",
        "light-base-negative",
        "fuji-superia-400",
    ]
});

/// Get a film preset by slug name
pub fn get_preset(slug: &str) -> Option<&'static FilmPreset> {
    FILM_PRESETS.get(slug)
}

/// Get the display name of a preset
pub fn get_preset_display_name(slug: &str) -> &str {
    match FILM_PRESETS.get(slug) {
        Some(preset) => &preset.name,
        None => slug,
    }
}

/// Get the default preset (Optimized Standard)
pub fn get_default_preset() -> Option<&'static FilmPreset> {
    get_preset("optimized-standard")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presets_load() {
        // Force initialization
        let count = FILM_PRESETS.len();
        assert!(count > 0, "Should have loaded at least one preset");
    }

    #[test]
    fn test_default_preset() {
        let preset = get_default_preset();
        assert!(preset.is_some(), "Default preset should exist");
    }
}
