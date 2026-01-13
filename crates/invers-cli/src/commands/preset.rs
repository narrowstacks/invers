use std::path::PathBuf;

/// List available film presets in the specified or default directory.
pub fn cmd_preset_list(dir: Option<PathBuf>) -> Result<(), String> {
    let dir = dir.unwrap_or_else(|| {
        invers_core::presets::get_presets_dir().unwrap_or_else(|_| PathBuf::from("profiles/film"))
    });

    println!("Listing presets in: {}", dir.display());
    match invers_core::presets::list_film_presets(&dir) {
        Ok(presets) => {
            if presets.is_empty() {
                println!("No presets found.");
            } else {
                for preset in presets {
                    println!("  {}", preset);
                }
            }
            Ok(())
        }
        Err(e) => Err(format!("Failed to list presets: {}", e)),
    }
}

/// Display details of a film preset (base offsets, tone curve, color matrix).
pub fn cmd_preset_show(preset: String) -> Result<(), String> {
    println!("Loading preset: {}", preset);

    // Try to load as file first
    let preset_path = PathBuf::from(&preset);
    let preset_obj = if preset_path.exists() {
        invers_core::presets::load_film_preset(&preset_path)?
    } else {
        // Validate preset name before constructing path to prevent path traversal
        invers_core::presets::validate_preset_name(&preset)?;
        // Try to find it in the presets directory
        let dir = invers_core::presets::get_presets_dir()
            .unwrap_or_else(|_| PathBuf::from("profiles/film"));
        let full_path = dir.join(format!("{}.yml", preset));
        invers_core::presets::load_film_preset(&full_path)?
    };

    println!("\nPreset: {}", preset_obj.name);
    println!(
        "Base Offsets (RGB): [{:.6}, {:.6}, {:.6}]",
        preset_obj.base_offsets[0], preset_obj.base_offsets[1], preset_obj.base_offsets[2]
    );

    println!("\nTone Curve:");
    println!("  Type: {}", preset_obj.tone_curve.curve_type);
    println!("  Strength: {:.6}", preset_obj.tone_curve.strength);

    println!("\nColor Matrix (3x3):");
    for row in &preset_obj.color_matrix {
        println!("  [{:.6}, {:.6}, {:.6}]", row[0], row[1], row[2]);
    }

    if let Some(notes) = &preset_obj.notes {
        println!("\nNotes: {}", notes);
    }

    println!();
    Ok(())
}

/// Create a new film preset template file with default values.
pub fn cmd_preset_create(output: PathBuf, name: String) -> Result<(), String> {
    println!("Creating new preset: {}", name);

    // Create a default preset
    let preset = invers_core::models::FilmPreset {
        name: name.clone(),
        base_offsets: [0.0, 0.0, 0.0],
        color_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        tone_curve: invers_core::models::ToneCurveParams {
            curve_type: "neutral".to_string(),
            strength: 0.5,
            toe_strength: 0.4,
            shoulder_strength: 0.3,
            toe_length: 0.25,
            shoulder_start: 0.75,
            params: std::collections::HashMap::new(),
        },
        notes: Some(format!("Film preset: {}", name)),
    };

    // Serialize to YAML
    let yaml_str =
        serde_yaml::to_string(&preset).map_err(|e| format!("Failed to serialize preset: {}", e))?;

    // Write to file
    std::fs::write(&output, yaml_str).map_err(|e| format!("Failed to write preset file: {}", e))?;

    println!("Preset created: {}", output.display());
    println!("You can now edit this file to customize the parameters.");
    println!();

    Ok(())
}
