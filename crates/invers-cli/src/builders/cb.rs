//! CB (curve-based) pipeline options builder.

use crate::parsers::{
    parse_cb_color_model, parse_cb_enhanced_profile, parse_cb_film_character,
    parse_cb_tone_profile, parse_cb_wb_preset,
};

/// Build CbOptions from parsed command line arguments
pub fn build_cb_options(
    tone_profile: Option<&str>,
    enhanced_profile: Option<&str>,
    color_model: Option<&str>,
    film_character: Option<&str>,
    wb_preset: Option<&str>,
) -> Result<invers_core::models::CbOptions, String> {
    let tone = parse_cb_tone_profile(tone_profile)?;
    let lut = parse_cb_enhanced_profile(enhanced_profile)?;
    let color = parse_cb_color_model(color_model)?;
    let film = parse_cb_film_character(film_character)?;
    let wb = parse_cb_wb_preset(wb_preset)?;

    let mut opts = invers_core::models::CbOptions::from_presets(
        tone,
        lut,
        color,
        film,
        invers_core::models::CbEnginePreset::V3_1,
    );
    opts.wb_preset = wb;
    Ok(opts)
}
