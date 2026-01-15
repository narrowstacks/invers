//! CB (curve-based) pipeline option parsing functions.

/// Parse CB tone profile from string
pub fn parse_cb_tone_profile(
    profile_str: Option<&str>,
) -> Result<invers_core::models::CbToneProfile, String> {
    match profile_str {
        None => Ok(invers_core::models::CbToneProfile::default()),
        Some(s) => match s.to_lowercase().replace('-', "_").as_str() {
            "standard" | "std" | "" => Ok(invers_core::models::CbToneProfile::Standard),
            "linear" => Ok(invers_core::models::CbToneProfile::Linear),
            "linear_gamma" | "lineargamma" => Ok(invers_core::models::CbToneProfile::LinearGamma),
            "linear_flat" | "linearflat" => Ok(invers_core::models::CbToneProfile::LinearFlat),
            "linear_deep" | "lineardeep" => Ok(invers_core::models::CbToneProfile::LinearDeep),
            "logarithmic" | "log" => Ok(invers_core::models::CbToneProfile::Logarithmic),
            "logarithmic_rich" | "log_rich" | "logrich" => {
                Ok(invers_core::models::CbToneProfile::LogarithmicRich)
            }
            "logarithmic_flat" | "log_flat" | "logflat" => {
                Ok(invers_core::models::CbToneProfile::LogarithmicFlat)
            }
            "all_soft" | "allsoft" | "soft" => Ok(invers_core::models::CbToneProfile::AllSoft),
            "all_hard" | "allhard" | "hard" => Ok(invers_core::models::CbToneProfile::AllHard),
            "highlight_hard" | "highlighthard" => {
                Ok(invers_core::models::CbToneProfile::HighlightHard)
            }
            "highlight_soft" | "highlightsoft" => {
                Ok(invers_core::models::CbToneProfile::HighlightSoft)
            }
            "shadow_hard" | "shadowhard" => Ok(invers_core::models::CbToneProfile::ShadowHard),
            "shadow_soft" | "shadowsoft" => Ok(invers_core::models::CbToneProfile::ShadowSoft),
            "autotone" | "auto" | "auto_tone" => Ok(invers_core::models::CbToneProfile::AutoTone),
            _ => Err(format!(
                "Unknown CB tone profile: '{}'. Valid: standard, linear, linear-gamma, \
                 linear-flat, linear-deep, log, log-rich, log-flat, soft, hard, \
                 highlight-hard, highlight-soft, shadow-hard, shadow-soft, auto",
                s
            )),
        },
    }
}

/// Parse CB enhanced profile (LUT) from string
pub fn parse_cb_enhanced_profile(
    profile_str: Option<&str>,
) -> Result<invers_core::models::CbEnhancedProfile, String> {
    match profile_str {
        None => Ok(invers_core::models::CbEnhancedProfile::default()),
        Some(s) => match s.to_lowercase().as_str() {
            "none" | "" => Ok(invers_core::models::CbEnhancedProfile::None),
            "natural" => Ok(invers_core::models::CbEnhancedProfile::Natural),
            "frontier" => Ok(invers_core::models::CbEnhancedProfile::Frontier),
            "crystal" => Ok(invers_core::models::CbEnhancedProfile::Crystal),
            "pakon" => Ok(invers_core::models::CbEnhancedProfile::Pakon),
            _ => Err(format!(
                "Unknown CB enhanced profile: '{}'. Valid: none, natural, frontier, crystal, pakon",
                s
            )),
        },
    }
}

/// Parse CB color model from string
pub fn parse_cb_color_model(
    model_str: Option<&str>,
) -> Result<invers_core::models::CbColorModel, String> {
    match model_str {
        None => Ok(invers_core::models::CbColorModel::default()),
        Some(s) => match s.to_lowercase().as_str() {
            "none" => Ok(invers_core::models::CbColorModel::None),
            "basic" | "" => Ok(invers_core::models::CbColorModel::Basic),
            "frontier" => Ok(invers_core::models::CbColorModel::Frontier),
            "noritsu" => Ok(invers_core::models::CbColorModel::Noritsu),
            "bw" | "blackandwhite" | "black_and_white" | "mono" => {
                Ok(invers_core::models::CbColorModel::BlackAndWhite)
            }
            _ => Err(format!(
                "Unknown CB color model: '{}'. Valid: none, basic, frontier, noritsu, bw",
                s
            )),
        },
    }
}

/// Parse CB film character from string
pub fn parse_cb_film_character(
    character_str: Option<&str>,
) -> Result<invers_core::models::CbFilmCharacter, String> {
    match character_str {
        None => Ok(invers_core::models::CbFilmCharacter::default()),
        Some(s) => match s.to_lowercase().replace('-', "_").as_str() {
            "none" => Ok(invers_core::models::CbFilmCharacter::None),
            "generic" | "genericcolor" | "generic_color" => {
                Ok(invers_core::models::CbFilmCharacter::GenericColor)
            }
            "kodak" | "portra" | "" => Ok(invers_core::models::CbFilmCharacter::Kodak),
            "fuji" | "fujifilm" => Ok(invers_core::models::CbFilmCharacter::Fuji),
            "cinestill_50d" | "cinestill50d" | "50d" => {
                Ok(invers_core::models::CbFilmCharacter::Cinestill50D)
            }
            "cinestill_800t" | "cinestill800t" | "800t" => {
                Ok(invers_core::models::CbFilmCharacter::Cinestill800T)
            }
            _ => Err(format!(
                "Unknown CB film character: '{}'. Valid: none, generic, kodak, fuji, \
                 cinestill-50d, cinestill-800t",
                s
            )),
        },
    }
}

/// Parse CB white balance preset from string
pub fn parse_cb_wb_preset(wb_str: Option<&str>) -> Result<invers_core::models::CbWbPreset, String> {
    match wb_str {
        None => Ok(invers_core::models::CbWbPreset::default()),
        Some(s) => match s.to_lowercase().replace('-', "_").as_str() {
            "none" => Ok(invers_core::models::CbWbPreset::None),
            "auto" | "autocolor" | "auto_color" | "avg" | "auto_avg" | "" => {
                Ok(invers_core::models::CbWbPreset::AutoColor)
            }
            "neutral" | "autoneutral" | "auto_neutral" => {
                Ok(invers_core::models::CbWbPreset::AutoNeutral)
            }
            "warm" | "autowarm" | "auto_warm" => Ok(invers_core::models::CbWbPreset::AutoWarm),
            "cool" | "autocool" | "auto_cool" => Ok(invers_core::models::CbWbPreset::AutoCool),
            "mix" | "automix" | "auto_mix" => Ok(invers_core::models::CbWbPreset::AutoMix),
            "standard" | "std" => Ok(invers_core::models::CbWbPreset::Standard),
            "kodak" => Ok(invers_core::models::CbWbPreset::Kodak),
            "fuji" | "fujifilm" => Ok(invers_core::models::CbWbPreset::Fuji),
            "cine_t" | "cinet" | "cinestill_t" | "tungsten" => {
                Ok(invers_core::models::CbWbPreset::CineT)
            }
            "cine_d" | "cined" | "cinestill_d" | "daylight" => {
                Ok(invers_core::models::CbWbPreset::CineD)
            }
            "custom" => Ok(invers_core::models::CbWbPreset::Custom),
            _ => Err(format!(
                "Unknown CB WB preset: '{}'. Valid: none, auto (default), neutral, warm, cool, \
                 mix, standard, kodak, fuji, cine-t, cine-d, custom",
                s
            )),
        },
    }
}
