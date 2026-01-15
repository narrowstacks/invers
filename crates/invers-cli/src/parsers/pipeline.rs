//! Pipeline and inversion mode parsing functions.

use super::parse_roi;

/// Parse pipeline mode from string
///
/// Supported values:
/// - "legacy" (default): Original Invers pipeline
/// - "research": Research-based pipeline with density balance before inversion
/// - "cb": CB-style curve-based pipeline
pub fn parse_pipeline_mode(mode_str: &str) -> Result<invers_core::models::PipelineMode, String> {
    match mode_str.to_lowercase().as_str() {
        "legacy" | "default" | "" => Ok(invers_core::models::PipelineMode::Legacy),
        "research" | "new" | "density" => Ok(invers_core::models::PipelineMode::Research),
        "cb" | "cbstyle" | "cb-style" => Ok(invers_core::models::PipelineMode::CbStyle),
        _ => Err(format!(
            "Unknown pipeline mode: '{}'. Valid options: legacy (default), research, cb",
            mode_str
        )),
    }
}

/// Parse inversion mode from string
///
/// Supported values:
/// - "mask-aware" / "mask" (default): Orange mask-aware inversion for color negative film
/// - "linear": Simple (base - negative) / base inversion
/// - "log" / "logarithmic": Density-based inversion
/// - "divide-blend" / "divide": Photoshop-style divide blend mode
/// - "bw" / "blackandwhite" / "grayscale": Simple B&W inversion with headroom
pub fn parse_inversion_mode(
    mode_str: Option<&str>,
) -> Result<Option<invers_core::models::InversionMode>, String> {
    match mode_str {
        None => Ok(None), // Use default from config
        Some(s) => match s.to_lowercase().as_str() {
            "mask-aware" | "mask" | "maskaware" => {
                Ok(Some(invers_core::models::InversionMode::MaskAware))
            }
            "linear" => Ok(Some(invers_core::models::InversionMode::Linear)),
            "log" | "logarithmic" => Ok(Some(invers_core::models::InversionMode::Logarithmic)),
            "divide-blend" | "divide" => {
                Ok(Some(invers_core::models::InversionMode::DivideBlend))
            }
            "bw" | "blackandwhite" | "black-and-white" | "grayscale" | "mono" => {
                Ok(Some(invers_core::models::InversionMode::BlackAndWhite))
            }
            _ => Err(format!(
                "Unknown inversion mode: '{}'. Valid options: mask-aware (default), linear, log, divide-blend, bw",
                s
            )),
        },
    }
}

/// Parse neutral point ROI for density balance calculation
///
/// # Arguments
/// * `roi_str` - Optional ROI string in format "x,y,width,height"
///
/// # Returns
/// A NeutralPointSample with the parsed ROI, or None if not provided
pub fn parse_neutral_roi(
    roi_str: &Option<String>,
) -> Result<Option<invers_core::models::NeutralPointSample>, String> {
    match roi_str {
        Some(s) if !s.is_empty() => {
            let roi = parse_roi(s)?;
            Ok(Some(invers_core::models::NeutralPointSample {
                roi: Some(roi),
                neutral_rgb: [0.0, 0.0, 0.0], // Will be sampled during processing
                auto_detected: false,
            }))
        }
        _ => Ok(None),
    }
}
