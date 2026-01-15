//! ScanProfile struct defining capture source characteristics.

use serde::{Deserialize, Serialize};

use crate::models::convert_options::InversionMode;

use super::{DemosaicHints, HslAdjustments, WhiteBalanceHints};

/// Scan profile defining capture source characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanProfile {
    /// Name of the scan profile
    pub name: String,

    /// Capture source type (e.g., "dslr", "mirrorless", "flatbed")
    pub source_type: String,

    /// Typical white level for this source
    pub white_level: Option<f32>,

    /// Typical black level for this source
    pub black_level: Option<f32>,

    /// Demosaic hints (for RAW sources)
    pub demosaic_hints: Option<DemosaicHints>,

    /// White balance hints
    pub white_balance_hints: Option<WhiteBalanceHints>,

    /// HSL adjustments specific to this scanner/source
    /// Used to compensate for scanner color characteristics
    #[serde(default)]
    pub hsl_adjustments: Option<HslAdjustments>,

    /// Default per-channel gamma values [R, G, B]
    /// Applied during levels adjustment
    #[serde(default)]
    pub default_gamma: Option<[f32; 3]>,

    /// Preferred inversion mode for this scanner type
    #[serde(default)]
    pub preferred_inversion_mode: Option<InversionMode>,
}
