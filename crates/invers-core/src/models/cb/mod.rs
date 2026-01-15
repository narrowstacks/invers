//! Curves-based pipeline types.
//!
//! Types for the curves-based processing pipeline, which uses curve-based
//! inversion with multiple white balance methods.

mod color_model;
mod enums;
mod histogram;
mod options;
mod presets;
mod tone_profile;

// Re-export all public types to maintain the existing public API
pub use color_model::{CbColorModel, CbColorModelParams, CbEnhancedProfile};
pub use enums::{CbLayerOrder, CbWbMethod, CbWbTonality};
pub use histogram::{CbChannelOrigins, CbHistogramAnalysis};
pub use options::CbOptions;
pub use presets::{CbEnginePreset, CbEngineSettings, CbFilmCharacter, CbSourceType, CbWbPreset};
pub use tone_profile::{CbToneProfile, CbToneProfileParams};
